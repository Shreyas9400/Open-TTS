"""
TTS Studio
Local inference interface for:
  - SpragAI/qwen3-tts-emotion-tags (Inline emotion tag control + 9 preset voices)
  - Qwen/Qwen3-TTS-12Hz-1.7B-Base (Zero-shot custom voice cloning)
  - fishaudio/openaudio-s1-mini (Zero-shot voice cloning + inline emotion markers, optional install)
  - Ollama API integration (gemma2:2b auto-rephraser & emotion tagger)
"""

import os
import sys
import time
import json
import hashlib
import urllib.request
import urllib.error
import threading
import traceback
from pathlib import Path
from datetime import datetime

# Suppress HuggingFace symlinks warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Fix emoji/unicode printing on Windows cp1252 console
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import gradio as gr
import torch
import soundfile as sf
import numpy as np
import tempfile
import librosa

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Model IDs
# ─────────────────────────────────────────────────────────────────────────────

MODEL_EMOTION_TAGS = "SpragAI/qwen3-tts-emotion-tags"
MODEL_VOICE_CLONE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_FISH_SPEECH = "fishaudio/openaudio-s1-mini"

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

FISH_CHECKPOINTS_DIR = Path("fish_checkpoints")
FISH_CHECKPOINTS_DIR.mkdir(exist_ok=True)

SUPPORTED_LANGUAGES = [
    "Auto", "English", "Chinese", "Japanese", "Korean", "French",
    "German", "Spanish", "Italian", "Portuguese", "Russian",
    "Arabic", "Hindi", "Dutch", "Polish", "Turkish",
    "Swedish", "Danish", "Finnish", "Norwegian", "Czech",
]

PRESET_SPEAKERS = [
    ("ryan", "Ryan (Male, English)"),
    ("aiden", "Aiden (Male, English)"),
    ("serena", "Serena (Female, English)"),
    ("vivian", "Vivian (Female, English)"),
    ("eric", "Eric (Male, English)"),
    ("dylan", "Dylan (Male, English)"),
    ("ono_anna", "Ono Anna (Female, Japanese/English)"),
    ("sohee", "Sohee (Female, Korean/English)"),
    ("uncle_fu", "Uncle Fu (Male, Chinese/English)"),
]

EMOTION_TAGS = [
    ("😡 Angry", "[Angry]"),
    ("😢 Sad", "[Sad]"),
    ("😊 Happy", "[Happy]"),
    ("⚡ Fast", "[Fast]"),
    ("🍃 Gentle", "[Gentle]"),
    ("🥱 Tired", "[Tired]"),
    ("😨 Fearful", "[Fearful]"),
    ("🤢 Disgusted", "[Disgusted]"),
    ("😲 Surprised", "[Surprised]"),
]

# Fish-Speech (OpenAudio S1) uses parenthesised inline markers instead of
# bracketed tags, and — unlike the SpragAI model above — the same model also
# does zero-shot voice cloning, so these markers work together with a
# reference clip.
FISH_EMOTION_MARKERS = [
    ("😡 Angry", "(angry)"),
    ("😢 Sad", "(sad)"),
    ("😊 Happy", "(happy)"),
    ("🤩 Excited", "(excited)"),
    ("🍃 Gentle", "(gentle)"),
    ("🤫 Whisper", "(whispering)"),
    ("😂 Laughing", "(laughing)"),
    ("😭 Crying", "(crying)"),
    ("📢 Shouting", "(shouting)"),
    ("😮‍💨 Sighing", "(sighing)"),
]


def resolve_model_path(model_id: str) -> str:
    """
    Return local directory or snapshot if available to avoid Windows symlink issues.
    """
    # 1. Local model_cache directory (for Base model)
    if "Base" in model_id:
        local_dir = Path("model_cache")
        if (local_dir / "model.safetensors").exists() and (local_dir / "config.json").exists():
            print(f"[cache] Using local model_cache: {local_dir.resolve()}")
            return str(local_dir.resolve())

    # 2. HF Hub snapshot cache
    try:
        from huggingface_hub import constants
        cache_root = Path(constants.HF_HUB_CACHE)
        repo_slug = "models--" + model_id.replace("/", "--")
        snap_dir = cache_root / repo_slug / "snapshots"
        if snap_dir.is_dir():
            snaps = sorted(snap_dir.iterdir())
            for snap in reversed(snaps):
                if (snap / "model.safetensors").exists() and (snap / "config.json").exists():
                    local_path = str(snap)
                    print(f"[cache] Using valid HF hub snapshot: {local_path}")
                    return local_path
    except Exception as e:
        print(f"[cache] Could not resolve HF hub path for {model_id}: {e}")

    # 3. Fall back to hub download
    print(f"[cache] Falling back to hub download: {model_id}")
    return model_id


# ─────────────────────────────────────────────────────────────────────────────
# Model Manager (singleton)
# ─────────────────────────────────────────────────────────────────────────────

class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.current_model_id = None
        self.is_loaded = False
        self.is_loading = False
        self.load_error = None
        self.device_info = self._get_device_info()
        self.load_log = []
        # Cache of pre-computed voice-clone prompts (reference audio + text
        # already encoded into the model's conditioning features), keyed by
        # a hash of the reference clip + transcript. Avoids re-encoding the
        # same reference clip on every single generation call.
        self._voice_clone_prompt_cache = {}
        self.compile_enabled = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_device_info(self):
        info = {"cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            info["vram_free_gb"] = round(
                (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9, 2
            )
        return info

    def log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        self.load_log.append(entry)
        try:
            print(entry)
        except UnicodeEncodeError:
            print(entry.encode("ascii", errors="replace").decode("ascii"))
        return entry

    def load_model(self, model_id: str, dtype_choice: str, attn_impl: str, compile_enabled: bool = False):
        if self.is_loading:
            return False, "Model is already loading..."
        if self.is_loaded and self.current_model_id == model_id:
            return True, f"Model {model_id} already loaded."

        self.is_loading = True
        self.load_error = None
        self.load_log = []
        self.compile_enabled = compile_enabled
        # A new model invalidates any cached voice-clone prompts from the
        # previous model instance.
        self._voice_clone_prompt_cache = {}

        try:
            # If switching models, unload previous to free VRAM
            if self.model is not None:
                self.log(f"Unloading previous model: {self.current_model_id}…")
                del self.model
                self.model = None
                self.is_loaded = False
                self.current_model_id = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

            self.log(f"Loading model: {model_id}")
            self.log(f"dtype={dtype_choice}  attn={attn_impl}  compile={compile_enabled}")

            dtype_map = {
                "bfloat16 (Recommended)": torch.bfloat16,
                "float16": torch.float16,
                "float32 (CPU / no GPU)": torch.float32,
            }
            dtype = dtype_map.get(dtype_choice, torch.bfloat16)

            if torch.cuda.is_available():
                device_map = "cuda:0"
            else:
                device_map = "cpu"
                dtype = torch.float32
                attn_impl = "sdpa"
                self.log("⚠️ No CUDA detected – using CPU (attn forced to sdpa).")

            self.log("Importing qwen_tts …")
            from qwen_tts import Qwen3TTSModel

            model_path = resolve_model_path(model_id)
            self.log(f"Model path: {model_path}")

            def _try_load(chosen_attn):
                kwargs = dict(device_map=device_map, dtype=dtype)
                if chosen_attn and chosen_attn != "default":
                    kwargs["attn_implementation"] = chosen_attn
                self.log(f"from_pretrained kwargs: {kwargs}")

                import queue as _queue
                result_q = _queue.Queue()

                def _load():
                    try:
                        m = Qwen3TTSModel.from_pretrained(model_path, **kwargs)
                        result_q.put(("ok", m))
                    except BaseException as _e:
                        result_q.put(("err", _e, traceback.format_exc()))

                t = threading.Thread(target=_load, daemon=True)
                t.start()
                dots = 0
                while t.is_alive():
                    t.join(timeout=5)
                    dots += 1
                    self.log(f"  … still loading {'.' * dots}")
                return result_q.get_nowait()

            self.log("Loading weights into GPU memory…")
            result = _try_load(attn_impl)

            # flash_attention_2 needs the `flash-attn` wheel installed and a
            # compatible GPU; fall back to sdpa automatically instead of
            # hard-failing the whole load.
            if result[0] == "err" and attn_impl == "flash_attention_2":
                self.log(f"⚠️ flash_attention_2 failed ({result[1]}); retrying with sdpa…")
                result = _try_load("sdpa")

            if result[0] == "err":
                raise result[1]

            self.model = result[1]
            self.current_model_id = model_id
            self.is_loaded = True
            self.is_loading = False

            if compile_enabled:
                self._try_compile_model()

            self.log(f"✅ Model '{model_id}' loaded successfully!")
            return True, "\n".join(self.load_log)

        except BaseException as e:
            self.is_loading = False
            self.load_error = str(e)
            self.log(f"❌ Load failed: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
            return False, "\n".join(self.load_log)

    def _try_compile_model(self):
        """
        Best-effort torch.compile of the underlying causal LM. qwen_tts wraps
        the actual decoder under one of a few common attribute names
        depending on version, so we probe for it rather than assuming one.
        Never fatal: a failure here just means we run uncompiled.
        """
        candidate_attrs = ["model", "language_model", "llm", "transformer", "decoder"]
        for attr in candidate_attrs:
            submodule = getattr(self.model, attr, None)
            if isinstance(submodule, torch.nn.Module):
                try:
                    compiled = torch.compile(submodule, mode="reduce-overhead")
                    setattr(self.model, attr, compiled)
                    self.log(f"⚡ torch.compile applied to '{attr}' (first generation will warm up / be slower).")
                    return
                except Exception as e:
                    self.log(f"⚠️ torch.compile on '{attr}' failed: {e}")
                    return
        self.log("⚠️ torch.compile skipped: no compatible submodule found on this qwen_tts version.")

    def generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str = "Auto",
        temperature: float = 0.9,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ):
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Please load the model first.")

        kwargs = dict(
            text=text,
            speaker=speaker,
            language=None if language == "Auto" else language,
        )
        gen_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        try:
            wavs, sr = self.model.generate_custom_voice(**kwargs, **gen_kwargs)
        except TypeError:
            wavs, sr = self.model.generate_custom_voice(**kwargs)

        return wavs, sr

    def _get_or_create_voice_clone_prompt(self, ref_audio_path: str, ref_text: str):
        """
        Returns (prompt_object_or_None, was_cache_hit). Newer qwen_tts builds
        expose create_voice_clone_prompt(), which encodes the reference clip
        into reusable conditioning features once instead of on every single
        generate call — this is the single biggest speed win for repeated
        generations against the same reference voice. Older builds without
        that method return (None, False) so the caller falls back to passing
        raw ref_audio/ref_text on every call.
        """
        if not hasattr(self.model, "create_voice_clone_prompt"):
            return None, False

        with open(ref_audio_path, "rb") as f:
            audio_bytes = f.read()
        digest = hashlib.sha256(audio_bytes + ref_text.encode("utf-8")).hexdigest()
        cache_key = (self.current_model_id, digest)

        cached = self._voice_clone_prompt_cache.get(cache_key)
        if cached is not None:
            return cached, True

        processed_path = preprocess_audio(ref_audio_path)
        try:
            prompt = self.model.create_voice_clone_prompt(ref_audio=processed_path, ref_text=ref_text)
        finally:
            try:
                Path(processed_path).unlink(missing_ok=True)
            except Exception:
                pass

        self._voice_clone_prompt_cache[cache_key] = prompt
        return prompt, False

    def generate_voice_clone(
        self,
        text: str,
        language: str,
        ref_audio_path: str,
        ref_text: str,
        temperature: float = 0.9,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ):
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Please load the model first.")

        lang = None if language == "Auto" else language
        gen_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        prompt, cache_hit = self._get_or_create_voice_clone_prompt(ref_audio_path, ref_text)

        if prompt is not None:
            try:
                wavs, sr = self.model.generate_voice_clone(
                    text=text, language=lang, voice_clone_prompt=prompt, **gen_kwargs
                )
                return wavs, sr, cache_hit
            except TypeError:
                # This qwen_tts build doesn't accept voice_clone_prompt after all;
                # fall through to the uncached per-call path below.
                pass

        processed_path = preprocess_audio(ref_audio_path)
        try:
            kwargs = dict(text=text, language=lang, ref_audio=processed_path, ref_text=ref_text)
            try:
                wavs, sr = self.model.generate_voice_clone(**kwargs, **gen_kwargs)
            except TypeError:
                wavs, sr = self.model.generate_voice_clone(**kwargs)
        finally:
            try:
                Path(processed_path).unlink(missing_ok=True)
            except Exception:
                pass

        return wavs, sr, False


# ─────────────────────────────────────────────────────────────────────────────
# Fish-Speech (OpenAudio S1) Manager — separate backend/engine from qwen_tts.
# One model does both zero-shot voice cloning AND inline emotion markers, so
# it runs alongside (not instead of) the Qwen models above.
# ─────────────────────────────────────────────────────────────────────────────

class FishSpeechManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.engine = None
        self.current_model_id = None
        self.is_loaded = False
        self.is_loading = False
        self.load_log = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        self.load_log.append(entry)
        try:
            print(entry)
        except UnicodeEncodeError:
            print(entry.encode("ascii", errors="replace").decode("ascii"))
        return entry

    def load_model(self, model_id: str, compile_enabled: bool = False):
        if self.is_loading:
            return False, "Fish-Speech model is already loading..."
        if self.is_loaded and self.current_model_id == model_id:
            return True, f"Fish-Speech model {model_id} already loaded."

        self.is_loading = True
        self.load_log = []

        try:
            self.log("Importing fish_speech … (pip install fish-speech)")
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.dac.inference import load_model as load_decoder_model
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
            from huggingface_hub import snapshot_download

            device = "cuda" if torch.cuda.is_available() else "cpu"
            precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            if not torch.cuda.is_available():
                self.log("⚠️ No CUDA detected – Fish-Speech will run on CPU (slow).")

            local_dir = FISH_CHECKPOINTS_DIR / model_id.split("/")[-1]
            if not (local_dir / "codec.pth").exists():
                self.log(f"Downloading checkpoint for {model_id} → {local_dir} (first run only)…")
                snapshot_download(repo_id=model_id, local_dir=str(local_dir))
            else:
                self.log(f"Using cached checkpoint: {local_dir}")

            self.log("Launching text2semantic worker thread…")
            llama_queue = launch_thread_safe_queue(
                checkpoint_path=str(local_dir),
                device=device,
                precision=precision,
                compile=compile_enabled,
            )

            self.log("Loading DAC decoder…")
            decoder_model = load_decoder_model(
                config_name="modded_dac_vq",
                checkpoint_path=str(local_dir / "codec.pth"),
                device=device,
            )

            self.engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder_model,
                precision=precision,
                compile=compile_enabled,
            )
            self.current_model_id = model_id
            self.is_loaded = True
            self.is_loading = False
            self.log(f"✅ Fish-Speech model '{model_id}' loaded successfully!")
            return True, "\n".join(self.load_log)

        except BaseException as e:
            self.is_loading = False
            self.log(f"❌ Load failed: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
            return False, "\n".join(self.load_log)

    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        temperature: float = 0.8,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
    ):
        if not self.is_loaded or self.engine is None:
            raise RuntimeError("Fish-Speech model is not loaded. Please load it first.")

        from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

        with open(ref_audio_path, "rb") as f:
            ref_bytes = f.read()

        req = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=ref_bytes, text=ref_text)],
            use_memory_cache="on",  # lets the engine itself dedupe repeat references
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            format="wav",
        )

        sr, audio = None, None
        for result in self.engine.inference(req):
            if result.code == "error":
                raise result.error
            if result.code == "final":
                sr, audio = result.audio

        if audio is None:
            raise RuntimeError("Fish-Speech returned no audio.")

        return audio, sr


# ─────────────────────────────────────────────────────────────────────────────
# Audio preprocessing (librosa-based, no SoX required)
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SR = 16_000


def preprocess_audio(input_path: str) -> str:
    audio, orig_sr = librosa.load(input_path, sr=None, mono=True)
    if orig_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(OUTPUTS_DIR))
    sf.write(tmp.name, audio, TARGET_SR, subtype="PCM_16")
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Rephraser & Tagging Engine
# ─────────────────────────────────────────────────────────────────────────────

def test_ollama_connection(ollama_url: str):
    try:
        url = f"{ollama_url.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "Qwen3TTS-Studio"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            return f"🟢 Ollama Connected! Models: {', '.join(models) if models else 'None'}"
    except Exception as e:
        return f"🔴 Ollama Connection Failed: {e}"


def rephrase_with_ollama(text: str, ollama_url: str = "http://localhost:11434", ollama_model: str = "gemma2:2b"):
    if not text or not text.strip():
        raise gr.Error("Please enter some text in the box before rephrasing.")

    prompt = f"""You are an expert expressive voice director for TTS dialogue.
Your task is to take the input dialogue, polish it slightly if needed for spoken delivery, and insert inline emotion tags to guide vocal acting.

Valid emotion tags:
[Angry] [Sad] [Happy] [Fast] [Gentle] [Tired] [Fearful] [Disgusted] [Surprised]

Example 1:
Input: What on earth were you thinking? I trusted you. Don't look at me like that.
Output: [Angry] What on earth were you thinking? [Sad] I trusted you. [Angry] Don't look at me like that.

Example 2:
Input: Welcome! It's so wonderful to see you. Rest now, everything will be fine.
Output: [Happy] Welcome! It's so wonderful to see you. [Gentle] Rest now, everything will be fine.

Now annotate this text. Return ONLY the final spoken text with emotion tags, nothing else:
Input: {text.strip()}
Output:"""

    try:
        url = f"{ollama_url.rstrip('/')}/api/generate"
        payload = json.dumps({
            "model": ollama_model.strip(),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.6}
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            response_text = data.get("response", "").strip()
            # Clean possible markdown block wrap
            if response_text.startswith("```") and response_text.endswith("```"):
                response_text = "\n".join(response_text.splitlines()[1:-1]).strip()
            if (response_text.startswith('"') and response_text.endswith('"')) or (response_text.startswith("'") and response_text.endswith("'")):
                response_text = response_text[1:-1].strip()
            return response_text, f"✅ Rephrased with {ollama_model} ({datetime.now().strftime('%H:%M:%S')})"
    except urllib.error.URLError as e:
        raise gr.Error(f"Could not connect to Ollama at {ollama_url}. Is Ollama running? Error: {e}")
    except Exception as e:
        raise gr.Error(f"Ollama rephrasing error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def format_device_info(info: dict) -> str:
    if info.get("cuda_available"):
        return (
            f"🟢 GPU: **{info['device_name']}** | "
            f"VRAM: {info['vram_total_gb']} GB total, "
            f"{info['vram_free_gb']} GB free"
        )
    return "🟡 CPU mode (no CUDA GPU detected)"


def save_output(wavs, sr) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUTS_DIR / f"output_{ts}.wav"
    sf.write(str(out_path), wavs[0], sr)
    return str(out_path)


def get_output_history():
    files = sorted(OUTPUTS_DIR.glob("*.wav"), reverse=True)[:30]
    return [str(f) for f in files]


def refresh_history():
    files = get_output_history()
    return gr.update(choices=files, value=files[0] if files else None)


def append_emotion_tag(current_text: str, tag: str) -> str:
    current = (current_text or "").rstrip()
    if not current:
        return f"{tag} "
    return f"{current} {tag} "


# ─────────────────────────────────────────────────────────────────────────────
# Gradio event handlers
# ─────────────────────────────────────────────────────────────────────────────

def load_model_handler(model_id, dtype_choice, attn_impl, compile_enabled):
    mgr = ModelManager.get_instance()
    if mgr.is_loaded and mgr.current_model_id == model_id:
        yield (
            gr.update(value=f"✅ {model_id} already loaded!", variant="secondary"),
            gr.update(value=f"✅ Model {model_id} is ready for inference."),
            gr.update(interactive=True),
            gr.update(value=f"🟢 Active Model: **{model_id}**"),
        )
        return

    yield (
        gr.update(value=f"⏳ Loading {model_id}…", variant="secondary", interactive=False),
        gr.update(value=f"Starting load for {model_id}…"),
        gr.update(interactive=False),
        gr.update(value=f"🟡 Loading **{model_id}**…"),
    )

    success, log_text = mgr.load_model(model_id, dtype_choice, attn_impl, compile_enabled)

    if success:
        yield (
            gr.update(value="✅ Model Loaded!", variant="primary", interactive=True),
            gr.update(value=log_text),
            gr.update(interactive=True),
            gr.update(value=f"🟢 Active Model: **{model_id}**"),
        )
    else:
        yield (
            gr.update(value="❌ Load Failed – retry", variant="stop", interactive=True),
            gr.update(value=log_text),
            gr.update(interactive=False),
            gr.update(value="🔴 Model Load Failed"),
        )


def generate_emotion_speech_handler(
    text,
    speaker,
    language,
    temperature,
    top_p,
    rep_penalty,
    progress=gr.Progress(track_tqdm=True),
):
    mgr = ModelManager.get_instance()

    if not mgr.is_loaded:
        raise gr.Error("No model loaded! Please go to '⚙️ Setup & Model' and load 'SpragAI/qwen3-tts-emotion-tags'.")

    if mgr.current_model_id != MODEL_EMOTION_TAGS:
        raise gr.Error(
            f"Active model is '{mgr.current_model_id}'. "
            f"Please go to '⚙️ Setup & Model' and switch to '{MODEL_EMOTION_TAGS}'."
        )

    if not text or not text.strip():
        raise gr.Error("Please enter some text to synthesize.")

    progress(0, desc="Generating expressive speech…")
    start_time = time.time()

    wavs, sr = mgr.generate_custom_voice(
        text=text.strip(),
        speaker=speaker,
        language=language,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rep_penalty,
    )

    elapsed = round(time.time() - start_time, 2)
    progress(1.0, desc="Done!")

    out_path = save_output(wavs, sr)
    audio_duration = round(len(wavs[0]) / sr, 2)
    rtf = round(elapsed / audio_duration, 3) if audio_duration > 0 else 0

    stats = (
        f"✅ Generated in **{elapsed}s** | "
        f"Duration: **{audio_duration}s** | "
        f"RTF: **{rtf}** | "
        f"Speaker: `{speaker}` | "
        f"Saved → `{out_path}`"
    )

    return out_path, stats


def generate_voice_clone_handler(
    text,
    language,
    ref_audio,
    ref_text,
    temperature,
    top_p,
    rep_penalty,
    progress=gr.Progress(track_tqdm=True),
):
    mgr = ModelManager.get_instance()

    if not mgr.is_loaded:
        raise gr.Error("No model loaded! Please go to '⚙️ Setup & Model' and load 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'.")

    if mgr.current_model_id != MODEL_VOICE_CLONE:
        raise gr.Error(
            f"Active model is '{mgr.current_model_id}'. "
            f"Please go to '⚙️ Setup & Model' and switch to '{MODEL_VOICE_CLONE}'."
        )

    if not text or not text.strip():
        raise gr.Error("Please enter some text to synthesize.")

    if ref_audio is None:
        raise gr.Error("Please upload a reference audio file for voice cloning.")

    if not ref_text or not ref_text.strip():
        raise gr.Error("Please provide the reference text (transcript of the reference audio).")

    progress(0, desc="Cloning voice & generating speech…")
    start_time = time.time()

    wavs, sr, cache_hit = mgr.generate_voice_clone(
        text=text.strip(),
        language=language,
        ref_audio_path=ref_audio,
        ref_text=ref_text.strip(),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rep_penalty,
    )

    elapsed = round(time.time() - start_time, 2)
    progress(1.0, desc="Done!")

    out_path = save_output(wavs, sr)
    audio_duration = round(len(wavs[0]) / sr, 2)
    rtf = round(elapsed / audio_duration, 3) if audio_duration > 0 else 0
    cache_note = "reused (fast)" if cache_hit else "computed"

    stats = (
        f"✅ Generated in **{elapsed}s** | "
        f"Duration: **{audio_duration}s** | "
        f"RTF: **{rtf}** | "
        f"Voice prompt: **{cache_note}** | "
        f"Saved → `{out_path}`"
    )

    return out_path, stats


def load_fish_speech_handler(model_id, compile_enabled):
    mgr = FishSpeechManager.get_instance()
    if mgr.is_loaded and mgr.current_model_id == model_id:
        yield (
            gr.update(value=f"✅ {model_id} already loaded!", variant="secondary"),
            gr.update(value=f"✅ Model {model_id} is ready for inference."),
            gr.update(interactive=True),
        )
        return

    yield (
        gr.update(value=f"⏳ Loading {model_id}…", variant="secondary", interactive=False),
        gr.update(value=f"Starting load for {model_id}… (first run also downloads the checkpoint)"),
        gr.update(interactive=False),
    )

    success, log_text = mgr.load_model(model_id, compile_enabled)

    if success:
        yield (
            gr.update(value="✅ Model Loaded!", variant="primary", interactive=True),
            gr.update(value=log_text),
            gr.update(interactive=True),
        )
    else:
        yield (
            gr.update(value="❌ Load Failed – retry", variant="stop", interactive=True),
            gr.update(value=log_text),
            gr.update(interactive=False),
        )


def generate_fish_speech_handler(
    text,
    ref_audio,
    ref_text,
    temperature,
    top_p,
    rep_penalty,
    progress=gr.Progress(track_tqdm=True),
):
    mgr = FishSpeechManager.get_instance()

    if not mgr.is_loaded:
        raise gr.Error("No Fish-Speech model loaded! Load it in this tab's Setup section first.")

    if not text or not text.strip():
        raise gr.Error("Please enter some text to synthesize.")

    if ref_audio is None:
        raise gr.Error("Please upload a reference audio file for voice cloning.")

    if not ref_text or not ref_text.strip():
        raise gr.Error("Please provide the reference text (transcript of the reference audio).")

    progress(0, desc="Cloning voice & generating expressive speech…")
    start_time = time.time()

    audio, sr = mgr.generate(
        text=text.strip(),
        ref_audio_path=ref_audio,
        ref_text=ref_text.strip(),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rep_penalty,
    )

    elapsed = round(time.time() - start_time, 2)
    progress(1.0, desc="Done!")

    out_path = save_output([audio], sr)
    audio_duration = round(len(audio) / sr, 2)
    rtf = round(elapsed / audio_duration, 3) if audio_duration > 0 else 0

    stats = (
        f"✅ Generated in **{elapsed}s** | "
        f"Duration: **{audio_duration}s** | "
        f"RTF: **{rtf}** | "
        f"Saved → `{out_path}`"
    )

    return out_path, stats


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-primary:      #0f1116;
  --bg-secondary:    #161922;
  --bg-card:         #1b1f2a;
  --border-color:    #2a2f3d;
  --text-primary:    #e6e8f0;
  --text-secondary:  #9aa1b8;
  --text-muted:      #6a7290;
  --accent:          #5b7cfa;
  --accent-strong:   #4a68e0;
  --success:         #34c98f;
  --warn:            #e0a83d;
  --radius-sm:       8px;
  --radius-md:       12px;
  --radius-lg:       16px;
  --font-sans:       'Inter', sans-serif;
  --font-mono:       'JetBrains Mono', monospace;
}

body, .gradio-container {
  background: var(--bg-primary) !important;
  font-family: var(--font-sans) !important;
  color: var(--text-primary) !important;
}

/* Calm, flat hero — no gradients, no animation, just a clear title + one-line summary */
.hero-header {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  padding: 20px 28px;
  margin-bottom: 16px;
}
.hero-header h1 {
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0 0 4px 0;
  letter-spacing: -0.01em;
}
.hero-header p {
  color: var(--text-secondary);
  font-size: 0.9rem;
  margin: 0;
  line-height: 1.5;
}

.section-title {
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border-color);
}

/* Emotion / marker quick-insert buttons */
.tag-bar-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 8px 0 14px 0;
  padding: 10px;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
}

/* Ollama assistant box — same neutral card style as everything else */
.ollama-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 14px 16px;
  margin-bottom: 14px;
}

/* Tab bar → a rounded segmented control instead of Gradio's default
   underlined-link row (targets ARIA roles, which are stable across
   Gradio versions, rather than internal class names, which aren't). */
.tab-wrapper {
  display: inline-flex !important;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 999px;
  padding: 4px;
  margin-bottom: 4px;
}
[role="tablist"] {
  gap: 2px;
}
button[role="tab"] {
  border: none !important;
  background: transparent !important;
  border-radius: 999px !important;
  padding: 8px 18px !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  color: var(--text-secondary) !important;
  box-shadow: none !important;
  transition: background 0.15s ease, color 0.15s ease !important;
}
button[role="tab"].selected {
  background: var(--accent) !important;
  color: white !important;
  font-weight: 600 !important;
}
.overflow-menu button {
  color: var(--text-secondary) !important;
}

/* Each tab panel is freshly mounted on switch, so this replays every time */
@keyframes ttsFadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0); }
}
[role="tabpanel"] {
  animation: ttsFadeIn 0.2s ease;
}

label {
  color: var(--text-secondary) !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container input[role="combobox"] {
  background: var(--bg-secondary) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-sans) !important;
  font-size: 0.92rem;
}
.gradio-container textarea:focus,
.gradio-container input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(91,124,250,0.2) !important;
  outline: none !important;
}

/* Recolor Gradio's own built-in pending/eta spinner to match our accent */
[data-testid="status-tracker"] {
  color: var(--accent) !important;
}

/* One consistent button style everywhere — no per-section gradients */
button.primary {
  background: var(--accent) !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  color: white !important;
  font-weight: 600 !important;
  font-family: var(--font-sans) !important;
  box-shadow: none !important;
  transition: background 0.15s ease, transform 0.1s ease !important;
}
button.primary:hover {
  background: var(--accent-strong) !important;
}
button.primary:active {
  transform: scale(0.98);
}
button.secondary {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-weight: 500 !important;
  font-family: var(--font-sans) !important;
  box-shadow: none !important;
}
button.secondary:hover {
  border-color: var(--accent) !important;
}
button.stop {
  background: transparent !important;
  border: 1px solid #e05a5a !important;
  color: #e05a5a !important;
  border-radius: var(--radius-sm) !important;
  box-shadow: none !important;
}

/* NOTE: elem_classes on gr.Button land directly on the <button> element
   itself (not a wrapper div), and these need to be at least as specific
   as (and declared after) the button.primary/secondary rules above to
   actually win, since emotion-tag chips carry the "secondary" variant
   class and the Ollama button carries "primary". */
button.emotion-btn {
  background: transparent !important;
  border: 1px solid var(--border-color) !important;
  color: var(--text-secondary) !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  padding: 5px 12px !important;
  border-radius: 20px !important;
  box-shadow: none !important;
  transition: border-color 0.15s ease, color 0.15s ease !important;
}
button.emotion-btn:hover {
  border-color: var(--accent) !important;
  color: var(--text-primary) !important;
}
button.ollama-rephrase-btn {
  background: var(--accent) !important;
  color: white !important;
  font-weight: 600 !important;
  border-radius: var(--radius-sm) !important;
  border: none !important;
  box-shadow: none !important;
  transition: background 0.15s ease !important;
}
button.ollama-rephrase-btn:hover {
  background: var(--accent-strong) !important;
}

.active-model-badge {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-left: 3px solid var(--success);
  border-radius: var(--radius-sm);
  padding: 8px 16px;
  font-size: 0.85rem;
  color: var(--text-primary);
  font-family: var(--font-mono);
  display: inline-block;
  margin-bottom: 12px;
}

.tip-box, .warn-box {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  font-size: 0.84rem;
  color: var(--text-secondary);
  line-height: 1.55;
}
.warn-box { border-left: 3px solid var(--warn); }
.tip-box { border-left: 3px solid var(--accent); }
"""

# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_ui():
    mgr = ModelManager.get_instance()
    dev_info_str = format_device_info(mgr.device_info)

    with gr.Blocks(title="TTS Studio") as demo:

        # ── Hero ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="hero-header">
          <h1>TTS Studio</h1>
          <p>
            Emotion tags (SpragAI) · Zero-shot voice cloning (Qwen3-TTS) ·
            Clone + emotion in one model (Fish-Speech) · Ollama auto-rephrasing
          </p>
        </div>
        """)

        # ── Global Status Banner ─────────────────────────────────────────────
        status_banner = gr.Markdown(
            "🟡 No model loaded yet. Go to **'⚙️ Setup & Model'** to load either Emotion or Voice Clone model.",
            elem_classes=["active-model-badge"]
        )

        with gr.Tabs():

            # ═════════════════════════════════════════════════════════════════
            # TAB 1: Emotion Tag Studio (SpragAI)
            # ═════════════════════════════════════════════════════════════════
            with gr.TabItem("🎭 Emotion Studio (SpragAI)"):
                with gr.Row(equal_height=False):

                    # Left Column: Inputs & Tag Bar
                    with gr.Column(scale=5):
                        gr.HTML('<div class="section-title">🎭 Emotion Script & Inline Tags</div>')

                        # Ollama Rephraser Box
                        with gr.Group(elem_classes=["ollama-card"]):
                            gr.Markdown(
                                "🤖 **AI Emotion Director (Ollama — gemma2:2b)**\n"
                                "Click below to automatically rephrase your text and insert appropriate emotion tags matching the tone."
                            )
                            with gr.Row():
                                ollama_url_input = gr.Textbox(
                                    label="Ollama API URL",
                                    value="http://localhost:11434",
                                    lines=1,
                                    scale=3,
                                )
                                ollama_model_input = gr.Textbox(
                                    label="Ollama Model",
                                    value="gemma2:2b",
                                    lines=1,
                                    scale=2,
                                )
                            with gr.Row():
                                ollama_rephrase_btn = gr.Button(
                                    "✨ Auto-Tag & Rephrase with Ollama",
                                    variant="primary",
                                    elem_classes=["ollama-rephrase-btn"],
                                    scale=3,
                                )
                                ollama_status = gr.Markdown("", scale=2)

                        emotion_target_text = gr.Textbox(
                            label="Target Text (use inline tags like [Happy], [Angry], etc.)",
                            placeholder="Example: [Happy] Welcome everyone! [Sad] Sadly, today is our last day here. [Gentle] But we will cherish the memories.",
                            value="[Happy] Welcome everyone! [Sad] Sadly, today is our last day here. [Gentle] But we will always cherish the memories.",
                            lines=6,
                            max_lines=20,
                            elem_id="emotion-target-text",
                        )

                        # Clickable Emotion Tag Chips
                        gr.HTML('<div class="section-title" style="margin-top:12px;">🏷️ Quick Insert Emotion Tags</div>')
                        with gr.Row(elem_classes=["tag-bar-wrap"]):
                            tag_btns = []
                            for label, tag_val in EMOTION_TAGS:
                                b = gr.Button(label, size="sm", elem_classes=["emotion-btn"])
                                tag_btns.append((b, tag_val))

                        for btn, tag_val in tag_btns:
                            btn.click(
                                fn=lambda txt, t=tag_val: append_emotion_tag(txt, t),
                                inputs=[emotion_target_text],
                                outputs=[emotion_target_text],
                            )

                        with gr.Row():
                            emotion_speaker = gr.Dropdown(
                                label="Preset Speaker Voice",
                                choices=[s[0] for s in PRESET_SPEAKERS],
                                value="ryan",
                                info="Select from the 9 fine-tuned character voices",
                            )
                            emotion_language = gr.Dropdown(
                                label="Language",
                                choices=SUPPORTED_LANGUAGES,
                                value="Auto",
                            )

                        gr.HTML("""
                        <div class="tip-box" style="margin-top:14px;">
                          💡 <strong>Emotion Tag Tips:</strong><br>
                          • Supported tags: <code>[Angry]</code>, <code>[Sad]</code>, <code>[Happy]</code>, <code>[Fast]</code>, <code>[Gentle]</code>, <code>[Tired]</code>, <code>[Fearful]</code>, <code>[Disgusted]</code>, <code>[Surprised]</code><br>
                          • Tags can be switched mid-sentence or per sentence.<br>
                          • Requires <strong>SpragAI/qwen3-tts-emotion-tags</strong> loaded in the Setup tab.
                        </div>
                        """)

                    # Right Column: Controls & Output
                    with gr.Column(scale=4):
                        gr.HTML('<div class="section-title">⚙️ Generation Parameters</div>')
                        with gr.Accordion("Fine-tuning controls", open=False):
                            em_temperature = gr.Slider(
                                label="Temperature",
                                minimum=0.1, maximum=2.0, value=0.9, step=0.05,
                                info="Higher = more dynamic / varied expressiveness",
                            )
                            em_top_p = gr.Slider(
                                label="Top-p",
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                            )
                            em_rep_penalty = gr.Slider(
                                label="Repetition penalty",
                                minimum=1.0, maximum=2.0, value=1.0, step=0.05,
                            )

                        generate_emotion_btn = gr.Button(
                            "🎭 Generate Emotion Speech",
                            variant="primary",
                            size="lg",
                            elem_id="generate-emotion-btn",
                        )

                        gr.HTML('<div class="section-title" style="margin-top:20px;">🔊 Output Audio</div>')
                        emotion_output_audio = gr.Audio(
                            label="Generated Speech",
                            type="filepath",
                            interactive=False,
                            elem_id="emotion-output-audio",
                        )
                        emotion_gen_stats = gr.Markdown(value="")

                # Wire Ollama Rephrase
                ollama_rephrase_btn.click(
                    fn=rephrase_with_ollama,
                    inputs=[emotion_target_text, ollama_url_input, ollama_model_input],
                    outputs=[emotion_target_text, ollama_status],
                )

                # Wire Emotion Generation
                generate_emotion_btn.click(
                    fn=generate_emotion_speech_handler,
                    inputs=[
                        emotion_target_text,
                        emotion_speaker,
                        emotion_language,
                        em_temperature,
                        em_top_p,
                        em_rep_penalty,
                    ],
                    outputs=[emotion_output_audio, emotion_gen_stats],
                )

            # ═════════════════════════════════════════════════════════════════
            # TAB 2: Zero-Shot Voice Clone Studio (Base Model)
            # ═════════════════════════════════════════════════════════════════
            with gr.TabItem("🎙️ Voice Clone (Base Model)"):
                with gr.Row(equal_height=False):

                    # Left Column: Inputs
                    with gr.Column(scale=5):
                        gr.HTML('<div class="section-title">📝 Target Text</div>')
                        clone_target_text = gr.Textbox(
                            label="Text to synthesize in cloned voice",
                            placeholder="Enter the text you want spoken in your reference voice…",
                            lines=5,
                            max_lines=20,
                            elem_id="clone-target-text",
                        )

                        with gr.Row():
                            clone_language = gr.Dropdown(
                                label="Language",
                                choices=SUPPORTED_LANGUAGES,
                                value="English",
                            )
                            clone_char_count = gr.Number(
                                label="Characters",
                                value=0,
                                interactive=False,
                                precision=0,
                            )

                        clone_target_text.change(
                            fn=lambda t: len(t or ""),
                            inputs=clone_target_text,
                            outputs=clone_char_count,
                        )

                        gr.HTML('<div class="section-title" style="margin-top:20px;">🎤 Reference Voice Sample</div>')
                        ref_audio = gr.Audio(
                            label="Reference Audio (WAV/MP3 — 5–15 sec recommended)",
                            type="filepath",
                            elem_id="ref-audio",
                        )
                        ref_text = gr.Textbox(
                            label="Reference Transcript (exact words spoken in audio)",
                            placeholder="Transcribe word-for-word what is said in the reference audio clip.",
                            lines=3,
                            elem_id="ref-text",
                        )

                        gr.HTML("""
                        <div class="tip-box">
                          💡 <strong>Voice Cloning Best Practices:</strong><br>
                          • Use 5–15 seconds of clean, isolated speech with no background noise.<br>
                          • The transcript must match the reference audio word-for-word.<br>
                          • Tone and emotion in the output will reflect the speaker's tone in the reference WAV.
                        </div>
                        """)

                    # Right Column: Controls & Output
                    with gr.Column(scale=4):
                        gr.HTML('<div class="section-title">⚙️ Generation Parameters</div>')
                        with gr.Accordion("Advanced parameters", open=False):
                            clone_temp = gr.Slider(
                                label="Temperature",
                                minimum=0.1, maximum=2.0, value=0.9, step=0.05,
                            )
                            clone_top_p = gr.Slider(
                                label="Top-p",
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                            )
                            clone_rep_penalty = gr.Slider(
                                label="Repetition penalty",
                                minimum=1.0, maximum=2.0, value=1.0, step=0.05,
                            )

                        generate_clone_btn = gr.Button(
                            "✨ Clone Voice & Generate",
                            variant="primary",
                            size="lg",
                            elem_id="generate-clone-btn",
                        )

                        gr.HTML('<div class="section-title" style="margin-top:20px;">🔊 Output Audio</div>')
                        clone_output_audio = gr.Audio(
                            label="Generated Speech",
                            type="filepath",
                            interactive=False,
                            elem_id="clone-output-audio",
                        )
                        clone_gen_stats = gr.Markdown(value="")

                generate_clone_btn.click(
                    fn=generate_voice_clone_handler,
                    inputs=[
                        clone_target_text,
                        clone_language,
                        ref_audio,
                        ref_text,
                        clone_temp,
                        clone_top_p,
                        clone_rep_penalty,
                    ],
                    outputs=[clone_output_audio, clone_gen_stats],
                )

            # ═════════════════════════════════════════════════════════════════
            # TAB 3: Fish-Speech (OpenAudio S1) — voice cloning + emotion markers
            # in a single model
            # ═════════════════════════════════════════════════════════════════
            with gr.TabItem("🐟 Clone + Emotion (Fish-Speech)"):
                gr.Markdown(
                    "One model does both: zero-shot voice cloning **and** inline emotion/tone markers. "
                    "Model weights are released under CC-BY-NC-SA-4.0 (non-commercial). "
                    "Needs `pip install fish-speech` — see the Guide tab."
                )
                with gr.Row(equal_height=False):

                    with gr.Column(scale=5):
                        gr.HTML('<div class="section-title">📝 Target Text & Emotion Markers</div>')
                        fish_target_text = gr.Textbox(
                            label="Text to synthesize (use markers like (happy), (whispering), etc.)",
                            placeholder="(happy) Welcome everyone! (gentle) We're really glad you could join us today.",
                            lines=6,
                            max_lines=20,
                            elem_id="fish-target-text",
                        )

                        gr.HTML('<div class="section-title" style="margin-top:12px;">🏷️ Quick Insert Markers</div>')
                        with gr.Row(elem_classes=["tag-bar-wrap"]):
                            fish_tag_btns = []
                            for label, marker_val in FISH_EMOTION_MARKERS:
                                b = gr.Button(label, size="sm", elem_classes=["emotion-btn"])
                                fish_tag_btns.append((b, marker_val))

                        for btn, marker_val in fish_tag_btns:
                            btn.click(
                                fn=lambda txt, t=marker_val: append_emotion_tag(txt, t),
                                inputs=[fish_target_text],
                                outputs=[fish_target_text],
                            )

                        gr.HTML('<div class="section-title" style="margin-top:20px;">🎤 Reference Voice Sample</div>')
                        fish_ref_audio = gr.Audio(
                            label="Reference Audio (10-30 sec recommended)",
                            type="filepath",
                            elem_id="fish-ref-audio",
                        )
                        fish_ref_text = gr.Textbox(
                            label="Reference Transcript (exact words spoken in audio)",
                            lines=3,
                            elem_id="fish-ref-text",
                        )

                    with gr.Column(scale=4):
                        gr.HTML('<div class="section-title">⚙️ Generation Parameters</div>')
                        with gr.Accordion("Advanced parameters", open=False):
                            fish_temp = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, value=0.8, step=0.05)
                            fish_top_p = gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, value=0.8, step=0.05)
                            fish_rep_penalty = gr.Slider(label="Repetition penalty", minimum=0.9, maximum=2.0, value=1.1, step=0.05)

                        gr.HTML('<div class="section-title" style="margin-top:20px;">📦 Load Model</div>')
                        fish_model_choice = gr.Dropdown(
                            label="Fish-Speech variant",
                            choices=[
                                (f"OpenAudio S1-mini ({MODEL_FISH_SPEECH})", MODEL_FISH_SPEECH),
                            ],
                            value=MODEL_FISH_SPEECH,
                            info="Gated on Hugging Face: accept the terms on the model page, then run `hf auth login`.",
                        )
                        fish_compile_checkbox = gr.Checkbox(label="⚡ Enable compile (experimental)", value=False)
                        fish_load_btn = gr.Button("🚀 Load Fish-Speech Model", variant="primary")
                        fish_load_log = gr.Textbox(
                            label="Load Log",
                            value="Fish-Speech model not loaded yet.",
                            lines=6,
                            interactive=False,
                        )

                        generate_fish_btn = gr.Button(
                            "🐟 Clone Voice & Generate",
                            variant="primary",
                            size="lg",
                        )

                        gr.HTML('<div class="section-title" style="margin-top:20px;">🔊 Output Audio</div>')
                        fish_output_audio = gr.Audio(
                            label="Generated Speech",
                            type="filepath",
                            interactive=False,
                        )
                        fish_gen_stats = gr.Markdown(value="")

                fish_load_btn.click(
                    fn=load_fish_speech_handler,
                    inputs=[fish_model_choice, fish_compile_checkbox],
                    outputs=[fish_load_btn, fish_load_log, fish_load_btn],
                )

                generate_fish_btn.click(
                    fn=generate_fish_speech_handler,
                    inputs=[
                        fish_target_text,
                        fish_ref_audio,
                        fish_ref_text,
                        fish_temp,
                        fish_top_p,
                        fish_rep_penalty,
                    ],
                    outputs=[fish_output_audio, fish_gen_stats],
                )

            # ═════════════════════════════════════════════════════════════════
            # TAB 4: Setup & Model Manager
            # ═════════════════════════════════════════════════════════════════
            with gr.TabItem("⚙️ Setup & Model"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.HTML('<div class="section-title">🖥️ Hardware</div>')
                        gr.Markdown(dev_info_str, elem_id="hw-info")

                        gr.HTML('<div class="section-title" style="margin-top:20px;">📦 Select Model to Load</div>')
                        model_choice = gr.Dropdown(
                            label="Model Architecture",
                            choices=[
                                (f"🎭 Emotion Tags Model ({MODEL_EMOTION_TAGS})", MODEL_EMOTION_TAGS),
                                (f"🎙️ Voice Clone Base Model ({MODEL_VOICE_CLONE})", MODEL_VOICE_CLONE),
                            ],
                            value=MODEL_EMOTION_TAGS,
                            info="Switch between the inline emotion model and the zero-shot voice clone base model",
                        )

                        dtype_choice = gr.Dropdown(
                            label="Data type (dtype)",
                            choices=[
                                "bfloat16 (Recommended)",
                                "float16",
                                "float32 (CPU / no GPU)",
                            ],
                            value="bfloat16 (Recommended)" if torch.cuda.is_available() else "float32 (CPU / no GPU)",
                        )
                        attn_impl = gr.Dropdown(
                            label="Attention implementation",
                            choices=["flash_attention_2", "sdpa", "default"],
                            value="flash_attention_2",
                            info="Falls back to sdpa automatically if flash-attn isn't installed / GPU isn't compatible.",
                        )
                        compile_checkbox = gr.Checkbox(
                            label="⚡ Enable torch.compile (experimental)",
                            value=False,
                            info="Speeds up steady-state generation after a slower first call. Safe to leave off if you hit compile errors.",
                        )

                        load_btn = gr.Button(
                            "🚀 Load Selected Model",
                            variant="primary",
                            elem_id="load-btn",
                        )

                        gr.HTML('<div class="section-title" style="margin-top:24px;">🤖 Ollama API Check</div>')
                        with gr.Row():
                            test_ollama_url = gr.Textbox(
                                label="Ollama Endpoint",
                                value="http://localhost:11434",
                                scale=3,
                            )
                            test_ollama_btn = gr.Button("🔍 Test Ollama", scale=1)
                        ollama_test_result = gr.Markdown("Click 'Test Ollama' to verify local Ollama service.")
                        test_ollama_btn.click(
                            fn=test_ollama_connection,
                            inputs=[test_ollama_url],
                            outputs=[ollama_test_result],
                        )

                    with gr.Column(scale=5):
                        gr.HTML('<div class="section-title">📋 Load Log</div>')
                        load_log = gr.Textbox(
                            label="",
                            value="Model not loaded. Select a model and click 'Load Selected Model' to begin.",
                            lines=18,
                            interactive=False,
                            elem_id="load-log",
                        )

                load_btn.click(
                    fn=load_model_handler,
                    inputs=[model_choice, dtype_choice, attn_impl, compile_checkbox],
                    outputs=[load_btn, load_log, load_btn, status_banner],
                )

            # ═════════════════════════════════════════════════════════════════
            # TAB 5: Output History
            # ═════════════════════════════════════════════════════════════════
            with gr.TabItem("📂 Output History"):
                gr.HTML('<div class="section-title">Recent Generations</div>')
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh History", variant="secondary")
                history_list = gr.Dropdown(
                    label="Select a file to preview",
                    choices=get_output_history(),
                    interactive=True,
                    elem_id="history-dropdown",
                )
                history_player = gr.Audio(
                    label="Preview Audio",
                    type="filepath",
                    interactive=False,
                    elem_id="history-player",
                )
                history_list.change(fn=lambda p: p, inputs=history_list, outputs=history_player)
                refresh_btn.click(fn=refresh_history, inputs=[], outputs=history_list)

            # ═════════════════════════════════════════════════════════════════
            # TAB 6: Documentation & Guide
            # ═════════════════════════════════════════════════════════════════
            with gr.TabItem("📖 Guide"):
                gr.Markdown("""
## Qwen3-TTS Voice & Emotion Studio — Comprehensive Guide

### 🎭 1. Emotion Tag Mode (`SpragAI/qwen3-tts-emotion-tags`)
This fine-tuned model introduces inline emotion tags across 9 preset voices.

#### Available Emotion Tags:
- `[Angry]` — Intense, confrontational delivery
- `[Sad]` — Melancholic, quiet or mournful tone
- `[Happy]` — Cheerful, uplifting, energetic
- `[Fast]` — Rapid, urgent pacing
- `[Gentle]` — Soft, calming, intimate delivery
- `[Tired]` — Sluggish, weary, exhausted delivery
- `[Fearful]` — Trembling, anxious, panicked
- `[Disgusted]` — Contemptuous, recoiling tone
- `[Surprised]` — Astonished, wide-eyed emphasis

#### Multi-Emotion Dialogue Example:
```text
[Angry] Why didn't anyone tell me? [Sad] I felt so alone. [Gentle] But thank you for listening now.
```

---

### 🤖 2. Ollama AI Rephraser (`gemma2:2b`)
- Connects directly to your local Ollama server at `http://localhost:11434`.
- Automatically analyzes your raw script and annotates sentences with the best matching emotion tags.
- Uses `gemma2:2b` for fast, lightweight local rephrasing without leaving your machine.

---

### 🎙️ 3. Zero-Shot Voice Clone Mode (`Qwen3-TTS-12Hz-1.7B-Base`)
- Clone any voice using a 5–15 second clean reference audio clip.
- Accompany the audio with an exact word-for-word transcript.
- The base model clones the voice's pitch, timbre, and natural delivery from the reference.
- Repeat generations against the **same** reference clip reuse a cached voice prompt instead of
  re-encoding the reference audio every time — look for "Voice prompt: reused (fast)" in the stats line.

---

### 🐟 4. Fish-Speech / OpenAudio S1 — Clone + Emotion in one model
Unlike the two models above, Fish-Speech does **both** zero-shot voice cloning and inline
emotion/tone markers in a single pass — no need to pick between cloning and expressiveness.

Install it separately (own dependency stack, heavier than the base install):
```bash
pip install fish-speech
```
The first "Load Fish-Speech Model" click downloads the checkpoint automatically (a few GB) into
`fish_checkpoints/`. Model weights are released under **CC-BY-NC-SA-4.0** (non-commercial use only).

#### Marker examples:
`(happy)` `(sad)` `(angry)` `(excited)` `(gentle)` `(whispering)` `(laughing)` `(crying)` `(shouting)` `(sighing)`

```text
(happy) Welcome everyone! (whispering) Come a little closer... (laughing) just kidding!
```

---

### ⚡ 5. Speed tips
- **Attention backend**: `flash_attention_2` is fastest on Ampere+ GPUs; the app now falls back to
  `sdpa` automatically if flash-attn isn't installed, so it's safe to leave on.
- **torch.compile**: toggle it on in the Setup tab for steady-state speedups; the first generation
  after loading will be noticeably slower while it compiles.
- **Voice-clone prompt caching**: reusing the same reference clip across multiple generations skips
  re-encoding it every time (Base model + Fish-Speech's own reference cache).
- **dtype**: `bfloat16` is the fastest safe option on modern GPUs; `float16` is a fallback for older cards.
                """)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  TTS Studio")
    print("  Supported Models:")
    print(f"    - {MODEL_EMOTION_TAGS} (Emotion Tags)")
    print(f"    - {MODEL_VOICE_CLONE} (Voice Clone)")
    print(f"    - {MODEL_FISH_SPEECH} (Voice Clone + Emotion, optional install)")
    print("=" * 60)

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
    )
