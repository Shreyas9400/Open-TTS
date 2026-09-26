"""
TTS Studio — backend engine.

All model loading / inference / Ollama logic lives here, completely
independent of any UI framework. server.py (FastAPI) exposes this over
HTTP for the vanilla HTML/CSS/JS frontend in static/. The legacy Gradio
UI (app.py) also imports from here.
"""

import os
import json
import hashlib
import urllib.request
import urllib.error
import threading
import traceback
import tempfile
from pathlib import Path
from datetime import datetime

# Suppress HuggingFace symlinks warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Fix emoji/unicode printing on Windows cp1252 console
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch
import soundfile as sf
import librosa

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Model IDs
# ─────────────────────────────────────────────────────────────────────────────

MODEL_EMOTION_TAGS = "SpragAI/qwen3-tts-emotion-tags"
MODEL_VOICE_CLONE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_FISH_SPEECH = "fishaudio/openaudio-s1-mini"
MODEL_FISH_SPEECH_FULL = "fishaudio/openaudio-s1"

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
# Model Manager (singleton) — Qwen3-TTS (emotion tags + voice clone)
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

def _ensure_fish_speech_project_root_marker():
    """
    fish_speech.models.dac.inference (and some sibling modules) call
    pyrootutils.setup_root(__file__, indicator=".project-root") at import
    time, which walks up from the module's own location looking for that
    marker file. That convention assumes fish-speech is run from inside its
    own git checkout, where the file is committed at the repo root — a
    `pip install fish-speech` has no such file anywhere on the path, so the
    import raises FileNotFoundError. Creating an empty marker directly
    inside the installed package directory satisfies the search without
    needing a git checkout.
    """
    try:
        import fish_speech
        pkg_dir = Path(fish_speech.__file__).resolve().parent
        marker = pkg_dir / ".project-root"
        if not marker.exists():
            marker.touch()
    except Exception:
        pass  # best-effort; if this fails, the real import below will surface the error


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
            _ensure_fish_speech_project_root_marker()
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
        req = urllib.request.Request(url, headers={"User-Agent": "TTS-Studio"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            return True, f"Ollama connected. Models: {', '.join(models) if models else 'none'}"
    except Exception as e:
        return False, f"Ollama connection failed: {e}"


def rephrase_with_ollama(text: str, ollama_url: str = "http://localhost:11434", ollama_model: str = "gemma2:2b"):
    if not text or not text.strip():
        raise ValueError("Please enter some text before rephrasing.")

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
            return response_text
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not connect to Ollama at {ollama_url}. Is Ollama running? Error: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama rephrasing error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_output(wavs, sr) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = OUTPUTS_DIR / f"output_{ts}.wav"
    sf.write(str(out_path), wavs[0], sr)
    return out_path


def get_output_history():
    files = sorted(OUTPUTS_DIR.glob("*.wav"), reverse=True)[:30]
    return files
