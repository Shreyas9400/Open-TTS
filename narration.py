"""
Batch-generate narration clips from a JSON script.

    python narration.py narration.json out/                                   # Qwen emotion-tags model, preset voice
    python narration.py narration.json out/ --model clone --ref-audio me.wav --ref-text "..."
    python narration.py narration.json out/ --model fish  --ref-audio me.wav --ref-text "..." --emotion sincere

JSON shape:
    {"project": "...", "scenes": [
        {"index": 0, "id": "intro", "file": "00-intro.mp3", "text": "...", "emotion": "joyful"},  # "emotion" optional
        ...
    ]}

Each scene is written to <output_dir>/<scene.file> exactly as named. The file
extension picks the encoding (.mp3 or .wav): the models return raw audio
samples, and each clip is encoded once, directly into that format - nothing is
converted from one audio format to another. Existing files are skipped unless
--overwrite is given. One voice, one set of generation settings and a fixed
seed are used for the whole run so the clips sound consistent.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

log = logging.getLogger("narration")

MODELS = ("emotion", "clone", "fish")

AUDIO_FORMATS = {".mp3": "MP3", ".wav": "WAV"}

GEN_DEFAULTS = {
    "emotion": dict(temperature=0.9, top_p=0.9, repetition_penalty=1.0),
    "clone": dict(temperature=0.9, top_p=0.9, repetition_penalty=1.0),
    "fish": dict(temperature=0.8, top_p=0.8, repetition_penalty=1.1),
}

# SpragAI emotion-tags model: the 9 tags it was fine-tuned on, written as [Tag].
QWEN_EMOTIONS = ["Angry", "Sad", "Happy", "Fast", "Gentle", "Tired", "Fearful", "Disgusted", "Surprised"]

# OpenAudio S1 markers from fish-speech's documentation, written as (marker).
# Anything outside this list is spoken aloud as plain text, so it's rejected up front.
FISH_EMOTIONS = [
    "angry", "sad", "disdainful", "excited", "surprised", "satisfied", "unhappy", "anxious",
    "hysterical", "delighted", "scared", "worried", "indifferent", "upset", "impatient", "nervous",
    "guilty", "scornful", "frustrated", "depressed", "panicked", "furious", "empathetic",
    "embarrassed", "reluctant", "disgusted", "keen", "moved", "proud", "relaxed", "grateful",
    "confident", "interested", "curious", "confused", "joyful", "disapproving", "negative",
    "denying", "astonished", "serious", "sarcastic", "conciliative", "comforting", "sincere",
    "sneering", "hesitating", "yielding", "painful", "awkward", "amused",
    "in a hurry tone", "shouting", "screaming", "whispering", "soft tone",
    "laughing", "chuckling", "sobbing", "crying loudly", "sighing", "panting", "groaning",
    "crowd laughing", "background laughter", "audience laughing",
]


class NarrationError(Exception):
    """A problem with the inputs, reported before any model is loaded."""


def _emotion_prefix(model: str, emotion):
    """Turn 'joyful' / '(joyful)' / '[Happy]' into the model's own tag syntax, or raise."""
    if not emotion:
        return ""
    name = str(emotion).strip().strip("[]()").strip()
    if model == "clone":
        raise NarrationError(
            f"emotion '{emotion}': the Qwen voice-clone model has no emotion tags "
            f"(use --model fish for cloning + emotions)"
        )
    if model == "emotion":
        match = next((e for e in QWEN_EMOTIONS if e.lower() == name.lower()), None)
        if not match:
            raise NarrationError(f"emotion '{emotion}' isn't supported by the emotion-tags model. "
                                 f"Use one of: {', '.join(QWEN_EMOTIONS)}")
        return f"[{match}] "
    match = next((e for e in FISH_EMOTIONS if e == name.lower()), None)
    if not match:
        raise NarrationError(f"emotion '{emotion}' isn't a documented Fish-Speech S1 marker "
                             f"(it would be read aloud). Examples: joyful, sincere, serious, soft tone, whispering")
    return f"({match}) "


def _load_scenes(json_path: Path, model: str, default_emotion):
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise NarrationError(f"can't read {json_path}: {e}")

    scenes = data.get("scenes") if isinstance(data, dict) else None
    if not isinstance(scenes, list) or not scenes:
        raise NarrationError(f"{json_path} has no 'scenes' list")

    if model == "clone" and default_emotion:
        _emotion_prefix(model, default_emotion)  # raises: an explicit --emotion can't be honored

    seen, prepared, ignored_emotions = {}, [], []
    for pos, scene in enumerate(scenes):
        where = f"scene #{pos} ({scene.get('id', '?') if isinstance(scene, dict) else '?'})"
        if not isinstance(scene, dict):
            raise NarrationError(f"{where} is not an object")
        name, text = scene.get("file"), scene.get("text")
        if not isinstance(name, str) or not name.strip():
            raise NarrationError(f"{where} has no 'file'")
        if not isinstance(text, str) or not text.strip():
            raise NarrationError(f"{where} ({name}) has no 'text'")
        # Filenames are used verbatim, so refuse anything that would land outside output_dir.
        if Path(name).name != name or name in (".", ".."):
            raise NarrationError(f"{where}: 'file' must be a plain filename, got '{name}'")
        ext = Path(name).suffix.lower()
        if ext not in AUDIO_FORMATS:
            raise NarrationError(f"{where}: unsupported extension '{ext}' in '{name}' (use .mp3 or .wav)")
        if name.lower() in seen:
            raise NarrationError(f"{where}: '{name}' is also used by scene #{seen[name.lower()]}")
        seen[name.lower()] = pos
        if model == "clone" and scene.get("emotion"):
            ignored_emotions.append(name)  # same JSON should still run on the clone model
            prefix = ""
        else:
            prefix = _emotion_prefix(model, scene.get("emotion") or default_emotion)
        prepared.append({"file": name, "text": prefix + text.strip(), "format": AUDIO_FORMATS[ext]})

    if ignored_emotions:
        log.warning(f"Ignoring 'emotion' on {len(ignored_emotions)} scene(s) ({', '.join(ignored_emotions)}): "
                    f"the Qwen voice-clone model has no emotion tags")
    return prepared


def _check_formats_writable(scenes):
    import soundfile as sf
    available = sf.available_formats()
    for fmt in {s["format"] for s in scenes}:
        if fmt not in available:
            raise NarrationError(
                f"this soundfile/libsndfile build can't write {fmt} "
                f"(soundfile {sf.__version__}, libsndfile {sf.__libsndfile_version__}). "
                f"Run: pip install -U soundfile"
            )


def _prepare_reference(ref_audio, ref_start, ref_end, workdir: Path):
    """Return a path to the reference clip, cut to [ref_start, ref_end] seconds if given."""
    if ref_start is None and ref_end is None:
        return str(ref_audio)
    import librosa
    import soundfile as sf
    start = ref_start or 0.0
    duration = None if ref_end is None else ref_end - start
    if duration is not None and duration <= 0:
        raise NarrationError("--ref-end must be greater than --ref-start")
    audio, sr = librosa.load(str(ref_audio), sr=None, mono=True, offset=start, duration=duration)
    trimmed = workdir / "reference_trimmed.wav"
    sf.write(str(trimmed), audio, sr)
    log.info(f"Reference trimmed to {start:.1f}-{start + len(audio) / sr:.1f}s ({len(audio) / sr:.1f}s)")
    return str(trimmed)


def _write_audio_atomically(audio, sr, dest: Path, fmt: str):
    """Write via a temp file + rename, so an interrupted run never leaves a
    half-written clip that a re-run would then skip as 'already exists'."""
    import soundfile as sf
    fd, tmp = tempfile.mkstemp(prefix=".narration-", suffix=".part", dir=str(dest.parent))
    os.close(fd)
    try:
        sf.write(tmp, audio, sr, format=fmt)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def generate_narration(
    json_path,
    output_dir,
    voice="ryan",
    model="emotion",
    ref_audio=None,
    ref_text=None,
    ref_start=None,
    ref_end=None,
    language="Auto",
    emotion=None,
    overwrite=False,
    seed=1234,
    temperature=None,
    top_p=None,
    repetition_penalty=None,
):
    """
    Generate one clip per scene of `json_path` into `output_dir`.

    model="emotion": Qwen3-TTS SpragAI emotion-tags model with preset speaker `voice`.
    model="clone":   Qwen3-TTS Base, cloning `ref_audio` (with transcript `ref_text`).
    model="fish":    Fish-Speech OpenAudio S1-mini, cloning `ref_audio` + emotion markers.

    `emotion` is prepended to every scene's text in the model's tag syntax; a
    scene's own "emotion" field overrides it. Returns a dict of counts.
    """
    if not logging.getLogger().handlers:  # called from Python without logging set up: still print progress
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    if model not in MODELS:
        raise NarrationError(f"model must be one of {MODELS}, got '{model}'")

    json_path, output_dir = Path(json_path), Path(output_dir)
    scenes = _load_scenes(json_path, model, emotion)
    _check_formats_writable(scenes)

    import backend  # heavy (torch); imported only once the inputs look valid

    if model == "emotion":
        speakers = [s[0] for s in backend.PRESET_SPEAKERS]
        if voice not in speakers:
            raise NarrationError(f"voice '{voice}' isn't a preset speaker. Use one of: {', '.join(speakers)}")
    else:
        if not ref_audio or not Path(ref_audio).is_file():
            raise NarrationError(f"--model {model} clones a voice: --ref-audio must point to an audio file")
        if not ref_text or not ref_text.strip():
            raise NarrationError(f"--model {model} needs --ref-text (the exact words spoken in the reference clip)")

    params = dict(GEN_DEFAULTS[model])
    for key, value in (("temperature", temperature), ("top_p", top_p), ("repetition_penalty", repetition_penalty)):
        if value is not None:
            params[key] = value

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(scenes)
    todo = [s for s in scenes if overwrite or not (output_dir / s["file"]).exists()]
    counts = {"generated": 0, "skipped": 0, "failed": 0}

    with tempfile.TemporaryDirectory() as workdir:
        engine = None
        if todo:  # don't spend minutes loading a model when every clip already exists
            reference = None if model == "emotion" else _prepare_reference(ref_audio, ref_start, ref_end, Path(workdir))
            engine = _load_engine(backend, model, voice, reference, ref_text, language, params)

        for pos, scene in enumerate(scenes, start=1):
            name = scene["file"]
            dest = output_dir / name
            prefix = f"[{pos}/{total}] {name} ->"

            if dest.exists() and not overwrite:
                log.warning(f"{prefix} skipped: already exists (pass --overwrite to regenerate)")
                counts["skipped"] += 1
                continue

            started = time.time()
            try:
                import torch
                torch.manual_seed(seed)  # same seed per clip: consistent delivery, reproducible re-runs
                audio, sr = engine(scene["text"])
                _write_audio_atomically(audio, sr, dest, scene["format"])
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.error(f"{prefix} FAILED: {type(e).__name__}: {e}")
                counts["failed"] += 1
                continue
            log.info(f"{prefix} generated ({len(audio) / sr:.1f}s audio in {time.time() - started:.1f}s)")
            counts["generated"] += 1

    _report_folder_contents(output_dir, scenes)
    log.info(f"Done: {counts['generated']} generated, {counts['skipped']} skipped, {counts['failed']} failed")
    return counts


def _load_engine(backend, model, voice, reference, ref_text, language, params):
    """Load the chosen model once and return text -> (audio, sample_rate)."""
    log.info(f"Loading model for --model {model} (this can take a while)…")
    if model == "fish":
        mgr = backend.FishSpeechManager.get_instance()
        ok, load_log = mgr.load_model(backend.MODEL_FISH_SPEECH)
        if not ok:
            raise NarrationError(f"Fish-Speech failed to load:\n{load_log}")
        return lambda text: mgr.generate(text=text, ref_audio_path=reference, ref_text=ref_text.strip(), **params)

    mgr = backend.ModelManager.get_instance()
    model_id = backend.MODEL_EMOTION_TAGS if model == "emotion" else backend.MODEL_VOICE_CLONE
    ok, load_log = mgr.load_model(model_id, "bfloat16 (Recommended)", "flash_attention_2")
    if not ok:
        raise NarrationError(f"{model_id} failed to load:\n{load_log}")

    if model == "emotion":
        def speak(text):
            wavs, sr = mgr.generate_custom_voice(text=text, speaker=voice, language=language, **params)
            return wavs[0], sr
    else:
        def speak(text):
            # The reference is encoded on the first scene and reused from cache after that.
            wavs, sr, _ = mgr.generate_voice_clone(text=text, language=language, ref_audio_path=reference,
                                                   ref_text=ref_text.strip(), **params)
            return wavs[0], sr
    return speak


def _report_folder_contents(output_dir: Path, scenes):
    expected = {s["file"] for s in scenes}
    present = {p.name for p in output_dir.iterdir() if p.is_file() and not p.name.startswith(".narration-")}
    missing = sorted(expected - present)
    extra = sorted(n for n in present - expected if Path(n).suffix.lower() in AUDIO_FORMATS)
    if missing:
        log.warning(f"Missing from {output_dir}: {', '.join(missing)}")
    if extra:
        log.warning(f"Audio files in {output_dir} not listed in the JSON (left untouched): {', '.join(extra)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch-generate narration clips from a JSON script.")
    parser.add_argument("json_path", help="narration JSON file")
    parser.add_argument("output_dir", help="folder to write the clips into")
    parser.add_argument("--model", choices=MODELS, default="emotion",
                        help="emotion = Qwen3 emotion-tags (preset voices); clone = Qwen3 Base voice clone; "
                             "fish = Fish-Speech S1-mini (voice clone + emotion markers). Default: emotion")
    parser.add_argument("--voice", default="ryan", help="preset speaker for --model emotion (default: ryan)")
    parser.add_argument("--ref-audio", help="reference clip to clone (--model clone / fish)")
    parser.add_argument("--ref-text", help="exact transcript of the reference clip (or of the trimmed part)")
    parser.add_argument("--ref-start", type=float, help="use the reference clip from this second")
    parser.add_argument("--ref-end", type=float, help="use the reference clip up to this second")
    parser.add_argument("--language", default="Auto", help="Qwen language, e.g. English (default: Auto)")
    parser.add_argument("--emotion", help="emotion for every scene, e.g. Happy (emotion model) or sincere (fish); "
                                          "a scene's own \"emotion\" field overrides it")
    parser.add_argument("--overwrite", action="store_true", help="regenerate clips that already exist")
    parser.add_argument("--seed", type=int, default=1234, help="random seed used for every clip (default: 1234)")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        counts = generate_narration(
            args.json_path, args.output_dir, voice=args.voice, model=args.model,
            ref_audio=args.ref_audio, ref_text=args.ref_text, ref_start=args.ref_start, ref_end=args.ref_end,
            language=args.language, emotion=args.emotion, overwrite=args.overwrite, seed=args.seed,
            temperature=args.temperature, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
        )
    except NarrationError as e:
        log.error(f"Error: {e}")
        return 2
    except KeyboardInterrupt:
        log.error("Interrupted - finished clips are kept; re-run to continue where it stopped.")
        return 130
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
