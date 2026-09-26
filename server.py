"""
TTS Studio — FastAPI backend for the vanilla HTML/CSS/JS frontend (static/).

All actual model logic lives in backend.py; this file is just the HTTP layer:
routes in, JSON/audio out. Run with:

    python server.py
"""

import io
import json
import logging
import re
import shutil
import tempfile
import threading
import traceback
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import backend
import narration

app = FastAPI(title="TTS Studio API")

UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "ref.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=str(UPLOADS_DIR))
    with tmp as f:
        shutil.copyfileobj(upload.file, f)
    return Path(tmp.name)


# ─────────────────────────────────────────────────────────────────────────────
# Static data the frontend needs to build its forms
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    mgr = backend.ModelManager.get_instance()
    return {
        "models": {
            "emotion": backend.MODEL_EMOTION_TAGS,
            "clone": backend.MODEL_VOICE_CLONE,
            "fish": backend.MODEL_FISH_SPEECH,
        },
        "languages": backend.SUPPORTED_LANGUAGES,
        "speakers": [{"id": s[0], "label": s[1]} for s in backend.PRESET_SPEAKERS],
        "emotion_tags": [{"label": l, "value": v} for l, v in backend.EMOTION_TAGS],
        "fish_markers": [{"label": l, "value": v} for l, v in backend.FISH_EMOTION_MARKERS],
        "device": mgr.device_info,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Qwen3-TTS model load / status
# ─────────────────────────────────────────────────────────────────────────────

class QwenLoadRequest(BaseModel):
    model_id: str
    dtype: str = "bfloat16 (Recommended)"
    attn_impl: str = "flash_attention_2"
    compile: bool = False


@app.post("/api/models/qwen/load")
def load_qwen_model(req: QwenLoadRequest):
    mgr = backend.ModelManager.get_instance()
    if mgr.is_loading:
        return {"started": False, "reason": "already loading"}

    thread = threading.Thread(
        target=mgr.load_model,
        args=(req.model_id, req.dtype, req.attn_impl, req.compile),
        daemon=True,
    )
    thread.start()
    return {"started": True}


@app.get("/api/models/qwen/status")
def qwen_status():
    mgr = backend.ModelManager.get_instance()
    return {
        "is_loading": mgr.is_loading,
        "is_loaded": mgr.is_loaded,
        "current_model_id": mgr.current_model_id,
        "load_error": mgr.load_error,
        "log": "\n".join(mgr.load_log),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fish-Speech model load / status
# ─────────────────────────────────────────────────────────────────────────────

class FishLoadRequest(BaseModel):
    model_id: str = backend.MODEL_FISH_SPEECH
    compile: bool = False


@app.post("/api/models/fish/load")
def load_fish_model(req: FishLoadRequest):
    mgr = backend.FishSpeechManager.get_instance()
    if mgr.is_loading:
        return {"started": False, "reason": "already loading"}

    thread = threading.Thread(
        target=mgr.load_model,
        args=(req.model_id, req.compile),
        daemon=True,
    )
    thread.start()
    return {"started": True}


@app.get("/api/models/fish/status")
def fish_status():
    mgr = backend.FishSpeechManager.get_instance()
    return {
        "is_loading": mgr.is_loading,
        "is_loaded": mgr.is_loaded,
        "current_model_id": mgr.current_model_id,
        "log": "\n".join(mgr.load_log),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Generation endpoints
# ─────────────────────────────────────────────────────────────────────────────

def _audio_response(wavs, sr, cache_hit: Optional[bool] = None):
    out_path = backend.save_output(wavs, sr)
    duration = round(len(wavs[0]) / sr, 2)
    return {
        "audio_url": f"/outputs/{out_path.name}",
        "duration_s": duration,
        "cache_hit": cache_hit,
    }


@app.post("/api/generate/emotion")
def generate_emotion(
    text: str = Form(...),
    speaker: str = Form(...),
    language: str = Form("Auto"),
    temperature: float = Form(0.9),
    top_p: float = Form(0.9),
    repetition_penalty: float = Form(1.0),
):
    mgr = backend.ModelManager.get_instance()
    if not mgr.is_loaded:
        raise HTTPException(400, f"No model loaded. Load '{backend.MODEL_EMOTION_TAGS}' first.")
    if mgr.current_model_id != backend.MODEL_EMOTION_TAGS:
        raise HTTPException(400, f"Active model is '{mgr.current_model_id}', not the emotion-tags model.")
    if not text.strip():
        raise HTTPException(400, "Please enter some text to synthesize.")

    try:
        wavs, sr = mgr.generate_custom_voice(
            text=text.strip(), speaker=speaker, language=language,
            temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

    return _audio_response(wavs, sr)


@app.post("/api/generate/clone")
def generate_clone(
    text: str = Form(...),
    language: str = Form("English"),
    ref_text: str = Form(...),
    temperature: float = Form(0.9),
    top_p: float = Form(0.9),
    repetition_penalty: float = Form(1.0),
    ref_audio: UploadFile = File(...),
):
    mgr = backend.ModelManager.get_instance()
    if not mgr.is_loaded:
        raise HTTPException(400, f"No model loaded. Load '{backend.MODEL_VOICE_CLONE}' first.")
    if mgr.current_model_id != backend.MODEL_VOICE_CLONE:
        raise HTTPException(400, f"Active model is '{mgr.current_model_id}', not the voice-clone model.")
    if not text.strip():
        raise HTTPException(400, "Please enter some text to synthesize.")
    if not ref_text.strip():
        raise HTTPException(400, "Please provide the reference transcript.")

    ref_path = _save_upload(ref_audio)
    try:
        wavs, sr, cache_hit = mgr.generate_voice_clone(
            text=text.strip(), language=language,
            ref_audio_path=str(ref_path), ref_text=ref_text.strip(),
            temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        ref_path.unlink(missing_ok=True)

    return _audio_response(wavs, sr, cache_hit=cache_hit)


@app.post("/api/generate/fish")
def generate_fish(
    text: str = Form(...),
    ref_text: str = Form(...),
    temperature: float = Form(0.8),
    top_p: float = Form(0.8),
    repetition_penalty: float = Form(1.1),
    ref_audio: UploadFile = File(...),
):
    mgr = backend.FishSpeechManager.get_instance()
    if not mgr.is_loaded:
        raise HTTPException(400, "No Fish-Speech model loaded. Load it first.")
    if not text.strip():
        raise HTTPException(400, "Please enter some text to synthesize.")
    if not ref_text.strip():
        raise HTTPException(400, "Please provide the reference transcript.")

    ref_path = _save_upload(ref_audio)
    try:
        audio, sr = mgr.generate(
            text=text.strip(), ref_audio_path=str(ref_path), ref_text=ref_text.strip(),
            temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        ref_path.unlink(missing_ok=True)

    return _audio_response([audio], sr)


# ─────────────────────────────────────────────────────────────────────────────
# Ollama rephraser
# ─────────────────────────────────────────────────────────────────────────────

class OllamaRephraseRequest(BaseModel):
    text: str
    url: str = "http://localhost:11434"
    model: str = "gemma2:2b"


@app.post("/api/ollama/rephrase")
def ollama_rephrase(req: OllamaRephraseRequest):
    try:
        rephrased = backend.rephrase_with_ollama(req.text, req.url, req.model)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"text": rephrased}


@app.get("/api/ollama/test")
def ollama_test(url: str = "http://localhost:11434"):
    ok, message = backend.test_ollama_connection(url)
    return {"ok": ok, "message": message}


# ─────────────────────────────────────────────────────────────────────────────
# Output history
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/outputs")
def list_outputs():
    files = backend.get_output_history()
    return [{"name": f.name, "url": f"/outputs/{f.name}"} for f in files]


# ─────────────────────────────────────────────────────────────────────────────
# Batch narration: JSON script → one clip per scene (runs narration.py's
# generate_narration in a background thread; the UI polls for progress)
# ─────────────────────────────────────────────────────────────────────────────

NARRATION_DIR = Path("narration_output")
NARRATION_DIR.mkdir(exist_ok=True)

# A root handler up front stops generate_narration() from calling
# logging.basicConfig(level=INFO), which would also turn on every other
# library's INFO chatter; its own progress lines still reach the console.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
narration.log.setLevel(logging.INFO)

_PROGRESS_LINE = re.compile(r"^\[\d+/\d+\] ")


class _NarrationJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.folder = None
        self.scene_files = []
        self.total = 0
        self.lines = []
        self.counts = None
        self.error = None


JOB = _NarrationJob()


class _JobLogHandler(logging.Handler):
    def emit(self, record):
        JOB.lines.append({"level": record.levelname.lower(), "text": record.getMessage()})


narration.log.addHandler(_JobLogHandler())


def _safe_folder(name: str) -> str:
    """Output subfolder name: always a single path component under NARRATION_DIR."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name or "").strip("-.") or "narration"


def _run_narration(json_path: Path, ref_path: Optional[Path], kwargs: dict):
    try:
        JOB.counts = narration.generate_narration(
            json_path, NARRATION_DIR / JOB.folder, ref_audio=str(ref_path) if ref_path else None, **kwargs
        )
    except narration.NarrationError as e:
        JOB.error = str(e)
    except Exception as e:
        traceback.print_exc()
        JOB.error = f"{type(e).__name__}: {e}"
    finally:
        json_path.unlink(missing_ok=True)
        if ref_path:
            ref_path.unlink(missing_ok=True)
        JOB.running = False


@app.get("/api/narration/options")
def narration_options():
    return {
        "qwen_emotions": narration.QWEN_EMOTIONS,
        "fish_emotions": narration.FISH_EMOTIONS,
        "output_root": str(NARRATION_DIR.resolve()),
    }


@app.post("/api/narration/start")
def narration_start(
    script: UploadFile = File(...),
    folder: str = Form(""),
    model: str = Form("emotion"),
    voice: str = Form("ryan"),
    language: str = Form("Auto"),
    emotion: str = Form(""),
    ref_text: str = Form(""),
    overwrite: bool = Form(False),
    seed: int = Form(1234),
    ref_audio: Optional[UploadFile] = File(None),
):
    with JOB.lock:
        if JOB.running:
            raise HTTPException(409, "A narration batch is already running.")

        raw = script.file.read()
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise HTTPException(400, f"Couldn't read the JSON script: {e}")
        scenes = data.get("scenes") if isinstance(data, dict) else None
        if not isinstance(scenes, list) or not scenes:
            raise HTTPException(400, "The JSON script has no 'scenes' list.")

        json_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=str(UPLOADS_DIR))
        with json_tmp as f:
            f.write(raw)
        ref_path = _save_upload(ref_audio) if ref_audio is not None and ref_audio.filename else None

        JOB.running = True
        JOB.folder = _safe_folder(folder or str(data.get("project", "")))
        JOB.scene_files = [s["file"] for s in scenes if isinstance(s, dict) and isinstance(s.get("file"), str)]
        JOB.total = len(scenes)
        JOB.lines, JOB.counts, JOB.error = [], None, None

    kwargs = dict(
        voice=voice, model=model, ref_text=ref_text or None, language=language,
        emotion=emotion or None, overwrite=overwrite, seed=seed,
    )
    threading.Thread(target=_run_narration, args=(Path(json_tmp.name), ref_path, kwargs), daemon=True).start()
    return {"started": True, "folder": JOB.folder, "total": JOB.total}


@app.get("/api/narration/status")
def narration_status():
    lines = list(JOB.lines)
    out_dir = NARRATION_DIR / JOB.folder if JOB.folder else None
    files = []
    if out_dir:
        for name in JOB.scene_files:
            p = out_dir / name
            if Path(name).name == name and p.is_file():
                # mtime in the URL so an overwritten clip isn't served from the browser cache
                files.append({"name": name, "url": f"/narration-files/{JOB.folder}/{quote(name)}?v={int(p.stat().st_mtime)}"})
    return {
        "running": JOB.running,
        "folder": JOB.folder,
        "output_dir": str(out_dir.resolve()) if out_dir else None,
        "total": JOB.total,
        "done": sum(1 for line in lines if _PROGRESS_LINE.match(line["text"])),
        "lines": lines,
        "counts": JOB.counts,
        "error": JOB.error,
        "files": files,
    }


@app.get("/api/narration/zip")
def narration_zip(folder: str):
    if _safe_folder(folder) != folder or not (NARRATION_DIR / folder).is_dir():
        raise HTTPException(404, "No such narration folder.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # mp3/wav don't compress further
        for p in sorted((NARRATION_DIR / folder).iterdir()):
            if p.is_file() and p.suffix.lower() in narration.AUDIO_FORMATS:
                z.write(p, p.name)
    return Response(
        buf.getvalue(), media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{folder}.zip"'},
    )


# Serve generated audio files and the static frontend. Registered last so
# they don't shadow the /api/* routes above.
app.mount("/outputs", StaticFiles(directory=str(backend.OUTPUTS_DIR)), name="outputs")
app.mount("/narration-files", StaticFiles(directory=str(NARRATION_DIR)), name="narration-files")
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  TTS Studio (HTML/CSS/JS frontend)")
    print("  http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
