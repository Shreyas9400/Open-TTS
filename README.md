# TTS Studio

A local inference interface for **Qwen3-TTS** and **Fish-Speech (OpenAudio S1)**, covering
inline emotion tags, zero-shot voice cloning, and both together in one model.
Runs entirely on your machine, no cloud required.

There are two UIs, sharing the same backend logic (`backend.py`):

| | Stack | Entry point | Status |
|---|---|---|---|
| **Recommended** | Plain HTML/CSS/JS frontend + FastAPI backend | `python server.py` → http://localhost:8000 | Primary UI |
| Legacy | Gradio | `python app.py` → http://localhost:7860 | Kept for anyone who prefers it |

The HTML/CSS/JS version exists because Gradio's component internals (nested wrapper
divs with hardcoded light-theme colors, class names that change between major versions)
made a fully custom dark theme an uphill fight — the hand-built frontend has no such
constraints and gives full control over the UI.

---

## ✨ Features

- 🎭 **Inline Emotion Tag Switching** (`SpragAI/qwen3-tts-emotion-tags`) with 9 preset character voices
- 🏷️ **Quick-Insert Emotion Chips** (`[Angry]`, `[Sad]`, `[Happy]`, `[Fast]`, `[Gentle]`, `[Tired]`, `[Fearful]`, `[Disgusted]`, `[Surprised]`)
- 🤖 **Ollama AI Auto-Rephraser** (`gemma2:2b`) — automatically analyzes dialogue and annotates sentences with matching emotion tags
- 🎙️ **Zero-shot voice cloning** from a 5–15 second WAV reference (`Qwen3-TTS-12Hz-1.7B-Base`)
- 🐟 **Fish-Speech / OpenAudio S1** — voice cloning *and* inline emotion/tone markers
  (`(joyful)`, `(soft tone)`, `(whispering)`, `(laughing)`, …) in a single model (optional install, see below)
- 📝 Reference transcript input for maximum clone accuracy
- 🌐 20+ supported languages
- ⚙️ Adjustable generation parameters (temperature, top-p, repetition penalty)
- 📂 Auto-saved output history with built-in preview player
- 🖥️ GPU auto-detection with dtype and attention backend selection
- ⚡ Speed: flash-attention with automatic sdpa fallback, optional `torch.compile`, and
  cached voice-clone prompts so repeat generations against the same reference clip
  skip re-encoding it every time
- 📖 Built-in guide and troubleshooting

---

## 🚀 Quick Start

### 1. Install dependencies

```bat
install.bat
```

Or manually:

```bash
# Install PyTorch with CUDA (adjust cu121 to your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -r requirements.txt

# Optional, faster attention on Ampere+ GPUs (falls back to sdpa
# automatically if not installed):
pip install flash-attn --no-build-isolation

# Optional, for the Fish-Speech "Clone + Emotion" feature:
pip install -r requirements-fish.txt

# For the recommended HTML/CSS/JS UI:
pip install -r requirements-web.txt
```

### 2. Run the app

**Recommended (HTML/CSS/JS + FastAPI):**
```bash
python server.py
```
Open **http://localhost:8000** in your browser.

**Legacy (Gradio):**
```bat
run.bat
```
or
```bash
python app.py
```
Open **http://localhost:7860** in your browser.

---

## 🎬 Batch Narration from a JSON Script

`narration.py` generates one clip per scene of a JSON script, named exactly as the scene's `file` field:

```json
{
  "project": "42-years-milestones",
  "scenes": [
    { "index": 0, "id": "intro", "file": "00-intro.mp3", "text": "42 Years of Growth and Expansion. 1984 to 2026." },
    { "index": 1, "id": "1984", "file": "01-1984.mp3", "text": "1984. Our journey began...", "emotion": "sincere" }
  ]
}
```

```bash
# Qwen3 emotion-tags model, preset voice (default: ryan)
python narration.py narration.json out --voice serena

# Qwen3 Base voice clone (no emotions)
python narration.py narration.json out --model clone --ref-audio me.wav --ref-text "Exact words in me.wav"

# Fish-Speech: voice clone + emotion markers; use only 5-20 s of a long reference
python narration.py narration.json out --model fish --ref-audio me.wav --ref-start 5 --ref-end 20 \
    --ref-text "Exact words spoken between 5 s and 20 s" --emotion sincere
```

- Progress is printed as `[n/total] file -> status`. Clips that already exist are skipped (so re-runs only fill gaps); pass `--overwrite` to regenerate them.
- The extension in `file` picks the format: `.mp3` or `.wav`, encoded once, directly from the model's audio.
- One voice, one set of settings and a fixed `--seed` are used for every clip, so the whole run sounds consistent.
- `--emotion` applies to every scene; a scene's optional `"emotion"` field overrides it. Emotion model: `Angry Sad Happy Fast Gentle Tired Fearful Disgusted Surprised`. Fish: documented S1 markers such as `joyful sincere serious "soft tone" whispering`. The clone model has no emotions, so scene emotions are skipped with a warning.
- A failed scene doesn't stop the run: it's reported, the exit code is 1, and re-running generates just that scene.
- Run `python narration.py --help` for all options (`--language`, `--temperature`, `--top-p`, `--repetition-penalty`).

---

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 8 GB    | 16 GB       |
| RAM       | 16 GB   | 32 GB       |
| Storage   | 8 GB    | 20 GB       |
| CUDA      | 11.8+   | 12.1+       |

> **CPU mode** works but is very slow. GPU is strongly recommended.

---

## 📁 Project Structure

```
Open-TTS/
├── backend.py               ← Model loading + inference logic (shared by both UIs)
├── server.py                ← FastAPI backend for the HTML/CSS/JS frontend (recommended)
├── static/                  ← The HTML/CSS/JS frontend itself
│   ├── index.html
│   ├── style.css
│   └── app.js
├── app.py                   ← Legacy Gradio UI (imports nothing from backend.py — self-contained)
├── requirements.txt
├── requirements-web.txt     ← Optional extra for the HTML/CSS/JS UI (fastapi, uvicorn)
├── requirements-fish.txt    ← Optional extra for the Fish-Speech feature
├── install.bat              ← One-click dependency installer (Windows)
├── run.bat                  ← Quick launch (Windows, Gradio UI)
├── outputs/                 ← Auto-created; holds generated WAV files
├── uploads/                 ← Auto-created; temporary reference-audio uploads (server.py)
├── fish_checkpoints/        ← Auto-created; Fish-Speech checkpoints download here
└── README.md
```

---

## 🎤 Voice Cloning Tips

1. Use **5–15 seconds** (Qwen) or **10–30 seconds** (Fish-Speech) of clean audio (no noise, music, or other speakers)
2. Transcript must be **word-for-word exact** — even punctuation matters
3. Same language for reference and target works best
4. If quality is poor, try a different reference clip
5. Reusing the same reference clip across multiple generations is fast — the app caches the
   encoded voice prompt instead of re-processing the clip every time

---

## ⚡ Speed Notes

- The Setup tab defaults to `flash_attention_2` and **automatically falls back to `sdpa`**
  if flash-attn isn't installed or your GPU isn't compatible — safe to leave on.
- `torch.compile` is available as an experimental toggle; the first generation after
  loading is slower while it compiles, then steady-state generation speeds up.
- Voice-clone generations against the same reference clip reuse a cached prompt
  (`Voice prompt: reused (fast)` in the stats line) instead of re-encoding the reference
  audio on every call.
- `bfloat16` is the fastest safe dtype on modern GPUs.

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: qwen_tts` | `pip install -U qwen-tts` |
| `ModuleNotFoundError: fish_speech` | `pip install -r requirements-fish.txt` (only needed for that feature) |
| `ModuleNotFoundError: fastapi` | `pip install -r requirements-web.txt` (only needed for `server.py`) |
| CUDA out of memory | Switch to `float16`; reduce text length |
| `ImportError: check_model_inputs` | `pip install transformers==4.57.3` |
| FlashAttention install fails | The app falls back to `sdpa` automatically; or select `sdpa`/`default` manually in Setup |
| Model not downloading | Check HF token: `huggingface-cli login` |

---

## 📦 Model Details

| Model | Voice cloning | Emotion control | License |
|---|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | ✅ Zero-shot | ❌ | Apache 2.0 |
| `SpragAI/qwen3-tts-emotion-tags` | ❌ (9 preset voices only) | ✅ Inline `[Tag]` markers | Apache 2.0 |
| `fishaudio/openaudio-s1-mini` | ✅ Zero-shot | ✅ Inline `(marker)` markers | CC-BY-NC-SA-4.0 (non-commercial) |
