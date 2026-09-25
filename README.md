# TTS Studio

A local inference interface for **Qwen3-TTS** and **Fish-Speech (OpenAudio S1)**, covering
inline emotion tags, zero-shot voice cloning, and both together in one model.
Built with Gradio — runs entirely on your machine, no cloud required.

---

## ✨ Features

- 🎭 **Inline Emotion Tag Switching** (`SpragAI/qwen3-tts-emotion-tags`) with 9 preset character voices
- 🏷️ **Quick-Insert Emotion Chips** (`[Angry]`, `[Sad]`, `[Happy]`, `[Fast]`, `[Gentle]`, `[Tired]`, `[Fearful]`, `[Disgusted]`, `[Surprised]`)
- 🤖 **Ollama AI Auto-Rephraser** (`gemma2:2b`) — automatically analyzes dialogue and annotates sentences with matching emotion tags
- 🎙️ **Zero-shot voice cloning** from a 5–15 second WAV reference (`Qwen3-TTS-12Hz-1.7B-Base`)
- 🐟 **Fish-Speech / OpenAudio S1** — voice cloning *and* inline emotion/tone markers
  (`(happy)`, `(whispering)`, `(laughing)`, …) in a single model (optional install, see below)
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

# Install app dependencies
pip install -r requirements.txt

# Optional, faster attention on Ampere+ GPUs (Setup tab tries this first,
# falls back to sdpa automatically if not installed):
pip install flash-attn --no-build-isolation

# Optional, for the Fish-Speech "Clone + Emotion" tab:
pip install -r requirements-fish.txt
```

### 2. Run the app

```bat
run.bat
```

Or:

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

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
├── app.py                  ← Main Gradio application
├── requirements.txt
├── requirements-fish.txt   ← Optional extra for the Fish-Speech tab
├── install.bat             ← One-click dependency installer (Windows)
├── run.bat                 ← Quick launch (Windows)
├── outputs/                ← Auto-created; holds generated WAV files
├── fish_checkpoints/       ← Auto-created; Fish-Speech checkpoints download here
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
| `ModuleNotFoundError: fish_speech` | `pip install -r requirements-fish.txt` (only needed for that tab) |
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
