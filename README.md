# Qwen3-TTS Voice Clone Studio

A local inference interface for **Qwen/Qwen3-TTS-12Hz-1.7B-Base** with full voice cloning support.
Built with Gradio — runs entirely on your machine, no cloud required.

---

## ✨ Features

- 🎭 **Inline Emotion Tag Switching** (`SpragAI/qwen3-tts-emotion-tags`) with 9 preset character voices
- 🏷️ **Quick-Insert Emotion Chips** (`[Angry]`, `[Sad]`, `[Happy]`, `[Fast]`, `[Gentle]`, `[Tired]`, `[Fearful]`, `[Disgusted]`, `[Surprised]`)
- 🤖 **Ollama AI Auto-Rephraser** (`gemma2:2b`) — automatically analyzes dialogue and annotates sentences with matching emotion tags
- 🎙️ **Zero-shot voice cloning** from a 5–15 second WAV reference (`Qwen3-TTS-12Hz-1.7B-Base`)
- 📝 Reference transcript input for maximum clone accuracy
- 🌐 20+ supported languages
- ⚙️ Adjustable generation parameters (temperature, top-p, repetition penalty)
- 📂 Auto-saved output history with built-in preview player
- 🖥️ GPU auto-detection with dtype and attention backend selection
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
pip install -U qwen-tts gradio soundfile numpy transformers accelerate huggingface-hub
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
qwen3-tts-studio/
├── app.py          ← Main Gradio application
├── requirements.txt
├── install.bat     ← One-click dependency installer (Windows)
├── run.bat         ← Quick launch (Windows)
├── outputs/        ← Auto-created; holds generated WAV files
└── README.md
```

---

## 🎤 Voice Cloning Tips

1. Use **5–15 seconds** of clean audio (no noise, music, or other speakers)
2. Transcript must be **word-for-word exact** — even punctuation matters
3. Same language for reference and target works best
4. If quality is poor, try a different reference clip

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: qwen_tts` | `pip install -U qwen-tts` |
| CUDA out of memory | Switch to `float16`; reduce text length |
| `ImportError: check_model_inputs` | `pip install transformers==4.57.3` |
| FlashAttention install fails | Use `sdpa` or `default` in Setup tab |
| Model not downloading | Check HF token: `huggingface-cli login` |

---

## 📦 Model Details

- **Model ID:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- **Size:** ~3.5 GB (downloaded once, cached)
- **Architecture:** Decoder-only LM + 12 Hz audio codec
- **License:** Apache 2.0
