@echo off
title Qwen3-TTS Voice Clone Studio — Setup
color 0A

echo ============================================================
echo   Qwen3-TTS Voice Clone Studio — Dependency Installer
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

echo [1/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [2/5] Installing PyTorch with CUDA 12.1 support...
echo       (Skip this step if you already have PyTorch installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet

echo [3/5] Installing qwen-tts and core dependencies...
pip install -U qwen-tts gradio soundfile numpy transformers accelerate huggingface-hub --quiet

echo [4/5] (Optional) Attempting FlashAttention 2 install...
echo       This may fail on some hardware — that is OK.
pip install flash-attn --no-build-isolation --quiet 2>nul || echo       [SKIP] FlashAttention 2 not installed (not required).

echo [5/5] Verifying installation...
python -c "import torch; print('  PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import gradio; print('  Gradio:', gradio.__version__)"

echo.
echo ============================================================
echo   Setup complete!  Run:  python app.py
echo ============================================================
echo.
pause
