"""Minimal test to see why from_pretrained crashes."""
import os, sys, traceback
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("Step 1: importing torch...")
import torch
print(f"  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  VRAM: {props.total_memory / 1e9:.2f} GB total")
    print(f"  Free: {(props.total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB free")

print("\nStep 2: importing qwen_tts...")
sys.stdout.flush()
sys.stderr.flush()
from qwen_tts import Qwen3TTSModel
print("  qwen_tts imported OK")

model_path = r"C:\Users\FlowwTech Industries\.gemini\antigravity-ide\scratch\qwen3-tts-studio\model_cache"
print(f"\nStep 3: loading from {model_path}")
print(f"  Files: {os.listdir(model_path)}")
sys.stdout.flush()
sys.stderr.flush()

try:
    print("\n  Calling from_pretrained (bfloat16, cuda:0)...")
    sys.stdout.flush()
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print("\n  ✅ MODEL LOADED SUCCESSFULLY!")
    print(f"  Type: {type(model)}")
except Exception as e:
    print(f"\n  ❌ Python exception: {type(e).__name__}: {e}")
    traceback.print_exc()
except BaseException as e:
    print(f"\n  ❌ BaseException: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\nDone.")
sys.stdout.flush()
