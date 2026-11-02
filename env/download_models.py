"""Pre-download the study's models into the HF cache.

Cloud, in no-GPU mode (cheaper):   python env/download_models.py
Local, OLMo-2-1B only (8GB GPU):   python env/download_models.py allenai/OLMo-2-0425-1B
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"D:\hf" if os.name == "nt" else "/root/autodl-tmp/hf")

from huggingface_hub import snapshot_download

MODELS = [
    "allenai/OLMoE-1B-7B-0924",
    "allenai/OLMoE-1B-7B-0924-Instruct",
    "allenai/OLMo-2-0425-1B",
]

for repo in sys.argv[1:] or MODELS:
    path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
    print(repo, "->", path, flush=True)
