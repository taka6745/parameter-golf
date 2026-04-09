#!/bin/bash
# setup.sh — pod environment setup. Idempotent.
#
# What it does:
#   1. Upgrade torch to 2.9.1+cu128 (matches the bundled FA3 wheel ABI)
#   2. Verify FA3 imports (the FA3 wheel is built against torch 2.9; the default
#      RunPod pytorch image ships torch 2.4.1 which fails with "undefined symbol:
#      aoti_torch_create_device_guard")
#   3. Install brotli + sentencepiece + huggingface_hub (used by data + train)
#   4. Verify CUDA + nvidia-smi
#
# Why torch 2.9: the flash-attn-3 wheel pre-installed in the runpod/pytorch:2.4
# image was actually built against torch 2.9.1+cu128 (`cu128torch291cxx11abitrue`).
# Importing it on torch 2.4 fails with the symbol error above. The cleanest fix
# is to upgrade torch to the version FA3 expects.

set -eu

echo "[setup] python: $(python3 --version)"
echo "[setup] pip: $(pip3 --version)"
echo "[setup] gpu(s):"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "  (nvidia-smi not available)"

# 1. torch 2.9.1+cu128
TORCH_VER=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "none")
if [ "$TORCH_VER" != "2.9.1+cu128" ]; then
    echo "[setup] current torch: $TORCH_VER → upgrading to 2.9.1+cu128 (matches FA3 wheel ABI)"
    pip install --quiet \
        torch==2.9.1 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
else
    echo "[setup] torch 2.9.1+cu128 already installed"
fi

# 2. Verify FA3 imports cleanly
echo "[setup] verifying flash-attn-3 import..."
python3 -c "
from flash_attn_interface import flash_attn_func
import torch
print(f'  torch: {torch.__version__}')
print(f'  cuda available: {torch.cuda.is_available()}')
print(f'  device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')
print('  flash_attn_3: OK')
"

# 3. Other deps
echo "[setup] installing brotli + sentencepiece + huggingface_hub..."
pip install --quiet brotli sentencepiece huggingface_hub numpy

# 4. Sanity check imports for the train script
echo "[setup] verifying train.py imports..."
python3 -c "
import sentencepiece
import brotli
import torch
import numpy as np
from huggingface_hub import hf_hub_download
print(f'  sentencepiece: {sentencepiece.__version__}')
print(f'  brotli: ok')
print(f'  numpy: {np.__version__}')
print(f'  huggingface_hub: ok')
"

echo "[setup] OK"
