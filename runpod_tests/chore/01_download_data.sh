#!/bin/bash
# 01_download_data.sh — Download FineWeb shards
# Runs on: any GPU pod
# Time: ~8 min (depends on network)

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=== DOWNLOAD FINEWEB DATA ==="
echo

# Download SP-1024 baseline data first (default tokenizer)
# This is needed by the standard train_gpt.py
python3 data/cached_challenge_fineweb.py \
    --variant sp1024 \
    --train-shards 10 \
    2>&1 | tail -20

echo
echo "✓ Data downloaded to data/datasets/"
ls -la data/datasets/
