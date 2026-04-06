#!/bin/bash
# 03_retokenize.sh — Re-export training data using BPE-8192
# Runs on: any pod (CPU work)
# Time: ~8 min for 10 shards

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=== RE-TOKENIZE WITH BPE-8192 ==="
echo

if [ -d "data/datasets/fineweb10B_bpe8192" ] && [ "$(ls data/datasets/fineweb10B_bpe8192/fineweb_train_*.bin 2>/dev/null | wc -l)" -ge 10 ]; then
    echo "✓ BPE-8192 data already exists, skipping"
    exit 0
fi

# Use the data prep script with BPE-8192 variant
python3 data/cached_challenge_fineweb.py \
    --variant bpe8192 \
    --train-shards 10 \
    --tokenizer data/tokenizers/fineweb_8192_bpe.model \
    2>&1 | tail -20

echo
echo "✓ Re-tokenized data ready"
ls -la data/datasets/fineweb10B_bpe8192/ 2>/dev/null || ls -la data/datasets/
