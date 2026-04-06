#!/bin/bash
# v01_smoke_test.sh — 50 steps with the BASE competition train_gpt.py
# Goal: verify CUDA port runs at all
# Hardware: 3060 (12 GB)
# Time: ~3 min
# Cost: ~$0.01

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=== V01: SMOKE TEST (base train_gpt.py) ==="
echo

# Use the DEFAULT competition config — no patches yet
# Just verify the unchanged script trains without errors
ITERATIONS=50 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=10 \
TRAIN_BATCH_TOKENS=8192 \
GRAD_ACCUM_STEPS=8 \
WARMUP_STEPS=10 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_gpt.py 2>&1 | tee /tmp/v01_smoke.log

# Check pass conditions
echo
echo "=== VALIDATION ==="

LOSS_INITIAL=$(grep 'step:1\b' /tmp/v01_smoke.log | grep -oE 'train_loss:[0-9.]+' | head -1 | cut -d: -f2)
LOSS_FINAL=$(grep 'step:50' /tmp/v01_smoke.log | grep -oE 'train_loss:[0-9.]+' | head -1 | cut -d: -f2)

echo "Initial loss: $LOSS_INITIAL"
echo "Final loss:   $LOSS_FINAL"

if [ -z "$LOSS_INITIAL" ] || [ -z "$LOSS_FINAL" ]; then
    echo "✗ FAIL: couldn't parse loss"
    exit 1
fi

if (( $(echo "$LOSS_FINAL < $LOSS_INITIAL" | bc -l) )); then
    echo "✓ PASS: loss decreased ($LOSS_INITIAL → $LOSS_FINAL)"
    exit 0
else
    echo "✗ FAIL: loss did not decrease"
    exit 1
fi
