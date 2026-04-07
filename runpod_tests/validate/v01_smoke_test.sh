#!/bin/bash
# v01_smoke_test.sh — Fast 50-step smoke test (base train_gpt.py)
# Goal: verify CUDA port runs and loss decreases. NOT a quality test.
# Hardware: any GPU with ≥10GB
# Time: ~30 sec on a 3080 Ti (down from 9 min before optimization)
# Cost: ~$0.005
#
# Speed tricks (from GPU_RESULTS.md findings):
#   1. TRAIN_SEQ_LEN=128 — 6.1x faster than 1024 (per gpu_speed_test.py)
#   2. TRAIN_BATCH_TOKENS=1024 — sane batch, kills microbatch overhead
#   3. VAL_BATCH_SIZE=131072 — fast val pass
#   4. VAL_LOSS_EVERY=0 SKIP_FINAL_EVAL=1 — no mid-training val
#   5. We don't care about quality here, just "does it run"

set -e
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=== V01: SMOKE TEST (base train_gpt.py, fast settings) ==="
echo

ITERATIONS=50 \
VAL_LOSS_EVERY=0 SKIP_FINAL_EVAL=1 \
TRAIN_LOG_EVERY=10 \
TRAIN_SEQ_LEN=128 \
TRAIN_BATCH_TOKENS=1024 \
GRAD_ACCUM_STEPS=1 \
VAL_BATCH_SIZE=131072 \
WARMUP_STEPS=10 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_gpt.py 2>&1 | tee runpod_tests/logs/v01_smoke.log

# Check pass conditions
echo
echo "=== VALIDATION ==="

LOSS_INITIAL=$(grep 'step:1\b' runpod_tests/logs/v01_smoke.log | grep -oE 'train_loss:[0-9.]+' | head -1 | cut -d: -f2)
LOSS_FINAL=$(grep 'step:50' runpod_tests/logs/v01_smoke.log | grep -oE 'train_loss:[0-9.]+' | head -1 | cut -d: -f2)

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
