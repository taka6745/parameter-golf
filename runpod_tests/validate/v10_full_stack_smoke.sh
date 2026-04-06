#!/bin/bash
# v10_full_stack_smoke.sh — Run 100 steps with the FULL stack
# Hardware: 3060
# Time: ~5 min
#
# This is the final validation: every patch applied, run for 100 steps,
# verify nothing crashes and loss decreases.

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=== V10: FULL STACK SMOKE TEST ==="
echo
echo "All patches: arch (LeakyReLU, softcap, wavelet) +"
echo "  n-gram bias + progressive seq + cosine LR + EMA"
echo "Running 100 steps with everything enabled..."
echo

PROGRESSIVE_SEQ=1 \
PHASE1_SEQ_LEN=128 \
PHASE1_LR_MULT=25.0 \
PHASE1_FRACTION=0.85 \
PHASE2_SEQ_LEN=1024 \
PHASE1_NGRAM_WEIGHT=0.40 \
PHASE2_NGRAM_WEIGHT=0.05 \
\
USE_NGRAM_BIAS=1 \
USE_WAVELET=1 \
USE_EMA=1 \
EMA_DECAY=0.997 \
\
NUM_LAYERS=11 \
MODEL_DIM=512 \
MLP_EXPANSION=3 \
\
ITERATIONS=100 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=20 \
TRAIN_BATCH_TOKENS=8192 \
GRAD_ACCUM_STEPS=8 \
WARMUP_STEPS=10 \
\
TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model" \
DATA_PATH="./data/datasets/fineweb10B_bpe8192" \
python3 train_gpt.py 2>&1 | tee /tmp/v10_full_stack.log

echo
echo "=== VALIDATION ==="

# Check for crashes
if grep -q 'Error\|Traceback\|RuntimeError' /tmp/v10_full_stack.log; then
    echo "✗ FAIL: errors detected"
    grep 'Error\|Traceback' /tmp/v10_full_stack.log | head -5
    exit 1
fi

# Check for NaN
if grep -q 'nan\|NaN\|inf\|Inf' /tmp/v10_full_stack.log; then
    echo "✗ FAIL: NaN/Inf detected"
    exit 1
fi

# Check loss decreased
INITIAL=$(grep 'step:' /tmp/v10_full_stack.log | grep -oE 'train_loss:[0-9.]+' | head -1 | cut -d: -f2)
FINAL=$(grep 'step:' /tmp/v10_full_stack.log | grep -oE 'train_loss:[0-9.]+' | tail -1 | cut -d: -f2)
echo "Loss: $INITIAL → $FINAL"

if [ -z "$FINAL" ]; then
    echo "✗ FAIL: no loss output"
    exit 1
fi

if (( $(echo "$FINAL < $INITIAL" | bc -l) )); then
    echo
    echo "✓ PASS: full stack runs cleanly, loss decreases"
    echo "  Ready to proceed to unknown/ tests"
    exit 0
else
    echo "✗ FAIL: loss did not decrease"
    exit 1
fi
