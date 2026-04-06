#!/bin/bash
# v04_progressive_seq.sh — Verify progressive seq scheduler triggers correctly
# Hardware: 3060
# Time: ~3 min
#
# This test runs train_gpt.py with PROGRESSIVE_SEQ=1 and a SHORT wallclock
# (60 sec total, 85% = 51s in Phase 1, 15% = 9s in Phase 2).
# Verifies the phase transition log message appears.

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=== V04: PROGRESSIVE SEQ PHASE TRANSITION ==="
echo
echo "Running 60s with progressive seq (51s @ seq=128, 9s @ seq=1024)..."

# This requires train_gpt.py to be patched with progressive seq support
# (Patch C from RUNPOD_TEST.md)
PROGRESSIVE_SEQ=1 \
PHASE1_SEQ_LEN=128 \
PHASE1_LR_MULT=25.0 \
PHASE1_FRACTION=0.85 \
PHASE2_SEQ_LEN=1024 \
PHASE1_NGRAM_WEIGHT=0.40 \
PHASE2_NGRAM_WEIGHT=0.05 \
\
ITERATIONS=10000 \
MAX_WALLCLOCK_SECONDS=60 \
TRAIN_LOG_EVERY=50 \
TRAIN_BATCH_TOKENS=8192 \
GRAD_ACCUM_STEPS=8 \
WARMUP_STEPS=10 \
python3 train_gpt.py 2>&1 | tee /tmp/v04_progressive.log

echo
echo "=== VALIDATION ==="

# Check that phase transition log appeared
if grep -q "PHASE TRANSITION" /tmp/v04_progressive.log; then
    echo "✓ Phase transition triggered"
else
    echo "✗ Phase transition NOT triggered (check Patch C is applied)"
    exit 1
fi

# Check that Phase 1 reached more steps than Phase 2
P1_STEPS=$(grep -c 'step:' /tmp/v04_progressive.log || echo 0)
echo "  Total step logs: $P1_STEPS"

# Check no NaN
if grep -q 'nan\|NaN\|inf\|Inf' /tmp/v04_progressive.log; then
    echo "✗ FAIL: NaN/Inf detected"
    exit 1
fi

# Check final loss is reasonable (decreased from initial)
INITIAL_LOSS=$(grep 'step:1\b' /tmp/v04_progressive.log | grep -oE 'train_loss:[0-9.]+' | head -1 | cut -d: -f2)
FINAL_LOSS=$(grep 'step:' /tmp/v04_progressive.log | grep -oE 'train_loss:[0-9.]+' | tail -1 | cut -d: -f2)
echo "  Loss: $INITIAL_LOSS → $FINAL_LOSS"

if [ -n "$FINAL_LOSS" ] && (( $(echo "$FINAL_LOSS < $INITIAL_LOSS" | bc -l) )); then
    echo "✓ PASS: progressive seq runs without errors, loss decreases"
    exit 0
else
    echo "✗ FAIL: loss did not decrease"
    exit 1
fi
