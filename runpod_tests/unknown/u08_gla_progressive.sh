#!/bin/bash
# u08_gla_progressive.sh — GLA + Progressive Seq combined
# Hardware: 1xH100
# Time: ~30 min
# Cost: ~$1.50
#
# WHY: u02 validates progressive seq with standard attention.
# u07 validates GLA at 50 steps.
# Neither tests the COMBINATION which could compound:
#
#   Standard + progressive seq → 8x more steps (validated on 3080Ti)
#   GLA + progressive seq → 8x × ~1.5x = 12x more steps (projected)
#
# The 8x in Phase 1 is already maxed out (seq=128 is short for any attention).
# The real win is in Phase 2: GLA at seq=1024 is much faster than standard,
# so Phase 2 could go from 1058 steps → 2000+ steps → fixes the 8x under-resource.
#
# REQUIRES: u07 must have passed (GLA installed)

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=========================================="
echo "U08: GLA + PROGRESSIVE SEQ COMBINED"
echo "=========================================="
echo

# Verify GLA is installed
python3 -c "import fla" 2>/dev/null || {
    echo "✗ FAIL: flash-linear-attention not installed"
    echo "  Run u07_gla_shootout.sh first"
    exit 1
}

# Verify u07 showed GLA isn't broken
if [ ! -f logs/u07/B_gla.log ]; then
    echo "✗ FAIL: u07 not run yet"
    echo "  Run u07_gla_shootout.sh first"
    exit 1
fi

mkdir -p logs/u08

NUM_LAYERS=${NUM_LAYERS:-11}
MLP_EXPANSION=${MLP_EXPANSION:-3}

echo "Testing GLA + progressive seq with same training budget."
echo "Compare against u02_run_B_progressive.log (standard + progressive)."
echo

env \
    ATTENTION_TYPE=gla \
    \
    PROGRESSIVE_SEQ=1 \
    PHASE1_SEQ_LEN=128 PHASE1_LR_MULT=25.0 PHASE1_FRACTION=0.85 \
    PHASE2_SEQ_LEN=1024 PHASE1_NGRAM_WEIGHT=0.40 PHASE2_NGRAM_WEIGHT=0.05 \
    \
    USE_NGRAM_BIAS=1 \
    USE_WAVELET=1 \
    USE_EMA=1 EMA_DECAY=0.997 \
    \
    NUM_LAYERS=$NUM_LAYERS \
    MODEL_DIM=512 \
    MLP_EXPANSION=$MLP_EXPANSION \
    \
    TRAIN_BATCH_TOKENS=524288 \
    GRAD_ACCUM_STEPS=1 \
    WARMUP_STEPS=10 \
    ITERATIONS=1000000 \
    MAX_WALLCLOCK_SECONDS=600 \
    TRAIN_LOG_EVERY=200 \
    \
    TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
    DATA_PATH=./data/datasets/fineweb10B_bpe8192 \
    python3 train_gpt.py 2>&1 | tee logs/u08/gla_progressive.log

echo
echo "=== U08 RESULT ==="

# Count steps achieved
TOTAL_STEPS=$(grep 'step:' logs/u08/gla_progressive.log | grep -v warmup | tail -1 | grep -oE 'step:[0-9]+' | head -1 | cut -d: -f2)
GLA_BPB=$(grep 'final_int8_zlib_roundtrip val_loss' logs/u08/gla_progressive.log | grep -oE 'val_bpb:[0-9.]+' | head -1 | cut -d: -f2)
echo "Steps achieved: $TOTAL_STEPS"
echo "Final val_bpb: $GLA_BPB"

# Compare with u02 standard + progressive
if [ -f logs/u02/run_B_progressive.log ]; then
    STD_STEPS=$(grep 'step:' logs/u02/run_B_progressive.log | grep -v warmup | tail -1 | grep -oE 'step:[0-9]+' | head -1 | cut -d: -f2)
    STD_BPB=$(grep 'final_int8_zlib_roundtrip val_loss' logs/u02/run_B_progressive.log | grep -oE 'val_bpb:[0-9.]+' | head -1 | cut -d: -f2)
    echo
    echo "=== COMPARISON ==="
    echo "u02 (standard + progressive): $STD_STEPS steps, $STD_BPB BPP"
    echo "u08 (GLA      + progressive): $TOTAL_STEPS steps, $GLA_BPB BPP"

    if [ -n "$STD_STEPS" ] && [ -n "$TOTAL_STEPS" ]; then
        SPEEDUP=$(python3 -c "print(f'{$TOTAL_STEPS / $STD_STEPS:.2f}')")
        echo "Step speedup: ${SPEEDUP}x"
    fi

    if [ -n "$STD_BPB" ] && [ -n "$GLA_BPB" ]; then
        DELTA=$(python3 -c "print(f'{float('$STD_BPB') - float('$GLA_BPB'):.4f}')")
        echo "Quality delta: $DELTA BPP (negative = GLA worse)"
        echo
        if (( $(echo "$DELTA > 0" | bc -l) )); then
            echo "✓ GLA + PROGRESSIVE WINS — use this for u04+"
        elif (( $(echo "$DELTA > -0.02" | bc -l) )); then
            echo "△ GLA roughly matches standard — keep standard for safety"
        else
            echo "✗ GLA + progressive loses — stick with standard attention"
        fi
    fi
fi
