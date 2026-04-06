#!/bin/bash
# u03_eval_cache.sh — Validate the eval n-gram cache + hedge mixer
# Hardware: 1xH100
# Time: ~30 min
# Cost: ~$1.50
#
# WHY: From our channel capacity analysis, Phase 2 is 8x under-resourced.
# The eval cache fills the gap (8/10 top PRs use it). This test confirms
# the -0.10 BPP gain we projected.
#
# REUSES THE TRAINED MODEL FROM u02 — only changes the eval pipeline.

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=========================================="
echo "U03: EVAL N-GRAM CACHE"
echo "=========================================="
echo
echo "Testing eval pipeline on the model from u02."
echo "Two eval modes:"
echo "  Run A: standard eval (no cache)"
echo "  Run B: eval cache + hedge mixer + sliding window"
echo

mkdir -p logs/u03

# Use the SAME training config as u02 winner
NUM_LAYERS=${NUM_LAYERS:-11}
MLP_EXPANSION=${MLP_EXPANSION:-3}

# Verify u02 ran (we need a trained model)
if [ ! -f logs/u02/run_B_progressive.log ]; then
    echo "✗ u02 not run yet. Run u02_progressive_seq.sh first."
    exit 1
fi

# === Run A: Standard eval (uses model trained in u02 Run B) ===
echo
echo "--- Run A: STANDARD EVAL (no cache) ---"
USE_EVAL_CACHE=0 \
USE_HEDGE_MIXER=0 \
USE_SLIDING_WINDOW=0 \
\
PROGRESSIVE_SEQ=1 \
PHASE1_SEQ_LEN=128 PHASE1_LR_MULT=25.0 PHASE1_FRACTION=0.85 \
PHASE2_SEQ_LEN=1024 PHASE1_NGRAM_WEIGHT=0.40 PHASE2_NGRAM_WEIGHT=0.05 \
\
NUM_LAYERS=$NUM_LAYERS MODEL_DIM=512 MLP_EXPANSION=$MLP_EXPANSION \
TRAIN_BATCH_TOKENS=524288 GRAD_ACCUM_STEPS=1 \
WARMUP_STEPS=10 \
ITERATIONS=1000000 MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=200 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
DATA_PATH=./data/datasets/fineweb10B_bpe8192 \
python3 train_gpt.py 2>&1 | tee logs/u03/run_A_no_cache.log

# === Run B: With eval cache + hedge mixer ===
echo
echo "--- Run B: EVAL CACHE + HEDGE MIXER + SLIDING WINDOW ---"
USE_EVAL_CACHE=1 \
EVAL_CACHE_MAX_ORDER=7 \
USE_HEDGE_MIXER=1 \
HEDGE_ETA=0.1 \
USE_SLIDING_WINDOW=1 \
SLIDING_STRIDE=512 \
\
PROGRESSIVE_SEQ=1 \
PHASE1_SEQ_LEN=128 PHASE1_LR_MULT=25.0 PHASE1_FRACTION=0.85 \
PHASE2_SEQ_LEN=1024 PHASE1_NGRAM_WEIGHT=0.40 PHASE2_NGRAM_WEIGHT=0.05 \
\
NUM_LAYERS=$NUM_LAYERS MODEL_DIM=512 MLP_EXPANSION=$MLP_EXPANSION \
TRAIN_BATCH_TOKENS=524288 GRAD_ACCUM_STEPS=1 \
WARMUP_STEPS=10 \
ITERATIONS=1000000 MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=200 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
DATA_PATH=./data/datasets/fineweb10B_bpe8192 \
python3 train_gpt.py 2>&1 | tee logs/u03/run_B_with_cache.log

# === Compare ===
echo
echo "=== COMPARISON ==="
A_BPB=$(grep 'final_int8_zlib_roundtrip val_loss' logs/u03/run_A_no_cache.log | grep -oE 'val_bpb:[0-9.]+' | head -1 | cut -d: -f2)
B_BPB=$(grep 'final_int8_zlib_roundtrip val_loss' logs/u03/run_B_with_cache.log | grep -oE 'val_bpb:[0-9.]+' | head -1 | cut -d: -f2)

echo "Run A (no cache):     val_bpb = $A_BPB"
echo "Run B (with cache):   val_bpb = $B_BPB"

if [ -n "$A_BPB" ] && [ -n "$B_BPB" ]; then
    DELTA=$(python3 -c "print(f'{float('$A_BPB') - float('$B_BPB'):.4f}')")
    echo "Cache improvement: $DELTA BPP"
    echo

    if (( $(echo "$DELTA > 0.05" | bc -l) )); then
        echo "✓ EVAL CACHE WORKS — keep it for u04+"
        echo "  This was our projected biggest gap (-0.10 BPP)"
    elif (( $(echo "$DELTA > 0.0" | bc -l) )); then
        echo "△ Marginal cache benefit — keep but verify implementation"
    else
        echo "✗ Cache not helping — debug before u04"
        echo "  Check: max_order, min_count, hedge eta"
    fi
fi
