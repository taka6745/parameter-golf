#!/bin/bash
# u04_full_stack.sh — Full stack: progressive seq + cache + TTT + per-layer quant
# Hardware: 1xH100
# Time: ~30 min (10 min training + 5 min eval + buffer)
# Cost: ~$1.50
#
# This is the maximum-effort single-seed run. Validates everything stacks.

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=========================================="
echo "U04: FULL STACK SINGLE SEED"
echo "=========================================="
echo
echo "Everything enabled:"
echo "  - 11L / 3x MLP / seq=2048 / QK 4.0 / XSA last 4"
echo "  - WaveletGPT, NorMuon, LeakyReLU(0.5)², EMA(0.997)"
echo "  - BPE-8192 tokenizer + n-gram bias (bigram, trigram, 4-gram)"
echo "  - DC500 categories"
echo "  - Progressive seq + high LR + cosine Phase 2"
echo "  - Eval cache (orders 2-7) + hedge mixer + sliding window"
echo "  - Score-First TTT (LoRA on last 2 layers)"
echo "  - Per-layer mixed precision GPTQ + Lloyd-Max codebook"
echo

mkdir -p logs/u04

# Final config — adjust based on u01/u02/u03 results
NUM_LAYERS=${NUM_LAYERS:-11}
MLP_EXPANSION=${MLP_EXPANSION:-3}
USE_PROGRESSIVE=${USE_PROGRESSIVE:-1}  # set to 0 if u02 showed it doesn't help

env \
    PROGRESSIVE_SEQ=$USE_PROGRESSIVE \
    PHASE1_SEQ_LEN=128 PHASE1_LR_MULT=25.0 PHASE1_FRACTION=0.85 \
    PHASE2_SEQ_LEN=1024 PHASE1_NGRAM_WEIGHT=0.40 PHASE2_NGRAM_WEIGHT=0.05 \
    \
    USE_NGRAM_BIAS=1 \
    USE_WAVELET=1 \
    USE_EMA=1 EMA_DECAY=0.997 \
    \
    USE_EVAL_CACHE=1 EVAL_CACHE_MAX_ORDER=7 \
    USE_HEDGE_MIXER=1 HEDGE_ETA=0.1 \
    USE_SLIDING_WINDOW=1 SLIDING_STRIDE=512 \
    \
    USE_SCORE_FIRST_TTT=1 \
    TTT_LR=1e-4 \
    TTT_LAYERS=last2 \
    TTT_EPOCHS=1 \
    \
    USE_MIXED_PRECISION=1 \
    QUANT_LAYERS_LOW="0,1,2,3:int7" \
    QUANT_LAYERS_MID="4,5,6:int6" \
    QUANT_LAYERS_HIGH="7,8,9,10:int5" \
    USE_LLOYD_MAX=1 \
    \
    NUM_LAYERS=$NUM_LAYERS \
    MODEL_DIM=512 \
    MLP_EXPANSION=$MLP_EXPANSION \
    SEQ_LEN=2048 \
    QK_GAIN=4.0 \
    USE_XSA=1 \
    XSA_LAYERS=4 \
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
    python3 train_gpt.py 2>&1 | tee logs/u04/full_stack.log

# Extract final score
echo
echo "=== U04 RESULT ==="
FINAL_BPB=$(grep 'final_int8_zlib_roundtrip val_loss' logs/u04/full_stack.log | grep -oE 'val_bpb:[0-9.]+' | head -1 | cut -d: -f2)
ARTIFACT_SIZE=$(grep 'Total submission size' logs/u04/full_stack.log | grep -oE '[0-9]+' | head -1)

echo "Final val_bpb:    $FINAL_BPB"
echo "Artifact size:    $ARTIFACT_SIZE bytes"

if [ -n "$FINAL_BPB" ]; then
    if (( $(echo "$FINAL_BPB < 1.00" | bc -l) )); then
        echo "✓ SUB-1.0 BPP — proceed to u05 (3-seed final)"
    elif (( $(echo "$FINAL_BPB < 1.08" | bc -l) )); then
        echo "✓ Beats merged SOTA (1.08) — proceed to u05"
    else
        echo "△ Above SOTA — review what didn't work before u05"
    fi
fi

if [ -n "$ARTIFACT_SIZE" ] && [ "$ARTIFACT_SIZE" -gt 16777216 ]; then
    echo "✗ ARTIFACT TOO BIG ($ARTIFACT_SIZE > 16777216) — must compress more"
fi
