#!/bin/bash
# u02_progressive_seq.sh — DOES PROGRESSIVE SEQ TRANSFER TO H100?
# Hardware: 3060 first (60s), then 1xH100 (10 min)
# Time: ~30 min total
# Cost: ~$1.50
#
# THE BIGGEST UNKNOWN. Mac validated -25% eval loss on 3080 Ti.
# We need to confirm this happens on H100 with the real train_gpt.py.

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=========================================="
echo "U02: PROGRESSIVE SEQ ON CUDA"
echo "=========================================="
echo
echo "This is THE big experiment. Two runs:"
echo "  Run A: standard training (baseline)"
echo "  Run B: progressive seq + cosine LR Phase 2"
echo "Difference between A and B is the gain."
echo

# Use the architecture winner from u01
NUM_LAYERS=${NUM_LAYERS:-11}
MLP_EXPANSION=${MLP_EXPANSION:-3}
echo "Architecture: ${NUM_LAYERS}L / ${MLP_EXPANSION}x MLP"
echo

mkdir -p logs/u02

# Detect hardware
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "GPU: $GPU_NAME"
echo

if [[ "$GPU_NAME" == *"3060"* ]] || [[ "$GPU_NAME" == *"3080"* ]] || [[ "$GPU_NAME" == *"4070"* ]]; then
    # Cheap GPU: just verify it doesn't crash
    WALL=120
    BATCH=1024
    GA=8
    echo "Cheap GPU detected — running 2 min smoke test only"
elif [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"A100"* ]]; then
    # Real test
    WALL=600
    BATCH=524288
    GA=1
    echo "Real GPU detected — running full 10 min test"
else
    WALL=300
    BATCH=4096
    GA=1
    echo "Unknown GPU, running medium test"
fi

COMMON="
NUM_LAYERS=$NUM_LAYERS
MODEL_DIM=512
MLP_EXPANSION=$MLP_EXPANSION
TRAIN_BATCH_TOKENS=$BATCH
GRAD_ACCUM_STEPS=$GA
WARMUP_STEPS=10
ITERATIONS=1000000
MAX_WALLCLOCK_SECONDS=$WALL
TRAIN_LOG_EVERY=200
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
DATA_PATH=./data/datasets/fineweb10B_bpe8192
"

# === Run A: baseline (no progressive seq) ===
echo
echo "--- Run A: BASELINE (standard training) ---"
env $COMMON \
    PROGRESSIVE_SEQ=0 \
    python3 train_gpt.py 2>&1 | tee logs/u02/run_A_baseline.log

# === Run B: progressive seq + cosine Phase 2 ===
echo
echo "--- Run B: PROGRESSIVE SEQ + COSINE PHASE 2 ---"
env $COMMON \
    PROGRESSIVE_SEQ=1 \
    PHASE1_SEQ_LEN=128 \
    PHASE1_LR_MULT=25.0 \
    PHASE1_FRACTION=0.85 \
    PHASE2_SEQ_LEN=1024 \
    PHASE1_NGRAM_WEIGHT=0.40 \
    PHASE2_NGRAM_WEIGHT=0.05 \
    python3 train_gpt.py 2>&1 | tee logs/u02/run_B_progressive.log

# === Compare ===
echo
echo "=== COMPARISON ==="
A_BPB=$(grep 'final_int8_zlib_roundtrip val_loss' logs/u02/run_A_baseline.log | grep -oE 'val_bpb:[0-9.]+' | head -1 | cut -d: -f2)
B_BPB=$(grep 'final_int8_zlib_roundtrip val_loss' logs/u02/run_B_progressive.log | grep -oE 'val_bpb:[0-9.]+' | head -1 | cut -d: -f2)

echo "Run A (baseline):    val_bpb = $A_BPB"
echo "Run B (progressive): val_bpb = $B_BPB"

if [ -n "$A_BPB" ] && [ -n "$B_BPB" ]; then
    DELTA=$(python3 -c "print(f'{float('$A_BPB') - float('$B_BPB'):.4f}')")
    echo "Improvement: $DELTA BPP"
    echo

    if (( $(echo "$DELTA > 0.05" | bc -l) )); then
        echo "✓ PROGRESSIVE SEQ WORKS — proceed to u03 with this config"
    elif (( $(echo "$DELTA > 0.0" | bc -l) )); then
        echo "△ Marginal improvement — proceed but be ready to fall back"
    else
        echo "✗ PROGRESSIVE SEQ HURTS — use baseline config for u03+"
    fi
fi
