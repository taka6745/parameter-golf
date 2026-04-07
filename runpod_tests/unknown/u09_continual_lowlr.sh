#!/bin/bash
# u09_continual_lowlr.sh — Two-stage continual training with reduced LR
# Hardware: 1xH100 (or 3060 for relative comparison)
# Time: ~30 min
# Cost: ~$1.50
#
# WHY: Mac validation showed continual training from a converged checkpoint
# WORKS at -0.0259 BPP (1.6929 → 1.6670) IF you reduce the LR ~4x in stage 2.
# Previous "stage 2 fails" finding was due to full LR pushing the model away
# from its minimum after optimizer reset. Low LR keeps it near the minimum
# and explores it more carefully.
#
# This is conceptually identical to our progressive seq Phase 2 (lower LR
# after a high-LR phase), but applied as a SEPARATE training stage rather
# than a continuous schedule. It might compound with progressive seq.
#
# Two runs:
#   Run A: single-stage training, 10 min straight (baseline)
#   Run B: stage 1 (8 min) + stage 2 with matrix_lr/4 (2 min)

set -e
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=========================================="
echo "U09: TWO-STAGE CONTINUAL TRAINING (LOW LR)"
echo "=========================================="
echo
echo "Mac validated -0.0259 BPP from this trick."
echo "Two runs:"
echo "  Run A: single-stage 10 min training (baseline)"
echo "  Run B: 8 min stage 1 + 2 min stage 2 at matrix_lr/4"
echo

mkdir -p logs/u09

NUM_LAYERS=${NUM_LAYERS:-11}
MLP_EXPANSION=${MLP_EXPANSION:-3}

# Detect hardware
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
echo "GPU: $GPU_NAME"
echo

if [[ "$GPU_NAME" == *"3060"* ]] || [[ "$GPU_NAME" == *"3080"* ]] || [[ "$GPU_NAME" == *"4070"* ]]; then
    BATCH=1024
    GA=8
    STAGE1_WALL=120  # 2 min
    STAGE2_WALL=30   # 30 sec
    TOTAL_WALL=150
    echo "Cheap GPU detected — running short test (2.5 min total)"
elif [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"A100"* ]]; then
    BATCH=524288
    GA=1
    STAGE1_WALL=480  # 8 min
    STAGE2_WALL=120  # 2 min
    TOTAL_WALL=600   # 10 min
    echo "Real GPU detected — running full 10 min test"
else
    BATCH=4096
    GA=1
    STAGE1_WALL=240
    STAGE2_WALL=60
    TOTAL_WALL=300
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
TRAIN_LOG_EVERY=200
TOKENIZER_PATH=./data/tokenizers/sp_bpe_1024.model
DATA_PATH=./data/datasets/fineweb10B_sp1024
"

# === Run A: single-stage baseline ===
echo
echo "--- Run A: SINGLE-STAGE BASELINE (${TOTAL_WALL}s straight) ---"
env $COMMON \
    MAX_WALLCLOCK_SECONDS=$TOTAL_WALL \
    MATRIX_LR=0.04 \
    python3 train_gpt.py 2>&1 | tee logs/u09/run_A_single.log

# === Run B Stage 1: normal LR for STAGE1_WALL ===
echo
echo "--- Run B Stage 1: ${STAGE1_WALL}s at matrix_lr=0.04 ---"
env $COMMON \
    MAX_WALLCLOCK_SECONDS=$STAGE1_WALL \
    MATRIX_LR=0.04 \
    SAVE_CHECKPOINT=logs/u09/stage1_checkpoint.pt \
    python3 train_gpt.py 2>&1 | tee logs/u09/run_B_stage1.log

# === Run B Stage 2: low LR continual ===
echo
echo "--- Run B Stage 2: ${STAGE2_WALL}s at matrix_lr=0.01 (4x lower) ---"
env $COMMON \
    MAX_WALLCLOCK_SECONDS=$STAGE2_WALL \
    MATRIX_LR=0.01 \
    LOAD_CHECKPOINT=logs/u09/stage1_checkpoint.pt \
    python3 train_gpt.py 2>&1 | tee logs/u09/run_B_stage2.log

# === Compare ===
echo
echo "=== U09 RESULTS ==="
A_BPB=$(grep 'final_int8_zlib_roundtrip val_loss' logs/u09/run_A_single.log | grep -oE 'val_bpb:[0-9.]+' | head -1 | cut -d: -f2)
B_STAGE1_BPB=$(grep 'final_int8_zlib_roundtrip val_loss' logs/u09/run_B_stage1.log | grep -oE 'val_bpb:[0-9.]+' | head -1 | cut -d: -f2)
B_STAGE2_BPB=$(grep 'final_int8_zlib_roundtrip val_loss' logs/u09/run_B_stage2.log | grep -oE 'val_bpb:[0-9.]+' | head -1 | cut -d: -f2)

echo "Run A (single-stage):    val_bpb = $A_BPB"
echo "Run B Stage 1 only:      val_bpb = $B_STAGE1_BPB"
echo "Run B Stage 1 + Stage 2: val_bpb = $B_STAGE2_BPB"

if [ -n "$A_BPB" ] && [ -n "$B_STAGE2_BPB" ]; then
    DELTA=$(python3 -c "print(f'{float('$A_BPB') - float('$B_STAGE2_BPB'):.4f}')")
    echo
    echo "Stage 2 vs single-stage delta: $DELTA BPP"
    if (( $(echo "$DELTA > 0.01" | bc -l) )); then
        echo "✓ Two-stage WORKS — low-LR continual stage helps"
        echo "  Mac validated this at -0.026 BPP. CUDA result: $DELTA BPP."
        echo "  Use this for u04 (full stack) and u05 (3-seed)."
    elif (( $(echo "$DELTA > 0" | bc -l) )); then
        echo "△ Marginal gain — keep an eye on it but don't depend on it"
    else
        echo "✗ Two-stage doesn't help on CUDA at this scale"
        echo "  Stick with single-stage progressive seq from u02."
    fi
fi

# Note about checkpointing
if [ ! -f logs/u09/stage1_checkpoint.pt ]; then
    echo
    echo "NOTE: SAVE_CHECKPOINT/LOAD_CHECKPOINT may not be supported by"
    echo "  the unpatched train_gpt.py. If Run B Stage 2 uses a fresh model"
    echo "  instead of loading from stage 1, the comparison is invalid."
    echo "  Patch train_gpt.py to support these env vars before trusting u09 results."
fi
