#!/bin/bash
# u07_gla_shootout.sh — Test GLA (Gated Linear Attention) vs standard attention
# Hardware: 1xH100
# Time: ~30 min (3 architectures × 10 min each)
# Cost: ~$1.50
#
# WHY: GLA is O(n) instead of O(n²). Could be 2-3x faster at long seq,
# allowing MORE training steps in the 10-min budget. From PLAN.md:
# "if GLA is >30% faster at same loss, SWITCH to GLA"
#
# This is the "risk it for the biscuit" experiment. Failed to install on
# 3080 Ti due to Triton version. Should work on H100 with newer PyTorch.
#
# REQUIRES: pip install flash-linear-attention

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=========================================="
echo "U07: GLA SHOOTOUT"
echo "=========================================="
echo
echo "Comparing 3 attention variants at 50 steps each:"
echo "  A: Standard transformer (control)"
echo "  B: GLA — Gated Linear Attention"
echo "  C: RWKV-7 — recurrent variant from same library"
echo

# Try to install flash-linear-attention
echo "Installing flash-linear-attention..."
pip install -q flash-linear-attention 2>&1 | tail -5 || {
    echo "✗ FAIL: couldn't install flash-linear-attention"
    echo "  Try: pip install flash-linear-attention --no-build-isolation"
    exit 1
}

# Verify it imports
python3 -c "import fla; print('fla version:', fla.__version__)" || {
    echo "✗ FAIL: fla doesn't import"
    exit 1
}

echo "✓ flash-linear-attention installed"
echo

mkdir -p logs/u07

COMMON="
NUM_LAYERS=11
MODEL_DIM=512
MLP_EXPANSION=3
TRAIN_BATCH_TOKENS=1024 VAL_BATCH_SIZE=131072 VAL_LOSS_EVERY=0 SKIP_FINAL_EVAL=1
GRAD_ACCUM_STEPS=1
WARMUP_STEPS=10
ITERATIONS=50
MAX_WALLCLOCK_SECONDS=0
TRAIN_LOG_EVERY=10


"

# === Run A: Standard transformer ===
echo "--- Run A: STANDARD ATTENTION ---"
env $COMMON \
    ATTENTION_TYPE=standard \
    SEQ_LEN=2048 \
    python3 train_gpt.py 2>&1 | tee logs/u07/A_standard.log

# === Run B: GLA ===
echo
echo "--- Run B: GLA ---"
env $COMMON \
    ATTENTION_TYPE=gla \
    SEQ_LEN=2048 \
    python3 train_gpt.py 2>&1 | tee logs/u07/B_gla.log

# === Run C: RWKV-7 ===
echo
echo "--- Run C: RWKV-7 ---"
env $COMMON \
    ATTENTION_TYPE=rwkv7 \
    SEQ_LEN=2048 \
    python3 train_gpt.py 2>&1 | tee logs/u07/C_rwkv7.log

# === Compare ===
echo
echo "=========================================="
echo "GLA SHOOTOUT RESULTS"
echo "=========================================="
echo
printf "%-20s  %12s  %14s\n" "Architecture" "ms/step" "loss@step50"
printf "%-20s  %12s  %14s\n" "----------------" "------------" "--------------"

for NAME in "A_standard" "B_gla" "C_rwkv7"; do
    MS=$(grep 'step:50' logs/u07/${NAME}.log | grep -oE 'step_avg:[0-9.]+' | tail -1 | cut -d: -f2)
    LOSS=$(grep 'step:50' logs/u07/${NAME}.log | grep -oE 'train_loss:[0-9.]+' | tail -1 | cut -d: -f2)
    printf "%-20s  %12s  %14s\n" "$NAME" "${MS:-N/A}" "${LOSS:-N/A}"
done

echo
echo "Decision rule: if any GLA/RWKV is >30% faster at similar loss, SWITCH."
echo "Otherwise stay with standard attention."
