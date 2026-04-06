#!/bin/bash
# u01_arch_sweep.sh — Find best architecture (layers × MLP expansion)
# Hardware: 3060 (cheap exploration), then 1xH100 to confirm winner
# Time: ~30 min on 3060, ~10 min for H100 confirmation
# Cost: ~$0.15 (3060) + $1.50 (H100 confirm)
#
# WHY: Mac said 9L > 11L at 1000 steps, but H100 has 7000+ steps and
# all top PRs use 11L. We need a real number, not extrapolation.

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=== U01: ARCHITECTURE SWEEP ==="
echo
echo "Testing 4 configs at 500 steps each on 3060:"
echo "  A: 9L,  2x MLP (Mac winner)"
echo "  B: 11L, 2x MLP (more layers only)"
echo "  C: 9L,  3x MLP (more MLP only)"
echo "  D: 11L, 3x MLP (both)"
echo

mkdir -p logs/u01

# Common config (no progressive seq, no eval cache — just vanilla)
COMMON="
TRAIN_BATCH_TOKENS=8192
GRAD_ACCUM_STEPS=8
WARMUP_STEPS=10
MAX_WALLCLOCK_SECONDS=0
ITERATIONS=500
TRAIN_LOG_EVERY=100
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
DATA_PATH=./data/datasets/fineweb10B_bpe8192
"

run_config() {
    local NAME=$1
    local LAYERS=$2
    local EXPANSION=$3
    echo
    echo "--- Config $NAME: ${LAYERS}L, ${EXPANSION}x MLP ---"
    env $COMMON \
        NUM_LAYERS=$LAYERS \
        MODEL_DIM=512 \
        MLP_EXPANSION=$EXPANSION \
        python3 train_gpt.py 2>&1 | tee logs/u01/config_${NAME}.log
}

run_config A 9  2
run_config B 11 2
run_config C 9  3
run_config D 11 3

# Compare final losses
echo
echo "=== RESULTS ==="
echo
for NAME in A B C D; do
    LOSS=$(grep 'step:500' logs/u01/config_${NAME}.log | grep -oE 'train_loss:[0-9.]+' | tail -1 | cut -d: -f2)
    VAL_BPB=$(grep 'val_bpb' logs/u01/config_${NAME}.log | grep -oE 'val_bpb:[0-9.]+' | tail -1 | cut -d: -f2 || echo "N/A")
    echo "Config $NAME: train_loss=$LOSS, val_bpb=$VAL_BPB"
done

echo
echo "Pick the WINNER and use those NUM_LAYERS / MLP_EXPANSION for u02-u05"
echo
echo "Note: 500 steps on 3060 is small. The winner here may NOT be the"
echo "winner at 7000 steps on H100. If A and D are within 0.02 BPP, prefer D"
echo "(more layers + MLP usually wins at scale)."
