#!/bin/bash
# u10_eval_temp_alpha.sh — Eval-time alpha + temperature stacking
# Hardware: 1xH100 (or 3060 for relative comparison)
# Time: ~30 min (mostly training, then 4 eval variants)
# Cost: ~$1.50
#
# WHY: Mac validated -0.0124 BPP from eval-time prob-space n-gram blend
# at α=0.06 + temperature T=0.93, stacked on top of a trained wavelet
# model (1.6929 → 1.6805). This is a ZERO-TRAINING-COST eval trick.
#
# It's similar to our hedge mixer (u03) but FIXED instead of adaptive:
#   p_eval = (1 - α) * softmax(model_logits / T) + α * p_ngram
# with α=0.06 and T=0.93.
#
# Worth testing because:
#   1. Cheap to validate (just changes eval, not training)
#   2. Stacks with everything else
#   3. Mac result was robust across 1000-step models
#
# Approach: train one model, then run 4 evals with different (α, T) combos.

set -e
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=========================================="
echo "U10: EVAL-TIME ALPHA + TEMPERATURE"
echo "=========================================="
echo
echo "Mac validated:"
echo "  α=0.00, T=1.00: 1.6929 BPP (baseline wavelet model)"
echo "  α=0.08, T=1.00: 1.6864 BPP (-0.0065)"
echo "  α=0.06, T=0.93: 1.6805 BPP (-0.0124)"
echo
echo "Train ONE model, then run 4 eval variants on it."
echo

mkdir -p logs/u10

NUM_LAYERS=${NUM_LAYERS:-11}
MLP_EXPANSION=${MLP_EXPANSION:-3}

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
echo "GPU: $GPU_NAME"

if [[ "$GPU_NAME" == *"3060"* ]] || [[ "$GPU_NAME" == *"3080"* ]] || [[ "$GPU_NAME" == *"4070"* ]]; then
    BATCH=1024
    GA=8
    WALL=120
elif [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"A100"* ]]; then
    BATCH=524288
    GA=1
    WALL=600
else
    BATCH=4096
    GA=1
    WALL=300
fi

# === Train the model ONCE ===
echo
echo "--- Training model (wallclock=${WALL}s) ---"
env \
    NUM_LAYERS=$NUM_LAYERS \
    MODEL_DIM=512 \
    MLP_EXPANSION=$MLP_EXPANSION \
    TRAIN_BATCH_TOKENS=$BATCH \
    GRAD_ACCUM_STEPS=$GA \
    WARMUP_STEPS=10 \
    ITERATIONS=1000000 \
    MAX_WALLCLOCK_SECONDS=$WALL \
    TRAIN_LOG_EVERY=200 \
    TOKENIZER_PATH=./data/tokenizers/sp_bpe_1024.model \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    SAVE_FINAL_CHECKPOINT=logs/u10/trained.pt \
    python3 train_gpt.py 2>&1 | tee logs/u10/train.log

# === Run 4 eval variants ===
declare -a CONFIGS=(
    "baseline:0.0:1.0"
    "alpha_only:0.06:1.0"
    "temp_only:0.0:0.93"
    "alpha_temp:0.06:0.93"
)

for CONFIG in "${CONFIGS[@]}"; do
    NAME="${CONFIG%%:*}"
    REST="${CONFIG#*:}"
    ALPHA="${REST%%:*}"
    TEMP="${REST#*:}"

    echo
    echo "--- Eval: $NAME (α=$ALPHA, T=$TEMP) ---"
    env \
        EVAL_ONLY=1 \
        EVAL_NGRAM_ALPHA=$ALPHA \
        EVAL_TEMPERATURE=$TEMP \
        LOAD_CHECKPOINT=logs/u10/trained.pt \
        TRAIN_BATCH_TOKENS=$BATCH \
        GRAD_ACCUM_STEPS=$GA \
        NUM_LAYERS=$NUM_LAYERS \
        MODEL_DIM=512 \
        MLP_EXPANSION=$MLP_EXPANSION \
        TOKENIZER_PATH=./data/tokenizers/sp_bpe_1024.model \
        DATA_PATH=./data/datasets/fineweb10B_sp1024 \
        python3 train_gpt.py 2>&1 | tee logs/u10/eval_${NAME}.log
done

# === Compare ===
echo
echo "=========================================="
echo "U10 RESULTS"
echo "=========================================="
echo
printf "%-20s  %-10s  %-10s  %-12s\n" "Config" "Alpha" "Temp" "val_bpb"
printf "%-20s  %-10s  %-10s  %-12s\n" "------" "-----" "----" "-------"

declare -a NAMES=("baseline" "alpha_only" "temp_only" "alpha_temp")
declare -a EXPECTED=("1.6929" "1.6864" "?" "1.6805")
BPB_VALUES=()

for i in "${!NAMES[@]}"; do
    NAME="${NAMES[$i]}"
    BPB=$(grep 'val_bpb' logs/u10/eval_${NAME}.log 2>/dev/null | grep -oE 'val_bpb:[0-9.]+' | tail -1 | cut -d: -f2)
    BPB_VALUES+=("$BPB")
    case $NAME in
        baseline)   ALPHA="0.0";  TEMP="1.0"  ;;
        alpha_only) ALPHA="0.06"; TEMP="1.0"  ;;
        temp_only)  ALPHA="0.0";  TEMP="0.93" ;;
        alpha_temp) ALPHA="0.06"; TEMP="0.93" ;;
    esac
    printf "%-20s  %-10s  %-10s  %-12s\n" "$NAME" "$ALPHA" "$TEMP" "${BPB:-N/A}"
done

# Compare baseline vs best combo
BASELINE_BPB=${BPB_VALUES[0]}
BEST_BPB=${BPB_VALUES[3]}

if [ -n "$BASELINE_BPB" ] && [ -n "$BEST_BPB" ]; then
    DELTA=$(python3 -c "print(f'{float('$BASELINE_BPB') - float('$BEST_BPB'):.4f}')")
    echo
    echo "Best (α+T) vs baseline delta: $DELTA BPP"
    if (( $(echo "$DELTA > 0.005" | bc -l) )); then
        echo "✓ EVAL TRICK WORKS — apply α=0.06, T=0.93 in u04 final eval"
    elif (( $(echo "$DELTA > 0" | bc -l) )); then
        echo "△ Marginal — might be noise"
    else
        echo "✗ Doesn't help at this scale (or at all on CUDA)"
    fi
fi

echo
echo "NOTE: This test requires train_gpt.py to support:"
echo "  - SAVE_FINAL_CHECKPOINT env var"
echo "  - LOAD_CHECKPOINT + EVAL_ONLY env vars"
echo "  - EVAL_NGRAM_ALPHA + EVAL_TEMPERATURE env vars"
echo "If these aren't patched in yet, this test won't be meaningful."
