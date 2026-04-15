#!/bin/bash
# A/B test: train on clean shard (0) vs messy shard (1) for 200 steps
# Then generate text from each to compare how "English" they sound
set -e
cd /Users/takodamundy/Documents/personal_repos/paramgolf
source .venv/bin/activate

STEPS=200
COMMON_ARGS="ITERATIONS=$STEPS TRAIN_BATCH_TOKENS=8192 GRAD_ACCUM_STEPS=8 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=524288 WARMUP_STEPS=5 TRAIN_LOG_EVERY=50 MAX_WALLCLOCK_SECONDS=0"

echo "========================================="
echo "TRAINING ON CLEAN SHARD (shard 0 - prose)"
echo "========================================="
PYTHONUNBUFFERED=1 \
DATA_PATH=/tmp/paramgolf_shard_clean \
RUN_ID=ab_clean \
OUT_DIR=logs \
ITERATIONS=$STEPS \
TRAIN_BATCH_TOKENS=8192 \
GRAD_ACCUM_STEPS=8 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=50 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_gpt_mlx.py 2>&1 | grep -E "step:|model_params|final_|val_bpb|saved_model|serialized"

echo ""
echo "========================================="
echo "TRAINING ON MESSY SHARD (shard 1 - blogs)"
echo "========================================="
PYTHONUNBUFFERED=1 \
DATA_PATH=/tmp/paramgolf_shard_messy \
RUN_ID=ab_messy \
OUT_DIR=logs \
ITERATIONS=$STEPS \
TRAIN_BATCH_TOKENS=8192 \
GRAD_ACCUM_STEPS=8 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=50 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_gpt_mlx.py 2>&1 | grep -E "step:|model_params|final_|val_bpb|saved_model|serialized"

echo ""
echo "========================================="
echo "GENERATING TEXT FROM BOTH MODELS"
echo "========================================="
PROMPTS=("The meaning of life is" "Once upon a time" "The president of the United States" "Scientists have discovered that")

for prompt in "${PROMPTS[@]}"; do
    echo ""
    echo "--- CLEAN model: '$prompt' ---"
    python3 generate.py --model logs/ab_clean_mlx_model.npz --prompt "$prompt" --tokens 80 --temperature 0.7 2>/dev/null
    echo ""
    echo "--- MESSY model: '$prompt' ---"
    python3 generate.py --model logs/ab_messy_mlx_model.npz --prompt "$prompt" --tokens 80 --temperature 0.7 2>/dev/null
done
