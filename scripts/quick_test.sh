#!/bin/bash
# Quick A/B test: runs a script for N steps and reports train_loss at each step
# Usage: ./quick_test.sh <script.py> <run_id> [steps=50]
set -e
SCRIPT="${1:-archive/train_gpt_mlx_v2.py}"
RID="${2:-quick_test}"
STEPS="${3:-50}"

source .venv/bin/activate
PYTHONUNBUFFERED=1 \
RUN_ID="$RID" \
ITERATIONS="$STEPS" \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=10 \
MAX_WALLCLOCK_SECONDS=0 \
python3 "$SCRIPT" 2>&1 | grep -E "step:|model_params|final_"
