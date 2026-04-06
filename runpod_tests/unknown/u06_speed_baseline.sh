#!/bin/bash
# u06_speed_baseline.sh — Pure ms/step measurement for our config on H100
# Hardware: 1xH100
# Time: ~5 min
# Cost: ~$0.30
#
# WHY: We measured 11L/seq=128 = 8.4 ms/step on RTX 3080 Ti.
# H100 should be 3-5x faster. We need to know the REAL step times
# before running u02-u04 so we can set wallclock budgets correctly.
#
# This test runs 100 steps at multiple (layers, seq_len) configs and
# reports ms/step. No quality measurement, just speed.

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=========================================="
echo "U06: H100 SPEED BASELINE"
echo "=========================================="
echo
echo "Measuring ms/step at different (layers, seq_len) configs."
echo "Each test: 100 steps, no eval. Just ms/step."
echo

mkdir -p logs/u06

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "GPU: $GPU_NAME"
echo

run_speed() {
    local NAME=$1
    local LAYERS=$2
    local SEQ=$3
    local BATCH=$4

    echo "--- $NAME: ${LAYERS}L, seq=${SEQ}, batch=${BATCH} ---"
    env \
        NUM_LAYERS=$LAYERS \
        MODEL_DIM=512 \
        MLP_EXPANSION=3 \
        SEQ_LEN=$SEQ \
        TRAIN_BATCH_TOKENS=$BATCH \
        GRAD_ACCUM_STEPS=1 \
        WARMUP_STEPS=10 \
        ITERATIONS=100 \
        MAX_WALLCLOCK_SECONDS=0 \
        TRAIN_LOG_EVERY=10 \
        TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
        DATA_PATH=./data/datasets/fineweb10B_bpe8192 \
        python3 train_gpt.py 2>&1 | tee logs/u06/${NAME}.log | tail -5

    # Extract step_avg from final step
    MS=$(grep 'step:100' logs/u06/${NAME}.log | grep -oE 'step_avg:[0-9.]+' | head -1 | cut -d: -f2)
    echo "  → ${MS} ms/step"
    echo
}

# Phase 1 configs (short seq, large effective batch)
run_speed "phase1_11L_s128" 11 128 524288
run_speed "phase1_9L_s128"  9  128 524288

# Phase 2 configs (long seq, smaller batch)
run_speed "phase2_11L_s1024" 11 1024 524288
run_speed "phase2_11L_s2048" 11 2048 524288
run_speed "phase2_9L_s1024"  9  1024 524288

# Standard configs for reference
run_speed "ref_11L_s1024" 11 1024 524288

echo
echo "=== SPEED SUMMARY ==="
echo
echo "Config              ms/step  steps in 510s (Phase 1)  steps in 90s (Phase 2)"
echo "------------------  -------  -----------------------  ----------------------"
for NAME in phase1_11L_s128 phase1_9L_s128 phase2_11L_s1024 phase2_11L_s2048 phase2_9L_s1024 ref_11L_s1024; do
    MS=$(grep 'step:100' logs/u06/${NAME}.log | grep -oE 'step_avg:[0-9.]+' | head -1 | cut -d: -f2)
    if [ -n "$MS" ]; then
        STEPS_510=$(python3 -c "print(int(510000 / $MS))")
        STEPS_90=$(python3 -c "print(int(90000 / $MS))")
        printf "%-18s  %7.2f  %19d  %22d\n" "$NAME" "$MS" "$STEPS_510" "$STEPS_90"
    fi
done

echo
echo "Use these numbers to:"
echo "  1. Predict total Phase 1 steps in u02 (should be tens of thousands)"
echo "  2. Predict total Phase 2 steps (should be 1000-3000)"
echo "  3. Verify our channel-capacity analysis (Phase 2 is 8x under-resourced)"
