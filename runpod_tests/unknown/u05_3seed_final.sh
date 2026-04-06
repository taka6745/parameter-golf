#!/bin/bash
# u05_3seed_final.sh — 3-seed validation on 8xH100 for submission
# Hardware: 8xH100
# Time: ~45 min (3 × 15 min runs)
# Cost: ~$15
#
# ONLY RUN THIS AFTER u04 IS GOOD.
# This produces the final submission scores.

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=========================================="
echo "U05: 3-SEED FINAL ON 8xH100"
echo "=========================================="
echo

# Verify we're on 8xH100
N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "$N_GPUS" -lt 8 ]; then
    echo "✗ FAIL: only $N_GPUS GPUs, need 8 for final submission"
    echo "  Switch to an 8xH100 pod first"
    exit 1
fi
echo "GPUs: $N_GPUS ✓"

# Verify u04 ran successfully
if [ ! -f logs/u04/full_stack.log ]; then
    echo "✗ FAIL: u04 not run yet. Validate single-seed first."
    exit 1
fi
U04_BPB=$(grep 'final_int8_zlib_roundtrip val_loss' logs/u04/full_stack.log | grep -oE 'val_bpb:[0-9.]+' | head -1 | cut -d: -f2)
echo "u04 result: $U04_BPB BPP"
if [ -z "$U04_BPB" ]; then
    echo "✗ FAIL: u04 has no val_bpb"
    exit 1
fi

mkdir -p logs/u05

# Use the EXACT config from u04
COMMON="
PROGRESSIVE_SEQ=1
PHASE1_SEQ_LEN=128 PHASE1_LR_MULT=25.0 PHASE1_FRACTION=0.85
PHASE2_SEQ_LEN=1024 PHASE1_NGRAM_WEIGHT=0.40 PHASE2_NGRAM_WEIGHT=0.05
USE_NGRAM_BIAS=1 USE_WAVELET=1 USE_EMA=1 EMA_DECAY=0.997
USE_EVAL_CACHE=1 EVAL_CACHE_MAX_ORDER=7
USE_HEDGE_MIXER=1 HEDGE_ETA=0.1
USE_SLIDING_WINDOW=1 SLIDING_STRIDE=512
USE_SCORE_FIRST_TTT=1 TTT_LR=1e-4 TTT_LAYERS=last2 TTT_EPOCHS=1
USE_MIXED_PRECISION=1 USE_LLOYD_MAX=1
NUM_LAYERS=11 MODEL_DIM=512 MLP_EXPANSION=3
SEQ_LEN=2048 QK_GAIN=4.0 USE_XSA=1 XSA_LAYERS=4
TRAIN_BATCH_TOKENS=524288 GRAD_ACCUM_STEPS=1
WARMUP_STEPS=10 ITERATIONS=1000000
MAX_WALLCLOCK_SECONDS=600 TRAIN_LOG_EVERY=200
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
DATA_PATH=./data/datasets/fineweb10B_bpe8192
"

# Run 3 seeds
for SEED in 42 314 999; do
    echo
    echo "=== SEED $SEED ==="
    env $COMMON SEED=$SEED \
        torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/u05/seed_${SEED}.log
done

# Compute mean and std
echo
echo "=========================================="
echo "FINAL SUBMISSION SCORES"
echo "=========================================="

python3 << 'PYEOF'
import re
import statistics

scores = []
for seed in [42, 314, 999]:
    with open(f"logs/u05/seed_{seed}.log") as f:
        for line in f:
            m = re.search(r"final_int8_zlib_roundtrip val_loss:[\d.]+ val_bpb:([\d.]+)", line)
            if m:
                scores.append(float(m.group(1)))
                break

if len(scores) == 3:
    mean = sum(scores) / 3
    std = statistics.stdev(scores)
    print(f"\nSeeds:  {scores}")
    print(f"Mean:   {mean:.4f}")
    print(f"Std:    {std:.4f}")
    print(f"Min:    {min(scores):.4f}")
    print(f"Max:    {max(scores):.4f}")
    print()

    if std < 0.01:
        print("✓ STABLE — submit the mean")
    elif std < 0.02:
        print("△ MODERATELY STABLE — submit mean, mention std in PR")
    else:
        print("✗ HIGH VARIANCE — investigate before submitting")
        print("  Possible causes: learning rate too high, bad seed sensitivity")

    print()
    print("Submission JSON:")
    print(f'  "val_bpb_mean": {mean:.6f},')
    print(f'  "val_bpb_seeds": [{scores[0]}, {scores[1]}, {scores[2]}],')
    print(f'  "val_bpb_std": {std:.6f}')
else:
    print(f"✗ Only got {len(scores)}/3 scores. Check logs.")
PYEOF
