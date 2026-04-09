#!/bin/bash
# run.sh — train + eval + serialize the submission model.
#
# Defaults to a 600s wallclock budget (the comp's 10-min limit). Override via:
#   MAX_WALLCLOCK_SECONDS=3000 bash run.sh   # full convergence run
#   SEED=1337 bash run.sh                    # different seed
#   DRY_RUN=1 bash run.sh                    # 60s smoke test
#
# Reads:
#   data/datasets/datasets/fineweb10B_sp8192/fineweb_*.bin
#   data/datasets/tokenizers/fineweb_8192_bpe.model (auto-built from /root/sp_models/)
#
# Writes:
#   logs/train_<run_id>.log
#   logs/run_<run_id>.log (this script's tee output)
#   final_model.pt  (fp32 EMA-applied)
#   final_model.int6.ptz  (int6 GPTQ + brotli — the submission artifact)

set -eu

REPO_DIR="${REPO_DIR:-/workspace/paramgolf}"
cd "$REPO_DIR"

mkdir -p logs

# === sanity-check shards ===
SHARDS_DIR="data/datasets/datasets/fineweb10B_sp8192"
NUM_SHARDS=$(ls "$SHARDS_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
NUM_VAL=$(ls "$SHARDS_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l)
if [ "$NUM_SHARDS" -lt 1 ] || [ "$NUM_VAL" -lt 1 ]; then
    echo "[run] ERROR: missing shards. Run get_data.sh first."
    exit 2
fi
echo "[run] $NUM_SHARDS train shards, $NUM_VAL val shard(s)"

# === bridge nested data paths into the layout train.py expects ===
# train.py looks for:
#   data/datasets/fineweb10B_sp8192/*.bin
#   data/tokenizers/fineweb_8192_bpe.model
# but get_data.sh writes to:
#   data/datasets/datasets/fineweb10B_sp8192/*.bin
#   data/datasets/tokenizers/fineweb_8192_bpe.model
# Symlink the bridges (idempotent).

mkdir -p data/tokenizers
if [ ! -L data/datasets/fineweb10B_sp8192 ]; then
    rmdir data/datasets/fineweb10B_sp8192 2>/dev/null || true
    ln -sfn datasets/fineweb10B_sp8192 data/datasets/fineweb10B_sp8192
fi
if [ ! -e data/tokenizers/fineweb_8192_bpe.model ] && [ -e data/datasets/tokenizers/fineweb_8192_bpe.model ]; then
    ln -sfn ../datasets/tokenizers/fineweb_8192_bpe.model data/tokenizers/fineweb_8192_bpe.model
fi

# === verify tokenizer loads ===
python3 -c "
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='data/tokenizers/fineweb_8192_bpe.model')
assert sp.vocab_size() == 8192, f'expected 8192, got {sp.vocab_size()}'
print(f'[run] tokenizer ok: vocab={sp.vocab_size()}')
"

# === env defaults (override on the command line) ===
SEED="${SEED:-42}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
TTT_ENABLED="${TTT_ENABLED:-1}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
DATA_DIR="${DATA_DIR:-./data/}"
VOCAB_SIZE="${VOCAB_SIZE:-8192}"

# torch.compile / TorchInductor first-run compile is 5+ min on H100 PCIe for our
# model shape, eats most of a 600s budget. Disable for fast iteration. Phase 3
# work re-enables compile and budgets the first-run cost via cache.
TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

# === Comp frontier env-var bumps (chunk 1, from PHASE1_NOVELTY_AUDIT.md) ===
# C2: 3-layer depth recurrence (was loop_start=4, loop_end=5 → 2 looped layers)
#     PR #1485 / #1471 / #1437 use loop_start=3, loop_end=5 → 3 looped layers,
#     each looped num_loops=2 times for ~17 virtual layers from 11 physical.
#     Expected delta: -0.005 to -0.01 BPB.
LOOP_START="${LOOP_START:-3}"
LOOP_END="${LOOP_END:-5}"
NUM_LOOPS="${NUM_LOOPS:-2}"

# C3: QK_GAIN_INIT bump 4 → 5. PR #1413/#1423/#1485/#1437/#1351/#1408 are at 5.0;
#     PR #1482 is at 5.25. The default 4 in PR #1477 is below the leaderboard curve.
#     Expected delta: -0.001 BPB.
QK_GAIN_INIT="${QK_GAIN_INIT:-5}"

# === DRY_RUN mode for fast smoke testing (60s wallclock, no TTT, no real eval) ===
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[run] DRY_RUN=1 — 60s smoke test"
    MAX_WALLCLOCK_SECONDS=60
    TTT_ENABLED=0
fi

echo "[run] config:"
echo "  SEED=$SEED"
echo "  MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"
echo "  TTT_ENABLED=$TTT_ENABLED"
echo "  TORCH_COMPILE_DISABLE=$TORCH_COMPILE_DISABLE"
echo "  TORCHDYNAMO_DISABLE=$TORCHDYNAMO_DISABLE"
echo "  TRAIN_LOG_EVERY=$TRAIN_LOG_EVERY"
echo "  VOCAB_SIZE=$VOCAB_SIZE"
echo "  LOOP_START=$LOOP_START LOOP_END=$LOOP_END NUM_LOOPS=$NUM_LOOPS  (C2: 3-layer recurrence)"
echo "  QK_GAIN_INIT=$QK_GAIN_INIT  (C3: bumped from 4)"

LOG="logs/run_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[run] launching train.py at $(date -u +%H:%M:%SZ)"
echo "[run] log: $LOG"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
TTT_ENABLED="$TTT_ENABLED" \
TORCH_COMPILE_DISABLE="$TORCH_COMPILE_DISABLE" \
TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE" \
TRAIN_LOG_EVERY="$TRAIN_LOG_EVERY" \
DATA_DIR="$DATA_DIR" \
VOCAB_SIZE="$VOCAB_SIZE" \
LOOP_START="$LOOP_START" \
LOOP_END="$LOOP_END" \
NUM_LOOPS="$NUM_LOOPS" \
QK_GAIN_INIT="$QK_GAIN_INIT" \
python3 -u submission/train.py 2>&1 | tee "$LOG"

echo
echo "[run] DONE $(date -u +%H:%M:%SZ)"
echo "[run] === val_bpb lines ==="
grep -E 'val_bpb' "$LOG"
echo
echo "[run] === artifact ==="
ls -la final_model.int6.ptz 2>/dev/null && echo "  size: $(stat -c %s final_model.int6.ptz 2>/dev/null || stat -f %z final_model.int6.ptz) bytes"
