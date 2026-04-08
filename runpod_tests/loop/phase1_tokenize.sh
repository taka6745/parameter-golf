#!/bin/bash
# Phase 1 tokenize launcher — runs on the H100 pod.
#
# Reuses an already-extracted docs_selected.jsonl (or a symlink to one) and an
# already-trained SentencePiece BPE 8192 model. Skips the 45 GB HF re-copy and
# the ~5 min tokenizer training entirely.
#
# Required pod state before invoking:
#   data/datasets/docs_selected.jsonl                      # symlink or real file (~45 GB)
#   data/datasets/tokenizers/fineweb_8192_bpe.model        # 370908 bytes
#   data/tokenizer_specs_8192.json                         # checked into repo
#
# Output:
#   data/datasets/datasets/fineweb10B_sp8192/fineweb_train_*.bin
#   data/datasets/datasets/fineweb10B_sp8192/fineweb_val_*.bin
#
# Logs to logs/phase1_tokenize.log on the pod.

set -eu
cd /workspace/paramgolf

mkdir -p logs

JSONL=data/datasets/docs_selected.jsonl
# Source model lives OUTSIDE the tokenizers_dir because the script unlinks
# the destination model_path before reading reuse_model_path. If they're the
# same file the source gets deleted out from under the reuse path. Stash the
# canonical model on container disk instead.
SP_MODEL=/root/sp_models/fineweb_8192_bpe.model
SPEC=data/tokenizer_specs_8192.json
LOG=logs/phase1_tokenize.log

echo "=== PHASE 1 TOKENIZE START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --- Pre-flight checks ---
for f in "$JSONL" "$SP_MODEL" "$SPEC"; do
    if [ ! -e "$f" ]; then
        echo "ERROR: missing required file: $f"
        exit 1
    fi
done
echo "JSONL: $(ls -la "$JSONL")"
echo "SP_MODEL: $(ls -la "$SP_MODEL")"

# --- Disk headroom check (refuse to run if volume is near full) ---
WORKSPACE_AVAIL_KB=$(df -k /workspace | awk 'NR==2 {print $4}')
WORKSPACE_AVAIL_GB=$((WORKSPACE_AVAIL_KB / 1024 / 1024))
echo "/workspace free: ${WORKSPACE_AVAIL_GB} GB"
if [ "$WORKSPACE_AVAIL_GB" -lt 30 ]; then
    echo "ERROR: /workspace has <30 GB free (have ${WORKSPACE_AVAIL_GB} GB). 24 GB of SP8192 shards must land here. Free space first."
    exit 2
fi

ROOT_AVAIL_KB=$(df -k / | awk 'NR==2 {print $4}')
ROOT_AVAIL_GB=$((ROOT_AVAIL_KB / 1024 / 1024))
echo "/ free: ${ROOT_AVAIL_GB} GB"
if [ "$ROOT_AVAIL_GB" -lt 5 ]; then
    echo "WARNING: container disk / has <5 GB free (${ROOT_AVAIL_GB} GB). Tokenize may not actually need much extra here, but be aware."
fi

# --- Run the tokenize ---
# MATCHED_FINEWEB_SKIP_HF_COPY=1 makes copy_from_hf_cache skip when the destination
# already exists. With our pre-staged JSONL symlink, this avoids the 45 GB re-copy.
# --reuse-sp-model 8192=... reuses our already-trained tokenizer.
echo "=== running download_hf_docs_and_tokenize.py ==="
MATCHED_FINEWEB_SKIP_HF_COPY=1 \
nohup python3 -u data/download_hf_docs_and_tokenize.py \
    --output-root data/datasets \
    --tokenizer-config "$SPEC" \
    --reuse-sp-model "8192=$SP_MODEL" \
    > "$LOG" 2>&1 &
TOKENIZE_PID=$!
echo "tokenize_pid=$TOKENIZE_PID"
echo "log=$LOG"
echo "=== launched, returning. monitor with: tail -f $LOG ==="
disown $TOKENIZE_PID || true
