#!/bin/bash
# get_data.sh — fetch FineWeb docs JSONL + tokenize SP8192 shards.
# Idempotent. Skips if shards already exist.
#
# Where things live (THE GOTCHA THAT BURNED US ON 2026-04-09):
#   /workspace                       = 50 GB persistent volume (network FS)
#   /                                = 100 GB ephemeral container disk (overlayfs)
#
# The 45 GB raw docs JSONL CANNOT live on /workspace — it would consume 90% of
# the volume quota and there'd be no room for the 24 GB SP8192 shards. We stash
# the JSONL on container disk at /root/paramgolf_bigdata/ and symlink it back
# into the repo so the tokenize script finds it at the expected path.
#
# After tokenization the JSONL can be deleted (the shards are derived from it)
# but we keep it around in case we want to re-tokenize with different params.

set -eu

REPO_DIR="${REPO_DIR:-/workspace/paramgolf}"
cd "$REPO_DIR"

DATA_BIG=/root/paramgolf_bigdata
DOCS_JSONL="$DATA_BIG/docs_selected.jsonl"
DOCS_SYMLINK="data/datasets/docs_selected.jsonl"

SP_MODEL_SRC=/root/sp_models/fineweb_8192_bpe.model
SP_MODEL_REPO="data/tokenizers/fineweb_8192_bpe.model"

SHARDS_DIR="data/datasets/datasets/fineweb10B_sp8192"

mkdir -p "$DATA_BIG" /root/sp_models data/datasets/tokenizers data/datasets/datasets

# === SHORT-CIRCUIT: tokenize already done? ===
if [ -d "$SHARDS_DIR" ] && [ "$(ls "$SHARDS_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)" -ge 100 ]; then
    NUM_SHARDS=$(ls "$SHARDS_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
    echo "[get_data] $NUM_SHARDS train shards already exist in $SHARDS_DIR — skipping tokenize"
    exit 0
fi

# === Step 1: get docs_selected.jsonl ===
if [ -f "$DOCS_JSONL" ] && [ "$(stat -c %s "$DOCS_JSONL" 2>/dev/null || stat -f %z "$DOCS_JSONL")" -gt 1000000000 ]; then
    echo "[get_data] $DOCS_JSONL already exists ($(du -sh "$DOCS_JSONL" | cut -f1)) — skipping download"
else
    echo "[get_data] downloading docs_selected.jsonl from HF Hub..."
    # Use os.link (hard link) instead of shutil.copy2 — same filesystem, single
    # inode, ZERO extra disk usage. The HF cache gets deleted afterward to free
    # the original cache copy. (shutil.copy2 doubled disk usage to 90 GB on
    # Pod L and OOM'd the 100 GB container disk before we got to tokenize.)
    python3 - <<'PYEOF'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
dst = Path('/root/paramgolf_bigdata/docs_selected.jsonl')
dst.parent.mkdir(parents=True, exist_ok=True)
src = hf_hub_download(
    repo_id='willdepueoai/parameter-golf',
    filename='docs_selected.jsonl',
    subfolder='datasets',
    repo_type='dataset',
)
src_path = Path(src).resolve()
if dst.exists():
    dst.unlink()
try:
    os.link(src_path, dst)  # hard link, 0 extra disk
    print(f'  hard-linked {src_path} -> {dst}')
except OSError:
    # Cross-FS or unsupported, fall back to copy
    shutil.copy2(src_path, dst)
    print(f'  copied (fallback) {src_path} -> {dst}')
print(f'  size: {dst.stat().st_size:,} bytes')
PYEOF
    # Now drop the HF cache copy — it's redundant once we have the linked file.
    # If get_data.sh re-runs, the existence check above will short-circuit before
    # we hit hf_hub_download, so deleting the cache here is safe.
    if [ -d /root/.cache/huggingface ]; then
        echo "[get_data] dropping /root/.cache/huggingface (redundant after hard link)"
        rm -rf /root/.cache/huggingface
    fi
fi

# === Step 2: symlink JSONL into repo path ===
mkdir -p "$(dirname "$DOCS_SYMLINK")"
if [ -L "$DOCS_SYMLINK" ] || [ -e "$DOCS_SYMLINK" ]; then
    rm -f "$DOCS_SYMLINK"
fi
ln -sfn "$DOCS_JSONL" "$DOCS_SYMLINK"
echo "[get_data] symlink: $DOCS_SYMLINK -> $(readlink "$DOCS_SYMLINK")"

# === Step 3: ensure SP model is in /root/sp_models OUTSIDE the destination tokenizers_dir ===
# (the download script's build_sentencepiece_tokenizer unlinks the destination
# model_path BEFORE checking reuse_model_path — if both point at the same file
# the source gets deleted out from under the reuse path. Stash separately.)
if [ ! -f "$SP_MODEL_SRC" ]; then
    if [ -f "$SP_MODEL_REPO" ]; then
        cp "$SP_MODEL_REPO" "$SP_MODEL_SRC"
        echo "[get_data] copied SP model from repo to $SP_MODEL_SRC"
    else
        echo "[get_data] WARNING: $SP_MODEL_SRC not found and not in repo. Tokenize will train a fresh SP model from the JSONL (~5-10 min extra)."
    fi
else
    echo "[get_data] $SP_MODEL_SRC already exists ($(stat -c %s "$SP_MODEL_SRC" 2>/dev/null || stat -f %z "$SP_MODEL_SRC") bytes)"
fi

# === Step 4: disk headroom check ===
WORKSPACE_AVAIL_GB=$(df -k /workspace | awk 'NR==2 {print int($4/1024/1024)}')
ROOT_AVAIL_GB=$(df -k / | awk 'NR==2 {print int($4/1024/1024)}')
echo "[get_data] /workspace free: ${WORKSPACE_AVAIL_GB} GB / / free: ${ROOT_AVAIL_GB} GB"
if [ "$WORKSPACE_AVAIL_GB" -lt 30 ]; then
    echo "[get_data] ERROR: /workspace has <30 GB free. Need ~24 GB for SP8192 shards. Free space first."
    exit 2
fi

# === Step 5: tokenize ===
echo "[get_data] launching tokenize ($(date -u +%Y-%m-%dT%H:%M:%SZ))..."
mkdir -p logs
REUSE_FLAG=()
if [ -f "$SP_MODEL_SRC" ]; then
    REUSE_FLAG+=(--reuse-sp-model "8192=$SP_MODEL_SRC")
fi

# MATCHED_FINEWEB_SKIP_HF_COPY=1 short-circuits copy_from_hf_cache when the
# destination JSONL exists (via our symlink). Without it the script would re-copy
# 45 GB from the HF cache to the repo path on every run.
MATCHED_FINEWEB_SKIP_HF_COPY=1 \
python3 -u data/download_hf_docs_and_tokenize.py \
    --output-root data/datasets \
    --tokenizer-config data/tokenizer_specs_8192.json \
    "${REUSE_FLAG[@]}" \
    2>&1 | tee logs/tokenize.log

NUM_SHARDS=$(ls "$SHARDS_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
NUM_VAL=$(ls "$SHARDS_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "[get_data] DONE tokenize: $NUM_SHARDS train shards, $NUM_VAL val shards"

if [ "$NUM_SHARDS" -lt 100 ]; then
    echo "[get_data] WARNING: only $NUM_SHARDS train shards (expected ~120). Check logs/tokenize.log for errors."
fi

# === Step 6: build n-gram tables ===
# Static log-prob tables for the NGRAM_BIAS / NGRAM_BACKOFF infrastructure.
# Loaded as non-persistent buffers by train.py — they do NOT count toward the
# 16 MB submission limit. Built from the tokenized shards (CPU-bound, 1-3 min).
# Skipped if outputs already exist.
echo "[get_data] building n-gram log-prob tables..."
NGRAM_VOCAB=8192 \
NGRAM_HASH_BUCKETS=16384 \
NGRAM_MAX_TOKENS=100000000 \
NGRAM_DATA_DIR="$SHARDS_DIR" \
NGRAM_OUT_DIR="data" \
python3 -u submission/build_ngrams.py 2>&1 | tee logs/build_ngrams.log

# Verify outputs
for f in data/bigram_tab_8192v.npy data/trigram_logprobs_8192v.npy data/fourgram_logprobs_8192v.npy; do
    if [ -f "$f" ]; then
        echo "[get_data]   ✓ $f ($(stat -c %s "$f" 2>/dev/null || stat -f %z "$f") bytes)"
    else
        echo "[get_data]   ✗ MISSING $f — NGRAM_BIAS will fall back to no-op at training time"
    fi
done
