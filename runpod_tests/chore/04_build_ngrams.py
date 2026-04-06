#!/usr/bin/env python3
"""
04_build_ngrams.py — Build bigram, trigram, 4-gram tables for SP-1024

Vectorized with np.bincount. Subsamples to MAX_TOKENS for speed (default 100M;
diminishing returns past ~50M for these tables). Set MAX_TOKENS=0 to use all.

Runs on: any pod (CPU work, but uses ~4 GB RAM for the count matrices)
Time: ~30 sec on 100M tokens, ~5 min on 1B tokens
Outputs:
    data/bigram_tab_1024v.npy
    data/trigram_logprobs_1024v.npy
    data/fourgram_logprobs_1024v.npy
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

VOCAB = 1024
HASH_BUCKETS = 2048
DATA_DIR = Path("data/datasets/fineweb10B_sp1024")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 100_000_000))  # 100M default; 0 = all


def load_tokens():
    print(f"Loading tokens from {DATA_DIR}...")
    if not DATA_DIR.exists():
        print(f"  ✗ {DATA_DIR} does not exist")
        print(f"  Did 01_download_data.sh run successfully?")
        sys.exit(1)

    shard_paths = sorted(DATA_DIR.glob("fineweb_train_*.bin"))
    if not shard_paths:
        print(f"  ✗ No fineweb_train_*.bin shards in {DATA_DIR}")
        sys.exit(1)

    all_tokens = []
    total = 0
    for shard_path in shard_paths:
        with open(shard_path, "rb") as f:
            f.read(1024)  # skip header
            tokens = np.frombuffer(f.read(), dtype=np.uint16)
        all_tokens.append(tokens)
        total += len(tokens)
        if MAX_TOKENS > 0 and total >= MAX_TOKENS:
            break

    tokens = np.concatenate(all_tokens)
    if MAX_TOKENS > 0 and len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
    print(f"  Loaded {len(tokens):,} tokens (limit={MAX_TOKENS:,})")
    return tokens.astype(np.int32, copy=False)


def build_bigram(tokens):
    """Vectorized bigram count using flat indexing into a (HASH_BUCKETS, VOCAB) table."""
    t0 = time.time()
    print(f"Building bigram (vectorized)...")
    prev = tokens[:-1]
    nxt = tokens[1:]
    h = (prev.astype(np.int64) * 36313) % HASH_BUCKETS
    flat = h * VOCAB + nxt
    counts = np.bincount(flat, minlength=HASH_BUCKETS * VOCAB).reshape(HASH_BUCKETS, VOCAB)
    counts = counts.astype(np.float64) + 0.1
    row_sums = counts.sum(axis=1, keepdims=True)
    log_probs = np.log(counts / row_sums).astype(np.float32)
    print(f"  done in {time.time() - t0:.1f}s")
    return log_probs


def build_trigram(tokens):
    t0 = time.time()
    print(f"Building trigram (vectorized)...")
    p2 = tokens[:-2].astype(np.int64)
    p1 = tokens[1:-1].astype(np.int64)
    nxt = tokens[2:]
    h = (p2 * 36313 + p1 * 27191) % HASH_BUCKETS
    flat = h * VOCAB + nxt
    counts = np.bincount(flat, minlength=HASH_BUCKETS * VOCAB).reshape(HASH_BUCKETS, VOCAB)
    counts = counts.astype(np.float64) + 0.1
    row_sums = counts.sum(axis=1, keepdims=True)
    log_probs = np.log(counts / row_sums).astype(np.float32)
    print(f"  done in {time.time() - t0:.1f}s")
    return log_probs


def build_fourgram(tokens):
    t0 = time.time()
    print(f"Building 4-gram (vectorized)...")
    p3 = tokens[:-3].astype(np.int64)
    p2 = tokens[1:-2].astype(np.int64)
    p1 = tokens[2:-1].astype(np.int64)
    nxt = tokens[3:]
    h = (p3 * 36313 + p2 * 27191 + p1 * 51497) % HASH_BUCKETS
    flat = h * VOCAB + nxt
    counts = np.bincount(flat, minlength=HASH_BUCKETS * VOCAB).reshape(HASH_BUCKETS, VOCAB)
    counts = counts.astype(np.float64) + 0.1
    row_sums = counts.sum(axis=1, keepdims=True)
    log_probs = np.log(counts / row_sums).astype(np.float32)
    print(f"  done in {time.time() - t0:.1f}s")
    return log_probs


def main():
    bigram_path = f"data/bigram_tab_{VOCAB}v.npy"
    trigram_path = f"data/trigram_logprobs_{VOCAB}v.npy"
    fourgram_path = f"data/fourgram_logprobs_{VOCAB}v.npy"

    if all(os.path.exists(f) for f in [bigram_path, trigram_path, fourgram_path]):
        print("✓ All n-gram tables already exist, skipping")
        return

    tokens = load_tokens()

    bigram = build_bigram(tokens)
    np.save(bigram_path, bigram)
    print(f"✓ Saved bigram: {bigram.shape}, {bigram.nbytes/1024/1024:.2f} MB")

    trigram = build_trigram(tokens)
    np.save(trigram_path, trigram)
    print(f"✓ Saved trigram: {trigram.shape}, {trigram.nbytes/1024/1024:.2f} MB")

    fourgram = build_fourgram(tokens)
    np.save(fourgram_path, fourgram)
    print(f"✓ Saved 4-gram: {fourgram.shape}, {fourgram.nbytes/1024/1024:.2f} MB")

    print("\n✓ All n-gram tables built")


if __name__ == "__main__":
    main()
