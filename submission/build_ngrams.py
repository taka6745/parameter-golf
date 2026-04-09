#!/usr/bin/env python3
"""build_ngrams.py — Build bigram/trigram/4-gram log-prob tables from tokenized shards.

Vectorized via np.bincount. Reads SP-tokenized .bin shards from
$NGRAM_DATA_DIR (default: data/datasets/datasets/fineweb10B_sp{vocab}/) and writes:
    data/bigram_tab_<vocab>v.npy
    data/trigram_logprobs_<vocab>v.npy
    data/fourgram_logprobs_<vocab>v.npy

Each table is shape (HASH_BUCKETS, vocab), float32 log-probabilities, computed
from add-0.1 smoothed counts of (hashed_context, next_token) pairs.

Hash: polynomial (prev * 36313 + cur * 27191 + ...) % HASH_BUCKETS.
HASH_BUCKETS=16384 default.

Memory: peak ~3 GB per table during count + log step.
Time: ~1-3 min total for 100M tokens at vocab=8192 on a typical pod.

Outputs are loaded as non-persistent buffers by submission/train.py — they do
NOT count toward the 16 MB submission size limit. They are rebuilt fresh on
every pod from the tokenized shards.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# === Config (env var overrides) ===
VOCAB = int(os.environ.get("NGRAM_VOCAB", "8192"))
HASH_BUCKETS = int(os.environ.get("NGRAM_HASH_BUCKETS", "16384"))
MAX_TOKENS = int(os.environ.get("NGRAM_MAX_TOKENS", "100000000"))  # 100M default; 0 = all
DATA_DIR = Path(os.environ.get("NGRAM_DATA_DIR", f"data/datasets/datasets/fineweb10B_sp{VOCAB}"))
OUT_DIR = Path(os.environ.get("NGRAM_OUT_DIR", "data"))
SHARD_HEADER_BYTES = 1024  # competition shard header (256 int32 values)


def load_tokens() -> np.ndarray:
    print(f"Loading tokens from {DATA_DIR}...", flush=True)
    if not DATA_DIR.exists():
        print(f"  ERROR: {DATA_DIR} does not exist. Tokenize first via get_data.sh.", flush=True)
        sys.exit(1)
    shard_paths = sorted(DATA_DIR.glob("fineweb_train_*.bin"))
    if not shard_paths:
        print(f"  ERROR: no fineweb_train_*.bin shards in {DATA_DIR}", flush=True)
        sys.exit(1)
    print(f"  found {len(shard_paths)} train shards, vocab={VOCAB}", flush=True)
    all_tokens: list[np.ndarray] = []
    total = 0
    for shard_path in shard_paths:
        with open(shard_path, "rb") as f:
            f.read(SHARD_HEADER_BYTES)  # skip header
            tokens = np.frombuffer(f.read(), dtype=np.uint16)
        all_tokens.append(tokens)
        total += len(tokens)
        if MAX_TOKENS > 0 and total >= MAX_TOKENS:
            break
    tokens = np.concatenate(all_tokens)
    if MAX_TOKENS > 0 and len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
    if int(tokens.max()) >= VOCAB:
        print(f"  WARNING: max token id {int(tokens.max())} >= VOCAB {VOCAB}. Clamping.", flush=True)
        tokens = np.clip(tokens, 0, VOCAB - 1)
    print(f"  loaded {len(tokens):,} tokens", flush=True)
    return tokens.astype(np.int32, copy=False)


def _hash_bigram(prev: np.ndarray) -> np.ndarray:
    return (prev.astype(np.int64) * 36313) % HASH_BUCKETS


def _hash_trigram(p2: np.ndarray, p1: np.ndarray) -> np.ndarray:
    return (p2.astype(np.int64) * 36313 + p1.astype(np.int64) * 27191) % HASH_BUCKETS


def _hash_fourgram(p3: np.ndarray, p2: np.ndarray, p1: np.ndarray) -> np.ndarray:
    return (
        p3.astype(np.int64) * 36313
        + p2.astype(np.int64) * 27191
        + p1.astype(np.int64) * 51497
    ) % HASH_BUCKETS


def _counts_to_logprobs(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.float64) + 0.1  # add-0.1 smoothing
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.log(counts / row_sums).astype(np.float32)


def build_bigram(tokens: np.ndarray) -> np.ndarray:
    t0 = time.time()
    print("Building bigram...", flush=True)
    prev = tokens[:-1]
    nxt = tokens[1:]
    h = _hash_bigram(prev)
    flat = h * VOCAB + nxt.astype(np.int64)
    counts = np.bincount(flat, minlength=HASH_BUCKETS * VOCAB).reshape(HASH_BUCKETS, VOCAB)
    out = _counts_to_logprobs(counts)
    print(f"  bigram done in {time.time()-t0:.1f}s, shape={out.shape}, {out.nbytes/1024/1024:.0f} MB", flush=True)
    return out


def build_trigram(tokens: np.ndarray) -> np.ndarray:
    t0 = time.time()
    print("Building trigram...", flush=True)
    p2 = tokens[:-2].astype(np.int64)
    p1 = tokens[1:-1].astype(np.int64)
    nxt = tokens[2:].astype(np.int64)
    h = _hash_trigram(p2, p1)
    flat = h * VOCAB + nxt
    counts = np.bincount(flat, minlength=HASH_BUCKETS * VOCAB).reshape(HASH_BUCKETS, VOCAB)
    out = _counts_to_logprobs(counts)
    print(f"  trigram done in {time.time()-t0:.1f}s, shape={out.shape}, {out.nbytes/1024/1024:.0f} MB", flush=True)
    return out


def build_fourgram(tokens: np.ndarray) -> np.ndarray:
    t0 = time.time()
    print("Building 4-gram...", flush=True)
    p3 = tokens[:-3].astype(np.int64)
    p2 = tokens[1:-2].astype(np.int64)
    p1 = tokens[2:-1].astype(np.int64)
    nxt = tokens[3:].astype(np.int64)
    h = _hash_fourgram(p3, p2, p1)
    flat = h * VOCAB + nxt
    counts = np.bincount(flat, minlength=HASH_BUCKETS * VOCAB).reshape(HASH_BUCKETS, VOCAB)
    out = _counts_to_logprobs(counts)
    print(f"  fourgram done in {time.time()-t0:.1f}s, shape={out.shape}, {out.nbytes/1024/1024:.0f} MB", flush=True)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bigram_path = OUT_DIR / f"bigram_tab_{VOCAB}v.npy"
    trigram_path = OUT_DIR / f"trigram_logprobs_{VOCAB}v.npy"
    fourgram_path = OUT_DIR / f"fourgram_logprobs_{VOCAB}v.npy"

    if all(p.exists() for p in (bigram_path, trigram_path, fourgram_path)):
        print(f"All n-gram tables already exist in {OUT_DIR}, skipping build", flush=True)
        return

    tokens = load_tokens()
    np.save(bigram_path, build_bigram(tokens))
    print(f"  saved {bigram_path}", flush=True)
    np.save(trigram_path, build_trigram(tokens))
    print(f"  saved {trigram_path}", flush=True)
    np.save(fourgram_path, build_fourgram(tokens))
    print(f"  saved {fourgram_path}", flush=True)
    print("All n-gram tables built.", flush=True)


if __name__ == "__main__":
    main()
