#!/usr/bin/env python3
"""Byte-level n-gram conditional entropy on FineWeb val (numpy fast path).

For each n=1..N_MAX, computes the empirical add-k conditional entropy
H(X | prev n-1 bytes) on the val sample using counts gathered from
the train sample. Reports BPB (bits per byte) and the marginal value
of each extra n-gram order.

Add-k smoothing (k=1, Laplace) is used instead of full Kneser-Ney for
speed — tells us the *direction* (does going from n=3 to n=4 buy a lot,
or are we hitting diminishing returns?) which is what we need for the
L09 backlog priority decision. Full KN would tighten the absolute number
but not change the marginal-value-per-n shape.

Run from repo root: python3 src/floor/kn_ngram.py [n_max=6] [train_mb=80] [val_mb=10]
"""
import sys
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6
TRAIN_MB = int(sys.argv[2]) if len(sys.argv) > 2 else 80
VAL_MB = int(sys.argv[3]) if len(sys.argv) > 3 else 10

VAL_PATH = Path("data/floor/val_sp1024.txt")
print(f"loading {VAL_PATH}", flush=True)
data = np.frombuffer(VAL_PATH.read_bytes(), dtype=np.uint8)
print(f"  total {len(data)/1e6:.1f} MB", flush=True)

train_n = TRAIN_MB * 1024 * 1024
val_n = VAL_MB * 1024 * 1024
assert train_n + val_n <= len(data), f"need {(train_n+val_n)/1e6:.1f} MB, have {len(data)/1e6:.1f}"
train = data[:train_n]
val = data[-val_n:]
print(f"  train slice: {len(train)/1e6:.1f} MB | val slice: {len(val)/1e6:.1f} MB", flush=True)

ALPHA = 256        # byte alphabet
K_SMOOTH = 1.0     # add-k smoothing constant (Laplace)

results = []
for n in range(1, N_MAX + 1):
    t0 = time.time()
    ctx_len = n - 1

    # Pack the n-gram into a single uint64 by base-256 encoding the bytes.
    # ngram_int = b0 * 256^(n-1) + b1 * 256^(n-2) + ... + b_{n-1}
    # Need ALPHA^n total; for n<=8 fits in uint64.
    if n > 8:
        raise SystemExit("n>8 doesn't fit in uint64; use a hash table approach")

    # Build train n-gram packed integers using numpy strided trick
    train_grams = np.zeros(len(train) - ctx_len, dtype=np.uint64)
    for k in range(n):
        train_grams += train[k:len(train) - ctx_len + k].astype(np.uint64) * (ALPHA ** (n - 1 - k))

    # Same for val
    val_grams = np.zeros(len(val) - ctx_len, dtype=np.uint64)
    for k in range(n):
        val_grams += val[k:len(val) - ctx_len + k].astype(np.uint64) * (ALPHA ** (n - 1 - k))

    # Count train n-grams. We need both:
    #   ctx_count(c)  = total count of contexts c (sum over all w of count(c,w))
    #   ngram_count(c,w) = count of context+next pair
    # Encode: ctx = ngram >> 8 (drop last byte). next_byte = ngram & 0xff.
    # ctx_count = histogram of (train_grams >> 8) over ALPHA^(n-1) bins.

    if ctx_len == 0:
        # n=1 unigram: ctx is empty, ngram == byte
        # P_smooth(w) = (count(w) + k) / (N + k * ALPHA)
        unigram_count = np.bincount(train_grams.astype(np.int64), minlength=ALPHA).astype(np.float64)
        N = unigram_count.sum()
        denom = N + K_SMOOTH * ALPHA
        log2_p = np.log2((unigram_count + K_SMOOTH) / denom)
        # Score val
        bits = -log2_p[val_grams.astype(np.int64)].sum()
        bpb = bits / len(val_grams)
    else:
        # n>=2. Use a sorted-keys + searchsorted approach: bin train_grams by (n-gram int)
        # so we can look up val_grams.
        # ngram count: dict-like via np.unique.
        train_sorted = np.sort(train_grams)
        # Use np.unique to get (unique_grams, counts)
        uniq_grams, gram_counts = np.unique(train_sorted, return_counts=True)
        # Count contexts
        ctx_arr = train_grams >> 8  # drop last byte
        uniq_ctx, ctx_counts = np.unique(np.sort(ctx_arr), return_counts=True)
        # Distinct nexts per context (for KN-like backoff later if we want)
        # For add-k smoothing we don't need it.

        # Score val n-grams:
        # For each val_gram = ctx*256 + w:
        #   c_ngram = count[val_gram] (0 if not seen)
        #   c_ctx = count[ctx]
        #   p = (c_ngram + k) / (c_ctx + k * ALPHA)
        # If ctx unseen: c_ctx=0 → p = 1/ALPHA (uniform)

        # n-gram count lookup
        idx = np.searchsorted(uniq_grams, val_grams)
        idx = np.clip(idx, 0, len(uniq_grams) - 1)
        found_ngram = uniq_grams[idx] == val_grams
        c_ngram = np.where(found_ngram, gram_counts[idx], 0).astype(np.float64)

        # ctx count lookup
        val_ctx = val_grams >> 8
        idx_c = np.searchsorted(uniq_ctx, val_ctx)
        idx_c = np.clip(idx_c, 0, len(uniq_ctx) - 1)
        found_ctx = uniq_ctx[idx_c] == val_ctx
        c_ctx = np.where(found_ctx, ctx_counts[idx_c], 0).astype(np.float64)

        # Smoothed probability
        denom = c_ctx + K_SMOOTH * ALPHA
        p = (c_ngram + K_SMOOTH) / denom
        log2_p = np.log2(p)
        bits = -log2_p.sum()
        bpb = bits / len(val_grams)

    elapsed = time.time() - t0
    results.append((n, bpb))
    print(f"  n={n}: BPB={bpb:.4f}  ({elapsed:.1f}s, {len(train_grams):,} train grams, {len(val_grams):,} val grams)", flush=True)

print("\n=== Add-k (k=1) smoothed n-gram BPB on FineWeb val ===", flush=True)
for n, b in results:
    print(f"  n={n}  BPB={b:.4f}", flush=True)

print("\n=== Marginal value of each extra n-gram order ===", flush=True)
for i in range(1, len(results)):
    delta = results[i-1][1] - results[i][1]
    print(f"  n={results[i][0]} vs n={results[i-1][0]}: ΔBPB = {delta:+.4f}", flush=True)
