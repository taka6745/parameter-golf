#!/usr/bin/env python3
"""
v02_ngram_bias.py — Verify n-gram bias loading + forward pass

Loads the precomputed bigram/trigram tables and applies them as a logit
bias to a simulated model output. Confirms shapes, hashing, and that
the bias actually changes predictions.

Expected: PASS in <30 sec.
"""

import torch
import numpy as np
import sys


import os
BATCH = 4
SEQ_LEN = 128

# Auto-detect which n-gram table exists (8192 or 1024)
if os.path.exists("data/bigram_tab_8192v.npy"):
    VOCAB = 8192
    HASH_BUCKETS = 16384
elif os.path.exists("data/bigram_tab_1024v.npy"):
    VOCAB = 1024
    HASH_BUCKETS = 2048
else:
    print("✗ No bigram table found. Run chore/04_build_ngrams.py first.")
    sys.exit(1)


def hash_bigram(prev):
    return (prev * 36313) % HASH_BUCKETS


def hash_trigram(prev2, prev1):
    return (prev2 * 36313 + prev1 * 27191) % HASH_BUCKETS


def main():
    print("=== V02: N-GRAM BIAS FORWARD PASS ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Vocab: {VOCAB}, hash buckets: {HASH_BUCKETS}")

    # Load tables (auto-detected vocab)
    print("Loading n-gram tables...")
    bigram = torch.from_numpy(np.load(f"data/bigram_tab_{VOCAB}v.npy")).to(device)
    trigram = torch.from_numpy(np.load(f"data/trigram_logprobs_{VOCAB}v.npy")).to(device)
    print(f"  bigram: {bigram.shape}, dtype={bigram.dtype}")
    print(f"  trigram: {trigram.shape}, dtype={trigram.dtype}")

    assert bigram.shape == (HASH_BUCKETS, VOCAB), f"Expected ({HASH_BUCKETS}, {VOCAB}), got {bigram.shape}"
    assert trigram.shape == (HASH_BUCKETS, VOCAB), f"Expected ({HASH_BUCKETS}, {VOCAB}), got {trigram.shape}"

    # Simulate a forward pass
    print("\nSimulating forward pass...")
    tokens = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)
    base_logits = torch.randn(BATCH, SEQ_LEN, VOCAB, device=device) * 0.1

    # Apply bigram bias for positions >= 1
    prev = tokens[:, :-1]  # (B, T-1)
    h_bi = hash_bigram(prev.long())  # (B, T-1)
    bi_bias = bigram[h_bi]  # (B, T-1, V)

    # Apply trigram bias for positions >= 2
    prev2 = tokens[:, :-2]  # (B, T-2)
    prev1 = tokens[:, 1:-1]
    h_tri = hash_trigram(prev2.long(), prev1.long())
    tri_bias = trigram[h_tri]  # (B, T-2, V)

    # Combine: w_bi * bigram + w_tri * trigram (zero-padded for early positions)
    w_bi, w_tri = 0.20, 0.15
    biased = base_logits.clone()
    biased[:, 1:, :] += w_bi * bi_bias
    biased[:, 2:, :] += w_tri * tri_bias

    print(f"  base logits range: [{base_logits.min():.4f}, {base_logits.max():.4f}]")
    print(f"  biased logits range: [{biased.min():.4f}, {biased.max():.4f}]")

    # Verify the bias actually changed predictions
    base_pred = base_logits.argmax(-1)
    biased_pred = biased.argmax(-1)
    changed = (base_pred != biased_pred).float().mean().item()
    print(f"  Predictions changed by bias: {changed:.1%}")

    # Should change a meaningful fraction (not all, not none)
    if 0.10 < changed < 0.99:
        print("\n✓ PASS: bias loads, applies correctly, changes predictions")
        return 0
    else:
        print(f"\n✗ FAIL: bias change rate {changed:.1%} is suspicious")
        return 1


if __name__ == "__main__":
    sys.exit(main())
