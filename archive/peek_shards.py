#!/usr/bin/env python3
"""Decode and print text samples from different training shards to compare quality."""
import numpy as np
import sentencepiece as spm
from pathlib import Path

sp = spm.SentencePieceProcessor(model_file="./data/tokenizers/fineweb_1024_bpe.model")

for shard_idx in range(10):
    path = Path(f"./data/datasets/fineweb10B_sp1024/fineweb_train_{shard_idx:06d}.bin")
    header = np.fromfile(path, dtype="<i4", count=256)
    tokens = np.fromfile(path, dtype="<u2", count=2000, offset=256 * 4)
    text = sp.decode(tokens.tolist())

    print(f"\n{'='*80}")
    print(f"SHARD {shard_idx} — first ~2000 tokens")
    print(f"{'='*80}")
    # Print first 1500 chars
    print(text[:1500])
    print(f"...")
