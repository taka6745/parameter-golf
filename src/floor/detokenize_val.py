#!/usr/bin/env python3
"""Detokenize FineWeb val (sp1024 or sp8192) → raw UTF-8 bytes file.

Run from repo root: python3 src/floor/detokenize_val.py [sp1024|sp8192] [bytes]
"""
import sys
import numpy as np
import sentencepiece as spm
from pathlib import Path

VOCAB = sys.argv[1] if len(sys.argv) > 1 else "sp1024"
LIMIT_BYTES = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = full file

if VOCAB == "sp1024":
    bin_path = Path("data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin")
    sp_path = Path("data/tokenizers/fineweb_1024_bpe.model")
elif VOCAB == "sp8192":
    bin_path = Path("data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin")
    sp_path = Path("data/tokenizers/fineweb_8192_bpe.model")
else:
    raise SystemExit(f"unknown vocab {VOCAB}")

out_path = Path(f"data/floor/val_{VOCAB}.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)

print(f"reading {bin_path} ({bin_path.stat().st_size/1e6:.1f} MB)")
# modded-nanogpt shard format: 256-int32 header (1024 bytes) then uint16 tokens
header = np.fromfile(bin_path, dtype="<i4", count=256)
assert int(header[0]) == 20240520 and int(header[1]) == 1, "bad header"
n_tok = int(header[2])
toks = np.fromfile(bin_path, dtype="<u2", count=n_tok, offset=256*4)
print(f"  → {len(toks):,} tokens (header n={n_tok})")

sp = spm.SentencePieceProcessor()
sp.load(str(sp_path))
print(f"  vocab size {sp.get_piece_size()}; max token id {int(toks.max())}")
assert int(toks.max()) < sp.get_piece_size(), "token id exceeds vocab"

# Decode in chunks to keep memory bounded
CHUNK = 100_000
out = open(out_path, "wb")
written = 0
for i in range(0, len(toks), CHUNK):
    chunk = toks[i:i+CHUNK].tolist()
    text = sp.decode(chunk)
    bs = text.encode("utf-8")
    out.write(bs)
    written += len(bs)
    if LIMIT_BYTES and written >= LIMIT_BYTES:
        break
out.close()

print(f"wrote {written:,} bytes to {out_path}")
print(f"  bytes/token = {written/len(toks):.4f} (or {written/min(len(toks), i+CHUNK):.4f} if truncated)")
