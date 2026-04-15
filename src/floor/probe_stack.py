#!/usr/bin/env python3
"""probe_stack.py — cheap under-utilisation probes on a .int6.ptz checkpoint.

Goal: find dead capacity in our 16MB codec. Each probe reports a headroom estimate
in BPB or bits. Runs without requiring the full model class — operates directly
on the deserialized state_dict.

Usage (on pod):
    python3 probe_stack.py <path/to/final_model_seed42.int6.ptz>

Output artifacts in ./probe_out/:
    P1_param_census.md     - bytes per module
    P2_weight_entropy.md   - effective bits vs allocated bits per module
    P5_embed_usage.md      - vocab-row usage on val sample
    P6_unembed_rank.md     - SVD of lm_head
"""
import io
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


CKPT = sys.argv[1] if len(sys.argv) > 1 else "final_model_seed42.int6.ptz"
VAL_BIN = sys.argv[2] if len(sys.argv) > 2 else None  # optional for P5
OUT = Path("probe_out"); OUT.mkdir(exist_ok=True)

TOTAL_CAP_BYTES = 16 * 1024 * 1024  # 16 MB

# ---------- Load ----------
_BSHF_MAGIC = b"BSHF"


def _byte_unshuffle(data: bytes) -> bytes:
    """Reverse the byte-shuffle applied by submission/train.py before compression.
    Format: b"BSHF" + 1 byte stride + payload.
    Payload is interleaved: first all bytes from pos 0 mod stride, then pos 1 mod stride, etc."""
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off:src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()


def try_load(path):
    blob = Path(path).read_bytes()
    # Canonical submission format: brotli → byte-unshuffle → torch.load
    # Fallbacks: lzma, zlib, raw (for other checkpoint formats)
    for name, decomp in [
        ("brotli+unshuffle", lambda b: _byte_unshuffle(__import__("brotli").decompress(b))),
        ("lzma+unshuffle",   lambda b: _byte_unshuffle(__import__("lzma").decompress(b))),
        ("brotli",           lambda b: __import__("brotli").decompress(b)),
        ("zlib",             lambda b: __import__("zlib").decompress(b)),
        ("raw",              lambda b: b),
    ]:
        try:
            raw = decomp(blob)
            obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            return obj, name, len(raw)
        except Exception as e:
            print(f"  try {name}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    raise RuntimeError("no decoder worked")


print(f"=== probe_stack: loading {CKPT} ({Path(CKPT).stat().st_size:,} bytes) ===", flush=True)
obj, decomp, raw_bytes = try_load(CKPT)
print(f"  decompressed with {decomp}, raw size {raw_bytes:,} bytes", flush=True)
print(f"  top-level type: {type(obj).__name__}", flush=True)
if isinstance(obj, dict):
    print(f"  top-level keys: {list(obj.keys())[:20]}", flush=True)

# The "state dict" in GPTQ-int6 format is usually {"model": {...}, "quant_meta": {...}}
# or a flat dict of tensors + meta. Find the tensor-dict.
def find_tensors(o, prefix=""):
    """Walk an object and yield (name, tensor) pairs."""
    if isinstance(o, torch.Tensor):
        yield prefix, o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from find_tensors(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from find_tensors(v, f"{prefix}[{i}]" if prefix else f"[{i}]")


tensors = dict(find_tensors(obj))
print(f"  found {len(tensors)} tensors in the checkpoint", flush=True)


# ---------- P1: parameter census ----------
print("\n=== P1: parameter census ===", flush=True)
rows = []
total_bytes = 0
for name, t in tensors.items():
    dtype = str(t.dtype).replace("torch.", "")
    bits = t.element_size() * 8
    numel = t.numel()
    bytes_ = t.element_size() * numel
    total_bytes += bytes_
    rows.append((name, tuple(t.shape), dtype, bits, numel, bytes_))

# Sort by descending bytes
rows.sort(key=lambda r: -r[5])

with open(OUT / "P1_param_census.md", "w") as f:
    f.write(f"# P1: Parameter Census\n\n")
    f.write(f"- Checkpoint: `{CKPT}`\n")
    f.write(f"- Decompression: {decomp} ({raw_bytes:,} raw bytes in torch.load'd payload)\n")
    f.write(f"- Tensor count: {len(tensors)}\n")
    f.write(f"- Total tensor bytes: {total_bytes:,} ({100*total_bytes/TOTAL_CAP_BYTES:.1f}% of 16MB cap)\n\n")
    f.write("| Module | Shape | dtype | bits | numel | bytes | % of cap |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|\n")
    for r in rows[:40]:
        f.write(f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]:,} | {r[5]:,} | {100*r[5]/TOTAL_CAP_BYTES:.2f}% |\n")
    if len(rows) > 40:
        f.write(f"\n…and {len(rows)-40} smaller tensors.\n")

for r in rows[:15]:
    print(f"  {r[0][:60]:<60} {r[1]!s:<25} {r[2]:<12} {r[5]:>12,}  {100*r[5]/TOTAL_CAP_BYTES:.2f}%", flush=True)
print(f"  TOTAL: {total_bytes:,} bytes ({100*total_bytes/TOTAL_CAP_BYTES:.1f}% of 16MB cap)", flush=True)


# ---------- P2: weight-bit entropy ----------
print("\n=== P2: weight-bit entropy ===", flush=True)
def weight_entropy(t, max_bins=256):
    """Shannon entropy of weight distribution, bits/weight."""
    x = t.detach().cpu().flatten()
    if x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        vals = x.numpy().astype(np.int64)
        # For ints, count unique values directly
        counts = np.bincount(vals - vals.min(), minlength=int(vals.max() - vals.min() + 1))
    else:
        # For floats, bin into max_bins equal-width bins
        v = x.numpy().astype(np.float64)
        if v.size == 0 or np.all(v == v[0]):
            return 0.0, 1
        counts, _ = np.histogram(v, bins=max_bins)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum()), int(counts.size)


entropy_rows = []
for r in rows[:40]:
    name = r[0]; t = tensors[name]
    alloc_bits = r[3]
    t0 = time.time()
    H, n_distinct = weight_entropy(t)
    wasted_bits_total = (alloc_bits - H) * r[4]
    entropy_rows.append((name, alloc_bits, H, n_distinct, r[4], wasted_bits_total, time.time()-t0))

entropy_rows.sort(key=lambda r: -r[5])  # sort by total wasted bits desc
with open(OUT / "P2_weight_entropy.md", "w") as f:
    f.write("# P2: Weight-Bit Entropy\n\n")
    f.write("`H` = Shannon entropy of weight values (bits/weight). If `H << alloc_bits`, ")
    f.write("we're paying for bits we're not using. `wasted_bits_total = (alloc - H) × numel`.\n\n")
    f.write("| Module | alloc_bits | H (bits/w) | distinct values | numel | wasted_total (bits) | wasted KB | MB_saved @int_down |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
    for r in entropy_rows:
        name, alloc, H, nd, numel, wasted, _ = r
        # Potential savings: re-quantize to next-lower precision (e.g. 6→4 bits)
        target_bits = max(1, math.ceil(H))
        mb_saved = (alloc - target_bits) * numel / 8 / 1e6
        f.write(f"| `{name}` | {alloc} | {H:.2f} | {nd} | {numel:,} | {wasted:,.0f} | {wasted/8/1024:,.1f} | {mb_saved:.2f} MB if quant→{target_bits}-bit |\n")

for r in entropy_rows[:15]:
    print(f"  {r[0][:50]:<50} alloc={r[1]:>2}  H={r[2]:.2f}  wasted={r[5]/8/1024:>8.1f} KB", flush=True)


# ---------- P5: embedding row usage (needs val) ----------
if VAL_BIN and Path(VAL_BIN).exists():
    print(f"\n=== P5: embedding row usage (val: {VAL_BIN}) ===", flush=True)
    # Expect competition's shard format: 256-int32 header + uint16 tokens
    with open(VAL_BIN, "rb") as f:
        header = np.fromfile(f, dtype="<i4", count=256)
    assert int(header[0]) == 20240520, "bad val shard header"
    n_tok = int(header[2])
    toks = np.fromfile(VAL_BIN, dtype="<u2", count=n_tok, offset=256*4)
    vocab_size = int(toks.max()) + 1
    counts = np.bincount(toks, minlength=vocab_size)
    total = counts.sum()
    unseen = int((counts == 0).sum())
    rare_frac = float((counts < total / (8 * vocab_size)).sum()) / vocab_size
    # Entropy of vocab distribution (bits/token)
    p = counts[counts > 0] / total
    H_token = float(-(p * np.log2(p)).sum())
    with open(OUT / "P5_embed_usage.md", "w") as f:
        f.write("# P5: Embedding Row Usage (vocab firing on val)\n\n")
        f.write(f"- Val tokens: {total:,}\n")
        f.write(f"- Vocab size: {vocab_size}\n")
        f.write(f"- Unseen rows: {unseen} ({100*unseen/vocab_size:.1f}%)\n")
        f.write(f"- Rare rows (<1/8 of uniform): {int(rare_frac*vocab_size)} ({100*rare_frac:.1f}%)\n")
        f.write(f"- Token distribution entropy: {H_token:.3f} bits/token\n\n")
        f.write("## Top 20 most-used tokens\n\n| token_id | count | freq |\n|---:|---:|---:|\n")
        for tid in counts.argsort()[::-1][:20]:
            f.write(f"| {tid} | {counts[tid]:,} | {counts[tid]/total:.4f} |\n")
        f.write("\n## Bottom 20 used (non-zero) tokens\n\n| token_id | count | freq |\n|---:|---:|---:|\n")
        nonzero = np.where(counts > 0)[0]
        for tid in nonzero[counts[nonzero].argsort()][:20]:
            f.write(f"| {tid} | {counts[tid]} | {counts[tid]/total:.7f} |\n")
    print(f"  unseen rows: {unseen}/{vocab_size}  rare rows: {100*rare_frac:.1f}%  H_token={H_token:.3f}", flush=True)
else:
    print(f"\n=== P5: SKIPPED (no val bin provided) ===", flush=True)


# ---------- P6: unembed low-rank ----------
print("\n=== P6: unembed low-rank ===", flush=True)
# Find biggest 2D tensor that looks like an output head (vocab-shaped)
lm_head_candidates = [(name, t) for name, t in tensors.items()
                       if t.ndim == 2 and t.is_floating_point()
                       and ('head' in name.lower() or 'embed' in name.lower() or 'out' in name.lower()
                            or 'unembed' in name.lower() or 'lm_' in name.lower())]
if not lm_head_candidates:
    # fallback: largest 2D float tensor
    lm_head_candidates = sorted(
        [(name, t) for name, t in tensors.items() if t.ndim == 2 and t.is_floating_point()],
        key=lambda p: -p[1].numel()
    )[:3]

with open(OUT / "P6_unembed_rank.md", "w") as f:
    f.write("# P6: Unembed / Head Low-Rank Analysis\n\n")
    f.write("For each candidate head/embed matrix, compute SVD and report cumulative variance.\n")
    f.write("If rank-k explains 95%, factoring to rank-k saves `(rows + cols - rows×cols/k×k)` params.\n\n")
    for name, t in lm_head_candidates[:3]:
        w = t.detach().cpu().float().numpy()
        if w.shape[0] > w.shape[1]:
            w = w.T  # put vocab dim first doesn't matter for SVD
        U, S, Vt = np.linalg.svd(w, full_matrices=False)
        var = S**2
        cum = var.cumsum() / var.sum()
        k50 = int(np.searchsorted(cum, 0.50) + 1)
        k90 = int(np.searchsorted(cum, 0.90) + 1)
        k95 = int(np.searchsorted(cum, 0.95) + 1)
        k99 = int(np.searchsorted(cum, 0.99) + 1)
        full = w.shape[0] * w.shape[1]
        factor_k95 = w.shape[0] * k95 + k95 * w.shape[1]
        savings = (full - factor_k95)
        f.write(f"## `{name}`  shape={t.shape}\n\n")
        f.write(f"- Full params: {full:,}\n")
        f.write(f"- k for 50% var: {k50}\n")
        f.write(f"- k for 90% var: {k90}\n")
        f.write(f"- k for 95% var: {k95}  →  rank-{k95} factor = {factor_k95:,} params "
                f"({100*savings/full:.1f}% savings)\n")
        f.write(f"- k for 99% var: {k99}\n\n")
        print(f"  {name}  {t.shape}  rank@95% = {k95} / {min(w.shape)}  savings@95% = {100*savings/full:.1f}%", flush=True)

print("\n=== probe_stack DONE ===", flush=True)
print(f"Artifacts in {OUT}/", flush=True)
