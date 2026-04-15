#!/usr/bin/env python3
"""probe_stack_v2.py — full stack-utilisation probes on a .int6.ptz checkpoint.

Adds to v1:
  P3 proxy: per-head attention weight norms (dead heads = near-zero)
  P4 proxy: MLP per-neuron weight norms (dead neurons = near-zero)
  P5 proper: embedding row usage from val token histogram
  P6 expanded: low-rank on all 2D modules, with dequantized weights
  Also dumps a machine-readable probe_summary.json

No forward pass needed — works from the quantized weights directly.
Fallbacks for non-GPTQ checkpoints also supported.

Usage:
    python3 probe_stack_v2.py <ckpt.ptz> [val.bin]
"""
import io, json, math, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else "final_model_seed42.int6.ptz"
VAL_BIN = sys.argv[2] if len(sys.argv) > 2 else None
OUT = Path("probe_out"); OUT.mkdir(exist_ok=True)
TOTAL_CAP_BYTES = 16 * 1024 * 1024

# Inferred from submission.json / record (matches 2026-04-10 config)
NUM_HEADS = 8
HEAD_DIM = 64
MLP_HIDDEN = 2048

# ---------- Load ----------
_BSHF_MAGIC = b"BSHF"
def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != _BSHF_MAGIC: return data
    stride = data[4]
    if stride < 2: return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload); out = np.empty(n, dtype=np.uint8); src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off:src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()

def try_load(path):
    blob = Path(path).read_bytes()
    for name, decomp in [
        ("brotli+unshuffle", lambda b: _byte_unshuffle(__import__("brotli").decompress(b))),
        ("lzma+unshuffle",   lambda b: _byte_unshuffle(__import__("lzma").decompress(b))),
        ("brotli",           lambda b: __import__("brotli").decompress(b)),
        ("zlib",             lambda b: __import__("zlib").decompress(b)),
        ("raw",              lambda b: b),
    ]:
        try:
            obj = torch.load(io.BytesIO(decomp(blob)), map_location="cpu", weights_only=False)
            return obj, name
        except Exception as e:
            print(f"  try {name}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    raise RuntimeError("no decoder worked")


print(f"=== probe_stack_v2: loading {CKPT} ({Path(CKPT).stat().st_size:,} bytes) ===", flush=True)
obj, decomp = try_load(CKPT)
print(f"  decompressed with {decomp}", flush=True)

# Expect {'w': {...}, 'm': {...}} from submission train.py. Handle flat fallback too.
if isinstance(obj, dict) and "w" in obj and "m" in obj:
    W = obj["w"]; M = obj["m"]
    print(f"  GPTQ format: {len(W)} weight entries, {len(M)} meta entries", flush=True)
else:
    W = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    M = {}
    print(f"  Flat format: {len(W)} tensors", flush=True)


def dequant(name_without_suffix):
    """Dequantize W[name.q] * W[name.scale] → float tensor. Returns None if not quantized."""
    q = W.get(name_without_suffix + ".q")
    s = W.get(name_without_suffix + ".scale")
    if q is None: return None
    if s is None: return q.float()
    s_f = s.float()
    if s_f.ndim > 0:
        return q.float() * s_f.view(q.shape[0], *[1] * (q.ndim - 1))
    return q.float() * float(s_f.item())


# Group tensors by module name (drop the .q / .scale suffix)
modules = {}
for k, v in W.items():
    if not isinstance(v, torch.Tensor): continue
    base = k[:-2] if k.endswith(".q") else k[:-6] if k.endswith(".scale") else k
    modules.setdefault(base, []).append((k, v))

# ---------- P1: parameter census ----------
print("\n=== P1: parameter census ===", flush=True)
rows = []
total_bytes_compressed_tensors = 0
for base, tensors in modules.items():
    total_numel = sum(t.numel() for _, t in tensors)
    total_bytes = sum(t.element_size() * t.numel() for _, t in tensors)
    total_bytes_compressed_tensors += total_bytes
    q_only = [(k, t) for k, t in tensors if k.endswith(".q")]
    shape = tuple(q_only[0][1].shape) if q_only else tuple(tensors[0][1].shape)
    dtype = str(q_only[0][1].dtype).replace("torch.", "") if q_only else str(tensors[0][1].dtype).replace("torch.", "")
    rows.append((base, shape, dtype, total_numel, total_bytes))
rows.sort(key=lambda r: -r[4])

(OUT / "P1_param_census.md").write_text(
    f"# P1: Parameter Census\n\n"
    f"- Checkpoint: `{CKPT}`  decompression: `{decomp}`\n"
    f"- Modules: {len(modules)}  Total uncompressed tensor bytes: {total_bytes_compressed_tensors:,} "
    f"({100 * total_bytes_compressed_tensors / TOTAL_CAP_BYTES:.1f}% of 16 MB — brotli shrinks this to fit)\n\n"
    f"| Module | q-shape | q-dtype | numel | bytes | % of cap |\n|---|---|---:|---:|---:|---:|\n" +
    "".join(f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]:,} | {r[4]:,} | {100*r[4]/TOTAL_CAP_BYTES:.2f}% |\n" for r in rows[:40])
)
for r in rows[:10]:
    print(f"  {r[0][:60]:<60} {str(r[1]):<14} {r[4]:>12,} {100*r[4]/TOTAL_CAP_BYTES:6.2f}%", flush=True)


# ---------- P2: weight-bit entropy (on raw quantized int tensor — unchanged) ----------
print("\n=== P2: weight-bit entropy (on quantized ints) ===", flush=True)
ent_rows = []
for base, tensors in modules.items():
    q_list = [(k, t) for k, t in tensors if k.endswith(".q")]
    if not q_list: continue
    _, q = q_list[0]
    if q.dtype not in (torch.int8, torch.int16, torch.uint8): continue
    vals = q.detach().cpu().flatten().numpy().astype(np.int64)
    counts = np.bincount(vals - vals.min(), minlength=int(vals.max() - vals.min() + 1))
    p = counts[counts > 0] / counts.sum()
    H = float(-(p * np.log2(p)).sum())
    alloc_bits = q.element_size() * 8
    wasted = (alloc_bits - H) * q.numel()
    ent_rows.append((base, alloc_bits, H, int(counts.size), q.numel(), wasted))
ent_rows.sort(key=lambda r: -r[5])

(OUT / "P2_weight_entropy.md").write_text(
    f"# P2: Weight-Bit Entropy (on raw quantized ints)\n\n"
    f"| Module | alloc_bits | H (bits/w) | distinct | numel | wasted (bits) | wasted (MB) |\n"
    f"|---|---:|---:|---:|---:|---:|---:|\n" +
    "".join(f"| `{r[0]}` | {r[1]} | {r[2]:.2f} | {r[3]} | {r[4]:,} | {r[5]:,.0f} | {r[5]/8/1e6:.2f} |\n" for r in ent_rows[:30])
)
print(f"  top-5 wasted:", flush=True)
for r in ent_rows[:5]:
    print(f"    {r[0][:55]:<55} alloc={r[1]}  H={r[2]:.2f}  wasted={r[5]/8/1e6:.2f} MB", flush=True)


# ---------- P3 proxy: attention per-head weight norms ----------
print("\n=== P3: attention head weight norms (dead-head proxy) ===", flush=True)
p3_rows = []  # (layer, head, c_q_norm, proj_norm)
for l in range(100):  # assume ≤100 blocks
    cq = dequant(f"blocks.{l}.attn.c_q.weight")
    proj = dequant(f"blocks.{l}.attn.proj.weight")
    if cq is None and proj is None: break
    for h in range(NUM_HEADS):
        # c_q: (embed_dim, embed_dim), head h owns rows [h*head_dim:(h+1)*head_dim]
        cq_norm = cq[h*HEAD_DIM:(h+1)*HEAD_DIM].norm().item() if cq is not None else float("nan")
        # proj: (embed_dim, embed_dim), head h owns columns [h*head_dim:(h+1)*head_dim]
        proj_norm = proj[:, h*HEAD_DIM:(h+1)*HEAD_DIM].norm().item() if proj is not None else float("nan")
        p3_rows.append((l, h, cq_norm, proj_norm))

# Normalize within each layer to spot outliers
by_layer = {}
for (l, h, cqn, pn) in p3_rows:
    by_layer.setdefault(l, []).append((h, cqn, pn))

(OUT / "P3_head_norms.md").write_text(
    f"# P3: Per-Head Attention Weight Norms (dead-head proxy)\n\n"
    f"Dead head proxy: head's fraction of layer's total ≪ 1/num_heads.\n"
    f"Uniform heads would each have share ≈ 1/{NUM_HEADS} = {100/NUM_HEADS:.1f}%.\n\n"
    f"| layer | head | c_q_norm | c_q_share_of_layer | proj_norm | proj_share_of_layer |\n"
    f"|---:|---:|---:|---:|---:|---:|\n" +
    "".join(
        f"| {l} | {h} | {cqn:.3f} | {100*cqn/sum(x[1] for x in by_layer[l] if not math.isnan(x[1])):.1f}% | "
        f"{pn:.3f} | {100*pn/sum(x[2] for x in by_layer[l] if not math.isnan(x[2])):.1f}% |\n"
        for (l, h, cqn, pn) in p3_rows
    )
)
# Summary
dead_head_flags = []
for l, heads in by_layer.items():
    cq_total = sum(x[1] for x in heads if not math.isnan(x[1])) or 1
    pr_total = sum(x[2] for x in heads if not math.isnan(x[2])) or 1
    for (h, cqn, pn) in heads:
        share_cq = cqn / cq_total if cq_total else 0
        share_pr = pn / pr_total if pr_total else 0
        avg_share = (share_cq + share_pr) / 2
        expected = 1.0 / NUM_HEADS
        if avg_share < 0.5 * expected:  # less than half expected weight
            dead_head_flags.append((l, h, avg_share))
print(f"  heads with <50% of expected norm share: {len(dead_head_flags)} / {len(p3_rows)}", flush=True)
for (l, h, s) in dead_head_flags[:10]:
    print(f"    layer {l} head {h}: share = {100*s:.2f}% (expected {100/NUM_HEADS:.1f}%)", flush=True)


# ---------- P4 proxy: MLP per-neuron weight norms ----------
print("\n=== P4: MLP per-neuron weight norms (dead-neuron proxy) ===", flush=True)
p4_layers = []  # (layer, dead_count, total_hidden, pct_dead)
all_neuron_norms = []
for l in range(100):
    fc = dequant(f"blocks.{l}.mlp.fc.weight")     # (hidden, embed)
    proj = dequant(f"blocks.{l}.mlp.proj.weight")  # (embed, hidden)
    if fc is None and proj is None: break
    hidden_dim = fc.shape[0] if fc is not None else proj.shape[1]
    # Each hidden neuron i has:
    #   fc row i (its input weights): fc[i, :]
    #   proj col i (its output weights): proj[:, i]
    fc_norms = fc.norm(dim=1).numpy() if fc is not None else np.full(hidden_dim, np.nan)
    proj_norms = proj.norm(dim=0).numpy() if proj is not None else np.full(hidden_dim, np.nan)
    joint = np.minimum(fc_norms, proj_norms) if fc is not None and proj is not None else np.nan_to_num(fc_norms + proj_norms, nan=0.0)
    threshold = np.percentile(joint, 10) * 0.3  # well below the 10th percentile
    dead = int((joint < threshold).sum())
    p4_layers.append((l, dead, hidden_dim, 100*dead/hidden_dim, float(joint.min()), float(joint.max()), float(joint.mean())))
    all_neuron_norms.append(joint)

(OUT / "P4_mlp_neurons.md").write_text(
    f"# P4: MLP Per-Neuron Weight Norms (dead-neuron proxy)\n\n"
    f"Dead neuron = `min(fc_row_norm, proj_col_norm)` below 0.3× layer's 10th percentile.\n\n"
    f"| layer | hidden | dead | %_dead | min_norm | max_norm | mean_norm |\n"
    f"|---:|---:|---:|---:|---:|---:|---:|\n" +
    "".join(f"| {l} | {h} | {d} | {pct:.1f}% | {mn:.3f} | {mx:.3f} | {avg:.3f} |\n"
            for (l, d, h, pct, mn, mx, avg) in p4_layers)
)
total_dead = sum(d for (_, d, _, _, _, _, _) in p4_layers)
total_hidden = sum(h for (_, _, h, _, _, _, _) in p4_layers)
print(f"  total dead neurons across all layers: {total_dead}/{total_hidden} ({100*total_dead/max(total_hidden,1):.1f}%)", flush=True)


# ---------- P5: embedding row usage (needs val) ----------
if VAL_BIN and Path(VAL_BIN).exists():
    print(f"\n=== P5: embedding row usage from val ===", flush=True)
    header = np.fromfile(VAL_BIN, dtype="<i4", count=256)
    assert int(header[0]) == 20240520, "bad val shard header"
    n_tok = int(header[2])
    toks = np.fromfile(VAL_BIN, dtype="<u2", count=n_tok, offset=256*4)
    vocab_size = int(toks.max()) + 1
    counts = np.bincount(toks, minlength=vocab_size)
    total = int(counts.sum())
    unseen = int((counts == 0).sum())
    uniform = total / vocab_size
    rare_frac = float((counts < uniform / 8).sum()) / vocab_size
    p = counts[counts > 0] / total
    H_token = float(-(p * np.log2(p)).sum())
    # Are the unseen rows ever used at all? Let's also check if their embedding row has non-trivial norm
    emb = dequant("tok_emb.weight")
    emb_row_norms = emb.norm(dim=1).numpy() if emb is not None else None
    # for rows with count=0, what's the distribution of embedding norms?
    if emb_row_norms is not None:
        unseen_idx = np.where(counts == 0)[0]
        if len(unseen_idx):
            u_norms = emb_row_norms[unseen_idx]
            stored_but_unused_mean = float(u_norms.mean())
            stored_but_unused_max = float(u_norms.max())
        else:
            stored_but_unused_mean = stored_but_unused_max = 0.0
        overall_norm_mean = float(emb_row_norms.mean())
    else:
        stored_but_unused_mean = stored_but_unused_max = overall_norm_mean = 0.0

    (OUT / "P5_embed_usage.md").write_text(
        f"# P5: Embedding Row Usage\n\n"
        f"- Val tokens scored: {total:,}  vocab: {vocab_size}\n"
        f"- Unseen rows: {unseen} ({100*unseen/vocab_size:.1f}%)\n"
        f"- Rare rows (<1/8 of uniform): {int(rare_frac*vocab_size)} ({100*rare_frac:.1f}%)\n"
        f"- Token-frequency entropy: {H_token:.3f} bits/token (uniform would be {math.log2(vocab_size):.3f})\n"
        f"- Unseen-row embedding norms: mean={stored_but_unused_mean:.3f}  max={stored_but_unused_max:.3f}  "
        f"(overall emb row norm mean: {overall_norm_mean:.3f})\n\n"
        f"## Top 20 most-used tokens\n| id | count | freq |\n|---:|---:|---:|\n" +
        "".join(f"| {tid} | {counts[tid]:,} | {counts[tid]/total:.4f} |\n" for tid in counts.argsort()[::-1][:20]) +
        f"\n## Bottom 20 used (non-zero) tokens\n| id | count |\n|---:|---:|\n" +
        "".join(f"| {tid} | {counts[tid]} |\n" for tid in np.where(counts > 0)[0][counts[np.where(counts > 0)[0]].argsort()][:20])
    )
    print(f"  unseen: {unseen}/{vocab_size}  rare: {100*rare_frac:.1f}%  H={H_token:.3f}  unused-embed-norm-mean: {stored_but_unused_mean:.3f} (vs overall {overall_norm_mean:.3f})", flush=True)
else:
    print("\n=== P5: SKIPPED (no val bin) ===", flush=True)


# ---------- P6: low-rank on large 2D dequant tensors ----------
print("\n=== P6: low-rank analysis (dequantized) ===", flush=True)
p6_rows = []
candidates = []
for base in modules:
    deq = dequant(base)
    if deq is None or deq.ndim != 2: continue
    if deq.numel() < 50_000: continue  # skip tiny
    candidates.append((base, deq))
candidates.sort(key=lambda p: -p[1].numel())

for base, w in candidates[:20]:
    t0 = time.time()
    w_np = w.detach().cpu().numpy().astype(np.float32)
    # Avoid OOM on big matrices with randomized SVD
    try:
        if min(w_np.shape) <= 1024:
            _, S, _ = np.linalg.svd(w_np, full_matrices=False)
        else:
            from numpy.linalg import svd
            _, S, _ = svd(w_np[:1024], full_matrices=False)  # approximate on first 1024 rows
    except Exception as e:
        print(f"  {base}: SVD failed: {e}", flush=True)
        continue
    var = S ** 2; cum = var.cumsum() / var.sum()
    k50 = int(np.searchsorted(cum, 0.50) + 1)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    full = w.numel()
    factor_k95 = w.shape[0] * k95 + k95 * w.shape[1]
    savings_pct = 100 * (full - factor_k95) / full if factor_k95 < full else 0.0
    p6_rows.append((base, tuple(w.shape), k50, k90, k95, k99, full, savings_pct, time.time()-t0))

(OUT / "P6_low_rank.md").write_text(
    f"# P6: Low-Rank Structure (dequantized)\n\n"
    f"| Module | shape | k50 | k90 | k95 | k99 | params | savings@k95 |\n|---|---|---:|---:|---:|---:|---:|---:|\n" +
    "".join(f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]:,} | {r[7]:.1f}% |\n" for r in p6_rows)
)
for r in p6_rows[:10]:
    print(f"  {r[0][:55]:<55} {str(r[1]):<14} k95={r[4]:>4d}/{min(r[1]):<4d} save={r[7]:5.1f}%", flush=True)


# ---------- machine-readable summary ----------
summary = {
    "checkpoint": CKPT,
    "decomp": decomp,
    "p1_modules": len(modules),
    "p1_total_uncompressed_bytes": total_bytes_compressed_tensors,
    "p2_top5_wasted_bits": [(r[0], r[2], r[5]) for r in ent_rows[:5]],
    "p3_dead_heads_count": len(dead_head_flags),
    "p3_dead_heads_list": dead_head_flags[:20],
    "p4_total_dead_neurons": total_dead,
    "p4_total_hidden": total_hidden,
    "p4_per_layer": p4_layers,
    "p6_low_rank_top": [(r[0], r[1], r[4], r[7]) for r in p6_rows[:10]],
}
if VAL_BIN and Path(VAL_BIN).exists():
    summary["p5"] = {
        "unseen_rows": unseen,
        "vocab_size": vocab_size,
        "rare_rows_pct": round(100*rare_frac, 2),
        "H_token": H_token,
        "unseen_embed_norm_mean": stored_but_unused_mean,
    }
(OUT / "probe_summary.json").write_text(json.dumps(summary, default=str, indent=2))

print("\n=== probe_stack_v2 DONE ===", flush=True)
print(f"artifacts in {OUT}/", flush=True)
