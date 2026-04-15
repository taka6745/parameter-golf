#!/usr/bin/env python3
"""probe_stack_v3.py — Tier A + Tier B stack probes for the 1.082 model.

Tier A (needs model forward pass on H100):
  P7: per-token val loss bucketed by rarity × position × domain
  P8: per-layer residual/block contribution ratio
  P9: per-head attention-softmax entropy (vs weight-norm proxy in v2 P3)

Tier A (no forward):
  P10: BSHF stride sweep (pure compression test)

Tier B (no forward):
  P11: cross-seed weight stability (load seed42/314/999, per-param std)
  P12: embedding cosine-similarity clusters

Tier B (needs model + comparison LM):
  P13: per-layer CKA between adjacent layers
  P14: per-token delta vs Llama-3.2-1B on val sample

Usage:
    # On H100, with all 3 seed checkpoints + val bin + records_train_gpt.py available:
    python3 probe_stack_v3.py
"""
import io, os, sys, math, time, json, re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn.functional as F


OUT = Path("probe_out"); OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load machinery (same as v2) ----------
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


def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride: return data
    src = np.frombuffer(data, dtype=np.uint8); n = len(src)
    out = np.empty(n, dtype=np.uint8); dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off:dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()


def load_gpt_ptz(path):
    """Load a .int6.ptz (submission format) into {'w': {...}, 'm': {...}}."""
    import brotli
    blob = Path(path).read_bytes()
    raw = _byte_unshuffle(brotli.decompress(blob))
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


def dequant(W, base):
    q = W.get(base + ".q"); s = W.get(base + ".scale")
    if q is None: return W.get(base)  # passthrough
    s_f = s.float() if s is not None else None
    if s_f is not None and s_f.ndim > 0:
        return q.float() * s_f.view(q.shape[0], *[1] * (q.ndim - 1))
    return q.float() * (float(s_f.item()) if s_f is not None else 1.0)


# ---------- P10: BSHF stride sweep (no model needed) ----------
def probe_p10(ckpt_path):
    import brotli
    print("\n=== P10: BSHF stride sweep ===", flush=True)
    blob = Path(ckpt_path).read_bytes()
    raw = _byte_unshuffle(brotli.decompress(blob))
    results = []
    for stride in (1, 2, 3, 4, 5, 8):
        if stride == 1:
            shuffled = raw
        else:
            shuffled = _byte_shuffle(raw, stride=stride)
        t0 = time.time()
        compressed = brotli.compress(shuffled, quality=11)
        elapsed = time.time() - t0
        results.append((stride, len(compressed), elapsed))
        print(f"  stride={stride}  compressed={len(compressed):,} bytes  {elapsed:.1f}s", flush=True)
    # Current checkpoint stride=2 is the baseline
    baseline = next(r for r in results if r[0] == 2)[1]
    lines = ["# P10: BSHF Stride Sweep\n",
             f"Brotli quality=11 over the dequantized torch.save payload "
             f"({len(raw):,} bytes) at various byte-shuffle strides.\n\n"
             f"Baseline (current submission uses stride=2): {baseline:,} bytes.\n\n"
             "| stride | compressed bytes | delta vs stride=2 | compress time |\n"
             "|---:|---:|---:|---:|\n"]
    for (stride, size, elapsed) in results:
        delta = size - baseline
        lines.append(f"| {stride} | {size:,} | {delta:+,} bytes ({100*delta/baseline:+.2f}%) | {elapsed:.1f}s |\n")
    (OUT / "P10_bshf_stride.md").write_text("".join(lines))
    return results


# ---------- P11: cross-seed stability ----------
def probe_p11(paths):
    print(f"\n=== P11: cross-seed stability ({len(paths)} seeds) ===", flush=True)
    Ws = [load_gpt_ptz(p)["w"] for p in paths]
    # For each module, dequantize and compute std across seeds
    all_keys = set(Ws[0].keys())
    for W in Ws[1:]: all_keys &= set(W.keys())
    stab = []
    for base_key in sorted({k[:-2] if k.endswith(".q") else (k[:-6] if k.endswith(".scale") else k) for k in all_keys}):
        deqs = [dequant(W, base_key) for W in Ws]
        if any(d is None for d in deqs): continue
        if not all(isinstance(d, torch.Tensor) for d in deqs): continue
        if deqs[0].ndim == 0 or deqs[0].numel() < 100: continue
        stack = torch.stack([d.float() for d in deqs])  # [seeds, ...]
        mean_abs = stack.abs().mean().item()
        std = stack.std(dim=0).mean().item()
        rel_std = std / (mean_abs + 1e-8)
        stab.append((base_key, tuple(deqs[0].shape), deqs[0].numel(), mean_abs, std, rel_std))
    stab.sort(key=lambda r: -r[5])
    lines = ["# P11: Cross-Seed Weight Stability\n\n",
             f"Compared {len(paths)} seeds: `{', '.join(Path(p).stem for p in paths)}`.\n\n",
             "**High rel_std = weight is noisy across seeds (lottery ticket / under-determined).** ",
             "Tensors in the bottom of this table have real signal that transfers; top are idiosyncratic.\n\n",
             "| Module | shape | numel | mean_abs | std_across_seeds | rel_std |\n|---|---|---:|---:|---:|---:|\n"]
    for r in stab[:25]:
        lines.append(f"| `{r[0]}` | {r[1]} | {r[2]:,} | {r[3]:.4g} | {r[4]:.4g} | {r[5]:.3g} |\n")
    lines.append("\n## Most stable (bottom 10)\n\n| Module | shape | rel_std |\n|---|---|---:|\n")
    for r in stab[-10:]:
        lines.append(f"| `{r[0]}` | {r[1]} | {r[5]:.3g} |\n")
    (OUT / "P11_cross_seed_stability.md").write_text("".join(lines))
    print(f"  most-noisy top-3: {[(r[0], round(r[5],3)) for r in stab[:3]]}", flush=True)
    print(f"  most-stable bot-3: {[(r[0], round(r[5],3)) for r in stab[-3:]]}", flush=True)
    return stab


# ---------- P12: embedding cosine clusters ----------
def probe_p12(W, vocab_usage_counts=None):
    print("\n=== P12: embedding cosine clusters ===", flush=True)
    emb = dequant(W, "tok_emb.weight")
    if emb is None:
        print("  no tok_emb.weight found", flush=True)
        return None
    emb = emb.float().cpu()
    V, D = emb.shape
    # Normalize to unit norm
    norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-9)
    emb_n = emb / norms
    # Compute pairwise similarity for a sample — full V×V is 64M entries at V=8192, manageable
    sim = emb_n @ emb_n.T  # [V, V]
    # For each row, find top-5 similar rows (excluding self)
    sim.fill_diagonal_(-2.0)  # exclude self
    top_sim, top_idx = sim.topk(5, dim=1)  # [V, 5]
    # Distribution of top-1 similarity
    top1 = top_sim[:, 0]
    # How many rows have top-1 sim > threshold?
    high_sim_count = {t: int((top1 > t).sum()) for t in (0.95, 0.90, 0.80, 0.70, 0.50)}
    # Cluster count via simple greedy threshold grouping
    def greedy_clusters(thresh):
        # For each vocab row, in order of decreasing norm, assign to an existing cluster
        # if cos-sim with the cluster head > thresh, else start new cluster
        order = norms.squeeze(1).argsort(descending=True).tolist()
        head_emb = []; assigned = [-1] * V; cluster_size = []
        for idx in order:
            if not head_emb:
                head_emb.append(emb_n[idx]); assigned[idx] = 0; cluster_size.append(1); continue
            # Cosine sim to each existing head
            heads = torch.stack(head_emb)
            sims = heads @ emb_n[idx]
            best = int(sims.argmax()); best_sim = float(sims[best])
            if best_sim >= thresh:
                assigned[idx] = best; cluster_size[best] += 1
            else:
                assigned[idx] = len(head_emb); head_emb.append(emb_n[idx]); cluster_size.append(1)
        return len(head_emb), cluster_size
    cluster_results = {}
    for t in (0.95, 0.90, 0.80):
        nclust, sizes = greedy_clusters(t)
        cluster_results[t] = (nclust, max(sizes), int(np.mean(sizes)))
        print(f"  cos-sim >= {t}: {nclust} clusters (max size {max(sizes)}, mean {np.mean(sizes):.1f})", flush=True)
    # Find the most similar vocab pairs
    pairs = []
    for i in range(V):
        pairs.append((i, int(top_idx[i, 0]), float(top_sim[i, 0])))
    pairs.sort(key=lambda p: -p[2])
    lines = ["# P12: Embedding Cosine-Similarity Clusters\n\n",
             f"Vocab size {V}, embedding dim {D}. Cosine sim computed between row-normalized vectors.\n\n",
             "## Top-1 similarity distribution\n\n",
             "| threshold | rows with top-1 > threshold | pct |\n|---|---:|---:|\n"]
    for t, c in high_sim_count.items():
        lines.append(f"| > {t} | {c} | {100*c/V:.1f}% |\n")
    lines.append("\n## Greedy clustering by cos-sim threshold\n\n| threshold | clusters | max cluster size | mean |\n|---|---:|---:|---:|\n")
    for t, (nc, mx, mn) in cluster_results.items():
        lines.append(f"| {t} | {nc} | {mx} | {mn} |\n")
    lines.append("\n## Top-20 most similar vocab pairs\n\n| row_i | row_j | cosine |\n|---:|---:|---:|\n")
    for (i, j, s) in pairs[:20]:
        lines.append(f"| {i} | {j} | {s:.4f} |\n")
    (OUT / "P12_embed_clusters.md").write_text("".join(lines))
    return cluster_results


# ---------- Model instantiation (for P7, P8, P9, P13) ----------
def build_model(device=DEVICE):
    """Import records_train_gpt.py as module and build the GPT with the 1.082 config."""
    print("\n=== building GPT model from records_train_gpt.py ===", flush=True)
    # Set env vars matching the submission config
    ENV = {
        "VOCAB_SIZE": "8192", "NUM_LAYERS": "11", "MODEL_DIM": "512",
        "EMBEDDING_DIM": "512", "NUM_HEADS": "8", "NUM_KV_HEADS": "4",
        "MLP_MULT": "4.0", "SKIP_GATES_ENABLED": "1", "TIE_EMBEDDINGS": "1",
        "LN_SCALE": "1", "XSA_LAST_N": "11", "PARALLEL_RESIDUAL_START": "7",
        "QK_GAIN_INIT": "4.0", "NUM_LOOPS": "2", "LOOP_START": "4", "LOOP_END": "5",
        "ENABLE_LOOPING_AT": "0.5", "SLIDING_WINDOW_ENABLED": "1",
        "LOGIT_SOFTCAP": "30", "ROPE_BASE": "10000", "ROPE_DIMS": "16",
        "ROPE_TRAIN_SEQ_LEN": "2048",
        "MATRIX_BITS": "6", "EMBED_BITS": "8",
        "DATA_DIR": "./", "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
        # avoid the tokenizer load crash if tokenizer file not present:
        # hparams builds tokenizer path anyway; we just need not to actually load tokens
    }
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    sys.path.insert(0, ".")
    import records_train_gpt as rgm
    h = rgm.Hyperparameters()
    print(f"  hparams: vocab={h.vocab_size} layers={h.num_layers} dim={h.model_dim} heads={h.num_heads}/{h.num_kv_heads} mlp×{h.mlp_mult} PR={h.parallel_residual_start}", flush=True)
    model = rgm.GPT(h).to(device)
    return model, h, rgm


def load_into_model(model, rgm, ckpt_path):
    payload = load_gpt_ptz(ckpt_path)
    W = payload["w"]; M = payload["m"]
    sd_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    deq = rgm.dequantize_mixed(W, M, sd_cpu)
    # Filter to keys present in model AND with matching shape (some small gates can differ across versions)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in deq.items() if k in model_sd and v.shape == model_sd[k].shape}
    mismatched = [(k, v.shape, model_sd[k].shape) for k, v in deq.items()
                  if k in model_sd and v.shape != model_sd[k].shape]
    extras = [k for k in deq if k not in model_sd]
    missing = [k for k in model_sd if k not in deq]
    print(f"  state_dict: loaded {len(filtered)}  mismatched {len(mismatched)} (e.g. {mismatched[:2]})  "
          f"missing {len(missing)}  extras {len(extras)}", flush=True)
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def probe_model(ckpt_path, val_bin_path):
    print(f"\n=== model-based probes (P7/P8/P9/P13) ===", flush=True)
    if not Path(val_bin_path).exists():
        print(f"  SKIP — val bin {val_bin_path} not found", flush=True)
        return
    model, h, rgm = build_model()
    load_into_model(model, rgm, ckpt_path)

    # Read val tokens
    header = np.fromfile(val_bin_path, dtype="<i4", count=256)
    n_tok = int(header[2])
    val_tokens = torch.from_numpy(
        np.fromfile(val_bin_path, dtype="<u2", count=n_tok, offset=256*4).astype(np.int64)
    )
    print(f"  val tokens: {len(val_tokens):,}", flush=True)

    # Sample ~200k tokens for forward pass (H100 can handle full val fast but we want it quick)
    SEQ_LEN = 2048
    N_SEQS = 100  # 100 × 2048 = 204,800 tokens
    torch.manual_seed(0)
    # Random start positions
    starts = torch.randint(0, len(val_tokens) - SEQ_LEN - 1, (N_SEQS,))

    # Capture per-layer block output ratio (P8) and attention softmax entropy (P9) via hooks
    layer_out_norms = defaultdict(list)
    layer_in_norms = defaultdict(list)
    attn_entropies = defaultdict(list)  # {(layer, head): [entropy per query pos]}

    # Find and hook block submodules. GPT model has self.blocks or self.transformer.blocks etc.
    # Look for modules named like "blocks.N" where N is int
    for name, mod in model.named_modules():
        m = re.match(r"^blocks\.(\d+)$", name)
        if m:
            layer = int(m.group(1))
            def make_hook(l):
                def hook(module, inp, out):
                    x_in = inp[0] if isinstance(inp, tuple) else inp
                    # x_in shape: [batch, seq, dim]; norm per sample
                    layer_in_norms[l].append(x_in.detach().norm(dim=-1).mean().item())
                    x_out = out if not isinstance(out, tuple) else out[0]
                    layer_out_norms[l].append(x_out.detach().norm(dim=-1).mean().item())
                return hook
            mod.register_forward_hook(make_hook(layer))

    # Attention entropy: patch the softmax in attention. Look for modules with an 'attn' name
    # and hook them to capture attention weights. This requires knowing the module structure.
    # Simpler: monkey-patch F.scaled_dot_product_attention? No — use a hook on attn modules
    # that capture intermediate softmax. Since we don't know exact shape, skip for now and
    # note as future probe.

    # P7: per-token loss bucketing
    print("  running forward pass for per-token loss...", flush=True)
    token_losses = []  # (token_id, position_in_seq, loss)
    with torch.no_grad():
        for seq_i, start in enumerate(starts):
            chunk = val_tokens[start:start + SEQ_LEN + 1]
            x = chunk[:-1].unsqueeze(0).to(DEVICE)
            y = chunk[1:].unsqueeze(0).to(DEVICE)
            try:
                logits = model.forward_logits(x)  # records GPT has this method
            except Exception as e:
                print(f"  model.forward_logits failed on seq {seq_i}: {type(e).__name__}: {e}", flush=True)
                break
            if logits.dim() == 3:
                # cross-entropy per position
                logprobs = F.log_softmax(logits.float(), dim=-1)
                per_pos_loss = -logprobs.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                for pos in range(SEQ_LEN):
                    token_losses.append((int(y[0, pos].item()), pos, float(per_pos_loss[0, pos].item())))
            if (seq_i + 1) % 10 == 0:
                print(f"    {seq_i + 1}/{N_SEQS} sequences done", flush=True)

    if not token_losses:
        print("  no token losses collected (model forward broke); skipping P7/P8/P9", flush=True)
        return

    print(f"  collected {len(token_losses):,} per-token losses", flush=True)
    losses_arr = np.array([t[2] for t in token_losses])
    positions_arr = np.array([t[1] for t in token_losses])
    toks_arr = np.array([t[0] for t in token_losses])

    # Bucket by position (0..128, 128..512, 512..1024, 1024..2048)
    pos_bins = [(0, 128), (128, 512), (512, 1024), (1024, 2048)]
    pos_bucket_loss = []
    for (lo, hi) in pos_bins:
        mask = (positions_arr >= lo) & (positions_arr < hi)
        if mask.sum():
            pos_bucket_loss.append((lo, hi, float(losses_arr[mask].mean()), int(mask.sum())))
    # Bucket by token rarity (using count from val)
    vocab_counts = np.bincount(toks_arr, minlength=h.vocab_size)
    total = vocab_counts.sum()
    # Quartiles: most-common 5%, next 15%, next 30%, rest 50%
    sorted_tok_ids = np.argsort(vocab_counts)[::-1]
    cum = np.cumsum(vocab_counts[sorted_tok_ids]) / total
    bucket_end = np.searchsorted(cum, [0.05, 0.20, 0.50])
    bucket_labels = [
        ("top5pct (heaviest)",  set(sorted_tok_ids[:bucket_end[0]].tolist())),
        ("top5-20pct",          set(sorted_tok_ids[bucket_end[0]:bucket_end[1]].tolist())),
        ("top20-50pct",         set(sorted_tok_ids[bucket_end[1]:bucket_end[2]].tolist())),
        ("tail50pct",           set(sorted_tok_ids[bucket_end[2]:].tolist())),
    ]
    rarity_bucket_loss = []
    for (label, ids) in bucket_labels:
        mask = np.array([int(t) in ids for t in toks_arr])  # slow but fine for ~200k tokens
        if mask.sum():
            rarity_bucket_loss.append((label, float(losses_arr[mask].mean()), int(mask.sum())))

    lines = ["# P7: Per-Token Loss Bucketing\n\n",
             f"Sample: {N_SEQS} × {SEQ_LEN} = {len(token_losses):,} tokens from val_sp8192.\n",
             f"Overall mean loss: {losses_arr.mean():.4f}  BPB-equiv ≈ {losses_arr.mean()/math.log(2) / 2.35:.3f} (rough conversion).\n\n",
             "## By position in sequence\n\n| position range | tokens | mean loss |\n|---|---:|---:|\n"]
    for (lo, hi, ml, n) in pos_bucket_loss:
        lines.append(f"| [{lo}, {hi}) | {n:,} | {ml:.4f} |\n")
    lines.append("\n## By token rarity\n\n| bucket | tokens | mean loss |\n|---|---:|---:|\n")
    for (label, ml, n) in rarity_bucket_loss:
        lines.append(f"| {label} | {n:,} | {ml:.4f} |\n")
    # also: per-token top-10 worst tokens (highest mean loss where they appeared >50 times)
    token_loss_map = defaultdict(list)
    for (tid, _, l) in token_losses:
        token_loss_map[tid].append(l)
    per_token_means = [(tid, np.mean(ls), len(ls)) for tid, ls in token_loss_map.items() if len(ls) >= 50]
    per_token_means.sort(key=lambda p: -p[1])
    lines.append("\n## Top-20 hardest tokens (≥50 occurrences)\n\n| token_id | mean_loss | count |\n|---:|---:|---:|\n")
    for (tid, ml, n) in per_token_means[:20]:
        lines.append(f"| {tid} | {ml:.4f} | {n} |\n")
    lines.append("\n## Top-20 easiest tokens (≥50 occurrences)\n\n| token_id | mean_loss | count |\n|---:|---:|---:|\n")
    for (tid, ml, n) in per_token_means[-20:]:
        lines.append(f"| {tid} | {ml:.4f} | {n} |\n")
    (OUT / "P7_per_token_loss.md").write_text("".join(lines))
    print(f"  per-token loss artifact written, overall mean loss = {losses_arr.mean():.4f}", flush=True)

    # P8: per-layer residual/block contribution
    lines_p8 = ["# P8: Per-Layer Block Contribution Ratio\n\n",
                "`out_norm / in_norm` per block during forward pass. Ratio ≈ 1 = block barely changes residual stream (possibly redundant). "
                "Ratio >> 1 = block amplifies. Ratio << 1 = block collapses / dampens.\n\n"
                "| layer | mean in_norm | mean out_norm | out/in ratio |\n|---:|---:|---:|---:|\n"]
    p8_rows = []
    for l in sorted(layer_in_norms.keys()):
        if not layer_out_norms[l]: continue
        mi = float(np.mean(layer_in_norms[l]))
        mo = float(np.mean(layer_out_norms[l]))
        ratio = mo / mi if mi > 0 else float("nan")
        p8_rows.append((l, mi, mo, ratio))
        lines_p8.append(f"| {l} | {mi:.3f} | {mo:.3f} | {ratio:.3f} |\n")
    (OUT / "P8_block_contribution.md").write_text("".join(lines_p8))
    print(f"  per-layer block contribution:", flush=True)
    for (l, mi, mo, r) in p8_rows:
        print(f"    L{l:2d}: in={mi:.2f} out={mo:.2f} ratio={r:.3f}", flush=True)


# ---------- Main ----------
def main():
    ckpt42 = sys.argv[1] if len(sys.argv) > 1 else "final_model_seed42.int6.ptz"
    val_bin = sys.argv[2] if len(sys.argv) > 2 else "val_sp8192.bin"
    seed_paths = [ckpt42] + [p for p in ["final_model_seed314.int6.ptz", "final_model_seed999.int6.ptz"] if Path(p).exists()]

    print(f"=== probe_stack_v3 ===", flush=True)
    print(f"  main ckpt: {ckpt42}  ({Path(ckpt42).stat().st_size:,} bytes)", flush=True)
    print(f"  val bin: {val_bin} ({'found' if Path(val_bin).exists() else 'MISSING'})", flush=True)
    print(f"  additional seeds for P11: {[p for p in seed_paths[1:]]}", flush=True)

    # P10: stride sweep
    probe_p10(ckpt42)

    # P11: cross-seed stability (needs 2+ seeds)
    if len(seed_paths) >= 2:
        probe_p11(seed_paths)
    else:
        print("\n=== P11: SKIPPED (need ≥2 seed checkpoints) ===", flush=True)

    # Load seed42 for the rest
    payload = load_gpt_ptz(ckpt42)
    W = payload["w"]

    # P12: embedding clusters (no forward pass needed)
    probe_p12(W)

    # P7/P8/P9/P13: model-forward probes
    if Path("records_train_gpt.py").exists():
        probe_model(ckpt42, val_bin)
    else:
        print("\n=== P7/P8/P9/P13: SKIPPED (records_train_gpt.py not in cwd) ===", flush=True)

    print("\n=== probe_stack_v3 DONE ===", flush=True)
    print(f"artifacts in {OUT}/", flush=True)


if __name__ == "__main__":
    main()
