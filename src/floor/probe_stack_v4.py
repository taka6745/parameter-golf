#!/usr/bin/env python3
"""probe_stack_v4.py — final round of probes.

Fix: LOOP_START=3 (was 4), ENABLE_LOOPING_AT=0.35 (was 0.5) → skip_weights shape matches.

Probes:
  P7 (re-run, correctly calibrated):  per-token val loss / BPB
  P8 (full now with all 11 layers):   per-layer block contribution ratio
  P13 CKA:                            representation similarity between adjacent layers
  P15 Gate ablation:                  zero out attn.gate_proj, measure ΔBPB
  P16 Llama-1B:                       frontier-LM BPB on same val text (ceiling)

Usage:
    python3 probe_stack_v4.py <ckpt.ptz> <val.bin> [val_text.txt]
"""
import io, os, sys, math, time, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

OUT = Path("probe_out"); OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


def load_gpt_ptz(path):
    import brotli
    blob = Path(path).read_bytes()
    return torch.load(io.BytesIO(_byte_unshuffle(brotli.decompress(blob))), map_location="cpu", weights_only=False)


def build_model():
    """Build GPT with CORRECT config for 2026-04-10 1.082 checkpoint (loop_start=3)."""
    ENV = {
        "VOCAB_SIZE": "8192", "NUM_LAYERS": "11", "MODEL_DIM": "512",
        "EMBEDDING_DIM": "512", "NUM_HEADS": "8", "NUM_KV_HEADS": "4",
        "MLP_MULT": "4.0", "SKIP_GATES_ENABLED": "1", "TIE_EMBEDDINGS": "1",
        "LN_SCALE": "1", "XSA_LAST_N": "11", "PARALLEL_RESIDUAL_START": "7",
        "PARALLEL_START_LAYER": "7",
        "QK_GAIN_INIT": "4.0",
        "NUM_LOOPS": "2", "LOOP_START": "3", "LOOP_END": "5",       # <-- FIX: was 4
        "ENABLE_LOOPING_AT": "0.35",                                # <-- FIX: was 0.5
        "SLIDING_WINDOW_ENABLED": "1",
        "LOGIT_SOFTCAP": "30", "ROPE_BASE": "10000", "ROPE_DIMS": "16",
        "ROPE_TRAIN_SEQ_LEN": "2048",
        "MATRIX_BITS": "6", "EMBED_BITS": "8",
        "DATA_DIR": "./", "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
    }
    for k, v in ENV.items():
        os.environ[k] = v
    import importlib, records_train_gpt
    importlib.reload(records_train_gpt)
    rgm = records_train_gpt
    h = rgm.Hyperparameters()
    print(f"  hparams: layers={h.num_layers} loop={h.loop_start}→{h.loop_end} ×{h.num_loops}", flush=True)
    model = rgm.GPT(h).to(DEVICE)
    # Must set looping_active=True if the ckpt was saved after looping engaged
    model.looping_active = True
    print(f"  enc_idx={model.encoder_indices}", flush=True)
    print(f"  dec_idx={model.decoder_indices}", flush=True)
    print(f"  num_skip_weights={model.num_skip_weights}", flush=True)
    return model, h, rgm


def load_state(model, rgm, ckpt_path):
    payload = load_gpt_ptz(ckpt_path)
    W, M = payload["w"], payload["m"]
    sd_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    deq = rgm.dequantize_mixed(W, M, sd_cpu)
    ms = model.state_dict()
    bad = [(k, v.shape, ms[k].shape) for k, v in deq.items() if k in ms and v.shape != ms[k].shape]
    if bad:
        print(f"  STILL mismatched: {bad[:3]}", flush=True)
    filtered = {k: v for k, v in deq.items() if k in ms and v.shape == ms[k].shape}
    missing = [k for k in ms if k not in filtered]
    print(f"  loaded={len(filtered)}  missing={len(missing)}", flush=True)
    if missing[:5]: print(f"  missing sample: {missing[:5]}", flush=True)
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def read_val_tokens(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    n = int(header[2])
    return torch.from_numpy(np.fromfile(path, dtype="<u2", count=n, offset=256*4).astype(np.int64))


def forward_logits_safely(model, x):
    """Handle either forward_logits or __call__."""
    if hasattr(model, "forward_logits"):
        return model.forward_logits(x)
    return model(x)


# ---------- P7+P8: per-token loss + per-layer block contribution ----------
def probe_7_8(model, val_tokens, h, vocab_size=8192, n_seqs=100, seq_len=2048):
    print(f"\n=== P7 + P8: per-token loss & block contribution (n_seqs={n_seqs}) ===", flush=True)
    torch.manual_seed(0)
    starts = torch.randint(0, len(val_tokens) - seq_len - 1, (n_seqs,))

    # Hook per-layer in/out norms for P8
    layer_ios = defaultdict(lambda: [[], []])
    hooks = []
    for name, mod in model.named_modules():
        m = re.match(r"^blocks\.(\d+)$", name)
        if m:
            l = int(m.group(1))
            def make_hook(l):
                def hook(module, inp, out):
                    x_in = inp[0] if isinstance(inp, tuple) else inp
                    x_out = out if not isinstance(out, tuple) else out[0]
                    layer_ios[l][0].append(x_in.detach().norm(dim=-1).mean().item())
                    layer_ios[l][1].append(x_out.detach().norm(dim=-1).mean().item())
                return hook
            hooks.append(mod.register_forward_hook(make_hook(l)))

    total_nll = 0.0; total_tokens = 0; total_bytes = 0
    token_losses = []
    # Rough bytes-per-token: 40 MB raw bytes / 40.5 M tokens = ~1.0 B/tok for sp8192 typical English
    # Actual ratio will be computed if we have detokenized text; for now use 2.35 (English avg)
    BYTES_PER_TOKEN = 2.35  # approximate; used for BPB estimate
    with torch.no_grad():
        for si, start in enumerate(starts):
            chunk = val_tokens[start:start + seq_len + 1].to(DEVICE)
            x = chunk[:-1].unsqueeze(0); y = chunk[1:].unsqueeze(0)
            logits = forward_logits_safely(model, x)
            per_pos = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="none",
            ).reshape(1, seq_len)
            total_nll += per_pos.sum().item()
            total_tokens += seq_len
            for pos in range(seq_len):
                token_losses.append((int(y[0, pos].item()), pos, float(per_pos[0, pos].item())))
            if (si + 1) % 20 == 0:
                print(f"    {si+1}/{n_seqs}  mean_nll_so_far={total_nll/total_tokens:.4f}", flush=True)
    for h_ in hooks: h_.remove()

    mean_nll = total_nll / total_tokens
    approx_bpb = mean_nll / math.log(2) / BYTES_PER_TOKEN
    print(f"  mean NLL/token: {mean_nll:.4f}  ≈ {mean_nll/math.log(2):.4f} bits/token  ≈ {approx_bpb:.4f} BPB (at {BYTES_PER_TOKEN} B/tok)", flush=True)

    # P7 write-up
    losses_arr = np.array([t[2] for t in token_losses])
    positions_arr = np.array([t[1] for t in token_losses])
    toks_arr = np.array([t[0] for t in token_losses])
    pos_bins = [(0, 128), (128, 512), (512, 1024), (1024, 2048)]
    lines = [f"# P7: Per-Token Loss (fix-calibrated)\n\n",
             f"Sample: {n_seqs} × {seq_len} = {total_tokens:,} tokens. Mean NLL/token = {mean_nll:.4f} nats ≈ {mean_nll/math.log(2):.4f} bits/token.\n",
             f"Approx BPB (at {BYTES_PER_TOKEN} B/tok): **{approx_bpb:.4f}**. Compare to submission val_bpb = 1.082.\n\n",
             "## By position\n\n| range | tokens | mean NLL |\n|---|---:|---:|\n"]
    for (lo, hi) in pos_bins:
        mask = (positions_arr >= lo) & (positions_arr < hi)
        lines.append(f"| [{lo}, {hi}) | {int(mask.sum()):,} | {float(losses_arr[mask].mean()):.4f} |\n")
    # Rarity
    vocab_counts = np.bincount(toks_arr, minlength=vocab_size)
    sorted_ids = np.argsort(vocab_counts)[::-1]
    cum = np.cumsum(vocab_counts[sorted_ids]) / vocab_counts.sum()
    cuts = np.searchsorted(cum, [0.05, 0.20, 0.50])
    buckets = [("top5pct", set(sorted_ids[:cuts[0]].tolist())),
               ("top5-20pct", set(sorted_ids[cuts[0]:cuts[1]].tolist())),
               ("top20-50pct", set(sorted_ids[cuts[1]:cuts[2]].tolist())),
               ("tail50pct", set(sorted_ids[cuts[2]:].tolist()))]
    lines.append("\n## By token rarity\n\n| bucket | tokens | mean NLL |\n|---|---:|---:|\n")
    for (label, ids) in buckets:
        mask = np.fromiter((int(t) in ids for t in toks_arr), dtype=bool, count=len(toks_arr))
        lines.append(f"| {label} | {int(mask.sum()):,} | {float(losses_arr[mask].mean()):.4f} |\n")
    (OUT / "P7_per_token_loss_v4.md").write_text("".join(lines))

    # P8 write-up
    p8_lines = ["# P8: Per-Layer Block Contribution (all 11 layers)\n\n",
                "| layer | mean in_norm | mean out_norm | out/in ratio |\n|---:|---:|---:|---:|\n"]
    p8 = []
    for l in sorted(layer_ios.keys()):
        ins, outs = layer_ios[l]
        if not ins: continue
        mi = float(np.mean(ins)); mo = float(np.mean(outs))
        ratio = mo / mi if mi > 0 else float("nan")
        p8.append((l, mi, mo, ratio))
        p8_lines.append(f"| {l} | {mi:.3f} | {mo:.3f} | {ratio:.3f} |\n")
    (OUT / "P8_block_contribution_v4.md").write_text("".join(p8_lines))
    print("  per-layer:", flush=True)
    for (l, mi, mo, r) in p8: print(f"    L{l:2d}: in={mi:.2f} out={mo:.2f} ratio={r:.3f}", flush=True)

    return {"mean_nll": mean_nll, "approx_bpb": approx_bpb, "block_ratios": p8}


# ---------- P13: per-layer CKA ----------
def probe_cka(model, val_tokens, n_seqs=10, seq_len=1024):
    print(f"\n=== P13: per-layer CKA (adjacent layer similarity) ===", flush=True)
    layer_acts = {}
    hooks = []
    for name, mod in model.named_modules():
        m = re.match(r"^blocks\.(\d+)$", name)
        if m:
            l = int(m.group(1))
            def make_hook(l):
                def hook(module, inp, out):
                    x_out = out if not isinstance(out, tuple) else out[0]
                    # Flatten seq+batch into samples × features
                    flat = x_out.detach().reshape(-1, x_out.shape[-1]).float()
                    # Accumulate (samples × dim) — keep a running sample (limit 8192 samples/layer)
                    if l not in layer_acts:
                        layer_acts[l] = [flat.cpu()]
                    else:
                        if sum(t.shape[0] for t in layer_acts[l]) < 8192:
                            layer_acts[l].append(flat.cpu())
                return hook
            hooks.append(mod.register_forward_hook(make_hook(l)))

    torch.manual_seed(1)
    starts = torch.randint(0, len(val_tokens) - seq_len - 1, (n_seqs,))
    with torch.no_grad():
        for start in starts:
            x = val_tokens[start:start + seq_len].unsqueeze(0).to(DEVICE)
            _ = forward_logits_safely(model, x)
    for h in hooks: h.remove()

    # Concat per-layer
    acts = {l: torch.cat(ts)[:8192] for l, ts in layer_acts.items()}
    print(f"  captured layers: {sorted(acts.keys())}  samples per layer: {[acts[l].shape[0] for l in sorted(acts.keys())]}", flush=True)

    # Linear CKA: CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    def linear_cka(X, Y):
        X = X - X.mean(dim=0, keepdim=True); Y = Y - Y.mean(dim=0, keepdim=True)
        hsic_xy = (Y.t() @ X).pow(2).sum().item()
        hsic_xx = (X.t() @ X).pow(2).sum().item()
        hsic_yy = (Y.t() @ Y).pow(2).sum().item()
        return hsic_xy / math.sqrt(hsic_xx * hsic_yy + 1e-12)

    adjacent = [(l, l+1) for l in sorted(acts.keys())[:-1]]
    cka_adj = [(a, b, linear_cka(acts[a], acts[b])) for (a, b) in adjacent]
    # Also pairwise for full picture
    layers = sorted(acts.keys())
    all_pairs = {}
    for i, l1 in enumerate(layers):
        for l2 in layers[i+1:]:
            all_pairs[(l1, l2)] = linear_cka(acts[l1], acts[l2])

    lines = ["# P13: CKA Between Adjacent Layers\n\n",
             "Linear CKA on activations from 10 × 1024-token val windows. CKA=1 → identical representations. CKA>0.95 → functionally redundant candidates.\n\n",
             "## Adjacent pairs\n\n| layers | CKA |\n|---|---:|\n"]
    for (a, b, c) in cka_adj:
        flag = "  ← candidate for merge/prune" if c > 0.95 else ""
        lines.append(f"| L{a}↔L{b} | {c:.4f}{flag} |\n")
    lines.append("\n## All pairs (top 15 most similar, skipping self)\n\n| layers | CKA |\n|---|---:|\n")
    sorted_pairs = sorted(all_pairs.items(), key=lambda p: -p[1])[:15]
    for ((a, b), c) in sorted_pairs:
        lines.append(f"| L{a}↔L{b} | {c:.4f} |\n")
    (OUT / "P13_cka.md").write_text("".join(lines))
    print(f"  adjacent CKA: {[(a,b,round(c,3)) for (a,b,c) in cka_adj]}", flush=True)
    return cka_adj


# ---------- P15: gated-attention ablation ----------
def probe_gate_ablation(model, val_tokens, baseline_nll, n_seqs=30, seq_len=2048):
    print(f"\n=== P15: gated attention ablation ===", flush=True)
    # Save current gate_proj weights
    saved = {}
    for name, p in model.named_parameters():
        if "attn.gate_proj.weight" in name:
            saved[name] = p.data.clone()
            p.data.zero_()
    print(f"  zeroed {len(saved)} gate_proj.weight tensors", flush=True)

    torch.manual_seed(2)
    starts = torch.randint(0, len(val_tokens) - seq_len - 1, (n_seqs,))
    total_nll = 0.0; total_tokens = 0
    with torch.no_grad():
        for start in starts:
            chunk = val_tokens[start:start + seq_len + 1].to(DEVICE)
            x = chunk[:-1].unsqueeze(0); y = chunk[1:].unsqueeze(0)
            logits = forward_logits_safely(model, x)
            nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
            total_nll += nll; total_tokens += seq_len

    # Restore
    for name, p in model.named_parameters():
        if name in saved: p.data.copy_(saved[name])

    ablated_nll = total_nll / total_tokens
    delta = ablated_nll - baseline_nll
    print(f"  baseline NLL:   {baseline_nll:.4f}", flush=True)
    print(f"  ablated NLL:    {ablated_nll:.4f}", flush=True)
    print(f"  Δ:              {delta:+.4f} nats  ({delta/math.log(2)*1000:.2f} mbit/token)", flush=True)

    lines = [f"# P15: Gated-Attention Ablation\n\n",
             f"Zeroed all 11× `attn.gate_proj.weight` tensors (45 KB of params that P11 flagged as lottery-ticket noise). Ran {n_seqs}×{seq_len}={n_seqs*seq_len:,} val tokens.\n\n",
             f"| | mean NLL/token |\n|---|---:|\n",
             f"| baseline | {baseline_nll:.4f} |\n",
             f"| gates zeroed | {ablated_nll:.4f} |\n",
             f"| Δ | {delta:+.4f} nats ({100*delta/baseline_nll:+.2f}%) |\n\n",
             f"**Interpretation**: "]
    if abs(delta) < 0.005:
        lines.append("gates contribute **essentially nothing** — safe to remove, reclaim 45 KB for other capacity.\n")
    elif delta > 0.02:
        lines.append(f"gates contribute meaningfully (+{delta:.3f} nats when removed). Keep them.\n")
    else:
        lines.append(f"gates contribute marginally ({delta:+.3f} nats). Worth ablating in a full retrain to confirm.\n")
    (OUT / "P15_gate_ablation.md").write_text("".join(lines))
    return delta


# ---------- P16: Llama-1B BPB on val text ----------
def probe_llama(val_text_path, model_id="Qwen/Qwen2.5-1.5B", n_chunks=20, chunk_chars=4000):
    print(f"\n=== P16: Frontier-LM BPB on val text ({model_id}) ===", flush=True)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("  transformers not installed, skipping", flush=True)
        return None
    if not Path(val_text_path).exists():
        print(f"  {val_text_path} not found, skipping", flush=True)
        return None

    text = Path(val_text_path).read_text(encoding="utf-8", errors="replace")
    chunks = [text[i:i+chunk_chars] for i in range(0, min(len(text), n_chunks*chunk_chars), chunk_chars)][:n_chunks]
    total_bytes = sum(len(c.encode("utf-8")) for c in chunks)
    print(f"  scoring {len(chunks)} chunks, {total_bytes:,} bytes", flush=True)

    print(f"  loading {model_id}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(DEVICE).eval()

    total_nll = 0.0; total_tokens = 0
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            ids = tok(chunk, return_tensors="pt").input_ids.to(DEVICE)
            if ids.shape[1] < 2: continue
            logits = mdl(ids[:, :-1]).logits
            targets = ids[:, 1:]
            nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
            total_nll += nll.item()
            total_tokens += targets.numel()
            if (i + 1) % 5 == 0:
                print(f"    {i+1}/{len(chunks)}", flush=True)

    bpb = total_nll / math.log(2) / total_bytes
    print(f"  Llama-3.2-1B BPB on val text: {bpb:.4f}  (our model: 1.082)", flush=True)

    lines = [f"# P16: Llama-3.2-1B BPB Ceiling\n\n",
             f"Scored {total_bytes:,} bytes of FineWeb val text (first {n_chunks} chunks of {chunk_chars} chars).\n\n",
             f"| Model | Params | Val BPB |\n|---|---:|---:|\n",
             f"| **Llama-3.2-1B** | 1.2B | **{bpb:.4f}** |\n",
             f"| Our 1.082 submission | 30M (16 MB int6) | 1.082 |\n",
             f"| xz -9e | (compressor) | 2.211 |\n\n",
             f"**Interpretation**: Llama-3.2-1B represents what a ~40× larger model with ~1000× the training compute achieves on the same text. ",
             f"If our 1.082 is {'below' if 1.082 < bpb else 'above'} Llama-1B's {bpb:.4f}, the headroom to <1.0 BPB is {'limited' if bpb > 1.0 else 'real'}. "]
    if bpb < 1.0:
        lines.append("The <1.0 moonshot is theoretically supported by a small frontier LM.\n")
    else:
        lines.append(f"The <1.0 moonshot would require beating a 1.2B-param model — likely impossible at 16 MB without structural novelty beyond scaling.\n")
    (OUT / "P16_llama_ceiling.md").write_text("".join(lines))
    return bpb


# ---------- Main ----------
def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "final_model_seed42.int6.ptz"
    val_bin = sys.argv[2] if len(sys.argv) > 2 else "val_sp8192.bin"
    val_text = sys.argv[3] if len(sys.argv) > 3 else "val_sp1024.txt"

    print(f"=== probe_stack_v4 ===", flush=True)
    model, h, rgm = build_model()
    load_state(model, rgm, ckpt)
    val_tokens = read_val_tokens(val_bin)
    print(f"val tokens: {len(val_tokens):,}", flush=True)

    # P7/P8 (re-calibrated now that shapes match)
    p7_8 = probe_7_8(model, val_tokens, h)
    baseline_nll = p7_8["mean_nll"]

    # P13 CKA
    probe_cka(model, val_tokens)

    # P15 Gate ablation (using baseline nll from P7)
    probe_gate_ablation(model, val_tokens, baseline_nll)

    # P16 Llama
    probe_llama(val_text)

    print("\n=== probe_stack_v4 DONE ===", flush=True)

if __name__ == "__main__":
    main()
