#!/usr/bin/env python3
"""
Nacrith-style entropy-adaptive n-gram mixer for eval.

Instead of fixed n-gram bias weights (0.2, 0.15, 0.1), this adjusts mixing
based on the model's confidence (entropy) at each position:
- High entropy (uncertain) → trust n-grams MORE
- Low entropy (confident) → trust n-grams LESS

The model was trained with fixed weights, so we test whether adaptive
mixing at eval time improves or hurts val_bpb.
"""
import glob, math, time, sys
from pathlib import Path
import numpy as np
import sentencepiece as spm
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).parent))
COMPUTE_DTYPE = mx.bfloat16

def load_data_shard(path):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    return np.fromfile(path, dtype="<u2", count=int(header[2]), offset=header_bytes).astype(np.int32)

def build_sp_luts(sp, V):
    base = np.zeros(max(int(sp.vocab_size()), V), dtype=np.int16)
    lead = np.zeros_like(base, dtype=np.bool_)
    bnd = np.ones_like(base, dtype=np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        bnd[tid] = False
        if sp.is_byte(tid): base[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"): lead[tid] = True; piece = piece[1:]
        base[tid] = len(piece.encode("utf-8"))
    return base, lead, bnd

def main():
    data_path = "./data/datasets/fineweb10B_sp8192"
    tok_path = "./data/tokenizers/fineweb_8192_bpe.model"
    V = 8192; seq_len = 1024

    # Load model
    from train_gpt_mlx import GPT, Hyperparameters, CastedLinear, RMSNormNoWeight, CausalSelfAttention, MLP, Block, rms_norm
    args = Hyperparameters()
    args.vocab_size = V

    model = GPT(vocab_size=V, num_layers=args.num_layers, dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_chunk_tokens=0, logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std, qk_gain_init=args.qk_gain_init)

    model_path = "logs/bpe8192_best_v2_mlx_model.npz"
    print(f"Loading {model_path}", flush=True)
    state = dict(mx.load(model_path))
    model.update(tree_unflatten(list(state.items())))

    # Load n-gram tables
    blp = mx.array(np.load("data/bigram_logprobs_8192v.npy"), dtype=mx.float32)
    tlp = mx.array(np.load("data/trigram_logprobs_8192v.npy"), dtype=mx.float32)
    flp = mx.array(np.load("data/fourgram_logprobs_8192v.npy"), dtype=mx.float32)
    print(f"N-gram tables loaded: bi={blp.shape} tri={tlp.shape} four={flp.shape}", flush=True)
    B_tri = tlp.shape[0]; B_four = flp.shape[0]

    sp = spm.SentencePieceProcessor(model_file=tok_path)
    base_lut, lead_lut, bnd_lut = build_sp_luts(sp, V)

    # Load val tokens
    val_files = sorted(glob.glob(f"{data_path}/fineweb_val_*.bin"))
    val_tokens = np.concatenate([load_data_shard(Path(f)) for f in val_files])
    usable = ((val_tokens.size - 1) // seq_len) * seq_len
    val_tokens = val_tokens[:usable + 1]
    total_seqs = usable // seq_len
    print(f"Val: {val_tokens.size} tokens, {total_seqs} seqs", flush=True)

    # Compile forward (get logits)
    def get_logits(x):
        h = model(x)
        h_flat = h.reshape(-1, model.tok_emb.weight.shape[1])
        logits = h_flat @ model.tok_emb.weight.astype(h_flat.dtype).T
        return model.softcap(logits)
    compiled = mx.compile(get_logits, inputs=model.state, outputs=model.state)

    # Eval with fixed vs adaptive mixing
    batch_seqs = 8
    results = {"fixed": {"loss": 0.0, "tokens": 0.0, "bytes": 0.0},
               "adaptive": {"loss": 0.0, "tokens": 0.0, "bytes": 0.0},
               "none": {"loss": 0.0, "tokens": 0.0, "bytes": 0.0}}

    t0 = time.perf_counter()
    for seq_start in range(0, total_seqs, batch_seqs):
        seq_end = min(seq_start + batch_seqs, total_seqs)
        chunk = val_tokens[seq_start * seq_len : seq_end * seq_len + 1]
        x_np = chunk[:-1].reshape(-1, seq_len)
        y_np = chunk[1:].reshape(-1, seq_len)
        x = mx.array(x_np, dtype=mx.int32)

        logits = compiled(x)
        mx.eval(logits)
        logits_np = np.array(logits.astype(mx.float32))

        y_flat = y_np.reshape(-1)
        N = len(y_flat)
        f = x_np.reshape(-1)
        p2 = np.concatenate([[0], f[:-1]])
        p3 = np.concatenate([[0, 0], f[:-2]])

        # N-gram biases
        h2 = f % blp.shape[0]
        bi_bias = np.array(blp[mx.array(h2)])
        h3 = (36313 * f + 27191 * p2) % B_tri
        tri_bias = np.array(tlp[mx.array(h3.astype(np.int32))])
        h4 = (36313 * f + 27191 * p2 + 51497 * p3) % B_four
        four_bias = np.array(flp[mx.array(h4.astype(np.int32))])

        # Compute neural entropy per token
        log_probs_raw = logits_np - np.log(np.exp(logits_np).sum(axis=-1, keepdims=True) + 1e-10)
        probs_raw = np.exp(log_probs_raw)
        entropy = -np.sum(probs_raw * log_probs_raw, axis=-1)  # [N]

        for mode in results:
            if mode == "fixed":
                biased = logits_np + 0.2 * bi_bias + 0.15 * tri_bias + 0.1 * four_bias
            elif mode == "adaptive":
                # Nacrith: alpha = 0.05 + 0.55 * sigmoid(2*(H-4.0))
                alpha = 0.05 + 0.55 / (1.0 + np.exp(-2.0 * (entropy - 4.0)))
                alpha = alpha[:, None]  # [N, 1]
                biased = logits_np + alpha * (0.6 * bi_bias + 0.45 * tri_bias + 0.3 * four_bias)
            else:  # no bias
                biased = logits_np

            lp = biased - np.log(np.exp(biased).sum(axis=-1, keepdims=True) + 1e-10)
            per_tok = -lp[np.arange(N), y_flat]
            results[mode]["loss"] += float(per_tok.sum())

        # Byte counting
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        b = base_lut[tgt_ids].astype(np.float64)
        b += (lead_lut[tgt_ids] & ~bnd_lut[prev_ids]).astype(np.float64)
        for mode in results:
            results[mode]["tokens"] += N
            results[mode]["bytes"] += b.sum()

        if (seq_start // batch_seqs) % 100 == 0:
            elapsed = time.perf_counter() - t0
            for m in results:
                if results[m]["tokens"] > 0:
                    bpb = (results[m]["loss"] / results[m]["tokens"] / math.log(2)) * (results[m]["tokens"] / results[m]["bytes"])
                    print(f"  {seq_start}/{total_seqs} {m}: bpb={bpb:.4f}", flush=True)

    print(f"\n=== FINAL RESULTS ===", flush=True)
    for m in results:
        loss = results[m]["loss"] / results[m]["tokens"]
        bpb = (loss / math.log(2)) * (results[m]["tokens"] / results[m]["bytes"])
        print(f"{m:>10}: val_loss={loss:.4f} val_bpb={bpb:.4f}", flush=True)
    print(f"Elapsed: {time.perf_counter()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
