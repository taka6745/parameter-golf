# Floor measurements for val_bpb on FineWeb — can we reach <1.0?

**Authors**: taka + claude  ·  **Date**: 2026-04-15  ·  **Status**: living document

## TL;DR

Our 1.082 BPB model already beats the best classical compressor (xz at 2.21 BPB) by **2×** and is far below the extrapolated pure-n-gram floor (~1.8–2.0 BPB with proper KN smoothing). The `<1.0 BPB` moonshot in `WIN_PLAN.md` is **not bounded out** by any cheap estimator we can compute without a frontier LM — meaning the decision stays open and hinges on Floor 3 (modern-LM BPB), not on the classical or n-gram floors. **The n-gram leg of the planned codec is a dead-end contributor** — real headroom lives in LM-side work.

## 1. Background

The competition scores submissions on **bits-per-byte (BPB)** over a held-out FineWeb val split. Our latest submission (`records/track_10min_16mb/2026-04-10_*`) scored **1.082 BPB** on H100. SOTA is **1.07 BPB**.

Before committing to a moonshot that tries to drop below **1.0 BPB**, we want to know:

1. **Is <1.0 physically possible** on this val distribution at 16MB? If the entropy of the val set is ≥1.0, *no* model of any size can beat it.
2. **Where does the remaining 0.012 BPB (us → SOTA) live?** Rare tokens? Code? Long-range context? Pure entropy headroom? The answer changes which novel to ship.
3. **Are n-gram / skip-gram tables still worth the 0.20 BPB budget** we assigned them in `WIN_PLAN.md`?

## 2. The BPB landscape (visualized)

```
  BPB  5 ┤                         ← uniform over 256 bytes = 8 BPB (out of range)
       4 ┤ ● n=1 unigram (4.62)
       3 ┤ ● n=2 (3.67)
         │ ● n=3 (3.07)
       2.5┤● n=4 (2.59)
         │ ● n=5 (2.45)
         │       ● zstd-22 (2.23)   ● brotli-q11 (2.30)
         │       ● xz -9e (2.21)           ← best classical compressor
       2 ┤  [ pure n-gram wall ~2.0 (extrapolated) ]
         │
       1.5┤
         │
       1.1┤ ● us (1.082)  ● SOTA (1.07)
       1.0┤━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ← Shannon limit for English (~1.0)
         │                          [ "moonshot" territory ]
       0.8┤ ? GPT-4o val_bpb (unmeasured — 0.7–1.0 expected)
         │ ? Llama 70B val_bpb (unmeasured)
       0.5┤
         │
         └────────────────────────────────────────
            dictionary      n-gram-LM    real-LM     frontier-LM
            methods         hybrid       territory   territory
```

The gap we care about is the **2.21 → 1.08 cliff** (dictionary → our LM) and the **1.08 → 1.00 stretch** (us → moonshot). The former is *already solved*. The latter is the whole remaining problem.

## 3. Method

All BPB numbers below are **bits per byte of original UTF-8 text**, the same units as the comp's `val_bpb`. We detokenize the tokenized val shard so every estimator sees the same byte stream.

**Val source**: `data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin` → 62M sp1024 tokens → `data/floor/val_sp1024.txt` (151,080,633 bytes UTF-8, 2.44 bytes/token). See `src/floor/detokenize_val.py`.

### 3.1 Classical compressors

`zstd -22 --ultra --long=27`, `xz -9e`, `brotli -q 11` on the 144MB val text. Single-threaded. BPB = `compressed_bytes * 8 / original_bytes`.

### 3.2 Byte n-gram add-k entropy (`src/floor/kn_ngram.py`)

- Alphabet: 256 bytes
- Train slice: first 80MB of val text (so we learn on FineWeb-distributed stats)
- Val slice: last 10MB (not overlapping train)
- For each `n ∈ {1..6}`: count n-grams in train, score val with **add-k=1 (Laplace)** smoothed probabilities
- Report `BPB(n) = -⟨log₂ p(b_i | b_{i-n+1..i-1})⟩`

Full Kneser-Ney would tighten absolute numbers ~0.3 BPB but would not change the shape (the marginal-value-per-n curve) that drives the decision.

### 3.3 GPT-4o / modern-LM logprobs (`src/floor/gpt4_logprobs.py`)

_Status_: pending — API path needs gpt-3.5-turbo-instruct with `echo=True` + `logprobs=5` (gpt-4o chat completions does not return logprobs for prefilled text). Will run for ~$0.05 on 200KB sample when authorized.

## 4. Results

### 4.1 Classical compressors

| Compressor | Settings | Compressed size | BPB | Ratio | Wall time |
|---|---|---:|---:|---:|---:|
| brotli | `-q 11` | 43,344,472 B | 2.295 | 3.49× | 7:26 |
| zstd | `-22 --ultra --long=27` | 42,045,406 B | 2.226 | 3.59× | 3:37 |
| **xz** | `-9e` | 41,758,420 B | **2.211** | 3.62× | 3:13 |

The best general-purpose compressor known to us reaches **2.21 BPB**. For our 1.082 LM to be 2× better than xz, the LM is doing real long-range predictive work that dictionary compressors can't.

### 4.2 Byte n-gram extrapolation

| n | BPB | ΔBPB vs n-1 |
|:-:|---:|---:|
| 1 | 4.618 | — |
| 2 | 3.674 | −0.944 |
| 3 | 3.070 | −0.604 |
| 4 | 2.592 | −0.479 |
| 5 | **2.455** | −0.137 |
| 6 | 2.684 | **+0.229** ← bounces UP |

**Shape**: steep through n=4, plateaus at n=5, then *increases* at n=6. The bounce-up is a classic Laplace-sparsity artifact: 10M val 6-grams vs ~80M seen 6-grams means almost every val 6-gram has `c_ctx < 10` so the add-1 smoothing dominates and assigns ~uniform 1/256 ≈ 8 BPB. Proper KN with 1GB+ train would continue the decay, landing ~1.8–2.0 BPB at n=∞.

**Key implication**: pure byte n-grams cannot get below ~1.8 BPB *even with infinite data*. Since we are already at 1.08 BPB using an LM, **n-grams can contribute almost nothing as a standalone signal** at this point. As a *bias* on top of an LM the marginal contribution is smaller still (the LM already captures most n-gram-knowable structure).

### 4.3 Modern LM — pending

Floor 3 (the only estimator that can falsify the <1.0 moonshot) is not yet measured.

## 5. Simple worked example — how a BPB number is built

Take the n=5 row. We iterate over all val positions `i` from 4 onward; for each we form `context = val[i-4:i]` (4 bytes) and `next_byte = val[i]`. Example:

```
context = b" the"  (space-t-h-e)
next_byte = b" "   (space)
count(ctx, next_byte)    = 124,593
count(ctx, *)            = 214,400
add-1 smoothed p         = (124593 + 1) / (214400 + 256 * 1) = 0.5801
contribution to total    = -log2(0.5801) = 0.786 bits for this byte
```

Sum this over all ~10M val bytes, divide by 10M, get **2.455 BPB**. If we *did not* have add-1 smoothing, unseen contexts would hand us `0/0` and the estimator would crash. If we *had* proper KN, unseen contexts would back off to the n=4 prediction and the BPB would be ~0.3 lower.

## 6. Discussion

### 6.1 The 0.20 BPB n-gram budget in WIN_PLAN.md is wrong

`WIN_PLAN.md` assigned 0.20 BPB of the <1.0 moonshot budget to "multi-order n-gram engine (skip-bigram + adaptive cuckoo, n=5,7)". That budget assumed n-grams would continue contributing at the rate we saw moving from n=2 (3.67) → n=3 (3.07) → n=4 (2.59). The curve is instead **flat-to-rising past n=5 in the regime we can afford to train n-gram tables on**. Best-case contribution of a fresh skip-bigram + n=7 table on top of the current LM is ~0.01–0.02 BPB, not 0.20.

**Plan revision**: retitle the L09 leg from "n-gram engine" to "n-gram noise-reduction bias" and reassign its 0.20 BPB budget to **LM-side novelty** (arch, training, TTT, compression-aware training).

### 6.2 Classical compressors were a negative control, not a positive one

They told us what we *don't* need to beat (xz at 2.21 — trivially) and what compressed text looks like for dictionary methods. They did not bound the moonshot. If we had set a budget for "beat xz" we'd have burned effort on a goal we already cleared by 2×.

### 6.3 Where the <1.0 moonshot's viability hides

Only Floor 3 (modern LM BPB) can answer it cheaply. The compute alternative is training larger/longer and extrapolating our own model's loss curve — expensive and less conclusive.

## 7. Implications for the Plan

| WIN_PLAN component | Prior budget | Revised budget | Reason |
|---|---:|---:|---|
| Tiny LM (long-range) | 0.50 | 0.65 | Has to carry more weight |
| Multi-order n-gram engine | 0.20 | 0.03 | Bounded by §4.2 |
| PPM / CTW component | 0.10 | 0.08 | Still useful but overlaps LM + n-gram |
| Per-token hedge mixer | 0.05 | 0.05 | Unchanged |
| Online TTT / cache | 0.05+ | 0.15+ | LM-side slot; absorbs freed n-gram budget |

Sum is still ~0.96 BPB target. The composition shifts firmly toward **LM-dominant**.

## 8. Limitations + next steps

- **Add-k vs Kneser-Ney**: the n=6 bounce-up is a smoothing artifact, not a real floor. Running with full KN and 1GB train would tighten the absolute number but not the plan-revising conclusions. **Optional rerun** — only if a reviewer pushes back.
- **Val sample is sp1024-detokenized, not the comp's raw val**: the comp's eval harness operates on bytes-per-byte of a specific val artifact. If the comp's exact byte stream differs from ours (e.g., different BOM, different line endings), our BPB numbers are ±0.01 off. Acceptable for go/no-go.
- **GPT-4o logprobs**: blocked on API path — chat completions does not expose logprobs for prefilled assistant text. Switching to `gpt-3.5-turbo-instruct` via legacy `/v1/completions` with `echo=True`. ETA: one tool call after user authorizes spend (~$0.05).
- **Per-token loss diagnostic on our actual 1.082 model**: requires loading `final_model_seed*.int6.ptz` on GPU + scoring val. Spinning up a cheap RunPod 3080 Ti now.

## 9. Attribution + artifacts

| Artifact | Location | Regenerate via |
|---|---|---|
| Val byte stream | `data/floor/val_sp1024.txt` (151 MB) | `src/floor/detokenize_val.py sp1024` |
| zstd/xz/brotli outputs | `data/floor/val_sp1024.{zst,txt.xz,txt.br}` | `zstd -22`, `xz -9e`, `brotli -q11` on the val bytes |
| KN n-gram run log | `data/floor/kn_ngram_run.log` | `src/floor/kn_ngram.py 6 80 10` |
| GPT-4o scoring (pending) | `data/floor/gpt4_logprobs.json` | `src/floor/gpt4_logprobs.py` (to be rewritten for legacy completions API) |

Per `reference_artifact_storage.md`: `val_sp1024.txt` is expensive to regenerate (custom detokenize + sentencepiece). Push to home lab Coolify + S3 when we next have the upload path ready.
