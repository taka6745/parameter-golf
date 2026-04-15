# Stack Utilisation Research — where are we wasting bits, params, and compute?

**Authors**: taka + claude  ·  **Date**: 2026-04-15  ·  **Status**: plan / in-progress

## TL;DR

We have a 16MB codec, 600s training budget, and a 3090 pod ($0.22/hr). The question: **which component of our stack is paying less BPB per bit/FLOP than it costs?** Under-utilised components are novelty targets — if a layer stores 6-bit weights with 2-bit effective entropy, there are 4 bits × that layer's param count *lying on the floor* for us to reallocate.

Running a battery of 10 probes (param census → weight entropy → attention heads → MLP neurons → embed rows → unembed rank → per-layer quant loss → GPU SM% → dataloader idle → CPU idle). Expected total runtime ~45 min. Each probe outputs a scalar "headroom" number (BPB or speed).

## 1. Background

`WIN_PLAN.md` says we need to claw back 0.012 BPB to hit SOTA and another 0.07 BPB to reach the moonshot. `FLOOR_RESULTS.md` says classical compressors and pure n-grams can't help us further. So the remaining BPB must come from **a better LM-dominant codec** — either by making the LM bigger (can't, 16MB cap), more efficient (better training), more expressive (better architecture), or by removing dead weight.

> **The claim**: the current stack has enough slack — unused param bits, idle FLOPs, dead heads, wasted vocab — that a rebuild reclaiming the slack could buy 0.02-0.05 BPB *before* adding any new novelty.

This doc catalogs what to measure, why, and in what order.

## 2. The stack layers (from `STACK_NOVELTY_TRACKER.md`)

```
┌───────────────────────────────────────────────────────────┐
│ L11  INFRA       — GPU kernels, CPU worker pool, mem BW    │
│ L10  TTT          — score-first test-time training         │
│ L09  N-GRAM ENGINE — tables, hashes, backoff framework     │
│ L08  N-GRAM BIAS   — bigram/trigram/etc. logits-bias       │
│ L07  COMPRESSION  — int6 QAT, GPTQ, brotli wrap            │
│ L06  EVAL         — 600s budget, sliding window, scoring   │
│ L05  TRAIN LOOP   — calibration, EMA, warmup, warmdown     │
│ L04  OPTIMIZER    — Muon / NorMuon / ParMuon               │
│ L03  MODEL ARCH   — 11L × 4xMLP, attention, RoPE, gated    │
│ L02  DATA LOAD    — sp1024/sp8192, prefetch, shuffle       │
│ L01  TOKENIZER    — BPE vocab, byte-fallback, init         │
└───────────────────────────────────────────────────────────┘
```

An under-utilised layer is one where the BPB-gained-per-unit-cost is below the stack-median.

## 3. Probes (ordered by cost, highest ROI first)

| # | Probe | Layer(s) | What it measures | Cost | Expected headroom (BPB or %) |
|--:|---|---|---|---|---|
| P1 | **Param census** | L03 | Bits of 16MB budget per submodule | ~10 s | frames every other probe |
| P2 | **Weight-bit entropy** | L03 × L07 | Effective bit-entropy of each module's quantized weights vs its allocated bits | ~1 min | **0.005–0.03 BPB** (re-quant savings) |
| P3 | **Attention-head entropy** | L03 | Entropy of each head's attention distribution (dead heads = flat or peaked) | ~5 min | **0.01–0.05 BPB** (prune/repurpose) |
| P4 | **MLP neuron firing rate** | L03 | % of MLP intermediate neurons that fire <1% of the time | ~5 min | **0.01–0.03 BPB** (shrink hidden dim) |
| P5 | **Embedding row usage** | L01 × L03 | Which vocab rows are ever looked up; how many are essentially dead | ~2 min | **0.005–0.02 BPB** (vocab collapse) |
| P6 | **Unembed low-rank** | L03 | Cumulative SVD variance of `lm_head`; k for 95/99% | ~1 min | **0.01–0.05 BPB** (factor to rank-k) |
| P7 | **Per-layer quant loss** | L07 | ΔBPB from dequant→requant per layer; find layers where int6 hurts most | ~10 min | **0.005–0.02 BPB** (mixed precision) |
| P8 | **GPU SM%** during train step | L11 | Avg SM utilisation during 30 train steps; idle = wasted FLOPs | ~5 min | **+20–40% train throughput** (→ more steps in 600s → lower BPB) |
| P9 | **Dataloader idle time** | L02 × L11 | Time GPU is blocked on dataloader per step | ~2 min | **+0–15% throughput** |
| P10 | **CPU idle during train** | L11 | # of vCPUs actually working; dead ones = missed precomputation | ~5 min | opens CPU-side n-gram / search novels |

## 4. Method — each probe, in one paragraph

### P1 Param census

Load `final_model_seed42.int6.ptz`. Iterate over `state_dict().items()`. Record `(name, shape, storage_dtype_bits, storage_numel)`. Compute bytes-per-module and percent of 16MB cap. Print sorted by descending bytes. Output: `data/floor/probe_param_census.md`.

### P2 Weight-bit entropy

For each module's weight tensor: if int6 quantized, look at histogram over the 64 bins. Shannon entropy `H = -Σ p_i log₂ p_i` gives the bits per weight actually carried. If a layer has `H = 3.1` over 6 allocated bits, ~2.9 bits per weight are unused. Multiply by param count → total wasted bits. Re-quant candidates.

### P3 Attention-head entropy

Forward pass on a 10k-byte val sample. Hook into attention modules, capture softmax outputs per head. For each head, compute:

- `H_pos = -Σ_j a_j log₂ a_j` averaged over query positions (entropy per head per sequence)
- Dead-head criterion: `H_pos < 0.5 * log₂(seq_len)` means head is peaked (potentially attending to a single position always); `H_pos > 0.95 * log₂(seq_len)` means head is uniform (not discriminating)

Both extremes indicate "this head may not be earning its keep." Params per head = (head_dim × d_model × 2 projections + head_dim × d_model × 2 output) — adds up fast.

### P4 MLP neuron firing

Same forward pass. Hook into MLP pre-activations. For each neuron: fraction of positions where `|activation| > threshold`. Dead neurons → zero-col-sparse MLP candidate for pruning.

### P5 Embedding row usage

For the full val tokens list (62M), count `bincount` over vocab. Rows with count < N_val / (8 * vocab_size) fire less than uniform — they're holding params for tokens that never appear. Report top-K most-idle vocab rows.

### P6 Unembed low-rank

SVD `lm_head.weight`. Plot cumulative variance. If rank-k accounts for 95% variance at `k = 100 << vocab_size`, factor `W ≈ U S V^T` with U = vocab×k, V = k×d_model. Parameter savings: `(vocab × d_model) - (vocab + d_model) × k` bits.

### P7 Per-layer quant loss

Baseline: run a 10-step forward pass, log loss. Then: for each module in turn, re-quantize its weights at lower precision (int4), re-run forward, compare loss. Repeat at int2. Layers where int4 loss ≈ int6 loss → safe to shrink. Layers where int4 loss is much worse → keep at int6 or int8.

### P8 GPU SM%

Launch `nvidia-smi dmon -s u -i 0 -c 60 > gpu_util.log` in background, run 30 train steps, kill. Post-process: mean SM%, min SM%, variance. Low mean = compute bound on something else (memory, kernel launch).

### P9 Dataloader idle

Time each train step with `torch.cuda.Event`s: t0 = pre-data-fetch, t1 = post-data-fetch, t2 = end-of-step. `(t1-t0)/(t2-t0)` is the fraction blocked on data.

### P10 CPU idle

`mpstat -P ALL 1 60` during training. Sum CPU% across all cores. Pod has 21 vCPUs; if we're using 2 fully, 19 are idle → 90% of CPU budget available for precomputation / n-gram tables / search.

## 5. Expected output shape

Each probe emits a single-line summary + a long-form artifact. The summary goes into this doc's "Results" section as:

```
| Probe | Finding | Headroom (BPB or %) | Action |
|---|---|---|---|
| P1 | embeddings=42% of cap, attn=28%, MLP=23%, lm_head=7% | 0 (frames) | — |
| P2 | attn.o_proj at 2.1 bit entropy / 6-bit alloc → 65% wasted | 0.012 BPB | shrink attn.o_proj to int4 |
| P3 | 4 of 36 heads are dead (entropy > 0.95 × log₂(seq)) | 0.018 BPB | prune, save params for deeper MLP |
| … | … | … | … |
| TOTAL | estimated free BPB from slack | 0.04 BPB | |
```

Every probe's long-form output ships to `data/floor/probe_<Pn>_<name>.{md,json}`.

## 6. Simple worked example — what P3 output looks like

Imagine the model has 11 layers × 4 heads = 44 heads. After forward pass:

```
layer_0:  H=[3.2, 3.1, 2.8, 3.0]   (good — all reasonably peaked)
layer_5:  H=[3.1, 4.8, 3.0, 3.2]   ← head 1 is near-uniform (seq=64, max_H=6)
layer_9:  H=[0.1, 0.2, 3.1, 3.0]   ← heads 0,1 are super-peaked (degenerate)
```

Dead heads: `(5,1)` too uniform (probably attending equally to everything = no signal), `(9,0)` and `(9,1)` too peaked (attending to one fixed position = maybe a BOS-token-attender). Three dead heads out of 44 = 6.8% of attention params = ~0.5MB of 16MB = **3% of total budget**. Reallocating those 0.5MB to a bigger MLP hidden dim could buy ~0.015 BPB.

## 7. Implementation — single script

All probes go in `src/floor/probe_stack.py`. Runs in one process. Uses the pod's PyTorch. Reads checkpoint from `records/track_10min_16mb/2026-04-10_*/final_model_seed42.int6.ptz`. Val sample: first 1MB of `data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin`. Writes artifacts to `data/floor/probe_*`.

## 8. Limitations

- **Forward-pass probes (P3/P4) are sample-dependent**: if the 1MB val slice doesn't exercise a rare pattern, a head that *would* fire for that pattern looks dead. Mitigate by running on a stratified 5MB sample covering multiple domains (prose/code/urls).
- **Per-layer quant loss (P7) ignores interactions**: shrinking one layer to int4 may damage the next. Mitigation: when a candidate emerges, validate with end-to-end val score before shipping.
- **GPU SM% (P8) requires real training**: can't measure on inference alone. Will need a 30-step smoke test with the real optimizer — adds ~5 min.
- **CPU probe (P10) assumes training-harness-level CPU load**: if we're running with 0 data-loading workers, CPU is trivially idle.

## 9. Next action

Bootstrap the pod (clone repo, push checkpoint), then run `probe_stack.py` in one shot. Estimated pod time: 45 min × $0.22/hr ≈ **$0.17**. Destroy pod after artifacts are pulled.
