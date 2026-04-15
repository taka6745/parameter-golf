# Plan: Stack-by-Stack Novelty Audit for Parameter Golf Submission

## Context

We're racing OpenAI's Parameter Golf **track_10min_16mb** challenge: 16 MB byte-level LM, 10 min train on 8×H100, BPB on FineWeb val, deadline Apr 30 2026. Only this track — no `track_60min_16mb`. The current SOTA merged record is `val_bpb = 1.11473` (abaybektursun). Our best so far is `train_loss = 2.4499` (SP6_seed1337, RTX 3080 Ti) — never converted to a real H100 val_bpb yet.

The user's brief: **stop porting from comp PRs just because they're from comp. Drive 3 comp-novel wins per ML-stack-layer (1 of the 3 being world-novel and PhD-defensible) across all 10 layers. Never let a cheap pod sit idle. Run experiments freely to see what works and what doesn't. Hymba, custom CUDA, custom anything is in scope. Track everything in physical files so context compaction can't lose state. Drive the loop with CronCreate Claude sessions every 5 minutes (or larger) so the campaign keeps moving even when no human is watching.**

Three hard performance gates on the final submission run: **all training data must be seen, all 10 min must be used, all 16 MB must be used.**

This plan is an open-ended novelty search with hard validation gates. The 10-layer × 3-win grid is the goal; the candidate pool inside each layer grows over time as cron-fired research subagents discover new ideas. A layer is locked when 3 wins land (1 world-novel); pods then drift to the next under-served layer. Pods are NEVER idle.

---

## Prime directives (load-bearing — every other section serves these)

**PD1 — Cheap pods are never idle.** All 6 cheap pods (B–G) plus the anchor (A) must always be running an experiment. The queue must always have ≥ 1 pending experiment per pod. If the queue drains, the next cron fire generates new candidates from the research backlog or via WebSearch literature mining. **Idle GPU is a P0 alarm**, not a quality-of-life issue. Tracked as Gate G5 below.

**PD2 — Find, don't just validate.** Each layer's "3 novelties" are seeded by Plan-A's roadmap, but the actual campaign goal is to **find** 3 comp-novel wins per layer, not validate a fixed list. Failed candidates get replaced from the research pool. Exploratory experiments ("see what is good and bad") are first-class — not every run has to be a confirmation; some are speculative probes that may turn into mini-papers if they pop.

**PD3 — One world-novel per layer must be PhD-defensible.** The single world-novel per layer must satisfy: (a) zero literature/code/comp hits, (b) measurable BPB or train_loss improvement on our stack, AND (c) **could be the central contribution of a ≥ 6-page workshop paper** — i.e. there's a clear hypothesis, a clear ablation, a clear theoretical or empirical reason it works. "Trivial unmotivated trick that happens to win 0.001 BPB" does not count.

**PD4 — Anything is in scope.** Hymba (Mamba+Attention), custom CUDA kernels, custom SentencePiece forks, custom GPTQ variants, modifications to PyTorch internals, custom Triton kernels — all allowed and encouraged where they unlock a real win. The only constraint is: it must compile + run in the 10 min budget on 8×H100, and the artifact must fit in 16 MB.

**PD5 — Compaction-resistant state.** Every campaign-relevant fact lives in a physical file in the repo. `STACK_NOVELTY_TRACKER.md` is the source of truth. `RESEARCH_BACKLOG.md` is the candidate pool. `RESEARCH_LOG.md` is the append-only log. Cron sessions read these files at the START and write at the END. No cron session relies on inheriting context — they all start fresh from the files. Context window compaction can never erase progress.

**PD6 — CronCreate loops keep the campaign alive.** The autonomous loop is driven by `CronCreate`-registered Claude sessions firing at intervals. The 5-minute monitor cron is the heartbeat; longer crons (30 min research, 1 h promote, 3 h audit) handle heavier work. See "CronCreate registrations" section below for the exact list.

**PD7 — Single track focus.** Only `track_10min_16mb`. Submissions to other tracks are out of scope.

**PD8 — Max out CPU + RAM, not just GPU.** Each pod has 8-16 vCPUs and 30-41 GB RAM that sit at <10% during typical training. Idle CPU is wasted compute. Every pod's run_forever loop should also run a CPU-side worker pool that pre-bakes artifacts (n-gram tables, Lloyd-Max codebooks, DCT decompositions, brotli/zstd sweeps, BPE-8192 build, results.jsonl analysis) IN PARALLEL with GPU training. The C5 monitor must check CPU util (target ≥50% during training) alongside GPU util. New gate G6 = CPU saturation. C90 build cron should prefer speed novelties that move work to CPU-side workers.

---

## The 10 stack layers (locked names, used everywhere below)

| ID | Layer | Where it lives in train_gpt.py | Currently active in default loop? |
|---|---|---|---|
| **L01** | Tokenizer | DATA_PATH / load_validation_tokens (lines 39, 207) | SP-1024 baseline only |
| **L02** | Data pipeline | TokenStream class, shard ordering | round-robin shards, no curriculum |
| **L03** | Embedding | nn.Embedding @ 669, tied head @ 695 | tied, fp32, no factorization |
| **L04** | Attention | CausalSelfAttention @ 555-603 | GQA (8h/4kv), softcap 30, no XSA, no gating |
| **L05** | Feedforward | MLP @ 606-617 | ReLU², 2× expansion |
| **L06** | Normalization & residuals | RMSNorm @ 500, Block @ 620-645 | RMSNorm, learnable resid_mix, no LN scale |
| **L07** | Loss | F.cross_entropy @ 724 | plain CE, no byte-weight, no MTP, no smoothing |
| **L08** | Optimizer | Muon @ 112-168 | Muon (NS=5), Adam scalars, momentum 0.95 |
| **L09** | N-gram engine | NGRAM_BIAS_MARKER @ patch line 302 | bigram+trigram+4gram bias ON (only USE_* on by default) |
| **L10** | Compression & eval | quantize_state_dict_int8 @ 342, eval_val @ 219 | int8 + zlib, no GPTQ, no EMA, no sliding eval |

The 27 patches in `runpod_tests/chore/08_patch_train_gpt.sh` map cleanly onto these 10 layers. Every USE_* env var defaults to 0, so today's "default" stack is a thin baseline.

---

## Per-layer novelty roadmap (seeds, not the full pool)

For each of the 10 layers, **three SEED novelties to start with** — these are the day-1 candidates pulled from RESEARCH_LOG/LESSONS/MINIPAPERS. **Slot #3 is the world-novel seed on every layer.** Italicized rows = port-from-merged-record (locked baseline, no novelty effort).

**The campaign goal is 3 confirmed wins per layer (1 of which is world-novel + PhD-defensible), NOT necessarily these specific seeds.** If a seed fails the promotion gate, the next research-cron fire pulls a replacement from `RESEARCH_BACKLOG.md` (the candidate pool, fed by literature subagents). Layers stay open until 3 wins land. See `Research-driven novelty discovery` section below.

### L01 — Tokenizer | Pod-D + Mac
*Port: SP-1024 baseline already locked. BPE-8192 promoted to novelty #1 below since rebuild cost is large.*

| # | Novelty | Hypothesis | Expected Δ | Falsifies if |
|---|---|---|---|---|
| 1 | **BPE-8192 + standard merges + n-gram tables rebuilt** | Mac LESSONS §18c shows -0.129 BPB at 500 steps; tables exist on disk but never built for vocab=8192 | -0.05 to -0.13 BPB | step-500 train_loss not below SP-1024 baseline by 0.020 |
| 2 | **Vocab-512 ultra-compact** for n-gram density | smaller vocab → bigram coverage densifies → more bias signal per byte | -0.01 BPB net (after n-gram tables grow) | tables don't fit 16 MB after rebuild |
| 3 | **🌍 Entropy-aware BPE merge ordering** | standard BPE merges by frequency; entropy-aware variant merges pairs whose joint distribution has lowest residual joint entropy after merge → maximize compression of bigram surprise into vocab | -0.05 train_loss vs vanilla BPE-8192 | held-out NLL on 5M tokens not ≥1% below frequency-BPE-8192 |

**World-novel justification (#3):** WebSearches for "entropy aware BPE merging language model 2025/2026" return only frequency / FLOPs-aware / EmByte / BLT. **No paper uses post-merge residual joint entropy as the merge selection criterion.** `gh search` for "entropy" + "tokenizer" + "merge" in openai/parameter-golf = 0 hits.

### L02 — Data pipeline | Pod-D
*Port: sliding window eval stride=64 from 1.1147 stack (already in train_gpt.py).*

| # | Novelty | Hypothesis | Expected Δ | Falsifies if |
|---|---|---|---|---|
| 1 | USE_COPRIME_STRIDE=1 (existing patch) | coprime-to-seq_len stride decorrelates batches | -0.005 train_loss | n=2 mean within ±0.005 of baseline |
| 2 | In-batch dedup (drop seqs whose first 256B already appeared this batch) | FineWeb has duplicate boilerplate | -0.005 train_loss | dedup count <2% of batch over 100 steps |
| 3 | **🌍 Shard-curriculum on byte entropy** — order shards low-to-high zstd-ratio | easy bytes first → curriculum → faster early loss drop → more effective steps in 10 min | -0.015 train_loss | reordered shards yield train_loss ≥ baseline at step 500 |

**World-novel justification (#3):** byte-LM curricula typically use sentence length or syntactic depth, not zstd compression ratio of the shard as difficulty. Searches for "curriculum learning compression ratio language model byte" → 0 specific hits.

### L03 — Embedding | Pod-E
*Port: int8 tok_emb (already in quantize_state_dict_int8 path), tied embeddings (default).*

This layer was flagged thin (only 2 backlog ideas). Plan-A WebSearches added EmByte (EMNLP findings 2025), TensorSLM (arxiv 2025), PALU/HiRA (ICLR 2025).

| # | Novelty | Hypothesis | Expected Δ | Falsifies if |
|---|---|---|---|---|
| 1 | **Dual-codebook K-means + int4 residual** | cluster the 1024 embeddings into 64 prototypes, store int4 residual; LESSONS §37 estimates 1.7 MB freed | -0.005 BPB indirect (freed bytes → bigger n-gram tables) | reconstruction MSE > 1e-4 vs fp32 |
| 2 | **Factorized 2-matmul tied head** (vocab × 64 + 64 × 512) | LESSONS §22 — bypass tied-head constraint, save 1.87 MB | -0.003 BPB after spending freed bytes | 2nd matmul slows step >5% |
| 3 | **🌍 Hadamard-rotated tied embeddings for byte-level LM** | precompose a fixed Walsh–Hadamard rotation R into tok_emb; logits become `(x·R)·(R^T·E)^T = x·E^T` exactly, but quantization noise after rotation is uniformly spread → lower int4/int6 GPTQ error | -0.004 BPB at int6, -0.012 at int4 | quant_error on rotated tok_emb not within 5% of int8 unrotated |

**World-novel justification (#3):** PALU/HiRA apply Hadamard rotations to KV-cache and LoRA adapters respectively. **No paper or comp PR applies Hadamard rotation to a tied input/output embedding for the purpose of making int4/int6 GPTQ noise isotropic on a byte-level vocab.** Searches for "Hadamard rotated tied embedding" / "Walsh-Hadamard quantization tied head byte language model" → 0 hits.

### L04 — Attention | Pod-A + Pod-G
*Port: XSA on last 4 layers + Partial RoPE 16/64 (USE_XSA, PARTIAL_ROPE markers).*

| # | Novelty | Hypothesis | Expected Δ | Falsifies if |
|---|---|---|---|---|
| 1 | USE_GATED_ATTENTION=1 re-validation under fixed batch | NeurIPS 2025; existing patch failed under broken batch | -0.005 train_loss | screen at proper batch still fails |
| 2 | MUONEQ_R / MOUSSE / DEPTH_RECUR backlog re-validation | existing patches, prior verdicts invalid | TBD | n=2 within seed noise |
| 3 | **🌍 Coprime-stride per-head RoPE bases** — each head uses a different prime base in its 16 partial-RoPE dims, so heads see slightly different positional spectra | reduces head redundancy by spreading positional information across the head set | -0.008 train_loss | step-500 train_loss ≥ partial-RoPE baseline + 0.005 (heads desynchronize) |

**Stretch architecture exploration (Pod-G, exploratory only — does NOT count toward the 3-wins-per-layer until validated):**

| # | Stretch | Why | Effort |
|---|---|---|---|
| S1 | **Hymba** (Mamba-2 + attention hybrid, PR #852, LESSONS §28) | claims 85 ms/step at 1.1189 BPB on H100; potentially massive throughput → way more steps in 10 min budget | requires `mamba-ssm` + `causal-conv1d` external CUDA libs; ~110 LOC for HymbaAttention class; high reward but compile-fragile |
| S2 | **Pure Mamba-2 layer mixed with attention** (every-other-layer Mamba) | linear-time attention alternative; if it works at our scale, frees compute for deeper model | similar to Hymba but cleaner; needs Mamba-2 install |
| S3 | **Custom Triton attention kernel** with our specific GQA shape (8h, 4kv, head_dim=64) | F.scaled_dot_product_attention has overhead at small batches; hand-tuned Triton can get 20-40% speedup | ~200 LOC Triton; benchmark first |
| S4 | **Differential transformer attention** (subtracted attention, Microsoft Oct 2024) | noise cancellation via subtraction; never tested at byte LM scale | high crash risk (λ blows up); requires custom forward |

**Stretches are scheduled when Pod-G is otherwise idle.** If a stretch passes S1 promotion gate, it gets promoted into the main 3-slot table for the layer (replacing whichever seed has the weakest delta).

**World-novel justification (#3):** Standard RoPE uses a single base; partial RoPE drops dims; multi-base RoPE has a few hits but always with the same base across heads. **Per-head distinct prime RoPE bases inside a partial-RoPE budget = 0 hits.** Comp PRs: 0 references to per-head distinct bases.

### L05 — Feedforward | Pod-A
*Port: 3× MLP expansion + LeakyReLU(0.5)² (already in 1.1147 stack, both in 08_patch).*

| # | Novelty | Hypothesis | Expected Δ | Falsifies if |
|---|---|---|---|---|
| 1 | USE_PARALLEL_RESIDUALS=1 re-validation | comp has it merged; our prior implementation regressed | -0.005 train_loss | still regresses |
| 2 | Swish²-LeakyReLU²-mix gate | combine two activations in parallel halves | -0.004 train_loss | within seed noise |
| 3 | **🌍 Norm-percentile dropout** — zero out FFN intermediate features whose row-norm is in the top 1% (kills explosive activation pathway documented in QK_GAIN failure) | targets the rare exploding-activation pathway, not random regularization | -0.006 train_loss | step-500 train_loss unchanged or worse |

**World-novel justification (#3):** Standard dropout = random elements. Structured dropout = random rows. **Norm-percentile dropout (zero only the rows whose post-activation norm is at the 99th percentile) is a precision tool, not regularization.** Searches for "norm-percentile dropout transformer" → 0 hits. Comp: 0 PRs.

### L06 — Normalization & residuals | Pod-A
*Port: LN_SCALE 1/√(layer+1) (in 1.1147 stack, already a patch).*

| # | Novelty | Hypothesis | Expected Δ | Falsifies if |
|---|---|---|---|---|
| 1 | USE_LN_SCALE=1 re-validation | already coded | -0.003 | not measurable |
| 2 | Per-layer learned residual scalar (init 1.0, ≤9 floats) | ReZero-style but per-layer | -0.004 | hurts late-training |
| 3 | **🌍 Asymmetric U-shape skip-mix init = 0.5** — `self.skip_weights` (line 673) defaults to ones; init at 0.5 instead | the model has a U-net-style skip; 0.5 = explicit information bottleneck claim | -0.006 train_loss | step-500 train_loss worse than init=1.0 by ≥0.005 |

**World-novel justification (#3):** Initialization papers for U-Net-style transformers use init=1.0 (preservation) or 0 (rezero). **Half = explicit information bottleneck = untested.** 0 papers, 0 PRs.

### L07 — Loss | Pod-F
*Port: logit_softcap=30 (already at line 668).*

| # | Novelty | Hypothesis | Expected Δ | Falsifies if |
|---|---|---|---|---|
| 1 | USE_BYTE_WEIGHT=1 at proper scale | LESSONS §3b validated on Mac, never on H100 stack | -0.003 BPB direct | seed-noise null |
| 2 | USE_MTP=1 (Patch 21) | DeepSeek-V3 multi-token prediction | -0.005 train_loss | still within noise |
| 3 | **🌍 Asymmetric label smoothing on frequent tokens only** — ε=0.01 only for tokens whose unigram log-prob > -3; rare tokens get hard targets | inverts standard recipe: BPB is dominated by rare/hard tokens, so force perfect confidence on those and only soften easy ones | -0.004 train_loss | doesn't beat baseline at step 500 |

**World-novel justification (#3):** Standard label smoothing applies uniform ε across all targets. **Smoothing only the easy tokens (so the model is forced to be perfectly confident on rare ones, which dominate BPB) is the opposite of the usual recipe.** 0 hits for "asymmetric label smoothing frequent tokens BPB". 0 PRs.

### L08 — Optimizer | Pod-B
*Port: Parallel Muon WD=0.04, NS_STEPS=4 (Turbo-Muon), EMA 0.997 (pending Patch 17).*

| # | Novelty | Hypothesis | Expected Δ | Falsifies if |
|---|---|---|---|---|
| 1 | USE_NORMUON=1 (Patch 25) | Mac claims -0.132 BPB | -0.01 train_loss | within noise under fixed batch |
| 2 | USE_MUONEQ_R=1 (Patch 18) | RESEARCH §35 | -0.005 | within noise |
| 3 | **🌍 Per-projection Muon LR split** — split the Muon param group so q.weight, k.weight, v.weight get different LRs (currently they share) | Q/K/V have different sensitivity to NS-orthogonalization; sharing LR is suboptimal | -0.005 train_loss | any of {Q,K,V}-only LR variants destabilizes in <300 steps |

**World-novel justification (#3):** Muon papers split LR by tensor type (matrix vs scalar). **Splitting within attention projections (Q vs K vs V) by sensitivity is not standard.** Comp: PR #1172 splits LR by layer, not by Q/K/V sublayer.

### L09 — N-gram engine | Pod-C
*Port: BigramHash 3072×112 + zstd-22 (1.1147 stack), trigram + 4-gram bias (USE_NGRAM_BIAS=1 default), tabulation hash (Patch 15, our mini-paper), signed hashing (LESSONS §34).*

This is our biggest leverage layer. Already 7 backlog ideas; we pick the strongest 3.

| # | Novelty | Hypothesis | Expected Δ | Falsifies if |
|---|---|---|---|---|
| 1 | USE_ENTROPY_ADAPTIVE_NGRAM=1 re-validation | Patch 14 prior verdict invalid (broken batch) | -0.005 train_loss | still null at fixed batch |
| 2 | Skip-bigram table (Q-R trick, +0.28 bits/tok signal) | LESSONS §31 | -0.005 BPB | held-out NLL not below trigram-only baseline |
| 3 | **🌍 Context-partitioned tabulation hash** — use a different tabulation table per `(prev3 mod 16)` slice → 16× n-gram capacity in same memory budget by partitioning input space at higher-order modulus | extends tabulation to k+1 independence for the cost of one bit of indexing memory; mini-paper extension | -0.008 train_loss | held-out NLL not at least 0.002 below current tabulation hash |

**World-novel justification (#3):** Pătraşcu–Thorup tabulation is published. **Partitioning the input space by a high-order context bit and using a separate tabulation table per partition is a new construction for n-gram bias tables.** Searches for "partitioned tabulation hash language model" → 0 hits. We are the only ones who shipped tabulation hash for n-grams at all (per audit).

### L10 — Compression & eval | Mac + Pod-G(spot)
*Port: int6 GPTQ per-row + int8 tok_emb + AR self-gen GPTQ (abaybektursun) + sliding window eval stride=64 + LZMA-9.*

| # | Novelty | Hypothesis | Expected Δ | Falsifies if |
|---|---|---|---|---|
| 1 | Brotli-11 over LZMA-9 wrapper | LESSONS §18 — saves 1.47 MB on n-gram tables vs zstd-22 | -0.005 BPB indirect (more table room) | artifact size delta < 200 KB |
| 2 | AR self-gen GPTQ (abaybektursun trick) port | autoregressive self-generated calibration data — top merged record | -0.003 BPB | calibration loop crashes or is too slow |
| 3 | **🌍 Per-row Hessian-aware rANS coding on GPTQ int6 codes** — entropy-coded post-quantization, with rANS prior derived from per-row GPTQ Hessian (not just observed frequencies) | zstd doesn't model per-row distribution; Hessian-derived prior captures the per-row skew GPTQ creates | 0.5–1.0 MB savings → -0.004 BPB indirect | artifact with rANS+GPTQ-int6 not at least 200 KB smaller than brotli-11+GPTQ-int6 |

**World-novel justification (#3):** rANS is well-known. GPTQ is well-known. **Combining them with per-row Hessian-derived prior frequencies as the rANS model is not in any LM compression paper.** Searches for "rANS GPTQ byte language model" → 0 hits. Awesome-LLM-Compression list does not include this combination. Comp: 0 PRs reference rANS at all. LESSONS §10 was a *negative* result for vanilla entropy coding — Hessian prior is what changes it.

**Stretch — custom CUDA / Triton (Pod-G + Mac, exploratory):**

| # | Stretch | Why | Effort |
|---|---|---|---|
| S1 | **Custom CUDA kernel for int6 GPTQ dequant + matmul fusion** | the dequant→matmul→quant cycle currently materializes int8 buffers; fused kernel skips them | ~300 LOC CUDA + bindings; benchmark vs torch path first |
| S2 | **Custom Triton kernel for n-gram bias gather** | the bigram/trigram/4gram gather is a memory-bound op; Triton kernel can co-locate gathers with attention output | ~150 LOC Triton |
| S3 | **Custom Brotli dictionary trained on our checkpoint distribution** | Brotli supports pre-trained dictionaries; training one on a corpus of our checkpoints can save 0.5-1.5 MB | uses existing brotli `-D dict` flag; Mac-side iteration |
| S4 | **Custom SentencePiece fork** with deterministic merge ordering for the BPE-8192 build | upstream sentencepiece has frequency-tie-breaking nondeterminism; fork to make builds reproducible | ~30 LOC patch to sentencepiece source |

These get scheduled on Pod-G when the regular L10 work is queue-blocked. If any stretch beats the main slot's delta, it gets promoted.

---

## Promotion gate (precise — applies to every novelty)

**Hybrid screen-then-confirm.** Each novelty must pass two stages:

| Stage | Hardware | Config | Metric | Required delta | n_seeds |
|---|---|---|---|---|---|
| **S1 — Screen** | one cheap pod | loop default (TRAIN_SEQ_LEN=1024, TRAIN_BATCH_TOKENS=65536, MAX_WALLCLOCK_SECONDS=900, SKIP_FINAL_EVAL=1, current champion stack ON) | mean train_loss | ≤ baseline − **0.012** | n=2 (seeds 42, 1337) |
| **S2 — Confirm** | 1× H100 spot | full SKIP_FINAL_EVAL=0, escalated config from H100_TEST_PLAN.md | `final_int8_zlib_roundtrip val_bpb` | ≤ baseline − **0.003** | n=1 (seed=42) |

- **Noise floor reasoning:** CHAMP_L5 5-seed σ ≈ 0.066 / √runs gives ~0.005 train_loss seed-variance at 900 s. We use **2.4σ ≈ 0.012** as the train_loss floor (same bar tabulation hash cleared in MINIPAPER_TABULATION_HASH.md).
- **Why hybrid:** S1 alone is blind to compression/quant regressions (NorMuon could destabilize int6). S2 alone burns ~$5/run × 30 = $150, blowing the cap. Hybrid kills 70%+ candidates cheaply.
- **"Improves the model"** = passes BOTH stages against the *current best stack*, not against the bare baseline. Once a novelty merges, the next candidate's baseline shifts upward.
- **Borderline rule:** if S1 lands in [-0.012, -0.005] (suspect), bump to n=3 seeds before deciding.

## World-novel definition (precise — applies to every #3 pick + any promoted stretch)

A novelty is **world-novel iff ALL of**:

1. **Literature check (0 hits)**: WebSearch with the specific name + 2 synonym variants returns 0 papers describing the exact mechanism. arxiv.org explicit search returns 0 hits. Google Scholar returns 0 hits. Semantic Scholar returns 0 hits.
2. **Code check (0 implementations)**: GitHub code search for the algorithm signature returns 0 hits. Awesome-LLM-Compression / Awesome-Transformer-Architectures lists do not include it. HuggingFace code search returns 0 hits.
3. **Comp check (0 PRs)**: `gh api repos/openai/parameter-golf/pulls --state all` text search returns 0 hits across open + closed + merged.
4. **Empirical check**: passes the promotion gate above (S1 train_loss + S2 val_bpb).
5. **PhD-defensibility check (PD3)**: the novelty must satisfy ALL of:
   - **Hypothesis**: there is a clear, falsifiable claim about *why* the technique works (not "we tried it and it won 0.001 BPB")
   - **Theoretical or empirical mechanism**: at least one of (a) a math derivation showing why the technique should help, (b) a controlled ablation isolating which component drives the gain, (c) a connection to existing theory that this technique extends
   - **Workshop-paper test**: could a competent ML PhD student write a 6-page workshop paper around this novelty with the experiments + ablations from our campaign? If "no, it's a one-line trick with no story", the novelty is demoted to comp-novel even if checks 1-4 pass.
   - **Reproducible**: the technique is implemented behind a clean env-var gate, the results.jsonl shows multi-seed evidence, and the mini-paper draft cites the exact log files.

If any of 1-5 fails at audit time, the novelty is **demoted** to "comp-novel". The layer must then **find a replacement** (cron pulls from `RESEARCH_BACKLOG.md`) so the "1-of-3 must be world-novel" invariant holds. The audit runs as a Mac MLX cron fire (90 min cadence) for any tracker row tagged `world_novel: auditing`.

---

## Research-driven novelty discovery (the candidate pool feeder)

The 3 seeded novelties per layer are the day-1 starting set, NOT the full search space. The campaign needs a continuous flow of new candidates so PD1 (no idle pods) never breaks. This is the C30 research cron's job, plus the C5 emergency-fill when a pod's queue runs dry.

**`RESEARCH_BACKLOG.md` schema (NEW file at repo root):**

One section per layer. Each section is a markdown table of candidates, ordered by priority. Cron pops from the top, pushes to the bottom of `STACK_NOVELTY_TRACKER.md` Section A as a new slot when a layer needs more candidates.

```
## L04 — Attention candidates

| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc |
|---|---|---|---|---|---|---|---|
| 1 | hymba_mamba2_hybrid | LESSONS §28 + PR #852 | Mamba-2 + attention; 85ms/step claim | -0.05 BPB | comp-novel | 110 | 20260408T0900Z |
| 2 | per_head_rope_coprime | Plan-A seed | per-head distinct prime RoPE bases | -0.008 train_loss | world-novel-candidate | 25 | 20260408T0900Z |
| 3 | differential_attention | RESEARCH_LOG fire #5 | softmax(QK)−λ·softmax(Q'K') subtraction | unknown | comp-novel | 80 | 20260408T1200Z |
| 4 | sliding_window_64 | comp PR audit | local attention only, no global | unknown | in-comp | 30 | 20260408T1430Z |
```

**Where candidates come from (in priority order):**

1. **Mining existing files** — RESEARCH_LOG.md / LESSONS.md / RESEARCH.md / MINIPAPER_*.md / HISTORY.md. Done by Plan-A explore agent on day 1; ~50 candidates already known.
2. **C30 research cron WebSearches** — every 30 min, for the 2 under-served layers, run 3 WebSearches + 1 GitHub code search. Filter for "byte-level LM" / "small LM" / "compressed LM" / "parameter-efficient" / "10-min training" terms. Output to RESEARCH_BACKLOG.
3. **Cross-domain pollination** — every C180 audit fire, the agent rotates a "domain-of-the-day" (compression theory, hash function design, audio codecs, signal processing, formal language theory, info theory, statistical physics, robust statistics, dynamical systems) and asks: "what techniques from this domain could apply to the worst-performing layer in our stack?"
4. **Failed-experiment retros** — when an experiment falsifies in screened-fail, the C30 cron asks "is there a related variant that addresses the failure mode?" and adds those to the backlog.
5. **Manual user injection** — the user can `git push` a row to RESEARCH_BACKLOG.md at any time and it will be picked up on next cron fire.

**Minimum backlog floor:** every layer must have **at least 5 untried candidates** in its RESEARCH_BACKLOG section at all times. If any layer drops below 5, the C30 cron prioritizes filling it before doing anything else. If the WebSearch returns nothing useful, C30 spawns up to 3 parallel Explore subagents (one per cross-domain perspective) to brainstorm.

**Exploratory experiments are first-class.** When a candidate is tagged `novelty_estimate: world-novel-candidate` but the agent isn't sure it will work, the experiment runs anyway as a "see what happens" probe. If it crashes or fails the gate, the result is logged in Section B as a negative result. Negative results have research value: they constrain the hypothesis space and can become mini-papers if there's a clean theoretical reason for the failure.

---

## Pod parallelization + bootstrap

7 RunPod pods + Mac MLX. Layer ownership lives in `STACK_NOVELTY_TRACKER.md` Section A `owner_pod` column; experiments dispatched via a new `pod_filter` field on each `experiments.json` entry.

| Pod | Hardware | $/h | Layers owned | Notes |
|---|---|---|---|---|
| **A** (`tyf0q5l1kgefgx`) | RTX 3080 Ti 12 GB | 0.30 | L04, L05, L06 | Anchor — keeps all data/ artifacts, results.jsonl history |
| **B** (`vwkkjkevpvyrfs`) | RTX 3090 24 GB | 0.28 | L08 | Optimizer stability work |
| **C** (`1yo8wu8n77nbv8`) | RTX 3090 24 GB | 0.28 | L09 | N-gram engine + tabulation |
| **D** (`1nqdd6aajwqofk`) | RTX 3090 24 GB | 0.26 | L01, L02 | Tokenizer + data pipeline |
| **E** (`9g10r6i4rst296`) | RTX 3090 24 GB | 0.27 | L03 | Largest mem for embedding factorization |
| **F** (`373y5iemxa5s9o`) | RTX 3090 24 GB | 0.27 | L07 | Loss experiments |
| **G** (`7yp2f7j6rm9unm`) | RTX 4070 Ti 12 GB | 0.22 | floating utility | Backup capacity, audit replay runs |
| **H100_spot** | 1× / 8× H100 | 2.69–23.40 | S2 confirm only | Spun up + killed per S2 escalation; never idle |
| **Mac MLX** | local | 0 | L10 (compression sweeps), audit subagents, BPE-8192 builds | Always running, never blocks pods |

### Bootstrap script (`runpod_tests/loop/bootstrap_new_pod.sh`, NEW)

Reuses `chore/00_setup_pod.sh`, `chore/01_download_data.sh`, `chore/08_patch_train_gpt.sh`, `loop/install_cron.sh`, `loop/run_forever.sh`. The script sequences as 4 dependency blocks, with intra-block `&` parallelism and inter-block `wait`:

1. **Block 1** (sequential start, then parallel): `git clone` → `git checkout main` → `bash chore/00_setup_pod.sh` (venv + pip)
2. **Block 2** (parallel): `scp` from Pod A in parallel — sp1024 shards + bigram/trigram/4gram/5gram tables (avoids 7× HF rate-limit fan-out)
3. **Block 3** (sequential): `bash chore/08_patch_train_gpt.sh` → run the patcher integrity check (Gate 4 below). Exit ≠ 0 short-circuits bootstrap; no loop start, no wasted hours.
4. **Block 4** (background): `echo $POD_ID > runpod_tests/loop/pod_id.txt` → `bash install_cron.sh` → `nohup bash run_forever.sh &`

### Dispatch model

One `experiments.json`, filtered per pod via a new `pod_filter` key. Tiny patch to `experiment_runner.pick_next()`: read `runpod_tests/loop/pod_id.txt` on startup; drop any experiment whose `pod_filter` (if present) doesn't include this pod's ID. Experiments without `pod_filter` are open to all pods (existing round-robin behavior preserved).

**Why one queue file:** single source of truth, live-editable via `git push`, existing `git pull --rebase --autostash` in `experiment_runner.main()` already picks up changes.

### Failure recovery

Three layers exist (`watchdog.sh` cron + `run_forever.sh` while-true + `pick_next` 3-crash-skip). One added: **stale in-flight detection.** Mac-side cron checks tracker rows with status `in-flight`; if `updated_utc` > 25 min stale (5× max wallclock), flips back to `pending` so another pod picks it up. Worst-case duplicate run costs <$0.10.

---

## Tracking files (the compaction-resistant state, per PD5)

Three physical markdown files at the repo root, all git-committed, all hand-editable, all regex-parseable. Every cron fire reads them at start, mutates them, writes them, commits, exits. **No campaign state ever lives only in conversation context.** Compaction can wipe the conversation; the campaign continues from these files on the next cron fire.

| File | Role | Mutated by |
|---|---|---|
| `STACK_NOVELTY_TRACKER.md` | source of truth: layer status, experiment ledger, audit log, promotion log, spend ledger, gate state | every cron (C5/C30/C60/C180/C720) |
| `RESEARCH_BACKLOG.md` | candidate pool: untried novelties per layer, prioritized | C30 (append), C5 (pop on G5 fire), human (manual injection) |
| `RESEARCH_LOG.md` | append-only narrative log: every decision, every pivot, every spend tick, every cron summary | every cron + manual notes |

**Compaction discipline (PD5):** the conversation context is treated as ephemeral. The truth lives in these three files. A new Claude session that wakes to a cron fire reads these files, makes its updates, commits, and exits — never assuming any prior context.

### `STACK_NOVELTY_TRACKER.md` schema (the master tracker)

Single git-committed markdown, hand-editable, regex-parseable, six fenced sections.

### Section A — Layer status table

```
| layer | slot | novelty_id | world_novel | status | tl_delta | bpb_delta | owner_pod | updated_utc |
|-------|------|------------|-------------|--------|----------|-----------|-----------|-------------|
| L04_attention | 1 | ATT_gated_head_sigmoid | no | confirmed-win | -0.012 | -0.004 | A | 20260408T1430Z |
| L04_attention | 3 | ATT_coprime_rope_bases | yes | screened-pass | -0.021 |  | B | 20260408T1510Z |
```

`status`: `pending` | `in-flight` | `screened-pass` | `screened-fail` | `confirmed-win` | `confirmed-fail` | `demoted`. `world_novel`: `yes` | `no` | `auditing`. Timestamps `YYYYMMDDTHHMMZ` UTC.

### Section B — Experiment ledger (append-only TSV)

```
ts_utc	pod_id	novelty_id	layer	env_diff	train_loss	n_seeds	log_path	results_id	exit_code
20260408T1512Z	B	ATT_coprime_rope_bases	L04_attention	USE_PARTIAL_ROPE=1,RR_PER_HEAD=1	3.1842	2	runpod_tests/loop/logs/RR2_xxx.log	B_0142	0
```

`env_diff` = comma-separated `k=v` (only keys differing from BASE_ENV). `results_id` = `<pod>_<counter>`. Append-only; never rewritten.

### Section C — Novelty audit log

One `### <novelty_id>` block each:

```
### ATT_coprime_rope_bases
websearch_terms: ["per-head distinct RoPE bases", "coprime rotary positional embedding", "multi-base RoPE attention"]
websearch_hits: 0
github_terms: ["RR_PER_HEAD", "coprime rope base"]
github_hits: 0
comp_pr_audit_utc: 20260408T1400Z
verdict: world-novel
verdict_reason: 0 hits anywhere
owner: MAC
```

### Section D — Promotion log

Append-only bullet list:
```
- 20260408T1630Z LOCK L04_attention winners=[ATT_xsa_last4, ATT_coprime_rope_bases] demoted=[ATT_gated_head_sigmoid]
```

### Section E — Spend ledger

```
| pod_id | hw | rate_usd_per_h | started_utc | hours | subtotal_usd | state |
|--------|----|---------------|-------------|-------|--------------|-------|
| A | RTX3080Ti | 0.30 | 20260408T0500Z | 9.3 | 2.79 | running |
...
total_session_usd: 7.93
prior_sessions_spent: 6.70
grand_total_usd: 14.63
soft_cap_usd: 25.00
hard_cap_usd: 36.00
```

### Section F — Performance gate status

```
| gate | last_checked_utc | last_value | threshold | state | red_flag_ct |
|------|------------------|------------|-----------|-------|-------------|
| G1_tokens_per_min | 20260408T1620Z | 28.3M | >=12.5M (3080Ti) | PASS | 0 |
| G2_gpu_idle_streak | 20260408T1620Z | 0 streaks | 0 streaks >5s | PASS | 0 |
| G3_artifact_bytes  | 20260408T1500Z | 16,770,112 B | >=16,252,928 B | PASS | 0 |
| G4_marker_count    | 20260408T1615Z | 29/29 | 29 expected | PASS | 0 |
```

`red_flag_ct` increments on PASS→FAIL transitions; reset only by human edit.

---

## Performance gates (the 3 user demands + hidden 4th)

### Gate G1 — All training data seen

**Local evidence:** `data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin` is 200,001,024 bytes uint16 = 100M tokens/shard. 10 train shards = ~1.0B train tokens at sp1024. (The "10B" branding is the upstream FineWeb-Edu pool, not our local count.)

**Target throughput:**
- 8×H100 target: 1.0B tok / 600 s = **1.67 M tok/s aggregate** (~208 K/s/H100)
- 3080 Ti proxy (Pod A): **≥ 12.5 M tok/min** (1/8 of aggregate). Stretch: 30 M tok/min.
- 3090/4070 Ti proxy (Pods B–G): **≥ 15 M tok/min**.

**Measurement:** `experiment_runner.parse_log()` extracts `tokens_per_min = TRAIN_BATCH_TOKENS * 60000 / ms_step`. Computed once per experiment from existing log line `step:N step_avg:MS`.

**Red flag:** `tokens_per_min < threshold` → append `G1_FAIL` to Section F, push to git, runner downgrades `MAX_WALLCLOCK_SECONDS=120` until investigated. The microscopic-batch bug from last session would have shown `tokens_per_min ≈ 320K` — a 40× shortfall — and tripped instantly.

### Gate G2 — Full 10 minutes used (no idle GPU)

**Measurement:** `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 5 > runpod_tests/loop/gpu_util.log &` launched as background by `run_forever.sh`. Polls every 5 s.

**Threshold:** any consecutive `util < 80%` for >5 s window during training, or cumulative idle fraction >5% over the run = fail. First 20 s after `step:0` excluded (warmup).

**Red flag:** `G2_FAIL` line written; experiment tagged `_idle=<fraction>` in Section B `env_diff`. Persistent G2 fails → stop queue, macOS notification via `osascript` (existing `local_pull_cron.sh` pattern).

### Gate G3 — Full 16 MB used

**Measurement (only on S2 confirm runs):** parse `final_int8_zlib_roundtrip` artifact bytes from log; fall back to `os.path.getsize` on the `*.int8.ptz` file.

**Threshold:** `16 * 1024 * 1024 - 0.5 * 1024 * 1024 = 16,252,928 bytes`. Smaller artifacts = compression slack on the table.

**Red flag:** `G3_SLACK` with `slack_bytes`. Recommend bump (in priority order): `NGRAM_TAB_DIM` +25%, `NUM_LAYERS` +1, or `MODEL_DIM` +32. Recommendation chosen by ranking expected BPB impact per byte. Tracker flags slack; human picks the lever for next S2.

### Gate G4 — Patcher integrity (hidden, from session trauma)

**Measurement:** at end of every patcher run, a python helper grep-checks the 29 expected markers against `train_gpt.py`:

```python
import sys, pathlib
src = pathlib.Path('train_gpt.py').read_text()
expected = ['NGRAM_BIAS_MARKER','NGRAM_GATE_MARKER','SMEAR_GATE_MARKER',
            'TABULATION_HASH_MARKER','GATED_ATTENTION_MARKER',
            'ENTROPY_ADAPTIVE_NGRAM_MARKER','ENGRAM_LITE_MARKER','MTP_MARKER',
            'PARTIAL_ROPE_MARKER','LN_SCALE_MARKER','PARALLEL_RESIDUALS_MARKER',
            'LEAKY_RELU_MARKER','BYTE_WEIGHT_MARKER','WAVELET_GPT_MARKER',
            'COPRIME_STRIDE_MARKER','DEPTH_RECUR_MARKER','NORMUON_MARKER',
            'XSA_MARKER','MUONEQ_R_MARKER','MOUSSE_MARKER',
            'SKIP_FINAL_EVAL_MARKER','SKIP_LAST_VAL_MARKER','SKIP_POST_LOOP_MARKER',
            'PROG_SEQ_INIT_MARKER','PHASE_TRANSITION_MARKER','PHASE_TRANSITION_CLAMP',
            'NS_STEPS_MARKER']
missing = [m for m in expected if m not in src]
print('MARKERS_PRESENT:', len(expected)-len(missing), '/', len(expected))
if missing: print('MISSING:', missing); sys.exit(2)
```

**Red flag:** missing markers → exit code 2 → watchdog counts repeated G4 exit-2s → after 3 in a row, disables the pod's cron via `crontab -r`. The Patch 22 EngramLite anchor break from last session would have tripped this on first cycle.

### Gate G5 — Queue saturation (PD1, the "no idle pod" gate)

**Measurement:** the C5 monitor cron checks every 5 min:
```
for pod_id in A B C D E F G:
    in_flight = (read STACK_NOVELTY_TRACKER.md Section A, count rows
                 with status=in-flight and owner_pod=pod_id)
    pending = (read experiments.json, count entries whose pod_filter
               includes pod_id and that haven't been run by this pod yet)
    if in_flight == 0 and pending == 0:
        ALERT(pod_id, "queue empty")
```

**Threshold:** **every pod must have at least 1 pending experiment AT ALL TIMES.** A pod with `in_flight=0 and pending=0` is a P0 alarm. A pod with `in_flight=0 and pending<2` is a yellow warning (the queue is draining and may go empty before the next cron fire).

**Red flag action (auto-recovery):**
1. C5 monitor detects empty queue for a pod
2. Reads `RESEARCH_BACKLOG.md` for the layer this pod owns
3. Pops the top 2 untried candidates
4. Generates `experiments.json` entries with the pod's `pod_filter`
5. Commits + pushes
6. Pod's existing `pull --rebase` cycle picks them up within seconds
7. Writes `G5_FAIL_RECOVERED` to Section F red_flag_ct

**If `RESEARCH_BACKLOG.md` is also empty for that layer:**
1. Spawns an emergency Explore subagent (single-shot, NOT a cron) with the C30 research prompt for that layer
2. Pushes the new candidates immediately
3. If the Explore subagent finds zero candidates → escalates to `MAYDAY_BACKLOG_EMPTY` in Section F (human alert)

**Cadence:** every 5 min via C5. Out-of-band check during any C30/C60/C180 fire.

---

## Mac MLX role (always running, never blocks pods)

| Task | Trigger | Output | Gating |
|---|---|---|---|
| Web research subagents (novelty audit) | every 90 min cron | Section C audit log + RESEARCH_LOG.md bullets | runs only if any tracker row has `world_novel: auditing` |
| BPE-8192 n-gram table builds (trigram, 4-gram, 5-gram, skip-bigram) | on-demand from Section A `owner_pod=MAC` rows | `data/*_8192v.npy` + `data/skipbigram_logprobs_8192v.npy` | uses existing `chore/04_build_ngrams.py` |
| SP-1024 skip-bigram table | once at Batch 2 start | `data/skipbigram_logprobs_1024.npy` | blocks novelty L09#2 |
| Tokenizer experiments (entropy-aware BPE merge, vocab-512 sweep) | one-off Plan-agent fire | new entries in `data/tokenizers/` | not used until S2 slot opens |
| Compression sweeps (rANS, brotli-11, LZMA-9 comparison) | on every new H100 confirm artifact pulled back | `records/track_10min_16mb/<name>.compare.json` | feeds Gate G3 recommendation |
| MLX exploratory smoke tests | proposed novelties before sending to a pod | BPB estimate attached to tracker row | Mac BPB ≈ pod BPB only correlationally — never used as S1 decision alone |

**Mac vs pod split rule:** anything that needs production `train_gpt.py`, CUDA, or int8 zlib roundtrip eval → **pod**. Anything read-only against static artifacts → **Mac**.

---

## CronCreate registrations (the autonomous campaign loop)

The campaign is driven by `CronCreate`-registered Claude sessions firing at fixed intervals. **Each cron is stateless: it reads the tracker files, does its job, writes the tracker files, commits, exits.** Context-window compaction can never lose state because nothing is held in conversation memory between fires. This satisfies PD5 (compaction-resistant state).

**Existing pod-side crons, kept unchanged:**
- `* * * * *` → `runpod_tests/loop/watchdog.sh` (pod-side `cron`, restarts `run_forever.sh` if dead)
- `run_forever.sh` while-true wraps `experiment_runner.py`

**Existing Mac-side cron, kept unchanged:**
- `*/10 * * * *` (system `cron`) → `runpod_tests/loop/local_pull_cron.sh` (anchor pull)

### New CronCreate Claude sessions (registered via the CronCreate tool, not system cron)

| ID | Frequency | Name | Payload purpose |
|---|---|---|---|
| **C5** | every 5 min | `monitor` | heartbeat: pod liveness, queue saturation (G5), gate state, idle alarm |
| **C30** | every 30 min | `research` | research backlog mining: WebSearch new candidates for under-served layers, append to `RESEARCH_BACKLOG.md`, update Section C audit log |
| **C60** | every 1 h | `promote` | layer promotion: any layer with 3 wins (1 world-novel) gets a Section D LOCK line; pods reassigned to next under-served layer |
| **C180** | every 3 h | `audit` | full re-audit: re-run novelty literature checks, recompute spend, summary commit, run interaction-screen for stacked-novelty conflicts |
| **C720** | every 12 h | `H100_confirm` | check tracker for novelties in `screened-pass` state; if any, spin up 1× H100 spot for one S2 confirm batch (max 3 per session per spend ceiling), then kill the H100 |

### C5 — monitor (every 5 min, the heartbeat)

```
cd /Users/takodamundy/Documents/personal_repos/paramgolf
1. git pull --rebase origin main
2. cat STACK_NOVELTY_TRACKER.md (read all 6 sections)
3. cat RESEARCH_BACKLOG.md (read candidate pool)
4. bash runpod_tests/loop/pull_all_pods.sh  (pull results from all 7 pods in parallel)
5. For each pod ssh and run:
     git branch --show-current
     git status --short
     nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
     ps -ef | grep run_forever
6. CHECK PD1 (G5 queue saturation):
     for each pod, count pending experiments matching its pod_filter
     if any pod has 0 pending → P0 ALARM:
       (a) read RESEARCH_BACKLOG.md
       (b) pop top 2 candidates for that pod's layer
       (c) commit to experiments.json (git add + push)
       (d) write G5_FAIL → G5_RECOVERED in Section F
     if any pod has been idle (no in-flight + no pending) for >5 min → P0
7. CHECK gates G1 G2 G3 G4 from gate_check.py output → update Section F
8. Parse last 100 rows of Section B; recompute tl_delta + bpb_delta per layer; update Section A
9. CHECK spend Section E:
     <$20 normal | $20-25 warn | $25-30 stop queue | $30+ kill cheap pods
10. git commit STACK_NOVELTY_TRACKER.md RESEARCH_BACKLOG.md experiments.json
    -m "C5 monitor <UTC>: <queue depth> pending across <n> pods"
11. git push
```

### C30 — research (every 30 min, the candidate pool feeder)

```
cd /Users/takodamundy/Documents/personal_repos/paramgolf
1. git pull --rebase origin main
2. cat STACK_NOVELTY_TRACKER.md Section A
3. Identify the 2 layers with FEWEST confirmed-wins (under-served)
4. For each under-served layer, spawn 1 Explore subagent with prompt:
     "Read RESEARCH_LOG.md, MINIPAPER_*.md, RESEARCH.md, LESSONS.md.
      Run 3 WebSearches for novel ideas in <layer> for byte-level LMs 2024-2026.
      Run 1 GitHub code search for nearby implementations.
      Return 3 candidate novelties with: name, hypothesis, expected delta,
      novelty estimate (world-novel/comp-novel/in-comp), code skeleton."
5. Append the returned candidates to RESEARCH_BACKLOG.md under the layer heading
6. For each new world-novel candidate, run the 5-check audit (literature/code/comp/empirical/PhD)
   and write a Section C audit block
7. git commit RESEARCH_BACKLOG.md STACK_NOVELTY_TRACKER.md
   -m "C30 research <UTC>: +<n> candidates for <layer>"
8. git push
```

### C60 — promote (every 1 h, the layer locker)

```
cd /Users/takodamundy/Documents/personal_repos/paramgolf
1. git pull --rebase origin main
2. Parse Section A: for each layer, count rows with status=confirmed-win
3. For any layer with ≥3 confirmed-wins AND ≥1 with world_novel=yes:
     append "20XXMMDDTHHMMZ LOCK <layer> winners=[...] demoted=[...]" to Section D
     mark all other slots for that layer as `demoted`
     reassign owner_pod for all pods that were on this layer to the next under-served layer
4. For any pending row >25 min stale → flip to pending (stale-in-flight recovery)
5. git commit STACK_NOVELTY_TRACKER.md
   -m "C60 promote <UTC>: locked <n> layers, <m> pods reassigned"
6. git push
```

### C180 — audit (every 3 h, the deep check)

```
cd /Users/takodamundy/Documents/personal_repos/paramgolf
1. git pull --rebase origin main
2. Re-run literature checks on every world-novel slot (catch new arxiv preprints)
3. Recompute spend ledger from RunPod billing API
4. For each LOCKed layer, run an interaction-screen experiment:
     queue 1 run with all winners from all locked layers stacked together (1 seed)
     if stacked train_loss > sum(individual deltas) + 0.01 → conflict alert
5. For each in-comp.com PR added since last audit, audit our novelty list for new collisions
6. Write a "AUDIT_<UTC>" summary block to Section C
7. git commit STACK_NOVELTY_TRACKER.md RESEARCH_LOG.md
   -m "C180 audit <UTC>: <findings>"
8. git push
```

### C720 — H100 confirm (every 12 h, the spot escalation)

```
cd /Users/takodamundy/Documents/personal_repos/paramgolf
1. git pull --rebase origin main
2. Parse Section A for rows with status=screened-pass and bpb_delta=null
3. If 0 rows → exit
4. If session spend > $25 → exit
5. Pick the top 3 by tl_delta (most promising)
6. runpodctl create pod (H100 spot, with --ports "22/tcp")
7. ssh in, run bootstrap, run S2 confirm for each (10 min each)
8. Pull final_int8_zlib_roundtrip val_bpb from each log
9. Update Section A with bpb_delta + status (confirmed-win or confirmed-fail)
10. runpodctl remove pod <id>  (KILL IMMEDIATELY)
11. git commit STACK_NOVELTY_TRACKER.md -m "C720 H100 <UTC>: <n> confirms"
12. git push
```

### Tracker I/O discipline

Every autonomous Claude session starts with:
```
cd /Users/takodamundy/Documents/personal_repos/paramgolf && git pull --rebase --autostash origin main
```
and ends with:
```
git add STACK_NOVELTY_TRACKER.md RESEARCH_BACKLOG.md [other touched files]
git commit -m "<cron_id> <UTC>: <one-line summary>"
git push
```

Section B is append-only so diffs stay clean. If a merge conflict happens between two cron fires, the later one does `git pull --rebase --autostash` and re-derives Section A from Section B (the ledger is the source of truth).

---

## Branch hygiene + git discipline (preflight in `run_forever.sh`)

New `preflight()` function added to top of `run_forever.sh`, called at top of the while-true loop:

```bash
preflight() {
    BR=$(git -C /workspace/paramgolf branch --show-current)
    if [ "$BR" != "main" ]; then
        echo "PREFLIGHT_FAIL: branch=$BR expected=main" >&2
        git -C /workspace/paramgolf checkout main || exit 3
    fi
    if [ -n "$(git -C /workspace/paramgolf status --porcelain)" ]; then
        echo "PREFLIGHT_WARN: dirty tree, restoring train_gpt.py from backup"
        [ -f /workspace/paramgolf/train_gpt.py.bak ] && \
            cp /workspace/paramgolf/train_gpt.py.bak /workspace/paramgolf/train_gpt.py
    fi
    git -C /workspace/paramgolf pull --rebase --autostash origin main
}
```

This catches the `sota-prikshit-hymba11-muon` bug from last session — pod flips back to main and continues instead of silently running wrong code.

---

## Spend ceiling

| Tier | Action |
|---|---|
| `< $20` | normal |
| `$20 ≤ x < $25` | warn; preemptively kill any pod with zero confirmed-wins this session |
| `$25 ≤ x < $30` | stop queue (commit empty experiments.json), pods idle, Mac+H100 confirms only |
| `$30 ≤ x < $34` | ssh kill `run_forever` on all cheap pods (`crontab -r`); only Pod A + H100 confirms remain |
| `≥ $34` | shutdown all pods except A (holds artifacts) |
| `≥ $36` | hard panic, all pods down, alert |

**H100 discipline:** spot, on-demand, start→run→kill <15 min. Each S2 confirm budgeted $4–6. Max 3 confirms across the session (≈$12–18). Additional confirms require manual go-ahead in Section E.

**Burn rate at 7 cheap pods running** ≈ $1.81/h. With ~$10 to soft-cap from current spend, that's ~5.7 h of full-fleet parallel work — enough for one full 4-batch sweep at 8 min S1 runs.

---

## Sequencing — open-ended parallel search with dependency hints

Per PD1 (no idle pods) and PD2 (find don't just validate), we do NOT use a strict batch schedule. All 7 cheap pods + Mac are working in parallel from minute 1. Layer ownership starts with the day-1 assignment table above, but layers re-balance as wins land:

**Initial assignment (day 1, when bootstrap completes):**
- Pod-A → L04 (attention) — has all instrumentation for attention work
- Pod-B → L08 (optimizer) — optimizer stability work
- Pod-C → L09 (n-gram engine) — biggest leverage layer
- Pod-D → L01 (tokenizer) + L02 (data) — owns tokenizer rebuild infrastructure
- Pod-E → L03 (embedding) — largest mem for factorization tests
- Pod-F → L07 (loss) — loss experiments
- Pod-G → L05 / L06 / stretch options (Hymba, Triton) — floating utility
- Mac → L10 (compression sweeps), candidate research, BPE-8192 builds

**Re-balance trigger (the C60 promote cron):**
- When a layer hits 3 confirmed wins (1 world-novel) → LOCK and reassign that pod to the most under-served layer
- Under-served = fewest confirmed wins + has untried RESEARCH_BACKLOG candidates
- Pods follow opportunity, not a fixed schedule

**Dependency hints (NOT hard sequencing — the system can violate them if a pod would otherwise be idle):**
- L01 tokenizer changes invalidate L09 n-gram tables → Pod-D rebuilds tables for both SP-1024 and BPE-8192 in parallel from day 1, so L09 work is never blocked
- L03 embedding shape depends on L01 vocab → if L01 hasn't locked, Pod-E uses SP-1024 vocab as the working baseline
- L10 compression novelty needs S2 artifacts → pulls from any LOCKed layer's S2 confirm artifact

**Open-ended scale:** the campaign runs until ALL 10 layers are LOCKed (each with 3 wins, 1 world-novel). Total experiment count is **NOT bounded by 78 S1 + 10 S2**. The number depends on how many candidates fail and need replacement from the backlog. Plan for ~120-200 S1 runs and 15-25 S2 confirms across the full campaign.

**Compute estimate:** 7 cheap pods running 24/7 at ~$0.27/h avg = **$45/day fleet cost**. S2 H100 confirms at ~$5/each × 20 = **$100**. **Total campaign cost: ~$145 over 3 days** at full burn. We have $29 remaining of the original $36 → **the campaign must run in spend-controlled mode** with the spend ceiling tiers below.

**Spend-controlled mode (mandatory at our budget):**
- C5 monitor enforces the spend ceiling every 5 min
- At $20: warn, kill any pod with 0 confirmed wins
- At $25: stop queue, only Mac + H100 confirms continue
- At $30: ssh kill all cheap pods, Pod A only
- At $34: ssh shutdown all but Pod A
- At $36: hard panic, all down

**This means the campaign WILL NOT finish in one go.** The realistic plan: spend ~$15-20 on the most leverage-dense layers (L09 n-gram, L01 tokenizer, L08 optimizer) in batch 1, get those LOCKed, then pause and wait for budget refill before continuing. The cron loop is designed to survive long pauses — when the user adds budget, the cron picks up where it left off from the tracker file.

---

## Risks and mitigations

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **Tokenizer rebuild invalidates L09 n-gram tables** — L01 shifts vocab, L09 needs full table rebuild | High | All L09 work invalidated | Lock L01 first (Batch 1). Build L09 tables on Pod-D for *both* SP-1024 and BPE-8192 in parallel from day 1. Don't start L09 novelty work until L01 has a winner. |
| 2 | **World-novel candidates fail audit** — we ship a "novel" patch, find a 2024 paper later | Medium | 1 layer needs replacement novelty | At each layer's first batch, spawn a Mac MLX subagent that does 3 fresh WebSearches + 1 GitHub code search before promoting to S2. Free, uses existing Explore subagent infra. |
| 3 | **Cheap-pod measurement too noisy for 0.012 train_loss differences** | Medium | False negatives kill real wins | n=2 baseline; bump to n=3 for borderline cases. Always confirm baseline unchanged at start of each batch. Verify TRAIN_BATCH_TOKENS=65536 + TRAIN_SEQ_LEN=1024 every cron fire (G1 gate). |
| 4 | **H100 spot price spike or unavailability** | Medium | Can't confirm wins, can't ship submission | Pre-cache one H100 image. Pod-G boot script ready. Fall back to A100 spot (cheaper, same memory). Track spot prices in Section E. |
| 5 | **Stack interactions kill stacked novelties** (LESSONS §3c: ByteWeight + SmearGate combined was WORSE than each alone) | High | Layer winner doesn't compose with other layers' winners | After each batch, run **interaction-screen runs**: pick the new champion from each layer in this batch, run them ALL stacked for 1 seed. If stacked train_loss > sum-of-individual-deltas + 0.01, identify conflict + demote weakest. This is the "stacking discount" check before crowning the campaign champion. C180 audit cron does this. |
| 6 | **Open-ended search never converges** — every candidate fails, RESEARCH_BACKLOG runs dry, no LOCKs land | Medium | Campaign stalls | Three guards: (a) C30 research cron has minimum-floor of 5 candidates per layer, (b) failed candidates feed back into the cross-domain pollination logic (failure-mode → variant addressing it), (c) C5 monitor escalates `MAYDAY_BACKLOG_EMPTY` to human if any layer goes below 1 candidate AND has 0 wins for >12 h. The user can `git push` an injection any time. |
| 7 | **CronCreate sessions miss fires or accumulate cost** | Medium | Campaign drifts off-track or burns cost on dead loops | C5 monitor checks last-fire-timestamp of each cron and re-registers if missed. Each cron has a hard timeout of 5 min (C5/C30) or 15 min (C60/C180/C720). All cron output goes to `RESEARCH_LOG.md` so the user can see what fired. |
| 8 | **PhD-defensibility check is too strict** — every world-novel candidate gets demoted, no layer locks | Low-medium | No mini-papers ship | The PhD check (PD3 + check #5) is intended to filter out "noise wins". If 3+ layers fail to find ANY world-novel after 2 days of campaign, relax the check: drop the "≥6-page workshop paper" bar, keep only "clear hypothesis + clear ablation". The user makes this call manually. |

---

## Critical files

**Existing (reused unchanged or with one-line patches):**
- `/Users/takodamundy/Documents/personal_repos/paramgolf/runpod_tests/loop/experiment_runner.py` — add `pod_filter` filter in `pick_next()` (≈7 lines)
- `/Users/takodamundy/Documents/personal_repos/paramgolf/runpod_tests/loop/run_forever.sh` — add `preflight()` function + Gate G2 background nvidia-smi
- `/Users/takodamundy/Documents/personal_repos/paramgolf/runpod_tests/loop/experiments.json` — every entry gains optional `pod_filter` key
- `/Users/takodamundy/Documents/personal_repos/paramgolf/runpod_tests/chore/08_patch_train_gpt.sh` — append marker integrity check at end
- `/Users/takodamundy/Documents/personal_repos/paramgolf/runpod_tests/chore/04_build_ngrams.py` — Mac uses for skip-bigram + BPE-8192 builds
- `/Users/takodamundy/Documents/personal_repos/paramgolf/train_gpt.py` — patcher target only, never directly edited
- `/Users/takodamundy/Documents/personal_repos/paramgolf/MINIPAPER_TABULATION_HASH.md` — reference for L09 #3 extension
- `/Users/takodamundy/Documents/personal_repos/paramgolf/PODS_SSH.md` — pod connection info (already created)

**New files to create (compaction-resistant state, per PD5):**
- `/Users/takodamundy/Documents/personal_repos/paramgolf/STACK_NOVELTY_TRACKER.md` — tracker schema (six sections), source of truth
- `/Users/takodamundy/Documents/personal_repos/paramgolf/RESEARCH_BACKLOG.md` — candidate pool, ten layer sections, fed by C30 cron
- `/Users/takodamundy/Documents/personal_repos/paramgolf/RESEARCH_LOG.md` — append-only narrative log (already exists from prior sessions; extend)
- `/Users/takodamundy/Documents/personal_repos/paramgolf/runpod_tests/loop/bootstrap_new_pod.sh` — 4-block bootstrap for Pods B–G
- `/Users/takodamundy/Documents/personal_repos/paramgolf/runpod_tests/loop/pull_all_pods.sh` — Mac-side multi-pod pull
- `/Users/takodamundy/Documents/personal_repos/paramgolf/runpod_tests/loop/gate_check.py` — Gates G1–G5 helper, writes Section F
- `/Users/takodamundy/Documents/personal_repos/paramgolf/runpod_tests/loop/pod_id.txt` — per-pod, gitignored
- `/Users/takodamundy/Documents/personal_repos/paramgolf/POD_HOSTS.env` — gitignored ssh aliases

**CronCreate registrations to install (via the CronCreate tool, not file edits):**
- C5 (every 5 min) — monitor heartbeat
- C30 (every 30 min) — research backlog feeder
- C60 (every 1 h) — layer promote
- C180 (every 3 h) — full audit
- C720 (every 12 h) — H100 confirm batch

**New mini-papers (one per world-novel that confirms):**
- `MINIPAPER_ENTROPY_BPE.md` (L01 #3)
- `MINIPAPER_BYTE_ENTROPY_CURRICULUM.md` (L02 #3)
- `MINIPAPER_HADAMARD_TIED_EMBED.md` (L03 #3)
- `MINIPAPER_COPRIME_PER_HEAD_ROPE.md` (L04 #3)
- `MINIPAPER_NORM_PCT_DROPOUT.md` (L05 #3)
- `MINIPAPER_ASYMMETRIC_SKIP_INIT.md` (L06 #3)
- `MINIPAPER_ASYMMETRIC_LABEL_SMOOTH.md` (L07 #3)
- `MINIPAPER_PROJECTION_LR_SPLIT.md` (L08 #3)
- `MINIPAPER_PARTITIONED_TABULATION.md` (L09 #3)
- `MINIPAPER_HESSIAN_RANS_GPTQ.md` (L10 #3)

---

## Verification (end-to-end test plan)

The campaign is **complete** iff ALL of the following hold:

1. **STACK_NOVELTY_TRACKER.md Section D** has 10 LOCK lines (one per layer L01–L10).
2. **Section A** has at least 30 rows in `confirmed-win` status (3 per layer minimum), of which **at least 10 have `world_novel: yes`** (1 per layer minimum).
3. **Each of the 10 world-novel slots** has a corresponding `MINIPAPER_<name>.md` committed at the repo root with: hypothesis, mechanism, ablation results, comparison to baseline, citations.
4. **Section F gates G1–G5** all in `PASS` state on the most recent C5 monitor cron.
5. **Final S2 confirm run** on H100 produces `final_int8_zlib_roundtrip val_bpb ≤ 1.10` (improves on current SOTA 1.11473) AND artifact size in `[16,252,928, 16,777,216]` bytes (G3 PASS — full 16 MB used).
6. **`tokens_per_min` on the final H100 run** ≥ 1.67 M tok/s aggregate (G1 PASS — all training data seen across the 10-min window).
7. **GPU idle streaks** = 0 on the final H100 run (G2 PASS — full 10 min used).
8. **All 7 pods** show non-zero contribution to Section B over the lifetime of the campaign (utilization gate, PD1).
9. **Total campaign spend** ≤ $36 in Section E.
10. **Section B has zero unexplained crashes** that weren't either (a) recovered by watchdog or (b) caught by the 3-strike skip rule.

The campaign is **partially complete** iff fewer than 10 layers are LOCKed but the budget is exhausted. In that state, the user pauses, refills budget, and the cron loop resumes from the partial state on the next fire (compaction-resistant per PD5).

### Day-1 smoke test (run BEFORE fanning out the campaign)

Sequence:
1. Bootstrap **only Pod-B** end-to-end via `bootstrap_new_pod.sh`. Verify Gate G4 reports `MARKERS_PRESENT: 29/29`.
2. Run **one** S1 experiment (e.g. `ATT_gated_head_sigmoid` re-validation under fixed batch).
3. Verify the row appears in Section B within 10 min via `pull_all_pods.sh`.
4. Verify Gate G1 reports `tokens_per_min ≥ 15M` for that run.
5. Verify Gate G2 reports 0 idle streaks.
6. Verify Gate G5 reports queue not empty after the run completes.
7. Manually fire the C5 monitor cron once (`Bash` invocation): verify it correctly reads the tracker, recomputes Section A from Section B, and pushes a clean commit.
8. Manually fire the C30 research cron once: verify it spawns an Explore subagent, returns ≥3 candidates, and appends them to RESEARCH_BACKLOG.md.
9. Only after ALL of 1–8 pass: fan out to Pods C–G in parallel via `bootstrap_new_pod.sh` × 5 in parallel.
10. Install all 5 CronCreate registrations and let the campaign run autonomously.

### Health check (any time during the campaign)

Run on Mac:
```
cd /Users/takodamundy/Documents/personal_repos/paramgolf
git pull --rebase origin main
python3 - <<'PY'
import re, pathlib
t = pathlib.Path('STACK_NOVELTY_TRACKER.md').read_text()
locks = re.findall(r'^- \d{8}T\d{4}Z LOCK (L\d+_\w+)', t, re.M)
wins  = re.findall(r'\| (L\d+_\w+) \| \d+ \| (\w+) \| (yes|no|auditing) \| confirmed-win', t)
print(f"Locked layers: {len(set(locks))}/10")
print(f"Confirmed wins: {len(wins)}")
print(f"World-novel wins: {sum(1 for w in wins if w[2]=='yes')}/10")
PY
```

If this prints `Locked: 10/10 | Wins: ≥30 | World-novel: ≥10`, the campaign is verified complete.

---

## Day-1 execution checklist (the order of operations after plan approval)

Step-by-step, what gets done in the FIRST 2 hours after the user approves this plan:

1. **Push the plan** — copy `/Users/takodamundy/.claude-personal/plans/refactored-splashing-crayon.md` into the repo as `STACK_NOVELTY_PLAN.md`, commit, push.
2. **Create the three tracking files** at repo root:
   - `STACK_NOVELTY_TRACKER.md` — six empty sections with the schema headers from above
   - `RESEARCH_BACKLOG.md` — ten layer sections, each pre-populated with the 3 seed novelties from the per-layer roadmap
   - (`RESEARCH_LOG.md` already exists — append a "Campaign start" block)
3. **Bootstrap Pod-B** end-to-end via `bootstrap_new_pod.sh` (the smoke test).
4. **Run the day-1 smoke test** (steps 2–8 from the Verification section).
5. **Bootstrap Pods C–G in parallel** (5 parallel `ssh` calls to the new pods) once Pod-B passes.
6. **Install the 5 CronCreate registrations** (C5, C30, C60, C180, C720) via the `CronCreate` tool. Each cron's payload is the matching prompt from the "CronCreate registrations" section above.
7. **Manually fire C5 once** to verify the heartbeat works end-to-end.
8. **Manually fire C30 once** to seed the candidate pool with literature mining.
9. **Pre-fill the experiments.json queue** with 14 experiments (2 per pod × 7 pods) so PD1 holds from minute 1.
10. **Commit + push** the initial state.
11. **Walk away** — the cron loop now runs the campaign autonomously; the user only needs to check in periodically and review locked layers.

After this checklist completes, the campaign is live and self-driving.

---

## In scope (explicitly NOT deferred — per PD4 "anything is in scope")

- **Hymba (Mamba+Attention hybrid)** — IN scope as L04 stretch S1. Requires `mamba-ssm` + `causal-conv1d` install; documented in LESSONS §28 + PR #852. If install on Pod-G works, Hymba is on the table.
- **Pure Mamba-2 layers** — IN scope as L04 stretch S2.
- **Custom CUDA kernels** — IN scope as L10 stretch S1 (int6 GPTQ dequant fusion) and L04 stretch S3 (Triton attention kernel).
- **Custom Triton kernels** for n-gram bias gather — IN scope as L10 stretch S2.
- **Custom SentencePiece fork** for deterministic merge ordering — IN scope as L10 stretch S4.
- **Custom Brotli pre-trained dictionaries** — IN scope as L10 stretch S3.
- **Differential transformer attention** — IN scope as L04 stretch S4.
- **PyTorch internals modifications** — allowed if they unlock a real win.

## Out of scope (deliberately deferred)

- **PR #1430 watch** — likely illegal under issue #677. Do not port. C180 audit fires keep watching until merged-and-not-reverted.
- **Multi-track submission** — only `track_10min_16mb`. No `track_60min_16mb` work.
- **Reverse-engineering competitors' artifacts** — out of scope; we only read public PRs and merged records.
