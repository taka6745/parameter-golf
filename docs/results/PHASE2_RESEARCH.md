# PHASE2_RESEARCH.md — Big Dreams for the Speed Path

**Written 2026-04-09 after 2 parallel research agents (comp PR audit + world-literature moonshots).**

**Thesis**: the speed path is the most underutilized section of openai/parameter-golf. The quality path has 170+ PRs, ~10 open world-novel claims, and diminishing returns (comp SOTA has moved 0.014 BPB in the last 2 weeks). The **speed path has maybe 30 PRs, 2-3 genuine novelties, and 10×+ room to grow**. Our current 1×H100 SXM rate is 0.31 steps/sec/GPU vs the comp's 4.17 = **13× slower per GPU**. The headroom is stupid.

**This file sets the ceiling, not the floor.** It lists every world-novel thing we could try. Shot prioritization is in PHASE2_PLAN.md.

---

## The 13× gap decomposed

| Cost term | Our penalty | Mitigation | Source |
|---|---|---|---|
| torch.compile disabled | 3-5× | Shot 1: re-enable + AOTInductor precompile | Phase 2 plan |
| FA3 not installed (SDPA fallback) | 1.3× | Shot 2: wheel sourcing or FA2 fallback | Phase 2 plan |
| `grad_accum_steps=8` for world_size=1 | **1.5-2×** | **Shot 0: drop to 1** — 5 LOC, free, untouched | world research §7A |
| Eval at stride=64, batch=1 | eats 3-5 min of 600s budget | **Shot 0b: batched + streaming KV** | world research §8A, §10A |
| Kernel launch overhead (~600/step × 1.3μs) | 16% of step budget | **Shot 14: training megakernel** (world-first) | world research §6A |
| Muon Newton-Schulz serial | 15× bigger than it should be | Shot 10: Parameter Banking (PR #399) | comp audit §10 |
| N-gram bias 9 kernel launches/layer | ~10% | Shot 4 (already planned) + ThunderKittens block fusion | both reports |
| **Per-GPU compute (residual)** | ~1.3× | — | fundamental |

**If we land even HALF of the mitigations, the 13× gap collapses to ~2× (which is just the per-GPU hardware ratio vs 8×GPU comp records).** The val_bpb improvement follows directly: 5× more steps → train_loss drops 0.5-1.0 → val_bpb drops 0.2-0.4.

---

## Free Wins (Shot 0 — do FIRST, no kernel work)

These are 4-8 hours of dev total for a combined **~3× speedup** and $0 risk.

### Shot 0a — Drop `grad_accum_steps` 8 → 1 (4h, 5 LOC)
- Current: 384 seqs × 2048 tokens / 8 = 48 seqs per microbatch, 8× kernel launches per step
- On H100 80GB, the full 384-seq batch (~786K tokens) fits easily — model is 35.9M params
- **We are paying 8× kernel-launch overhead for literally no reason.** This was inherited from an 8×GPU distributed config where grad_accum made sense.
- **Expected 30-50% per-step speedup, 5 LOC change, zero risk, zero code review required.**
- Source: world research §7A

### Shot 0b — Eval batched + streaming KV cache (1d, 250 LOC)
- Current sliding-window eval: 625K sequential forward passes at B=1, stride=64 → 10-15 min on H100 SXM
- Each window shares 1984/2048 = **97% of its context with the previous window** — we recompute the same attention 32× per token
- **Fix**: batch 32-64 windows per forward, carry the KV cache across windows (StreamingLLM arXiv 2309.17453 pattern)
- **Expected: 5-15× eval speedup, saves 3-5 min of the 600s budget → directly becomes more training steps**
- World-novel for our shape — no comp PR does this (they use the correct-but-slow block-stride)
- Source: world research §8A, §10A

### Shot 0c — SkyLadder progressive seq_len 256→2048 (1d, 80 LOC)
- Start training at `train_seq_len=256`, double every ~100 steps to 2048
- SkyLadder (NeurIPS 2025, arXiv 2503.15450) reports **22% faster + 1-3.7% benchmark gain**
- Already in Mac SETUP.md §35 backlog — we planned this before and never shipped
- Also enables **sequence packing with block-diagonal attention mask** (FA3 native support) for +5-15% on top
- Source: world research §3A, §12G

**Shot 0 total expected: ~3× speedup with 2-3 days of work and no custom kernels.** This alone converts 180 steps → ~540 steps in 600s, dropping val_bpb meaningfully.

---

## Comp-Port Wins (Shot 9-15 additions from comp audit)

The comp PR audit found 6 big techniques our Phase 2 plan missed. Each is a validated comp-port (verified n=1 or n=3 in a comp PR) with known speedup numbers.

### Shot 9 — FA3 varlen + window attention + mixed seq_len (from PRs #1354, #1212)
- **PR #1212 holds the fastest step in the leaderboard at 69.6 ms/step** using `flash_attn_varlen_func` + `window_size=512` on alternating layers + mixed seq_len ACROSS GPUs in the same training step
- **Mixed seq_len across GPUs in same step** is world-novel — every other paper assumes uniform seq_len
- Document-isolated TTT only legal because varlen training (co-designed legality by @samacqua)
- Effort: high (requires rebuilding the data loader + FA3 varlen API)
- Expected: matches PR #1212's record pace if we pull it off

### Shot 10 — Parameter Banking + Parallel Muon (PR #399 by @abaybektursun)
- Restructures 66 nn.Linear into 4 contiguous 3D banks → Newton-Schulz becomes one `torch.bmm` → **optimizer time 19.7 ms → 1.3 ms (15×!)**
- DDP-free async reduce_scatter / all_gather
- Architecture-agnostic — works with our 10-patch stack unchanged
- **World-novel**: NOT in modded-nanogpt
- Effort: medium, ~200 LOC

### Shot 11 — CUTLASS EVT backward (PRs #1105, #1420 by @abaybektursun)
- **Branch-free backward using the precomputed-act-grad identity `post = 0.5 · act_grad · pre`** — this identity itself looks world-novel, NOT in standard CUTLASS examples
- Saves 1.6 ms/step on top of the Shot 4 Triton TMA forward
- Requires sm_90a header build (Hopper-only)
- Pair with Shot 4 in one commit

### Shot 12 — Async prefetch + pinned + dedicated copy stream + memmap multi-shard loader (PRs #591, #726, #1420)
- Daemon thread builds CPU batches into a `queue.Queue(maxsize=2)` while GPU runs current step
- Double-buffered val H2D + `VAL_BYTECOUNT_DEVICE=cpu` to move BPB byte counting off GPU
- Memmap multi-shard with merged slab copies (critical for cloud virtio disks)
- Coprime-stride sampling across shards
- **Critical at 1×H100 because our compute/IO ratio favors this more than at 8×H100**
- Effort: medium, ~150 LOC

### Free Inductor patch
- Set `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` and/or pin torch 2.9.1+cu128 — worth **+5.93 ms/step (+8.8%) = +57 training steps in 600s**
- PR #1420's author landed two upstream PyTorch patches (pytorch#179494, #179422) to fix Inductor regressions specifically caused by this competition
- **This is FREE.** One env var. Ship immediately after Shot 1.

### Shot 13 — Triton KV-cache eval backend (PRs #1149, #1153)
- **2.6-2.7× faster int8/qjl score+apply with prewarm**
- Our 10-15 min sliding-window eval is the hidden bottleneck
- Composes with Shot 0b (batched + streaming KV) for a multiplicative eval win

### Shot 14 — Fused Softcap+CE megakernel for eval (PR #915)
- **1.94× vs torch.compile, 7.51× vs eager**
- Eliminates the B×T×V fp32 capped-logits intermediate that Inductor cannot kill
- Eval-only kernel via `torch.utils.cpp_extension.load_inline`
- Effort: low, ~100 LOC

### Shot 15 — Train-data GPTQ calibration (PR #1219, comp-organizer-approved)
- Replaces 220s AR self-gen calibration with 14s training-data Hessian
- **+200s of training budget = +2000 extra steps at our current rate = +6000 steps at the Shot 0 rate**
- Massive reallocation. Probably the biggest single "find" in the comp speed path.
- Effort: low, ~30 LOC

---

## The World-First Opportunities (big dreams)

These are the items that don't exist in ANY comp PR and don't exist in the open literature for our specific shape. Each is a potential mini-paper.

### 🏆 Megadream 1 — Training Megakernel (5-7 days, 500-1500 LOC)

**Current state**: HazyResearch / Mirage / MegaQwen have shipped persistent-SM-scheduler megakernels for **inference**. Nobody has built one for **fwd + bwd + optimizer training of a small-batch transformer**.

**The math**: 1.3 μs per kernel launch × ~600 launches per training step = **~800 μs of pure launch overhead per step = 16% of our step budget**. At 5000 steps in 600s (post-Shot 0), that's 800 seconds of pure overhead we could reclaim.

**The construction**:
1. Single CUDA kernel that holds all 11 transformer layers + optimizer state in shared memory
2. Persistent threadblocks that iterate the layer stack without returning to host
3. Backward pass woven into the same kernel via a second phase
4. Optimizer step (Muon Newton-Schulz + AdamW) in the same launch
5. Starts from ThunderKittens templates (HazyResearch TK kernel DSL for H100)

**Risks**:
- Backward sync is the unknown — no public megakernel does training
- Custom numerics may drift val_bpb by >ε (would need ε-tolerance verification)
- Debugging is brutal — single kernel = single-stack traceback

**Upside**: **1.5-2.5× on top of Shot 3 (CUDAGraph)**, world-first for training, publishable mini-paper, moonlights as a PhD-defensible research contribution.

**Verdict**: ATTEMPT this if Shots 0-8 land cleanly and we have 5+ days to spare. It's the highest-ceiling item in the entire Phase 2 plan.

### 🏆 Megadream 2 — Streaming-KV sliding-window eval (1-2 days, 250 LOC)

(Already in Shot 0b above but worth re-emphasizing.)

**Why it's world-novel**: every public sliding-window perplexity implementation does either (a) recompute per window or (b) block-stride. Nobody does **streaming KV cache over overlapping windows where each new window appends 64 tokens to a persistent cache and scores only the tail**.

**The construction**:
1. Keep a persistent KV cache of length 2048
2. Each "window" appends 64 new tokens + evicts the oldest 64
3. Score only the 64 new tokens
4. Do this for 625K windows in a single long loop with batched forwards

**Upside**: **5-15× eval speedup.** Eval phase currently eats 20-25% of the 600s budget. Recovering that buys direct more training steps.

**Risk**: attention mask edge cases at the seam between windows. Verifiable by running both eager and streaming versions on the same val data and comparing val_bpb to 6 decimal places.

### 🏆 Megadream 3 — Fuzzy LR Bandit per microbatch (4h, 80 LOC)

**The user's "dial-in" hint operationalized.**

**Construction**:
1. At each microbatch, sample LR from `{0.5×, 1×, 2×} × base_lr` with Thompson sampling weights
2. Track loss-delta per sampled LR
3. Online bandit (UCB or Thompson) picks the next sample weight distribution
4. Converges to the locally-optimal LR schedule without requiring an offline LR search

**Why it's novel**: no comp PR does per-microbatch LR sampling; LR schedules are pre-computed. Related: Hyperband, successive halving. This extends them to the finest granularity.

**Expected**: 1-2% train loss improvement at the same wallclock. Small in isolation, meaningful stacked with other wins.

**Effort**: trivial. 80 LOC.

### 🏆 Megadream 4 — CPU n-gram precompute thread (4h, 50 LOC)

**The user's "precompute things with CPU while GPU runs" hint operationalized.**

**Construction**:
1. Background thread on the CPU that, for each upcoming batch, pre-computes the 3 n-gram hash tensors (bigram, trigram, fourgram) and the bias values
2. These get written into a shared GPU tensor pool via `cuda.Stream` + pinned memory
3. The GPU forward pass gathers from the pool instead of computing the hashes inline
4. Saves 3 kernel launches per forward = ~100 μs per step × ~5000 steps = 500 ms of wallclock

**Why it's novel**: N-gram bias precompute has never been overlapped with GPU training in any comp PR. Small-LM-specific because the n-gram tables fit the H100 bandwidth budget.

**Expected**: 3-7% per-step speedup. Composes with the fused n-gram kernel (Shot 4) as an alternative path.

### 🏆 Megadream 5 — GPU-resident successive halving (200 LOC)

**The user's "like GPU tests" hint operationalized.**

**Construction**:
1. Inside the 600s budget, run 4 model replicas with different LR/momentum/wd configs for the first 100 steps
2. Pick the winning config (lowest loss at step 100)
3. Continue training the winner for the remaining 5000 steps
4. The "loss" of the dropped replicas is mostly shared compute (same forward passes, different optimizers)
5. Cost: ~5% of training budget for configuration search

**Why it's novel**: nobody does GPU-resident hyperparam search inside a fixed wallclock training run. Hyperband is offline; this is online.

**Expected**: 1-3% val_bpb improvement from a smarter LR choice. Bigger if the default LR is badly tuned for our specific patch stack.

**Risk**: CUDAGraph re-capture when switching configs. Solvable.

### 🏆 Megadream 6 — AOTInductor precompile + binary ship (1 day, 100 LOC)

**Kill the torch.compile cold-start permanently.**

**Construction**:
1. Compile ONCE on a cheap pod: `torch._inductor.aoti_compile_and_package(model, example_inputs) → .so`
2. Ship the `.so` file as part of the submission artifact (or download from blob storage)
3. Load the compiled `.so` on the real pod via `torch._export.aot_load(...)` — zero compile time
4. ABI-stable across PyTorch minor versions

**Why it's a dream**: removes the 5+ min compile penalty that made us disable compile in Phase 1 in the first place. Cache-portable across pods of the same SM architecture (so compile on a 3090-sm86 for 3090 targets; compile on an H100-sm90 for H100 targets).

**Source**: PyTorch AOTInductor, ezyang Aug 2025 blog post.

---

## Extreme Moonshots (may or may not work)

These are the "if we had infinite time" ideas. Listed for completeness.

1. **Two-stage train**: 300s cheap surrogate (fp8 + smaller hidden) + 300s full model warm-started from surrogate. TinyBERT-style distillation. High risk.
2. **Learned LR prediction network**: a tiny policy network that predicts the next LR from current loss + gradient norms. Requires pre-training the policy. Probably not worth it.
3. **JIT 100 steps as one CUDAGraph + replay**: 5-10% on top of Shot 3. Easy, small upside.
4. **Surrogate teacher with logits cached on disk**: pre-generate teacher logits for the first N training batches, use them as targets instead of running a teacher forward pass. Requires a teacher model.
5. **FP4 inference paths for eval**: Hopper has FP8; Blackwell has FP4. Not on our hardware.
6. **Eidetic training**: each training step stores every gradient; at the end pick the Pareto-optimal subset via L2 projection. Too memory-hungry.
7. **Loss-function surgery**: replace cross-entropy with Kraft-inequality Huffman loss during late training (matches the brotli-compressed submission path). Speculative but philosophically aligned with the compression objective.

---

## Revised Phase 2 shot ordering (post-research)

**Tier 0 — Free wins (do BEFORE any kernel work)**

| Shot | Technique | Effort | Speedup |
|---|---|---|---|
| **0a** | Drop `grad_accum_steps 8 → 1` | 4 h / 5 LOC | 1.3-1.5× |
| **0b** | Eval batched + streaming KV | 1 d / 250 LOC | 5-15× eval |
| **0c** | SkyLadder progressive seq_len 256→2048 | 1 d / 80 LOC | 1.22× + quality |
| **Free** | `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` + torch 2.9.1 pin | 5 min | 1.09× |
| **0d** | Shot 15: train-data GPTQ calibration | 30 LOC | +200s budget |

**Tier 1 — Kernel work (existing Phase 2 plan)**

| Shot | Technique | Effort | Speedup |
|---|---|---|---|
| 1 | torch.compile re-enable + AOTInductor precompile | 1-2 d | 3-5× |
| 2 | FA3 sourcing (wheel / source / FA2 fallback) | 1 d | 1.3× |
| 3 | CUDAGraph capture | 2 d | 1.5-2× |

**Tier 2 — Comp-port wins (new Shots 9-15)**

| Shot | Technique | Effort | Speedup |
|---|---|---|---|
| 9 | FA3 varlen + window + mixed seq_len (PR #1212) | 2-3 d | 1.5-2× |
| 10 | Parameter Banking + Parallel Muon (PR #399) | 1-2 d | Muon 15× → step ~1.2× |
| 11 | CUTLASS EVT backward (PRs #1105, #1420) | 2 d | 1.1× |
| 12 | Async prefetch + dedicated copy stream (PRs #591, #726) | 1 d | 1.05-1.15× |
| 13 | Triton KV-cache eval backend (PRs #1149, #1153) | 1 d | eval 2.6× |
| 14 | Fused Softcap+CE megakernel (PR #915) | 1 d | eval 1.94× |

**Tier 3 — Big dreams / world-firsts**

| Shot | Technique | Effort | Speedup |
|---|---|---|---|
| 16 | Flash-Muon + Gram Newton-Schulz | 1 d | Muon 25-50% |
| 17 | Fuzzy LR bandit per microbatch (world-novel) | 4 h | quality 1-2% |
| 18 | CPU n-gram precompute thread (world-novel) | 4 h | 3-7% step |
| 19 | GPU-resident successive halving (world-novel) | 2-3 d | quality 1-3% |
| 20 | **Training megakernel (Mirage/TK, world-first)** | 5-7 d | 1.5-2.5× on top of 3 |

**Tier 4 — Moonshots (only if everything else lands)**

| Shot | Technique | Effort | Speedup |
|---|---|---|---|
| 21 | Two-stage train (surrogate + full) | 3-5 d | speculative |
| 22 | Sequence packing block-diag mask + VSL bucketing | 2 d | 1.05-1.15× |
| 23 | Predictive batch scheduling | 1-2 d | 1.06-1.13× |
| 24 | ZenFlow stall-free CPU optimizer offload | 2 d | 1.05-1.15× |

---

## Stacked expected impact (if everything works)

| After stage | Approx steps in 600s | Approx val_bpb | Gap to comp |
|---|---|---|---|
| Phase 1 baseline (current) | 180 | ~1.4-1.6 (undertrained EMA) | +0.3-0.5 |
| + Tier 0 (free wins) | **~540** | ~1.25-1.35 | +0.17-0.27 |
| + Tier 1 (compile + FA3 + CUDAGraph) | ~2000 | ~1.15-1.22 | +0.07-0.14 |
| + Tier 2 (comp-port wins) | ~4000 | ~1.10-1.15 | +0.02-0.07 |
| + Tier 3 Megadream 1 (training megakernel) | **~8000** | ~1.08-1.12 | **matches comp 1×H100 ratio** |
| + Tier 3 all | ~10000 | ~1.06-1.10 | **AHEAD of comp on 1×H100** |

**At 10000+ steps we're past the comp record's training budget (comp = 20000 steps on 8×H100 = 2500 steps/GPU)**. If we can hit 10000 steps on 1×H100, we're doing 4× more per-GPU training than the comp record. That's where the val_bpb can actually DROP BELOW comp.

---

## What to do RIGHT NOW

1. **Let Pod L finish the Phase 1 dry run** (in flight, ~03:30 UTC). Get a baseline val_bpb number so we know what we're improving from.
2. **Spin a cheap 3090 pod** ($0.22/h) for Phase 2 Tier 0 work.
3. **Ship Tier 0 shots in one day** — all 5 items are low-LOC, low-risk. That alone should give ~3× speedup.
4. **Measure each Tier 0 shot's val_bpb delta vs Phase 1** to verify the invariant (ε ≤ 0.005).
5. **Decide: Tier 1-2 comp ports next, or go straight for Megadream 1 (training megakernel)**. The latter is the ambitious path; the former is the safe path.

---

**Source reports**:
- `/tmp/phase2_comp_speed_audit.md` — comp PR audit (22 PRs surveyed, ~2000 words)
- `/tmp/phase2_world_speed_research.md` — world-literature research (~2500 words)
- Agent continuations available via SendMessage IDs `a03f17fe154b5a74f` and `a21d2ab968e16510a` for follow-up questions.

**The single most surprising finding**: the comp PR audit says **the eval path holds the biggest speed wins currently, not training**. Our 10-15 min sliding-window eval is the hidden bottleneck. Tier 0b (batched + streaming KV) + Tier 2 Shot 13 (Triton KV-cache eval) + Tier 2 Shot 14 (fused softcap+CE megakernel) together save 5-8 min per eval pass, which is more wallclock than any single training-side patch in the comp would buy us at the current rate.

**The second most surprising finding**: we can probably drop `grad_accum_steps=8 → 1` and get 30-50% speedup for 5 lines of code. That's Shot 0a. Do it before anything else.
