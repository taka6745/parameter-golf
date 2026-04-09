# Phase 2 Plan — Speed work (SAME model, faster execution)

Written 2026-04-09 after Phase 1 dry run identified the speed problem. Owner: Claude + Takoda.

## 🚨 URGENT: Shot 0e — Fix NGR_LOG_FREQ_INV serialization bug (BEFORE any speed work)

**Discovered 2026-04-09 from Pod L Phase 1 dry run**. This is a **CORRECTNESS bug**, not a speed bug, but it must land before Phase 2 speed work because the speed work's val_bpb verification invariant (ε ≤ 0.005 drift) can't be measured against a broken baseline.

**Symptom**:
```
post-prequant-ttt   val_bpb: 1.24108   ← model looks GREAT unquantized
quantized           val_bpb: 3.86174   ← quantized is DESTROYED (-2.62 BPB gap!)
```

Normal GPTQ int6 quant gap is 0.005-0.02 BPB. We're seeing **2.62 BPB**. Something is wrong.

**Root cause**: The n-gram table buffers in `GPT.__init__` are registered with `persistent=False` so they don't bloat the submission state_dict. But **NGR_LOG_FREQ_INV mutates them in place on first forward** (multiplies every bucket value by `1/log(2+count)`). The flow breaks on serialize → deserialize:

1. Training runs → `_bigram_tab`, `_trigram_tab`, `_fourgram_tab` get MUTATED in place
2. `serialize()` saves `base_model.state_dict()` → **mutated tables NOT included** (persistent=False)
3. `deserialize()` creates a fresh `GPT(h)` → `__init__` reloads tables from disk → **UNMUTATED**
4. Eval runs with the original-freq tables, but the model was trained expecting the mutated-freq ones → **massive mismatch**

CMP_QUANT_VALUE_DEDUP probably adds a small amount too, but the primary culprit is NGR_LOG_FREQ_INV losing its state on save.

**Fix options (ranked)**:

1. **Option A — save multipliers separately, re-apply on load** (best, ~30 LOC, world-novel claim survives):
   - Compute the multipliers once on first forward (current behavior)
   - Also store them in `self._nlfi_bigram_mult`, `_nlfi_trigram_mult`, `_nlfi_fourgram_mult` as `persistent=True` small tensors (shape `[16384]` fp32 = 64 KB each, ~200 KB total, fits the 16 MB cap easily)
   - On `deserialize()`, the `__init__` reloads the tables fresh, THEN the state_dict load restores the multipliers, THEN a new helper method `_nlfi_reapply_if_needed()` re-multiplies the tables using the saved multipliers
   - Total submission size impact: +200 KB. Still well under 16 MB cap.
   - The world-novel L09 claim survives: we still do inverse-log-frequency bucket suppression, just with the multipliers persisted so quantized eval matches trained eval.

2. **Option B — disable NGR_LOG_FREQ_INV for the submission path** (~1 LOC):
   - Set `USE_NGR_LOG_FREQ_INV=0` in `submission/run.sh` by default
   - Keep the code in place so the patch can be re-enabled in research runs
   - LOSES the world-novel claim but guarantees quant gap recovers

3. **Option C — disable both NGR_LOG_FREQ_INV AND CMP_QUANT_VALUE_DEDUP**, re-run, confirm baseline (~2 LOC):
   - Pure diagnostic run to prove the two patches are the culprits
   - Expected: quant gap drops to ~0.05 BPB, val_bpb lands around 1.30 (comp-parity region)
   - USE FIRST as a diagnostic, THEN pick A or B for the real fix

**Implementation plan**:

1. First: Option C diagnostic run (2 env vars flipped, re-run on a fresh cheap pod, measure the quant gap)
2. If C confirms: Option A (save multipliers, ~30 LOC) — keeps the world-novel claim
3. If A is too risky under time pressure: Option B fallback

**This shot blocks**: all downstream Phase 2 speed work, because the val_bpb invariant (ε ≤ 0.005 drift vs Phase 1 baseline) can't be measured against a 3.86 baseline. We need a clean baseline first.

---

## TL;DR

- **Same model as Phase 1.** Architecture, hyperparams, env vars, all 10 patches — unchanged.
- **Different implementation**: torch.compile, FA3, custom kernels, CUDAGraph. Same math, faster execution.
- **The win mechanism**: speedup → more training steps in the comp's 600s wallclock budget → lower train_loss → lower val_bpb. Phase 1 hit 180 steps; with 5× speedup we'd hit ~900, with 15× → ~2700.
- **Hardware**: cheap 3090/4070 Ti pods, NOT H100. The H100 rule resumes after Phase 1 ends.
- **Gate**: Phase 2 can start once **Shot 0e** (NGR_LOG_FREQ_INV serialization fix) has landed AND Phase 1 has produced a clean baseline val_bpb (≤ ~1.30 expected after the fix).

---

## Phase 1 baseline numbers (the floor we're improving)

Phase 1 dry run on 1×H100 SXM (Pod L `55fzwdfhbg9n4u`) with the 10-patch differentiated stack:

- **Hardware**: 1× H100 SXM 80GB HBM3 ($2.99/h)
- **Tokens/sec at the start**: ~285,000 (slowed to ~245,000 by step 180 due to memory pressure)
- **Steps in 600s wallclock**: ~180 (vs comp records' 20,000 on 8×H100)
- **Per-GPU rate**: 0.31 steps/sec (vs comp's 4.17 steps/sec/GPU = ~13× slower)
- **Final train_loss at step 180**: ~3.90
- **Final val_bpb**: TBD (waiting on the dry run to land)

The 13× per-GPU slowdown breaks down approximately as:
1. **No torch.compile** — ~3-5× per-step penalty (compile disabled to avoid the 5+ min first-run cost)
2. **No FA3** — ~30-50% penalty (SDPA fallback because flash-attn-3 wheel isn't on PyPI and Pod L's image didn't pre-install it)
3. **N-gram bias forward overhead** — ~5-10% per step (3 hash lookups + 3 gathers + mix per forward)
4. **3-layer recurrence (+13% layers)** — encoder/decoder go from 15 to 17 logical layers (loop on [3,4,5] × 3 vs [4,5] × 3)
5. **Bf16 SDPA inefficiency on our shape** — small model on a big GPU, kernel launch overhead dominates

## Goal

Get the SAME model trained for ~5-15× more wallclock-effective steps within the comp's 600s budget, **without changing any hyperparam or any patch**. Submit-ready stack with submit-ready throughput.

**Success criterion**: val_bpb drops to within **0.05-0.10 BPB of the comp records (PR #1477 = 1.0822)** when run on the same 1×H100 SXM hardware. We won't match the 8×H100 records on a 1×GPU pod, but we should get into the 1.10-1.18 region with the speed work.

**Stretch criterion**: ms/step matches PR #1485's measured speed when normalized for hardware (their 8×H100 records work out to ~250ms/step; we should hit ~250-400ms/step on 1× SXM for the same code).

---

## Hardware (cheap pods only)

Per the original PHASE1_PLAN.md narrow rule, **the H100 ban resumes after Phase 1 ends**. Phase 2 runs on cheap GPUs:

- **Primary**: 1× RTX 4070 Ti (~$0.30/h) for the speedup measurements. Hopper-class fast path, sub-$0.50 hourly burn. Best for quick iteration.
- **Secondary**: 1× RTX 3090 (~$0.22/h) for compile-cache builds and overnight kernel compiles.

**No H100 unless**:
1. We need to validate the FINAL submission stack in the actual submission hardware path (1× H100 SXM, ~$3/h, ONE-shot validation only)
2. User explicitly approves an H100 burn

**Total Phase 2 budget**: $5-10 (vs Phase 1's $5 burn). Most of Phase 2 is dev work + small validation runs.

---

## Shot sequence (priority order, each gates the next)

### Shot 1 — torch.compile re-enable (~30 min, $0.20)

**Goal**: prove that re-enabling `torch.compile` doesn't break our 10-patch stack and measure the per-step speedup vs eager mode.

**The current state**: `submission/run.sh` sets `TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1` to skip the 5+ min first-run inductor compile. This was a Phase 1 hack to get a number out fast.

**Code changes**:
- New env var `WARM_COMPILE_CACHE=1` triggers a SHORT warm-up run that populates the inductor compile cache (uses fewer iterations + a tiny dataset slice). Then the real run reuses the cache.
- Update `submission/run.sh` to default `TORCH_COMPILE_DISABLE=0` and run the warm-up run automatically on the first invocation per pod.
- The torch.compile cache lives at `~/.cache/torch/inductor` — bring it across runs.
- Possibly add `torch._inductor.config.fx_graph_remote_cache = True` to share the cache across pods via blob storage (stretch).

**Risks**:
- 10-patch stack may not all compile cleanly. n-gram bias gather, gated_attention sigmoid, NORM_PCT_DROPOUT quantile — these are unusual ops that may fall out of dynamo and recompile per shape.
- `fullgraph=True` will fail loudly if any patch breaks the graph. We'll have to selectively `dynamic=False` or `mode="reduce-overhead"` where needed.

**Success criterion**: per-step time drops by ≥3× vs the Phase 1 dry run baseline (currently ~3.3 sec/step → target ≤1.1 sec/step).

**Stop condition**: if the compile crashes mid-run on any patch we own, disable compile for that specific code path via `@torch._dynamo.disable` and continue. Don't let compile failures block Phase 2.

### Shot 2 — FA3 sourcing (~60 min, $0.30)

**Goal**: get the flash-attn-3 fast path working so we drop the SDPA fallback.

**Three options in priority order**:
1. **Build from source**: clone Dao-AILab/flash-attention, build the FA3 wheel against torch 2.9.1+cu128. ~20 min compile time on a cheap pod with fast CPU. The build dependencies (CUDA toolkit, ninja) need to be installed.
2. **Find the private wheel**: the runpod/pytorch:2.4.0 image SOMETIMES pre-installs `flash_attn_3-3.0.0+20260303.cu128torch291cxx11abitrue.ceb109` as a private wheel (Pod K had it, Pod L didn't). Find what determines this and either use the right image variant or find the wheel URL and pip-install directly.
3. **Use flash-attn 2 instead**: the older Flash Attention 2 IS on PyPI as `flash-attn`. It supports H100 (and has a Hopper-specific kernel since version 2.5+). Speed is between SDPA and FA3.

**Code changes**: probably none — `train.py` already imports `flash_attn_interface` with try/except. If we get FA3 working, the import succeeds and the fast path activates. If we go with FA2, we add a second try/except above the FA3 one for `flash_attn` (the FA2 module name) and call it via the FA2 API.

**Success criterion**: per-step time drops another 30-50% on top of Shot 1's compile speedup. Combined Shot 1 + Shot 2 should give ≥5× total speedup vs Phase 1 baseline.

**Stop condition**: if the build fails or the wheel isn't sourceable, fall back to flash-attn 2. Don't sink more than 1 hour on FA3-specific issues.

### Shot 3 — Persistent CUDAGraph capture (~90 min, $0.40)

**Goal**: eliminate per-iteration kernel launch overhead by capturing the entire forward+backward+optimizer step as a single CUDAGraph and replaying it each iteration.

**Approach**: PyTorch's `torch.cuda.graph` API. Capture a single iteration after warmup, replay it for all subsequent iterations. Requires:
- Static input shapes (we have this — `train_seq_len=2048` is fixed)
- No host synchronization mid-iteration (need to audit our patches — Pre-Quant TTT, NORM_PCT_DROPOUT quantile, NGR_LOG_FREQ_INV in-place mutation may all sync)
- Memory pre-allocation for all activations

**Code changes**:
- New helper `make_train_step_graph(model, optimizers, sample_x, sample_y)` that captures one step then returns a callable
- Replace the inner `for step in ...` loop with `train_step_graph()` calls
- NORM_PCT_DROPOUT uses `torch.quantile` which has a host sync — either pre-compute the threshold off-graph or replace with `torch.kthvalue`
- NGR_LOG_FREQ_INV's lazy one-time table mutation breaks graph capture — move the mutation OUT of the forward (do it explicitly before the graph capture)
- Pre-Quant TTT runs after main training so it's outside the graph — fine

**Risks**: Cudagraph capture is finicky. The 10-patch stack has lots of dynamic-looking ops (quantile, scatter_add for NLFI, etc.) that may break capture.

**Success criterion**: another 1.5-2× speedup on top of Shots 1+2. Total ≥8× vs Phase 1 baseline.

### Shot 4 — Fused n-gram bias gather + attention Triton kernel (~3-4 h, $1.00)

**Goal**: write a custom Triton kernel that fuses the n-gram bias hash + gather + bias-add directly into the final logits projection. Eliminates 4 small kernel launches per layer per forward.

**Why**: the current ngram bias path materializes 3 hash tensors (B*S each), 3 gather outputs (B*S*V each = ~64 MB at our shape), then a sum. That's ~250 MB of intermediate tensors per forward, all from kernels too small to saturate the H100. A fused kernel computes the bias inline with the matmul.

**Code changes**:
- New file `submission/kernels/ngram_bias_kernel.py` with a Triton implementation
- Modified `forward_logits` to call the fused kernel when `USE_NGRAM_BIAS_FUSED=1`
- Keep the eager fallback for debugging

**Success criterion**: ngram bias overhead drops from ~10% per step to ~1%. Composes with Shots 1-3 for a ~10-12× total speedup.

**Risks**: writing custom Triton is high-effort. Skip if Shots 1-3 already get us to acceptable performance.

### Shot 5 — GPTQ int6 dequant + matmul fusion (~2 h, $0.50)

**Goal**: at inference time (post-train, during the final eval), the int6-quantized model dequantizes weights to bf16 in a separate kernel before the matmul. Fusing the dequant into the matmul saves a memory roundtrip per layer.

**Approach**: bitsandbytes-style int8 matmul, but for int6. Either:
- Use existing `torch.matmul` with int6 weights via a custom autograd function that dequants on-the-fly
- Use a Triton matmul kernel that takes int6 input and bf16 output (~80 LOC)

**Success criterion**: eval phase speedup ~30%. Doesn't help training (training uses fp32/bf16), but the eval phase is currently the slowest part of run.sh (~10-15 min for sliding window stride=64 at our current speed).

**Risks**: GPTQ int6 storage format isn't standard — we use int8 backing storage with int6 range. May need to repack to actual int6 for the kernel.

### Shot 6 — Custom SDPA replacement tuned for our shape (~3 h, $0.80)

**Goal**: write a Triton SDPA tuned specifically for `(num_heads=8, num_kv_heads=4, head_dim=64, softcap=30)` GQA with logit softcap. The generic SDPA has a lot of branching for shapes that aren't ours.

**Skip if**: Shot 2 (FA3) lands cleanly. FA3 already does this for us.

### Shot 7 — Int8 tabulation hash GPU gather (~1 h, $0.25)

**Goal**: NGR_LOG_FREQ_INV currently runs once on first forward (CPU sync to compute the multiplier). Replace with a GPU tabulation hash that runs in O(1) instead of O(table_size).

**Skip if**: NGR_LOG_FREQ_INV doesn't materially affect val_bpb in Phase 2 measurements. (It's a world-novel patch but never validated on SP8192.)

### Shot 8 — FP8 compute paths (~4 h, $1.00)

**Goal**: matmul in FP8, accumulate in bf16. Hopper supports FP8 natively. Theoretically 2× speedup over bf16.

**Hardware requirement**: H100 SXM (FP8 instructions are Hopper-only). Means a ONE-shot H100 burn at the end of Phase 2. Alternatively use 4070 Ti's FP8 (Ada generation) for testing.

**Skip if**: Shots 1-3 already get us to ~10× speedup. Diminishing returns vs the dev complexity.

---

## Phase 2 stop conditions

1. **Total per-step speedup ≥ 5×** vs Phase 1 baseline → declare Phase 2 done, move to actual submission run on H100 SXM
2. **Spend ≥ $10** → stop, lock in the speedup we have
3. **Any shot crashes the 10-patch stack and can't be fixed in <1 h** → disable that shot, move to the next, document the gap in `PHASE2_TROUBLESHOOTING.md`

## What Phase 2 explicitly does NOT do

- ❌ Add new patches to the model (model is locked from Phase 1)
- ❌ Change training hyperparams (no LR sweeps, no optimizer changes)
- ❌ 8×H100 distributed work (separate phase, after Phase 2 lands)
- ❌ Train new tokenizers (SP8192 is locked)
- ❌ Re-validate Phase 1 patches (n=2 confirmation is a separate concern)

## Phase 2 → Submission gate

Phase 2 is done when:
- ≥5× speedup achieved (verified on a cheap pod)
- All 10 patches still work end-to-end (no patch was disabled to get the speedup)
- A 1×H100 SXM run produces a val_bpb in the **1.10-1.18** range (within 0.10 of comp records)

After Phase 2: spin a fresh 1×H100 SXM pod, run the Phase 2-optimized stack with the full 600s budget, capture the final val_bpb. **That's the submission**.

## Files Phase 2 will create

- `PHASE2_PLAN.md` (this file)
- `PHASE2_TROUBLESHOOTING.md` — append-only log of what broke + what fixed it (mirror of PHASE1_TROUBLESHOOTING.md format)
- `PHASE2_RESULTS.md` — append-only ledger of per-shot speedup measurements
- `submission/kernels/` — new directory for any Triton kernels we write
- `submission/setup_compile_cache.sh` — wrapper to populate the inductor cache via a warm-up run
- Possibly `submission/train_phase2.py` — IF we end up forking train.py for kernel-fused ops; otherwise modify in place and gate via env vars

## Files Phase 2 will modify

- `submission/run.sh` — toggle `TORCH_COMPILE_DISABLE=0`, add `WARM_COMPILE_CACHE` step, possibly add new env vars for FA3/Triton paths
- `submission/setup.sh` — add FA3 install step (build from source or wheel install)
- `submission/train.py` — selective `@torch._dynamo.disable` on patches that break compile, possibly call new fused kernels

---

## Timing reality check

The previous Phase 1 was supposed to take ~30 min. It took ~4 hours due to disk topology + FA3 ABI + torch.compile + other surprises. **Assume Phase 2 will take 2-3× longer than the optimistic estimates above.** Realistic Phase 2 wall clock: 6-12 hours of dev + ~$5-8 of cheap-pod burn.

If the user wants a faster Phase 2: skip Shots 4-8 entirely. Just do Shots 1+2 (compile + FA3) for a ~5× speedup. Ship that.
