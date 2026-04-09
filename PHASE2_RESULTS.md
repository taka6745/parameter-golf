# PHASE2_RESULTS.md — append-only speedup + val_bpb ledger

**Comp**: openai/parameter-golf
**Phase**: 2 (speed work)
**Plan**: PHASE2_PLAN.md
**Model invariant**: Phase 1 locked-in stack (train.py at 731 lines, 10 patches, git HEAD 3dfc868)

Each row: shot id, hardware, wallclock, steps achieved, ms/step, val_bpb, artifact_bytes, speedup vs Phase 1 baseline, status, timestamp.

| shot | hardware | wallclock | steps | ms/step | tok/s | val_bpb | artifact_bytes | speedup | status | utc |
|---|---|---|---|---|---|---|---|---|---|---|
| (P1 baseline) | 1×H100 SXM 80GB | 600s | 180 | ~3300 | ~280K | TBD | TBD | 1.0× | Phase 1 dry run | 20260409T0230Z approx |

---

## Phase 1 baseline context

Phase 1 hit 180 steps in 600s because:
- `torch.compile` disabled (~3-5× penalty)
- FA3 not installed, SDPA fallback (~30-50% penalty)
- N-gram bias forward overhead (~5-10%)
- 3-layer recurrence adds 13% more layers
- Small model on a big GPU — kernel launch overhead dominates

**Per-GPU rate**: 0.31 steps/sec (vs comp records' 4.17 steps/sec/GPU = ~13× slower).

## Comp anchors (the target)

| PR | stack | val_bpb | hardware |
|---|---|---|---|
| #1485 | 1477 + 3L recurrence + Pre-Quant AdamW TTT + EMA 0.9965 + QK5 | **1.0679** | 8×H100 SXM |
| #1477 | SP8192 + Parallel Residuals + Score-First TTT | 1.0822 | 8×H100 SXM |
| #1482 | SP8192 + Pre-Quant TTT QK 5.25 8ep freeze-1 | 1.0787 | 8×H100 SXM |

**Phase 2 target on 1×H100 SXM**: val_bpb in the **1.10-1.18 range** (within 0.10 of comp records). Won't match 8× because we're 1/8 the raw compute, but we should close most of the gap relative to the 8× vs 1× ratio once the code path is optimized.

---

## Shot-by-shot results

### Shot 1 — torch.compile re-enable
<!-- fill in when run -->

### Shot 2 — FA3 sourcing
<!-- fill in when run -->

### Shot 3 — Persistent CUDAGraph capture
<!-- fill in when run -->

### Shot 4 — Fused n-gram bias Triton kernel
<!-- fill in when run -->

### Shot 5 — GPTQ int6 dequant + matmul fusion
<!-- fill in when run -->

### Shot 6 — Custom SDPA replacement
<!-- fill in when run (probably skipped if FA3 lands in Shot 2) -->

### Shot 7 — Int8 tabulation hash GPU gather
<!-- fill in when run (probably skipped) -->

### Shot 8 — FP8 compute paths
<!-- fill in when run (probably skipped) -->

---

## Cumulative speedup tracker

| after shot | ms/step | vs P1 baseline | steps in 600s | val_bpb | Δ val_bpb vs P1 |
|---|---|---|---|---|---|
| P1 (baseline) | ~3300 | 1.0× | 180 | TBD | — |
| +S1 (compile) | TBD | TBD | TBD | TBD | TBD |
| +S2 (FA3) | TBD | TBD | TBD | TBD | TBD |
| +S3 (CUDAGraph) | TBD | TBD | TBD | TBD | TBD |
| +S4 (fused ngram) | TBD | TBD | TBD | TBD | TBD |
| +S5 (GPTQ fusion, eval only) | TBD | TBD | TBD | TBD | TBD |
| Phase 2 done | **target ≥5× / ≤660 ms/step / ≥900 steps / val_bpb 1.10-1.18** | | | | |
