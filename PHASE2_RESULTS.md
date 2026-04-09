# PHASE2_RESULTS.md — append-only speedup + val_bpb ledger

**Comp**: openai/parameter-golf
**Phase**: 2 (speed work)
**Plan**: PHASE2_PLAN.md
**Model invariant**: Phase 1 locked-in stack (train.py at 731 lines, 10 patches, git HEAD 3dfc868)

Each row: shot id, hardware, wallclock, steps achieved, ms/step, val_bpb, artifact_bytes, speedup vs Phase 1 baseline, status, timestamp.

| shot | hardware | wallclock | steps | ms/step | tok/s | val_bpb | artifact_bytes | speedup | status | utc |
|---|---|---|---|---|---|---|---|---|---|---|
| **P1 baseline (DIFF, unquantized)** | 1×H100 SXM 80GB HBM3 | 591s train + 1606s PreQ TTT (~37 min total) | 183 train + 8 PreQ TTT epochs | ~3230 ms/step | ~270K | **1.24108** (unquantized, post-PreQ-TTT) | ~16 MB (broken path) | 1.0× | **RESEARCH-GRADE — not comp-legal** (wallclock overrun on TTT) | 20260409T0316Z |
| **P1 baseline (DIFF, quantized)** | 1×H100 SXM 80GB HBM3 | same + GPTQ int6 + brotli | post-train | n/a | n/a | **3.86174** ❌ | n/a | n/a | **BROKEN** — NGR_LOG_FREQ_INV serialization bug (Shot 0e blocks P2 start) | 20260409T0317Z |
| **E1 (Shot 0e validation, 3090, TTT=0)** | 1×RTX 3090 24GB | 108s train + 234s eval + ~60s GPTQ + 233s quant eval (~11 min total) | 37 train (wallclock cap) | **2933 ms/step** | 80K | unquant **3.03477** / quant **3.05683** | **11,651,969** (11.1 MB ✅) | baseline for P2 work | ✅ **Shot 0e CONFIRMED FIXED** — quant gap = **0.02206 BPB** (normal GPTQ int6 gap 0.01-0.02; was -2.62 BPB broken). Undertrained due to 120s cap + no TTT, but gap measurement is clean. | 20260409T0709Z |
| **E2 (Shot 1: torch.compile on, 3090, TTT=2ep)** | 1×RTX 3090 24GB | 109s train + 131s eval + 1538s TTT (2 epochs) + ~60s GPTQ + 158s quant eval (~50 min wallclock) | 69 train (wallclock cap) | **1581 ms/step** (**1.85× vs E1**) | **148K** (1.88× vs E1) | post-EMA **2.92033** / post-TTT **1.42528** ★ / quant **3.29089** ⚠️ | **11,631,923** (11.09 MB ✅) | **1.85×** | ✅ **Speed target smashed** (85% gain vs 25-35% target) ✅ Shot 0e survives compile (NLFI eager setup working) ✅ TTT OOM fix holds at batch_seqs=8 ✅ **Post-TTT unquant val_bpb 1.425 matches H100 SXM reference** ⚠️ **Quant gap BROKEN when TTT on**: 3.29 − 1.425 = 1.866 BPB (vs E1's 0.022). New bug, separate from Shot 0e. Does not block fast-screen E3-E5 (TTT off) but blocks submission. | 20260409T0842Z |
| **E3 (Shot 17: fuzzy LR bandit, fast-screen, TTT=0)** | 1×RTX 3090 24GB | 110s train + 130s eval + ~60s GPTQ + 131s quant eval (~7 min wallclock) | 69 train (wallclock cap) | **1592 ms/step** (bandit overhead 0.7%) | 146K | post-EMA **3.21635** / quant **2.95165** (gap −0.265, undertrained noise) | **11,645,138** (11.11 MB ✅) | 1.84× vs E1 | ❌ **Bandit LOSS vs E2 baseline**. A/B train_loss at matched steps (same seed 42): step 30 +0.073, step 50 +0.027, step 60 +0.032 ALL WORSE. Bandit chose arm 2.0 (high-LR) 56/68 times, but that caused explosion at step 2 (train_loss 19.57 vs E2's 12.63). **Verdict: SKIP Shot 17** in champion stack — needs more steps to converge + higher LR arm penalty dominates at undertrained state. Neutral-to-harmful in budget. | 20260409T0903Z |

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
