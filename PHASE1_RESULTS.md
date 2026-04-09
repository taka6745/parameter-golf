# PHASE1_RESULTS.md — append-only shot ledger

**Comp**: openai/parameter-golf
**Pod**: `9lfji49c6ngy9a` (paramgolf-phase1-h100, NVIDIA H100 PCIe 80GB, RunPod-rented)
**Plan**: PHASE1_PLAN.md
**Trainer**: `train_gpt_phase1.py` (clean decoded PR #1477, no patcher hunks)
**Cost cap**: $15

Each row: shot id, env diff, wallclock, val_bpb, artifact_bytes, ms/step, status,
timestamp. Append only.

| shot | env | wallclock | val_bpb | artifact_bytes | ms/step | status | utc |
|---|---|---|---|---|---|---|---|
| **R1 in-training** | seed=42, TTT=1, MAX_WALLCLOCK_SECONDS=600, TORCH_COMPILE_DISABLE=1 | 593086 ms (cap hit, 129/20000 steps) | **1.7059** (in-training) | n/a (no quant yet) | ~4400 ms/step (eager mode, no compile) | partial: train phase only | 20260409T0033Z |
| **R1 quantized** | same as R1 + EMA 0.997 + GPTQ int6 + brotli-11 | n/a (post-train) | **3.3166** ★ EMA junk — only 129 steps | **16041642** (41 KB OVER 16 MB limit) | n/a | **REJECTED — over size limit + undertrained EMA** | 20260409T0050Z |

## R1 analysis

- The 1.7059 in-training number proves the pipeline works (model is learning).
- The 3.3166 quantized number is junk because EMA decay 0.997 needs ~1000+ steps to forget the random init; we only had 129 steps.
- The 16.04 MB submission size is 41 KB over the 16 MB limit — would be rejected.
- Both issues resolve with longer training: more steps → EMA converges → smaller delta-from-fp32 → quantization works correctly → both val_bpb and size land in spec.
- **Action**: R3 (next) needs `MAX_WALLCLOCK_SECONDS >= 3000` so EMA can converge. Don't worry about size at this stage — that's a quantization-quality issue downstream of training.


---

## Comp anchors (for comparison — not our runs)

| PR | stack | val_bpb | hardware |
|---|---|---|---|
| #1477 | SP8192 + Parallel Residuals + Score-First TTT | **1.0822** | 8×H100 |
| #1476 | SP8192 + QK5 + Legal TTT | 1.0842 | 8×H100 |
| #1471 | SP8192 + SDClip + 3-Layer Depth Recurrence + EMA | 1.0866 | 8×H100 |
| #1019 | AR Self-Gen GPTQ + XSA-all + BigramHash3072 | 1.1147 | 8×H100 |

## Our pre-Phase-1 baseline (overnight cheap-pod 4070 Ti S2)

| stack | val_bpb | n_seeds | hardware |
|---|---|---|---|
| STACK_GATED_LEGAL_TTT (2-component minimal) | **1.3711** | 2 | 4070 Ti |

Phase 1 success criterion: get on H100 + SP8192 stack and land within 0.02-0.05 BPB
of the comp anchor (i.e. ~1.10-1.15 expected on 1×H100 with our smaller batch).
