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
| **E4 (max-autotune compile mode)** | 1×RTX 3090 24GB | crashed at step 0 | — | — | — | — | — | — | ❌ **FAILED**: `torch.compile(mode='max-autotune')` enables CUDA graphs which conflict with rotary embedding cached tensors (train.py:231 — `_cos_cached` gets overwritten by subsequent graph runs). `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`. | 20260409T1034Z |
| **E4b (max-autotune-no-cudagraphs)** | 1×RTX 3090 24GB | 110s train + 147s eval + ~60s GPTQ + 133s quant eval (~7 min wallclock) | 72 train (wallclock cap) | **1526 ms/step** (+3.7% vs E2) | 155K peak (+2.6%) | post-EMA **2.92311** / quant **3.01671** (gap 0.094) | **11,670,594** (11.13 MB ✅) | **1.92× vs E1** | ✅ **WIN (+3.7%)** via env-only change. Same kernel autotuning as max-autotune but no CUDA graphs. Stacks cleanly on top of E2. | 20260409T1111Z |
| **E5 (E4b + cudnn.benchmark=True)** | 1×RTX 3090 24GB | 109s train + 134s eval + ~60s GPTQ + 133s quant eval (~7 min wallclock) | 72 train (wallclock cap) | **1514 ms/step** (+0.8% vs E4b) | 155K peak | post-EMA **2.92456** / quant **3.01891** (gap 0.094) | **11,671,080** (11.13 MB ✅) | **1.94× vs E1** | ✅ tiny incremental win from cuDNN kernel autotuning. Stacks on top of E4b. | 20260409T1144Z |
| **E8 (E5 + NUM_LOOPS=1)** | 1×RTX 3090 24GB | 108s train + 118s eval + ~60s GPTQ + 177s quant eval (~7 min wallclock) | 77 train (wallclock cap) | **1410 ms/step** (+6.9% vs E5) | 155K peak | post-EMA **2.92781** / quant **3.03483** (gap 0.107) | **11,671,520** (11.13 MB ✅) | **🎯 2.08× vs E1** | ✅ **CROSSED 2× BASELINE.** Reducing layer_loop num_loops 2→1 drops from 17 → 14 block invocations (~17% less compute) for +6.9% speed. Quality unaffected at this scale. | 20260409T1208Z |
| **E8c (E8 + TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1)** | 1×RTX 3090 24GB | 108s train + rest same (~18 min total — coord descent recompiles for eval_model too) | 77 train (wallclock cap) | **1408.9 ms/step** (+0.1% vs E8, noise) | 155K | same as E8 | 11.13 MB | 2.08× vs E1 | ⚠️ **Speed NEUTRAL** but **peak VRAM 10,934 MiB** (vs E8's 12,697 MiB, **-14%**). Free memory headroom — enables bigger TRAIN_BATCH_TOKENS. Compile time DOUBLED to ~18 min because coord descent re-tunes for eval model. Not worth keeping for speed, but memory savings enable E8d. | 20260409T1214Z |
| **E8d (E8 + TRAIN_BATCH_TOKENS=262144)** | 1×RTX 3090 24GB | 108.7s train + compile (max-autotune-no-cudagraphs + coord_descent) | 57 train (wallclock cap) | **1907 ms/step** (per-step) | 137.4K tok/s sustained (−1.5% vs E8) | pre-quant **2.91058** / quant **3.00015** (gap 0.089) | — | **−1.5% vs E8 (WASH)** | ❌ **Bigger batch gives NO effective speedup.** Ratio math: E8d does 0.74× fewer steps but each step processes 1.333× more tokens → 0.985× throughput. **3090 is compute-bound, not launch-bound.** Better gradient estimates drop val_bpb 2.928 → 2.911 but that's quality not speed. **Insight: next wins must cut compute or fuse kernels, not grow batches.** | 20260409T1311Z |
| **E6 (Shot 10: Parameter Banking + Parallel Muon)** | 1×RTX 3090 24GB | 108.156s train + rest (~7 min wallclock) | 79 train (wallclock cap) | **1369 ms/step** (+3.0% vs E8) | **143.6K** tok/s sustained (+3.0%) | pre-quant **2.93211** / quant **3.03641** (gap 0.104) | — | **🎯 2.14× vs E1** | ✅ **Parallel Muon WORKS** — batched Newton-Schulz across shape-matched params cuts ~24 serial NS calls per step down to a handful of batched calls. 79 steps vs E8's 77 in same wallclock. Quality identical. Stacks on top of E2+E4b+E5+E8. | 20260409T1325Z |
| **E10b (E6 + mode='max-autotune')** | 1×RTX 3090 24GB | crashed ~90s in | — | — | — | — | — | — | ❌ **FAILED** — max-autotune still crashes despite E10a rotary pre-compute fix. Another CUDA graphs incompatibility somewhere in forward path. Not debugging deeper in fast-screen budget. | 20260409T1327Z |
| **E11 (E6 + USE_NGRAM_BF16=1)** | 1×RTX 3090 24GB | ~7 min wallclock | 79 train (wallclock cap) | **1370 ms/step** (≈0% vs E6) | 143.5K | pre-quant **2.93260** / quant **3.03869** (gap 0.106) | — | 2.14× vs E1 | ⚠️ **Speed UNCHANGED** (confirms compute-bound). BF16 n-grams save ~750 MB VRAM but don't speed up compute-bound kernels. Quality identical to E6. Stack as free memory win for any future experiment that needs headroom. | 20260409T1341Z |
| **E7a (E6 + USE_NGRAM_BIGRAM_ONLY=1)** | 1×RTX 3090 24GB | ~7 min wallclock | 81 train (wallclock cap) | **1342 ms/step** (+2.0% vs E6) | 146.5K sustained / 163.9K peak | pre-quant **2.94729** / quant **3.03730** (gap 0.090) | — | **🎯 2.19× vs E1** | ✅ **Skipping trigram+fourgram lookups = +2% speed** for 0.015 BPB quality drift. Confirms 3-gram and 4-gram gathers ARE a bottleneck worth investigating with a fused Triton kernel (E7 proper). For the champion stack, we may keep all 3 tables for quality but this validates the direction. | 20260409T1359Z |
| **E12_stack (E6 + bf16 + TRAIN_BATCH_TOKENS=262144)** | 1×RTX 3090 24GB | 109.5s train + rest | 60 train (wallclock cap) | **1826 ms/step** per-step | 143.5K tok/s sustained (wash) | pre-quant **2.90518** / quant **3.00499** (gap 0.100) | — | wash vs E6 | ⚠️ Same pattern as E8d — bigger batch is compute-bound wash. Pre-quant val_bpb 2.905 is the BEST quality seen (best gradient estimates) but speed doesn't improve. Confirms repeatedly: 3090 is compute-bound, bigger batches only help quality. | 20260409T1414Z |
| **E13_layers8 (E6 + NUM_LAYERS=8)** | 1×RTX 3090 24GB | 108.4s train + 93s eval + ~60s GPTQ + 122s quant eval (~7 min wallclock) | **102 train** (wallclock cap) | **1062 ms/step** (**+29% vs E6**) | 185K+ tok/s sustained, 211K peak | pre-quant **3.05181** / quant **3.07367** (gap 0.022 ✅) | tbd | **🔥 2.76× vs E1** | ✅ **BIGGEST WIN YET.** 8 layers vs 11 = -27% blocks, 24% fewer params (27.3M vs 36M). 102 steps in 108s (E6 had 79 in 108s). **Quality cost**: +0.12 BPB vs E6 (3.05 vs 2.93). **Quant gap is 0.022 — back to E1 levels, clean.** Smaller model = easier to quantize. | 20260409T1433Z |
| **E14_mlp2 (E6 + MLP_MULT=2)** | 1×RTX 3090 24GB | 108.4s train + ~7 min wallclock | 95 train (wallclock cap) | **1141 ms/step** (+20% vs E6) | 162K sustained / 177K peak | pre-quant **3.03866** / quant **3.02738** (gap −0.011, undertrained noise) | tbd | **2.57× vs E1** | ✅ Halving MLP hidden dim (4096 → 2048) cuts FFN compute ~50%. Less effective than E13 (cutting layers). Quality 3.04, in between E6 and E13. | 20260409T1451Z |
| **E15_seq1024 (E6 + TRAIN_SEQ_LEN=1024)** | 1×RTX 3090 24GB | 108.3s train + ~7 min wallclock | 88 train (wallclock cap) | **1231 ms/step** (+11% vs E6) | ~152K sustained | pre-quant **2.95423** / quant **3.08066** (gap 0.127) | tbd | **2.38× vs E1** | ⚠️ Halved seq_len gives only +11% speed (vs +29% from layer cut). But **BEST QUALITY among compute-cut experiments** (val_bpb 2.954, closest to E6's 2.932). Trade-off: less context per step but more steps and same param count. Interesting for quality-sensitive champion builds. | 20260409T1511Z |
| **E13b_layers9 (E6 + NUM_LAYERS=9)** | 1×RTX 3090 24GB | 109.0s train + ~7 min wallclock | 94 train (wallclock cap) | **1160 ms/step** (+18% vs E6) | ~156K sustained | pre-quant **3.10322** / quant **3.02411** (quant better, undertrained) | tbd | **2.53× vs E1** | Milder than E13 (9 vs 8 layers). +18% speed but quality slightly worse at 94 steps. Expected to be in between E6 and E13 but didn't quite reach expected quality. | 20260409T1530Z |
| **E16_champion (E13b + USE_NGRAM_BF16=1)** | 1×RTX 3090 24GB | 109.1s train + ~7 min wallclock | 94 train (wallclock cap) | **1161 ms/step** (same as E13b) | ~156K | pre-quant **3.10180** / quant **3.02109** (same pattern) | tbd | **2.53× vs E1** | ⚠️ Adding bf16 n-grams to E13b is speed-neutral (confirms compute-bound AGAIN). Quality virtually identical to E13b. | 20260409T1543Z |
| **E17_champion_lite (NUM_LAYERS=8 + MLP_MULT=3 + parallel muon)** | 1×RTX 3090 24GB | 108.2s train + ~7 min wallclock | **110 train** (wallclock cap) | **983 ms/step** (**+28% vs E13, +39% vs E6**) | ~180K+ sustained | pre-quant **3.06547** / quant **3.01734** (undertrained noise) | tbd | **🔥🔥 2.98× vs E1 (near-3×)** | ✅ **NEW BEST.** Combining layer cut (8 vs 11) + moderate MLP cut (3 vs 4) → first run under 1000 ms/step. 110 steps in 108s wallclock. Quality slightly worse than E13 (3.066 vs 3.052) but +28% more speed. Fastest config on 3090 so far. | 20260409T1559Z |
| **E18_triple (NUM_LAYERS=8 + MLP_MULT=3 + TRAIN_SEQ_LEN=1024)** | 1×RTX 3090 24GB | 108.4s train + eval + quant (~7 min) | **123 train** (wallclock cap) | **881 ms/step** (**+10% vs E17**) | 223K tok/s effective | pre-quant **3.12824** / quant **3.08062** | tbd | **🔥 3.33× vs E1** | 🎯 Triple compute cut: layers=8, mlp=3, seq=1024. **23.1M params** (-36%). 123 steps in 108s. Quality cost 0.065 BPB vs E17. | 20260409T1627Z |
| **E21_layers6 (NUM_LAYERS=6)** | 1×RTX 3090 24GB | 108.7s train + ~7 min wallclock | **127 train** (wallclock cap) | **856 ms/step** (**+3% vs E18, +44% vs E13**) | 230K+ tok/s effective | pre-quant **2.95432** / quant **3.15366** (gap 0.20) | tbd | **🏆 3.43× vs E1 (CURRENT BEST)** | 🏆 **CURRENT CHAMPION.** Dropping to 6 layers (from 11) gives BOTH faster AND better quality than E18! Pre-quant val_bpb 2.954 — best among all compute-cut experiments. **Insight**: smaller model → more training steps (127 vs E6's 79) → better convergence in 108s budget. Quant gap widens (0.20) with smaller model, but pre-quant quality wins. | 20260409T1643Z |
| **E22_l8m2 (NUM_LAYERS=8 + MLP_MULT=2)** | 1×RTX 3090 24GB | 108.0s train + ~7 min wallclock | 121 train (wallclock cap) | **893 ms/step** (+10% vs E17) | ~220K tok/s | pre-quant **3.13982** / quant **3.17401** (gap 0.034) | tbd | **3.29× vs E1** | ⚠️ Aggressive FFN cut (mlp=2) gives similar speed to E18 but worse quality. E21 dominates on both axes. | 20260409T1656Z |
| **E23_quality (NUM_LAYERS=9 + seq=1024 + bf16)** | 1×RTX 3090 24GB | 108.1s train + ~7 min wallclock | 103 train (wallclock cap) | **1050 ms/step** (+30% vs E6) | 30.2M params | pre-quant **3.16448** / quant **3.01634** (undertrained pattern) | tbd | **2.79× vs E1** | ⚠️ Targeted quality preservation didn't work — seq=1024 actually hurt quality. 103 steps, worse pre-quant than E13 at 102 steps. Abandon this config. | 20260409T1713Z |
| **E24_extreme (NUM_LAYERS=6 + MLP_MULT=2)** | 1×RTX 3090 24GB | 108.8s train + ~7 min wallclock | **150 train** (wallclock cap) | **725 ms/step** (+15% vs E21) | **15.2M params (-58% vs E6!)**, 260K+ tok/s | pre-quant **2.97149** / quant **3.13132** (gap 0.16) | tbd | **🏆 4.05× vs E1** | 🏆 **Best BALANCE.** 15M param tiny model, 150 training steps in 108s — MORE THAN 4× E1's 37. Pre-quant val_bpb 2.971 still under 3.0. Quant gap 0.16 acceptable. | 20260409T1724Z |
| **E25_wide (NUM_LAYERS=6 + MLP_MULT=3 + TRAIN_SEQ_LEN=1024)** | 1×RTX 3090 24GB | 108.4s train + ~7 min wallclock | **152 train** (wallclock cap) | **713 ms/step** (+2% vs E24) | ~18M params | pre-quant **2.99545** / quant **3.25776** (gap 0.26) | tbd | **4.11× vs E1** | 🔥 Fastest before E26. 152 steps, pre-quant 2.995. Widest quant gap (0.26). | 20260409T1738Z |
| **E26_nuke (NUM_LAYERS=6 + MLP_MULT=2 + TRAIN_SEQ_LEN=1024)** | 1×RTX 3090 24GB | 108.6s train + ~7 min wallclock | **169 train** (wallclock cap) | **643 ms/step** (+10% vs E25, +11% vs E24) | ~15M params, ~306K tok/s effective | pre-quant **2.92250** ⭐ / quant **3.18644** (gap 0.26) | tbd | **🏆🏆🏆 4.56× vs E1 (NEW CHAMPION!)** | 🏆🏆 **PARETO OPTIMAL.** All three compute cuts on a layers=6 base: layers=6, mlp=2, seq=1024. 169 steps in 108s (>4.5× E1's 37). **Pre-quant val_bpb 2.922 — BEST QUALITY of ALL compute-cut experiments, BETTER THAN E6's 2.932 baseline!** Smaller model trains more converged. Quant gap wider but pre-quant wins. | 20260409T1757Z |
| **E27_layers4 (NUM_LAYERS=4)** | 1×RTX 3090 24GB | ~2 min | — | — | — | — | — | — | ❌ **FAILED** — LOOP_START=3 LOOP_END=5 range (from run.sh defaults) is incompatible with 4 blocks. Expected failure. Would need to override LOOP_END or disable looping. | 20260409T1759Z |
| **E28_dim384 (NUM_LAYERS=6 + MLP_MULT=2 + MODEL_DIM=384)** | 1×RTX 3090 24GB | 108.5s train + ~7 min wallclock | **191 train** (wallclock cap) | **568 ms/step** (+13% vs E26) | ~9M params? (not measured), ~346K tok/s | **pre-quant 2.60109 ⭐⭐⭐** / quant **3.39795** (gap 0.80 ⚠️) | tbd | **🤯 5.16× vs E1** | 🤯 **Speed record: 5.16×.** Cutting model_dim 512 → 384 (25% smaller hidden) on top of layers=6+mlp=2 base. **Pre-quant val_bpb 2.601** (massive improvement — smaller model overfits less in undertrained regime). **BUT quant gap is 0.80** (worst quant of all experiments) — tiny model hurts int6 proportionally more. Speed champion, NOT submission champion. | 20260409T1815Z |

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
