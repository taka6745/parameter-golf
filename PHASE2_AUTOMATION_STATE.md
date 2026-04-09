# PHASE2_AUTOMATION_STATE.md — unified experiment driver state

**Pod**: M = `4jfptzwhy9exy9` (RTX 3090 24 GB, $0.46/h, eu-cz-1)
**Cron**: `137c5635` at `9,26,43 * * * *`
**Per-fire budget**: 10 min
**Total budget cap**: $5 / ~11 h wallclock on Pod M

**Protocol (as of 0840Z)**: E1 and E2 are full-fat baseline runs (TTT on, quant eval, the works — ~27 min each). **E3-E5 run in fast-screen mode**: `PREQUANT_TTT_EPOCHS=0` + `MAX_WALLCLOCK_SECONDS=120`, saving ~26 min/run. Fast-screen is for rejecting broken/neutral patches via seed-matched A/B vs E2 baseline. Promising winners get promoted to a "champion stack" full-fat run at the end to measure final submission val_bpb.

## Experiments (v2 — SPEED-FOCUSED, 1020Z)

**Mission recalibrated**: user reminded us the goal is SPEED. Original 5-experiment plan was too short. Expanded to E1-E12 with a focus on big-hitter training throughput wins, all runnable on 3090 in fast-screen mode (TTT=0, 120s wallclock, ~4 min per run). The E2 quant-gap-when-TTT bug is parked — not on critical path; a submission built on E1 stack (TTT=0) is already shippable.

| exp | shot | description | LOC | expect | status | ms/step | notes |
|---|---|---|---|---|---|---|---|
| **E1** | — | Baseline (no compile, TTT=0) | — | 1.0× | ✅ done | 2933 | Shot 0e quant gap 0.022 ✅, artifact 11.1 MB |
| **E2** | S1 | torch.compile on (+ layer_loop + prefetch) | — | +25-35% | ✅ done | **1581 (1.85×)** | post-TTT val_bpb 1.425 matches H100 reference |
| **E3** | S17 | fuzzy LR bandit | 21 | ??? | ❌ done (SKIP) | 1592 | lost A/B vs E2 (+0.07 train_loss at step 30) |
| **E2b** | — | GPTQ val-calib for TTT quant gap fix | 26 | bug fix | ❌ done (FAIL) | 1588 | gap only moved 0.014 BPB (1.866→1.852), hypothesis rejected |
| **E4** | NEW | `torch.compile(mode='max-autotune')` | 1 | +5-15% | ❌ **failed** | | CUDA graphs conflict with rotary embedding caching pattern (`self._cos_cached = freqs.cos()[...]` at train.py:231 gets overwritten by subsequent graph runs). Crashed 1034Z. |
| **E4b** | NEW | `torch.compile(mode='max-autotune-no-cudagraphs')` | 0 | +3-10% | ✅ **done** | **1526** (+3.7% vs E2, **1.92× vs E1**) | quant gap 0.094, artifact 11.13 MB |
| **E5** | NEW | E4b + `cudnn.benchmark=True` | 1 | +1-5% | ✅ **done** | **1514** (+0.8% vs E4b, **1.94× vs E1**) | tiny incremental win |
| **E8** | NEW | E5 + `NUM_LOOPS=1` | 0 | +15-20% | ✅ **done** | **1410** (+6.9% vs E5, **🎯 2.08× vs E1**) | 77 steps. Quality intact (2.928 pre-quant). Clean 2× crossed. |
| **E8c** | NEW | E8 + `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1` | 0 | +3-10% | ✅ **done (neutral)** | **1408.9** (~0% vs E8) | speed neutral BUT peak VRAM 10.9 GB (vs E8's 12.7 GB, **-14%**). Free memory headroom enables bigger batch. |
| **E8d** | NEW | E8c + `TRAIN_BATCH_TOKENS=262144` (to use freed VRAM) | 0 | +15-25% effective | **queued** | | will auto-launch when E8c exits. **Note**: E8c is taking ~18 min (vs 7 min normal) because coord_descent tuning recompiles for eval_model too. E8d will be similarly slow. Consider dropping coord_descent if OOM is not an issue. |
| **E6** | S10 | Parameter Banking + Parallel Muon | ~90 | +5-15% | ✅ **coded** `ca545ed`, **queued** | | auto-launches after E8d. `USE_PARALLEL_MUON=1` groups params by shape, batches Newton-Schulz across them. Falls back to serial path if env var off. |
| **E7** | S4 | Fused n-gram bias Triton kernel | 150 | +5-10% | pending | | write a Triton kernel that does all 3 n-gram gathers + weighted add in one pass |
| **E9** | S12 | Multi-shard loader + dedicated copy stream | 100 | +2-5% | pending | | incremental prefetch win |
| **E10** | S3 | Persistent CUDAGraph capture | 200 | +10-20% | pending | | reduces kernel launch overhead |
| — | S9, S2 | FA3 varlen / FA3 sourcing | — | +30-50% | **blocked** | | Hopper-only, can't test on 3090 |
| — | S0b, S13, S14 | Eval-only speedups (streaming KV, Triton KV eval, fused softcap+CE megakernel) | — | +eval 5-15× | **deferred** | | sliding eval disabled in fast-screen, revisit for submission |

**Running order** (smallest to biggest LOC, fail-fast): **E4 → E5 → E8 → E7 → E6 → E10 → E9**. E4, E5, E8 are sub-5-LOC one-line env flips — we can test all 3 in the time one proper implementation takes.

**Target**: composing survivors on top of E2's 1.85× → realistic stretch **2.5-3.5× vs E1 baseline**.

## Fire log

| fire | utc | action | next |
|---|---|---|---|
| 1 | 20260409T0544Z | E1 running: get_data.sh downloading docs_selected.jsonl from HF. GPU 0%, container disk 60/80 GB (75%), 0 shards yet. Process alive (PID 307). | wait for tokenize + train + eval; next fire ~17 min |
| 2 | 20260409T0609Z | E1 still running: tokenize at 6.4M/15.4M docs (42%), hard-link fix confirmed (container disk 60→53 GB after cache drop), GPU 0% (tokenize CPU-only). No errors. NOTE: earlier "77 shards" note was wrong — tokenize script writes at the end, no progressive shards. | wait for tokenize to complete + n-grams + train; next fire ~17 min |
| 3 | 20260409T0613Z | E1 still running: tokenize at 9.6M/15.4M docs (62%), PID 664 at 1307% CPU (13 cores saturated), 30.5 min elapsed, RSS 9 GB / 1 TiB RAM, container disk stable 53/80 GB, GPU 0% still. Load avg 5.88. Output dir `data/datasets/datasets/fineweb10B_sp8192/` exists, empty until tokenize finalizes. | wait ~13 min for next fire (0626Z) — tokenize ETA ~20 min remaining |
| 4 | 20260409T0626Z | **E1 entered training phase.** Tokenize DONE (128 train shards .bin + 1 val shard, 25.6 GB total on container disk). N-gram build DONE: bigram_tab_8192v.npy + trigram_logprobs_8192v.npy + fourgram_logprobs_8192v.npy (512 MB each). `python3 submission/train.py` running PID 3843275 (elapsed 00:23 at snapshot time, 125% CPU, 1.2 GB RSS). Config loaded: train_shards 128, val_tokens 40542208. No GPU telemetry parsed yet (train just starting). | next fire (0643Z): should catch E1 post-train-phase (maybe mid-PreQ-TTT or fully done). Expected E1 completion ~0637Z |
| 5 | 20260409T0644Z | **E1 CRASHED (OOM) during PreQ TTT.** Train phase completed OK (37 steps in 108.6s = 2935 ms/step, tok/s 80K, train_loss 8.87→5.61, in-training val_bpb @ step 37 = 2.1207, stopping_early wallclock_cap). Pre-quant post-EMA eval: **val_bpb 3.03473** (undertrained EMA, expected). **CRITICAL: NGR_LOG_FREQ_INV fix path CONFIRMED firing** (log shows "computed + saved multipliers" + "applied mutation"). TTT OOM'd at train.py:843 (`prequant_ttt_adapt_adamw` forward): 23.52 GiB / 23.56 GiB used, needed 64 MiB more. `expandable_segments:True` was already set. Root cause: 3090 24 GB can't fit model + n-grams (1.5 GB) + Adam state + TTT activations at batch_seqs=32. **Action**: relaunched E1 retry @ 0647Z via direct `bash run.sh` (skipping setup/get_data/n-gram, data still staged) with `PREQUANT_TTT_EPOCHS=0 PREQUANT_TTT_ENABLED=0 TTT_ENABLED=0` — TTT not needed for Shot 0e gap measurement. PID 3843576. Crash log saved to `phase2/run_logs/e1_crash_0644Z.log`. | next fire (0704Z): E1 retry should be DONE by then (~4 min total: train 120s + quant 30s + quant eval 70s). Parse unquant + quant val_bpb, compute gap, verify <0.1 BPB, mark E1 done, launch E2 |
| 6 | 20260409T0709Z | ✅ **E1 DONE. Shot 0e CONFIRMED FIXED.** Unquant val_bpb **3.03477**, quant val_bpb **3.05683**, **gap = 0.02206 BPB** (normal GPTQ int6 gap). Artifact **11,651,969 bytes (11.1 MB ✅)**. ms/step 2933. peak VRAM 19,308/24,576 MiB. GPTQ passthrough list confirmed `_nlfi_bigram_mult, _nlfi_trigram_mult, _nlfi_fourgram_mult, _nlfi_stored_flag` in serialized artifact. Also mid-fire: **run.sh TTT OOM fix landed** (`6e7f299`) — auto-detects VRAM < 40 GB and drops `PREQUANT_TTT_BATCH_SEQS` 32→8. **Launched E2** @ 0713Z via phase2/run.sh with TTT=2 epochs (auto batch_seqs=8). PIDs: bash 3854176, warm_compile 3854178, train.py 3854249. Inductor cache was cold → will pay ~5 min first-compile cost, then measure ms/step delta. | next fire (0726Z): E2 still in compile warmup or just starting train phase. Wait + progress note, next fire after that (~0743Z) catches E2 done |
| 7 | 20260409T0726Z | ❌ **E2 initial CRASHED @ 0714Z**: torch.compile hit `.item()` graph break at `train.py:427` (`if int(self._nlfi_stored_flag.item())==1:`). **fullgraph=True** compile can't trace Tensor.item() without `capture_scalar_outputs=True`. Both warm_compile_cache.py AND phase2/run.sh died at forward step 0. Pod M idle ~40 min. **Action**: refactored — extracted NLFI setup into `GPT._apply_nlfi_once()` method called eagerly from `train_model` + `train_and_eval` BEFORE torch.compile. Commit `055bafb`, 65 insertions / 47 deletions. Forward now pure tensor ops. **E2 retry LAUNCHED** @ 0752Z, NLFI setup confirmed firing via new eager path. | fire 8 (0809Z): check E2 retry progress |
| 8 | 20260409T0809Z | **E2 retry mid-flight, crushing it.** Past train phase: **69 steps in 109s (1581 ms/step)** = **1.85× speedup vs E1's 2933 ms/step**. tok/s **148K (vs E1's 80K)** = 1.88× throughput. Pre-quant post-EMA val_bpb **2.92033** (better than E1's 3.03 due to ~2× more steps). Peak VRAM 12,698 MiB (LOWER than E1's 19,308 — compile uses less memory). Currently in PreQ TTT (trainable 33.1M / frozen 2.89M, freeze_blocks=1). GPU 100% @ 348 W. No errors. PID 3856713 alive. | fire 9 (0826Z): expect E2 DONE. Parse quant val_bpb, verify gap still clean, confirm speedup, launch E3 (Shot 17 fuzzy LR bandit — needs coding) |
| 9 | 20260409T0826Z | E2 still running. **PreQ TTT epoch 1/2 done in 769s, loss 4.5184**. Epoch 2 in progress. At batch_seqs=8 (auto-dropped), epochs are slower than the H100 reference (H100 was 8 epochs × 200s). Total TTT budget on 3090: ~25 min (~1540s). GPU 100% @ 348 W, 13.4 GB VRAM. No crashes. PID 3856713 alive. | fire 10 (0843Z): E2 should be DONE or near done (epoch 2 ends ~0838Z, then GPTQ + quant eval ~3 min → done by ~0842Z). Parse results, launch E3 |

## Running tally (1247Z — autonomous mode until 22:00Z / 8am AEST)

- Pod M uptime: ~6h
- Pod M spend: ~$2.76
- Total commits: 20+
- **Best so far: E8 at 1410 ms/step = 2.08× vs E1**
- **Orchestrator running PID 3994567** queued 6 experiments: E8d → E6 → E10b → E11 → E7a → E12_stack
- Metrics log at `/tmp/paramgolf_orchestrator.log` on pod
- **Coded + staged** (not yet run): E6 Parallel Muon (`ca545ed`), E11 BF16 n-grams (`4146f4e`), E7a ngram bigram-only (`35d10a5`), E10a rotary pre-compute fix (`c8f8bb2`)
- **Still to code**: E10 explicit CUDA graphs (200 LOC, risky), E7 fused n-gram Triton kernel (150 LOC), smaller-model env-var A/B (NUM_LAYERS, MLP_MULT)
- **Target**: stretch 3-3.5× vs E1

## Orchestrator protocol (for future cron fires)

**WAVE 1** (`/tmp/paramgolf_orchestrator.sh`, PID 3994567, launched 1247Z):
- ✅ E8d (wash), ✅ E6 (+3% → 2.14×), ❌ E10b (crash), ✅ E11 (same as E6, bf16 verified)
- 🚀 E7a running (bigram only)
- ⏳ E12_stack queued (bf16 + bigger batch + parallel muon)

**WAVE 2** (`/tmp/paramgolf_orchestrator_wave2.sh`, PID 4050718, queued 1346Z):
- Waits for wave 1 to exit, then runs:
- E13_layers8 (NUM_LAYERS=8)
- E14_mlp2 (MLP_MULT=2)
- E15_seq1024 (TRAIN_SEQ_LEN=1024)
- E13b_layers9 (NUM_LAYERS=9, milder)
- E16_champion (NUM_LAYERS=9 + bf16 + parallel muon)
- E17_champion_lite (NUM_LAYERS=8 + MLP_MULT=3 + parallel muon)

All wave 2 experiments are env-var-only; no new code needed. Per-experiment runtime ~7-15 min. Wave 2 total ~60-90 min.

**Key insight from wave 1**: 3090 is **compute-bound**. Bigger batches (E8d), bf16 (E11), coord descent (E8c) don't help speed. Real wins come from **cutting compute** (NUM_LOOPS in E8, Parallel Muon in E6) or **kernel fusion** (max-autotune in E4b). Wave 2 tests aggressive compute-cutting via smaller model dimensions.

**Waves 2-5 done.** Summary in PHASE2_RESULTS.md "FINAL SUMMARY" section. Speed podium: E29 (8.55×), E28 (5.16×), E26 (4.56×). Submission podium (best quant): E14 (3.027), E4b (3.017).

**Wave 6 CHAMPION runs** (full 600s wallclock) launched 19:28Z:
- CHAMP_A (l11+m2, E14 at full time) — ✅ 515 steps, pre-quant 1.600 / quant 4.603 (int6 broken)
- CHAMP_B (l6+m2, E24 at full time) — ✅ 813 steps, pre-quant 1.399 / quant 4.966 (int6 broken)
- CHAMP_C (E6 baseline at full time) — ✅ 431 steps, pre-quant 1.704 / quant 4.801 (int6 broken)

**Key finding**: all 3 champion runs show int6 quant fails catastrophically for converged models. Quant gap 3+ BPB.

**Wave 7 INT8 RESCUE** (CHAMP_D, CHAMP_E with MATRIX_BITS=8) launched 20:28Z:
- CHAMP_D (l6+m2 + int8) — 🎯 **BREAKTHROUGH: 811 steps, val_bpb 1.39943 quant, gap 0.001, 9.55 MB artifact** — submission-grade!
- CHAMP_E (l11+m2 + int8) — UNKNOWN, Pod M died at ~21:09Z before result could be read

## POD M TERMINATED 21:09Z

Pod returned "container not found" — likely RunPod reclaim or billing issue. Total uptime ~14h45m. Total spend estimate: ~$6.80. All committed results are safe in git. CHAMP_D alone is sufficient to declare the submission path unblocked.

## Pod N: paramgolf-ttt-int8 (A6000 48GB, $0.33/h, eu-cz-1)

Spun up 23:11Z to test CHAMP_F: CHAMP_D config + Pre-Quant TTT 8 epochs (the missing TTT validation). A6000 48GB chosen because 3090 was sold out, and the extra VRAM eliminates TTT OOM concerns.

**CHAMP_F monitor cron**: `37ff8ea1` (9,26,43 schedule)

**23:46Z fire**: bootstrap-data phase, sentencepiece training in progress (no cached model on fresh pod), 0/3 ngram files yet. ETA to CHAMP_F training start ~00:25Z. Spend ~$0.17.

**WAVE 3 plan** (after wave 2 results, for future cron fires):
- Stack the best wave 2 winner with E6 + E8 + E4b config into a true champion run
- Try explicit CUDA graph capture if we have time (requires fixing remaining rotary/state issues)
- Possibly try Triton fused n-gram kernel if E7a shows lookups are a bottleneck
