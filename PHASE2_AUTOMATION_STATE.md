# PHASE2_AUTOMATION_STATE.md — unified experiment driver state

**Pod**: M = `4jfptzwhy9exy9` (RTX 3090 24 GB, $0.46/h, eu-cz-1)
**Cron**: `137c5635` at `9,26,43 * * * *`
**Per-fire budget**: 10 min
**Total budget cap**: $5 / ~11 h wallclock on Pod M

## Experiments

| exp | description | status | val_bpb (unquant / quant) | ms/step | log | notes |
|---|---|---|---|---|---|---|
| **E1** | Shot 0e validation: Phase 1 stack + fix, `bash run.sh` direct, no compile, TTT=0 | ✅ **done** | 3.03477 / **3.05683** | **2933** | `phase2/run_logs/e1_crash_0644Z.log` (initial crash) | **Shot 0e FIXED** — quant gap 0.02206 BPB (was -2.62). Artifact 11.1 MB ✅. 37 steps in 120s cap. |
| E2 | Phase 2 Shot 1 (torch.compile on) via phase2/run.sh | **running (PreQ TTT)** | 2.92033 / — | **1581** | `/tmp/paramgolf_bootstrap.log` | retry @ 0752Z after NLFI refactor `055bafb`. **Train done**: 69 steps in 109s, ms/step **1581 (1.85× vs E1's 2933)**, tok/s **148K (vs E1's 80K)**. Pre-quant val_bpb 2.92033. Peak VRAM 12.7 GB. Now in PreQ TTT. |
| E3 | Code + test Shot 17 (fuzzy LR bandit, ~80 LOC) | pending | | | | needs coding |
| E4 | Code + test Shot 0b (streaming KV eval, ~250 LOC) | pending | | | | needs coding |
| E5 | Code + test Shot 10 (Parameter Banking + Parallel Muon, ~200 LOC) | pending | | | | needs coding |

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

## Running tally

- Pod M uptime: ~145 min (~2h25m)
- Pod M spend: ~$1.11
- Total commits by driver: 10 (fires 1-8 + refactor)
- **E1 DONE** @ 0709Z: Shot 0e fixed, quant gap 0.02206 BPB ✅
- **E2 RETRY RUNNING** @ 0752Z: after `.item()` refactor, now in PreQ TTT. Interim: ms/step **1581 (1.85× speedup)**, pre-quant val_bpb 2.92. Expected done ~0820-0830Z.
- **Headline speedup**: 1.85× from torch.compile alone (before FA3, CUDAGraph, fused ngram, etc)
