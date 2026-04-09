# PHASE2_AUTOMATION_STATE.md — unified experiment driver state

**Pod**: M = `4jfptzwhy9exy9` (RTX 3090 24 GB, $0.46/h, eu-cz-1)
**Cron**: `137c5635` at `9,26,43 * * * *`
**Per-fire budget**: 10 min
**Total budget cap**: $5 / ~11 h wallclock on Pod M

## Experiments

| exp | description | status | val_bpb (unquant / quant) | ms/step | log | notes |
|---|---|---|---|---|---|---|
| **E1** | Shot 0e validation: Phase 1 stack + fix, `bash run.sh` direct, no compile, TTT=0 | **running (retry)** | — / — | — | `/tmp/paramgolf_bootstrap.log` | initial run @ 0544Z crashed OOM in PreQ TTT (3090 24 GB). **RETRY** @ 0647Z with `PREQUANT_TTT_EPOCHS=0 PREQUANT_TTT_ENABLED=0 TTT_ENABLED=0`. NGR_LOG_FREQ_INV fix path confirmed firing in crash log. PID 3843576 |
| E2 | Phase 2 Shot 1 (torch.compile on) via phase2/run.sh direct | pending | | | | waits on E1 done. TTT OOM fix LANDED in run.sh `6e7f299` — auto-detects VRAM < 40 GB and drops `PREQUANT_TTT_BATCH_SEQS` 32→8 |
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
| 5 | 20260409T0644Z | **E1 CRASHED (OOM) during PreQ TTT.** Train phase completed OK (37 steps in 108.6s = 2935 ms/step, tok/s 80K, train_loss 8.87→5.61, in-training val_bpb @ step 37 = 2.1207, stopping_early wallclock_cap). Pre-quant post-EMA eval: **val_bpb 3.03473** (undertrained EMA, expected). **CRITICAL: NGR_LOG_FREQ_INV fix path CONFIRMED firing** (log shows "computed + saved multipliers" + "applied mutation"). TTT OOM'd at train.py:843 (`prequant_ttt_adapt_adamw` forward): 23.52 GiB / 23.56 GiB used, needed 64 MiB more. `expandable_segments:True` was already set. Root cause: 3090 24 GB can't fit model + n-grams (1.5 GB) + Adam state + TTT activations at batch_seqs=32. **Action**: relaunched E1 retry @ 0647Z via direct `bash run.sh` (skipping setup/get_data/n-gram, data still staged) with `PREQUANT_TTT_EPOCHS=0 PREQUANT_TTT_ENABLED=0 TTT_ENABLED=0` — TTT not needed for Shot 0e gap measurement. PID 3843576. Crash log saved to `logs/phase2_e1_crash_0644Z.log`. | next fire (0704Z): E1 retry should be DONE by then (~4 min total: train 120s + quant 30s + quant eval 70s). Parse unquant + quant val_bpb, compute gap, verify <0.1 BPB, mark E1 done, launch E2 |

## Running tally

- Pod M uptime: ~64 min
- Pod M spend: ~$0.49
- Total commits by driver: 4 (fires 1, 2, 3, 4)
- E1 phase: **retry running @ 0647Z** (initial crashed OOM in TTT at 0643Z). Data staged (skipping tokenize/ngram on retry). ETA: train 120s + quant + quant eval = **~4 min to E1 retry complete (~0652Z)**
- **Key datapoint from crashed run**: ms/step = 2935 on 3090 (no compile, no FA3). tok/s = 80K. 37 steps in 120s budget. The Shot 0e NGR_LOG_FREQ_INV fix is confirmed firing in forward_logits (log lines present).
- **TTT OOM FIX LANDED** (commit `6e7f299`): `submission/run.sh` now auto-detects GPU VRAM via nvidia-smi and drops `PREQUANT_TTT_BATCH_SEQS` from 32 to 8 when < 40 GB. H100 80 GB path unchanged. E2+ experiments will get real TTT-enabled val_bpb numbers on the 3090.
- **E1 retry status at 0654Z**: past train (37 steps OK), past EMA, past pre-quant eval (val_bpb 3.0348), currently in **GPTQ quantization**. Peak memory 19,308 MiB / 24,576 MiB. **Shot 0e fix confirmed in serialization**: GPTQ passthrough list shows `_nlfi_bigram_mult, _nlfi_trigram_mult, _nlfi_fourgram_mult, _nlfi_stored_flag` as float16 passthrough — persistent buffers are being saved alongside quantized weights. Quantized eval ~2-3 min away.
