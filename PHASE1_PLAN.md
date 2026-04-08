# Phase 1 Plan — Validate the target stack on H100

Written 2026-04-09 (post NIGHT_MODE termination). Updated to H100 hardware 2026-04-09. Owner: Claude + Takoda.

Phase 1 purpose: **validate the real target stack** (SP8192 + Score-First TTT + parallel residuals + AR self-gen GPTQ + int6 GPTQ + brotli compression) end-to-end on H100 hardware — the same architecture the comp records use — and lock in the measured val_bpb as our new baseline. The current 1.3711 plateau champion is NOT our submission target: it's on SP1024, which is obsolete on the merged leaderboard.

## H100 narrow rule override

The durable "NO H100 EVER" rule is overridden for Phase 1 specifically, approved by user 2026-04-09. Scope:

- H100 is used ONLY for Phase 1 (validating the target stack via porting from PR #1477 / #1019 / #1471 / #1476)
- PR #1477 **requires FlashAttention 3** which is Hopper-only — this is the technical justification for H100 over 3090/4070 Ti for this specific workload
- Kill the pod IMMEDIATELY once `TARGET_STACK` lands n=2 confirmed
- Phase 2 (torch.compile) and Phase 3 (kernels) revert to cheap 3090/4070 Ti pods — the no-H100 rule resumes after Phase 1 terminates
- Total Phase 1 H100 burn cap: **$15** (hard stop)

## Goal

End Phase 1 with a row in STACK_NOVELTY_TRACKER.md Section A labelled `TARGET_STACK`, n=2 confirmed on 1×H100, with:
- val_bpb measured via `final_int8_zlib_roundtrip` (or brotli equivalent) — same measurement path comp records use
- artifact size ≤ 16 MB
- ms/step recorded as the REAL Phase 3 kernel A/B reference (matches target hardware)

Expected landing range: **1.08-1.15 BPB** on H100 (much closer to comp records since we're on the same hardware now). Comp anchors: #1019 (1.1147), #1471 (1.0866), #1476 (1.0842), #1477 (1.0822). That's 0.22-0.29 BPB improvement over the 1.3711 plateau, achieved purely by porting comp wins — no kernel work yet.

**Note**: since we're on H100 + FlashAttention 3, our numbers will be DIRECTLY comparable to the comp record numbers, not through a "cheap-pod scaling factor." If we port correctly, we should land within 0.02-0.05 of the comp record's own number for the same stack.

## Pre-flight (DONE 2026-04-09)

Mac-side files verified on disk:

| file | status | bytes |
|---|---|---|
| `data/tokenizers/sp8192.model` | ✅ | 131,502 |
| `data/tokenizers/fineweb_8192_bpe.model` | ✅ | 370,908 |
| `data/tokenizers/sp8192.vocab` | ✅ | exists |
| `data/datasets/fineweb10B_sp8192/` | ✅ 129 shards | 24 GB total |
| `data/bigram_logprobs_8192v.npy` | ✅ | 512 MB |
| `data/trigram_logprobs_8192v.npy` | ✅ | 512 MB |
| `data/fourgram_logprobs_8192v.npy` | ✅ | 512 MB |
| `data/skipbigram_logprobs_8192v.npy` | ✅ | 512 MB |
| `data/bigram_tab_8192v.npy` | ✅ | 512 MB |
| `data/trigram_tab_8192v.npy` | ✅ | 512 MB |
| `data/fourgram_tab_8192v.npy` | ✅ | 512 MB |
| `data/fivegram_logprobs_4k_8192v.npy` | ✅ | 128 MB |
| `data/sevengram_cms_8192v.npy` | ✅ | 128 MB |
| `data/tokenizer_specs_8192.json` | ✅ | vocab=8192 |

PR source diffs fetched to `/tmp/`:
- `pr1477.diff` — SP8192 + Parallel Residuals + Score-First TTT record (1.0822)
- `pr1019.diff` — AR Self-Gen GPTQ + XSA-all + BigramHash3072 record (1.1147)
- PR #461 — Score-First TTT framework (needs fetch)
- PR #1260 — parallel residuals implementation (needs fetch)

**Data deployment problem**: 24 GB of SP8192 shards is too big to SCP over the RunPod proxy in reasonable time. Three options:

1. **Download from HuggingFace on the pod** via `data/cached_challenge_fineweb.py --variant sp8192` (used in PR #1477 reproduction) — fastest if HF has the shards cached
2. **Tokenize on the pod** — copy the 131 KB `sp8192.model` via SCP, pod downloads raw FineWeb-Edu, runs `encode_as_ids` on each document, writes shards. CPU-bound, ~15-25 min.
3. **Chunked SCP via base64+tar** — works but slow (~30-60 min for 24 GB over proxy)

Default to option 1, fall back to option 2 if HF dataset doesn't exist.

## Fleet

**1 × NVIDIA H100 80GB** (PCIe preferred, SXM acceptable), ~$2.00-2.70/h. Pod label: `paramgolf-phase1-h100`.

Reasoning:
1. PR #1477's reproduction requires **FlashAttention 3** (Hopper-only kernels). This alone forces H100 for a faithful reproduction.
2. 1 × H100 ≈ 8 × 3090 on our workload scale (small model, compute-bound on matmul at vocab=8192). Each Shot runs in 10-15 min wallclock instead of 30-60 min on 3090.
3. Same kernel path as the comp records → our measured val_bpb is directly comparable to PR #1477's 1.0822 without a cross-hardware fudge factor.
4. Validation happens on the submission target hardware, removing one class of "it worked on 3090 but fails on H100" risk for the final submission.

Dropped Pod β (reference baseline) — the 1.3711 champion baseline is already in the tracker from last night's S2 runs; re-measuring is waste.

**Kill discipline**: pod terminated immediately after `TARGET_STACK` lands n=2. No drifting into Phase 2/3 on H100.

Total Phase 1 burn target: **$5-12**, hard cap **$15**. Phase 2/3 revert to cheap 3090 fleet (separate budget).

## Shot sequence (ordered, each gates the next)

### Shot 1 — SP8192 deployment (45-60 min, $0.25)

**Goal**: prove SP8192 loads end-to-end on the pod and our current champion stack (gated + LEGAL_TTT) runs on it without code crashes.

**Code changes**:
- New patcher hunk `SP8192_MARKER` in `runpod_tests/chore/08_patch_train_gpt.sh` gated behind `USE_SP8192=1`:
  - Change `DATA_PATH` default from `fineweb10B_sp1024` → `fineweb10B_sp8192` when env flag set
  - Change `VAL_TOKENS_PATH` default similarly
  - Change `vocab_size` default 1024 → 8192
  - Change n-gram bias loader paths to load `*_8192v.npy` when flag set
  - Keep the old paths unchanged (non-breaking — baseline still runs on SP1024)
- New file `runpod_tests/chore/09_deploy_sp8192.sh` — handles the data deployment on the pod side (HF download OR on-pod tokenization fallback)
- New experiment entry in `experiments.json`: `SP8192_CHAMPION_seed42` with `USE_SP8192=1 USE_GATED_ATTENTION=1 USE_LEGAL_TTT=1 MAX_WALLCLOCK_SECONDS=1500 SKIP_FINAL_EVAL=0`

**Success criterion**: produces `final_int8_zlib_roundtrip val_bpb ≤ 1.30`. Anything in that range confirms SP8192 works end-to-end and becomes the new baseline for subsequent shots. Expected number based on LESSONS §18c: 1.15-1.25.

**Stop condition**: if val_bpb > 1.30, STOP and debug tokenizer loading. Don't proceed to Shot 2.

### Shot 2 — SP8192 champion n=2 (30 min wall-clock, runs parallel to Shot 3 port, $0.15)

Queue `SP8192_CHAMPION_seed1337` with same env. Confirms the Shot 1 number is not a single-seed lucky draw.

### Shot 3 — Port Score-First TTT from PR #461 / #1477 (60-90 min dev, $0.30 pod time)

**Read first**: PR #461 (framework) and PR #1477 (usage). The key code to port lives in PR #461's `train_gpt.py` — it's a test-time gradient procedure different from LEGAL_TTT. Likely structure: for each context batch at eval time, do `k` forward passes with different candidates, score each, pick the best, commit. Different from LEGAL_TTT's "per-batch gradient steps on context/target."

**Code changes**:
- New patcher hunk `SCORE_FIRST_TTT_MARKER` in `08_patch_train_gpt.sh` gated behind `USE_SCORE_FIRST_TTT=1`
- Add env vars: `SCORE_FIRST_TTT_EPOCHS=3` (matches PR #1477 default), `SCORE_FIRST_TTT_LR`, `SCORE_FIRST_TTT_CANDIDATES`
- Keep LEGAL_TTT alongside as a separate flag — we want to ablate both
- New experiment entry: `SP8192_SCORE_FIRST_TTT_seed42` with `USE_SP8192=1 USE_GATED_ATTENTION=1 USE_SCORE_FIRST_TTT=1 USE_LEGAL_TTT=0`

**Success criterion**: val_bpb ≤ 1.15. PR #1477 reports 1.0822 on 3-seed 8×H100. On cheap-pod smaller batch the number should be worse but still clearly below the Shot 1 SP8192-only baseline.

### Shot 4 — Port parallel residuals from PR #1260 / #1412 / #1477 (30-60 min dev, $0.20)

**Read first**: our existing `USE_PARALLEL_RESIDUALS=1` patch in `08_patch_train_gpt.sh` — the SP1024 test of this failed (S2 1.4235 BIG FAIL, demoted). The PR #1477 stack uses `PARALLEL_START_LAYER=7` meaning parallel residuals only kick in from layer 7+. Our patch may have enabled it everywhere; fix to match.

**Code changes**:
- Modify the existing `PARALLEL_RESIDUALS_MARKER` patcher hunk to support `PARALLEL_START_LAYER` env var (default 7)
- New experiment: `SP8192_SCORE_FIRST_TTT_PARALLEL_RESID_seed42` — same as Shot 3 + `USE_PARALLEL_RESIDUALS=1 PARALLEL_START_LAYER=7`

**Success criterion**: val_bpb drops from Shot 3 by 0.003-0.010 (PR #1477 shows ~0.0006 improvement from adding parallel residuals on top of Score-First TTT, but we're at much smaller model so the effect may be slightly bigger or smaller).

### Shot 5 — Port AR self-gen GPTQ from PR #1019 (60-90 min dev, $0.25)

**Read first**: `/tmp/pr1019.diff` — 2588 lines, have the full record including `train_gpt.py`. The calibration loop: after training completes, model generates 64 sequences × 2048 tokens at temp=0.8, fixed seed, then Full-Hessian GPTQ uses those sequences as calibration data. No val data touched.

**Code changes**:
- New patcher hunk `AR_GPTQ_MARKER` gated behind `USE_AR_GPTQ=1`
- New function `calibrate_gptq_ar_selfgen()` in train_gpt.py that runs after training, before compression
- Only affects `final_int8_zlib_roundtrip` path — no training-time impact
- New experiment: `SP8192_SCORE_FIRST_TTT_PARALLEL_RESID_AR_GPTQ_seed42`

**Success criterion**: val_bpb drops 0.003-0.010 vs Shot 4 (per-record claims 0.003-0.005 from AR calibration alone).

### Shot 6 — Int6 GPTQ (30-60 min dev, $0.15)

**Read first**: existing `quantize_state_dict_int8` at `train_gpt.py:342`. Comp records use int6 per-row for matrix weights (keep int8 for tok_emb).

**Code changes**:
- Add `GPTQ_BITS` env var (default 8, set to 6 for this shot)
- Modify `quantize_state_dict_int8` to dispatch on `GPTQ_BITS`
- Per-row int6 quantization with correct dequant path
- New experiment: `SP8192_FULL_STACK_INT6_seed42`

**Success criterion**: artifact bytes drop from ~16.7 MB → ~15.0-15.5 MB (frees ~1.5 MB budget), val_bpb within 0.005 of Shot 5 (int6 noise shouldn't destroy quality).

### Shot 7 — Brotli-11 compression (15 min dev, $0.05)

Cheapest shot. Replace zlib with brotli in the roundtrip path.

**Code changes**:
- `pip install brotli` in bootstrap
- Add `COMPRESSION_METHOD` env var (default `zlib`, set to `brotli` for this shot)
- Modify `final_int8_zlib_roundtrip` to branch on method
- New experiment: `SP8192_FULL_STACK_BROTLI_seed42`

**Success criterion**: artifact bytes drop another 100-300 KB vs Shot 6. val_bpb should be bit-identical (lossless compression).

### Shot 8 — N=2 final target stack confirm (60 min, $0.30)

Take the best combination of Shots 3-7 and run with both seed=42 AND seed=1337. Lock as `TARGET_STACK` in tracker.

Also run **one timed profiling run** on Pod α with `torch.profiler` enabled to record:
- ms/step breakdown by kernel
- Which operators dominate (likely: attention, dequant, n-gram gather)
- CPU idle fraction during training (G6 gate)

These numbers feed directly into Phase 3 kernel planning.

## File deliverables

New files created in Phase 1:
- `PHASE1_PLAN.md` (this file)
- `runpod_tests/chore/09_deploy_sp8192.sh` — SP8192 data deployment on pod
- `runpod_tests/chore/10_port_score_first_ttt.sh` — Score-First TTT port
- `runpod_tests/chore/11_port_ar_gptq.sh` — AR self-gen GPTQ port
- `runpod_tests/chore/12_int6_brotli.sh` — int6 + brotli compression port
- `PHASE1_RESULTS.md` — per-shot log of val_bpb, artifact bytes, ms/step, issues

Modified files:
- `runpod_tests/chore/08_patch_train_gpt.sh` — add SP8192_MARKER, SCORE_FIRST_TTT_MARKER, AR_GPTQ_MARKER, GPTQ_BITS, BROTLI_COMPRESSION markers
- `runpod_tests/loop/experiments.json` — add 8 new experiment entries (Shots 1-8)
- `STACK_NOVELTY_TRACKER.md` — new row `TARGET_STACK` in Section A at the end of Phase 1

## Cost + time (1 × H100 @ ~$2.40/h)

| phase | time | cost |
|---|---|---|
| Pre-flight (done) | 15 min | $0 |
| Pod spin-up + bootstrap + FA3 install | 15-20 min | $0.80 |
| SP8192 data deploy (HF pull or re-tokenize) | 10-15 min | $0.60 |
| Shot 1 SP8192 + current champion stack | 12 min run | $0.48 |
| Shot 2 seed1337 confirm | 12 min run (sequential) | $0.48 |
| Shot 3 Score-First TTT port + run | 30-40 min dev + 12 min run | $2.00 |
| Shot 4 parallel residuals port + run | 20 min dev + 12 min run | $1.28 |
| Shot 5 AR-GPTQ port + run | 40-50 min dev + 12 min run | $2.40 |
| Shot 6 int6 GPTQ port + run | 20 min dev + 12 min run | $1.28 |
| Shot 7 brotli port + run | 10 min dev + 12 min run | $0.88 |
| Shot 8 n=2 final confirm + profiler | 30-40 min | $1.40 |
| Debug buffer | 20-30 min | $1.00 |
| **Phase 1 H100 total** | **~3-4 h wall clock** | **~$12.60 (buffer to $15 hard cap)** |

Stretch goal: **2 hours** if every port goes clean-first-try.

## Stop conditions

1. **SP8192 loading fails** (Shot 1 val_bpb > 1.30): stop, debug tokenizer path before burning more compute
2. **Score-First TTT port regresses** (Shot 3 val_bpb > Shot 1 by > 0.02): stop, re-read PR #461 diff, the port is likely wrong
3. **int6 GPTQ destroys quality** (Shot 6 val_bpb > Shot 5 by > 0.01): revert to int8 for Shot 7, don't block the rest
4. **Pod cost burn exceeds $8**: stop, escalate to user

## What Phase 1 explicitly does NOT include

- ❌ torch.compile (Phase 2)
- ❌ Custom CUDA/Triton kernels (Phase 3)
- ❌ CUDAGraph capture (Phase 3)
- ❌ 8×H100 validation (comp submission path, much later)
- ❌ FP8 work (Phase 3, needs Ada card)
- ❌ New research / literature mining (backlog saturated from last night)
- ❌ Any stack-search beyond porting known comp wins (PD3: only port wins from top-10 records)

## Post-Phase 1 → Phase 2 gate

**Phase 2 = all speed work**, on cheap 3090 / 4070 Ti hardware (NO H100). Phase 2 subsumes what I previously separated as "Phase 2 compile" and "Phase 3 kernels" — they're both speed and should be one phase. Phase 2 can start only once Phase 1 has `TARGET_STACK` n=2 confirmed AND the Phase 1 H100 pod has been terminated.

### Phase 2 shot ordering

Phase 2 Shot 1 is the trivial lever; Shots 2+ are the actual kernel work:

1. **torch.compile re-enable** (~30 min, 1 pod) — the stack currently has `USE_TORCH_COMPILE=0` on every experiment because of an old breakage. Find why, fix it, measure the compile-on vs compile-off ms/step delta on the Phase 1 target stack. Expected 15-35% free throughput.
2. **Persistent CUDAGraph capture** for full forward+backward+opt.step — eliminate per-iter kernel launch overhead.
3. **Fused n-gram bias gather + GQA attention Triton** — the n-gram bias currently materializes an int buffer in a separate kernel. Fuse with the attention logit computation.
4. **GPTQ int6 dequant + matmul fusion** — currently dequantizes weights to fp16 in a separate kernel before matmul. Fuse.
5. **Custom SDPA replacement** tuned for (8h, 4kv, head_dim=64, softcap=30) GQA shape.
6. **Int8 tabulation hash GPU gather** — move CPU-side n-gram table lookup onto GPU.
7. **FP8 compute paths** (requires 4070 Ti or 4090 for Ada) — matmul in FP8, accumulate in bf16.
8. **CPU-side worker pool** — saturate the 8-16 vCPU / 30 GB RAM with parallel n-gram table builds, result analysis, checkpoint compression sweeps.

Mandatory ordering:
- Shot 1 (compile) gates everything else — custom kernels have to beat the compile-optimized path, not the eager path
- Phase 1 gates Phase 2 entirely — kernel work on an unstable target stack gets invalidated every time the target changes
- Phase 2 hardware reverts to cheap 3090/4070 Ti — the H100 rule resumes after Phase 1 terminates
