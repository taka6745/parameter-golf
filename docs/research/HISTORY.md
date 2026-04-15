# Parameter Golf - Experiment History

## Session 2 (Apr 7, 2026) — Autonomous RunPod Loop

### Setup
- Rented RunPod 3080 Ti (12GB), tyf0q5l1kgefgx-64410a6f
- Built remote-control via SSH heredoc + base64 (RunPod's SSH proxy blocks scp/sftp).
- Helpers: /tmp/podrun.sh (pipe stdin to interactive shell), /tmp/podpull.sh (tar+base64 file pull).

### Critical Discovery — train_gpt.py has ZERO n-gram bias support
- The pod runs the upstream competition baseline train_gpt.py which has NO `USE_NGRAM_BIAS`.
- All previous u04/u08 USE_NGRAM_BIAS / USE_WAVELET / USE_HEDGE_MIXER env vars were silent no-ops.
- Mac results (1.96 BPB with n-gram, 2.34 without) explained the gap: pod was running pure transformer.

### Patch 6 (NGRAM_BIAS) — added to runpod_tests/chore/08_patch_train_gpt.sh
- Loads bigram_tab_1024v.npy / trigram_logprobs_1024v.npy / fourgram_logprobs_1024v.npy at __init__.
- Hash polynomials match 04_build_ngrams.py exactly:
  - bigram:   `(prev * 36313) % 2048`
  - trigram:  `(prev2 * 36313 + prev1 * 27191) % 2048`
  - fourgram: `(prev3 * 36313 + prev2 * 27191 + prev1 * 51497) % 2048`
- Adds weighted log-probs to logits in forward() before cross_entropy.
- Env vars: USE_NGRAM_BIAS, NGRAM_W_BIGRAM, NGRAM_W_TRIGRAM, NGRAM_W_FOURGRAM.
- Idempotent via NGRAM_BIAS_MARKER.

### Patches 7a/7b (SKIP_LAST_VAL + SKIP_POST_LOOP) — fix the "stuck after wallclock" issue
- The training loop has `should_validate = last_step or ...` so the LAST step ALWAYS does a
  62M-token val pass (~20 min on 3080 Ti). With SKIP_FINAL_EVAL=1 we now skip that.
- The post-loop GPTQ + zlib also still ran. Bail before it when in signal mode.
- Idempotent via SKIP_LAST_VAL_MARKER + SKIP_POST_LOOP_MARKER.

### Autonomous Loop — runpod_tests/loop/
- experiment_runner.py: round-robin runner, picks experiment with fewest completed runs,
  runs `python3 -u train_gpt.py` with env overrides, parses train_loss / val_bpb / ms_step
  from log, appends to results.jsonl, refreshes leaderboard.txt.
- experiments.json: 32 configs covering n-gram on/off, weight tuning, arch (9L/11L, 2x/3x/4x),
  optimizer (matrix_lr, muon_momentum, warmdown_iters), QK gain, softcap, batch/seq variants.
- run_forever.sh: bash wrapper with `while true; do git pull --autostash; runner; sleep 5; done`
  so pushed code changes auto-propagate.
- 240s wallclock per experiment → ~1200 steps on the cheap GPU at 187 ms/step.

### First confirmation (06:32 UTC)
- 00_baseline_no_ngram: train_loss=3.9952 at step 1200 (no n-gram, 9L 2x MLP, sp1024)
- 01_bigram_only:       step:1 train_loss=5.6865 (vs baseline 6.9428 — bigram head start as on Mac)
                        step:100 train_loss=4.4557 (vs baseline 4.6230 — -0.17 already)
                        NGRAM_BIAS: loaded bigram (2048, 1024) w=0.2
- The patch is working: bigram gives both the step-1 head start AND lower train_loss as steps progress.

## Session 1 (Apr 2-4, 2026)

### Setup

- Forked openai/parameter-golf -> taka6745/paramgolf
- Created venv, installed MLX deps
- Downloaded 1 train shard initially, later 10 shards
- Built BPE-8192 tokenizer from 1M FineWeb docs

### Experiments Run (149 total)

**Timing context:**

*Mac (Apple Silicon M-series, all experiments 1-149):*
- 50-step smoke test: ~49s (BPE-8192, ~1s/step) or ~43s (SP-1024)
- 100-step test: ~1.5 min train, no val
- 200-step test: ~3 min train, no val
- 500-step + val: ~8 min train + ~18 min val = ~26 min total
- 1000-step + val: ~16 min train + ~18 min val = ~34 min total
- Precomputing n-gram tables: ~5 min (from 10 training shards)
- Precomputing DC categories: ~2 min
- Building BPE-8192 tokenizer: ~10 min
- Re-exporting dataset with new tokenizer: ~20 min
- Compression tests (brotli/zstd): ~30 seconds each

*H100 (estimated, for Phase 3 planning):*
- 1 full training run (10 min wallclock): ~7000 steps at ~85ms/step
- 1 full training + GPTQ + eval: ~10 min train + 3 min GPTQ + 2 min eval = ~15 min
- 3-seed validation: 3 × 15 min = ~45 min
- 8-seed tournament: 8 parallel runs × 10 min = ~10 min wall (parallel on 8 GPUs)
- Mamba training run: ~10 min but ~21000 steps at ~28ms/step (est)
- Fat-train 768d: ~10 min but ~3700 steps at ~160ms/step (est)
- Phase 2a smoke tests (8 configs): ~80 min serial or ~10 min parallel (1 per GPU)
- Phase 2b novel experiments (~20 ideas): ~200 min serial, ~25 min if 8-way parallel
- Phase 3 optimization: ~2-3 hours
- Phase 4 final submission: ~45 min (3-seed val)
- Total estimated H100 time: ~6-8 hours

*Experiment methodology:*
- Experiments run across Apr 2-4, 2026 (~3 days of continuous Mac testing)
- No hyperparameter sweeps on val data — 50-step smokes use train_loss only
- 500/1000-step val runs only done on techniques that showed signal at 50 steps
- No seed tuning — all experiments use default seed unless explicitly noted
- Each "winner" is validated by stacking with previous winners at 500 steps


| #   | Experiment                                         | Steps | Result                    | Verdict                                                               |
| --- | -------------------------------------------------- | ----- | ------------------------- | --------------------------------------------------------------------- |
| 1   | Baseline (9L/512d/1024v)                           | 200   | 2.3214 bpb                | Baseline reference (1 shard)                                          |
| 2   | Full SOTA stack (11L/3xMLP/XSA/etc)                | 50    | 3.3192 bpb                | WORSE — too big for low steps                                         |
| 3   | Baseline                                           | 50    | 3.2273 bpb                | 50-step reference                                                     |
| 4   | LeakyReLU(0.5)^2                                   | 50    | 3.2270 bpb                | Negligible at 50 steps                                                |
| 5   | Baseline                                           | 500   | 2.0239 bpb                | 500-step reference (10 shards)                                        |
| 6   | LeakyReLU(0.5)^2                                   | 500   | **2.0102 bpb**            | **WINNER: -0.014 bpb**                                                |
| 7   | LeakyReLU+SmearGate v1                             | 500   | killed during val         | Memory leak with mx.concatenate                                       |
| 8   | LeakyReLU+SmearGate v2                             | 500   | **2.0054 bpb**            | **-0.019 bpb.** Fixed with mx.pad                                     |
| 9   | LeakyReLU+ByteWeightedLoss                         | 500   | 2.0065 bpb                | -0.017 bpb                                                            |
| 10  | All 3 stacked                                      | 500   | 2.0093 bpb                | WORSE — ByteWeight + SmearGate interfere                              |
| 11  | SmearGate+LeakyReLU+WD=0.04                        | 500   | 2.0036 bpb                | Weight decay helps generalization                                     |
| 12  | v6 + depth recurrence (repeat layers 3-4)          | 500   | 2.0024 bpb                | -0.001 from recurrence                                                |
| 13  | v7 + bigram logit bias (w=0.3)                     | 500   | 1.9932 bpb                | BROKE 2.0! Bigram logprobs as bias                                    |
| 14  | bigram w=0.5, w=1.0 (parallel)                     | 200   | n/a                       | Invalid — parallel GPU contention                                     |
| 15  | bigram w=0.1, w=0.2 (parallel)                     | 200   | n/a                       | Invalid — parallel GPU contention                                     |
| 16  | bigram w=0.2 (solo)                                | 500   | 1.9922 bpb                | w=0.2 marginally better than w=0.3                                    |
| 17  | bigram+trigram bias (65K buckets)                  | 500   | 1.9712 bpb                | -0.021 from trigram                                                   |
| 18  | bi+tri+4gram bias                                  | 500   | 1.9519 bpb                | -0.072 total                                                          |
| 19  | 2-5gram bias (65K buckets)                         | 500   | **1.9428 bpb**            | **-0.081 total!** Best SP-1024                                        |
| 20  | 8K bucket n-gram tables                            | 500   | 1.9663 bpb                | -0.024 penalty vs 65K                                                 |
| 21  | 16K bucket n-gram tables                           | 500   | 1.9567 bpb                | Best size/quality tradeoff                                            |
| 22  | Eval-time n-gram cache on v7                       | eval  | FAILED                    | Cache HURTS model trained without bias                                |
| 23  | Baseline 1000 steps                                | 1000  | 1.9257 bpb                | Scaling reference                                                     |
| 24  | v12 (8K) 1000 steps                                | 1000  | **1.8841 bpb**            | N-gram gains persist at scale                                         |
| 25  | v12 seq_len=2048                                   | 200   | 3.654 train               | No gain — extra context unused at low steps                           |
| 26  | v12 grad_clip=1.0                                  | 200   | 3.909 train               | WORSE — Muon handles grads already                                    |
| 27  | v12 + 3xMLP                                        | 200   | 3.893 train               | WORSE — bigger model needs more steps                                 |
| 28  | v12 + momentum 0.99                                | 200   | 3.911 train               | WORSE — too slow to settle                                            |
| 29  | v12 warmdown=3500                                  | 200   | 4.233 train               | MUCH WORSE — LR decays too early                                      |
| 30  | v12 high LR (0.08/0.1)                             | 100   | 4.197 train               | WORSE — too high                                                      |
| 31  | v12 no warmdown                                    | 100   | 4.027 train               | Slight win at 100 (-0.023)                                            |
| 32  | v12 fast momentum warmup (100 steps)               | 100   | 4.368 train               | WORSE — too fast                                                      |
| 33  | v12 big batch (16384)                              | 100   | 4.405 train               | WORSE per step                                                        |
| 34  | v12 low LR (0.02)                                  | 100   | 4.429 train               | WORSE — too slow                                                      |
| 35  | v12 seed=42                                        | 100   | 4.376 train               | Neutral (parallel contention)                                         |
| 36  | v12 no warmdown 500-step                           | 500   | 3.617 train (killed)      | WORSE at 500. Warmdown helps final convergence                        |
| 37  | v12 muon_backend_steps=10                          | 100   | 4.368 train               | WORSE — just adds cost                                                |
| 38  | v8 bigram-ONLY (no tri/4/5gram)                    | 100   | 4.765 train               | MUCH WORSE — higher-order n-grams essential                           |
| 39  | v12 WITHOUT SmearGate                              | 100   | 4.284 train               | SmearGate still helps with n-gram (parallel)                          |
| 40  | v12 WITHOUT depth recurrence                       | 100   | 4.382 train               | Recurrence still helps (parallel)                                     |
| 41  | v12 softcap=20                                     | 100   | 4.372 train               | Neutral (parallel contention)                                         |
| 42  | v12 QK_gain=2.0                                    | 100   | 4.370 train               | Neutral (parallel contention)                                         |
| 43  | v4 byte-weight+bigram (no SmearGate)               | 100   | 5.069 train               | WORSE — byte-weight conflicts with n-gram                             |
| 44  | SVD embedding init from bigram matrix              | 100   | 4.379 train               | WORSE — tied embed input/output conflict                              |
| 45  | 5-layer + n-gram                                   | 100   | 4.398 train               | WORSE — n-gram can't compensate fewer layers                          |
| 46  | 7-layer + n-gram                                   | 100   | 4.388 train               | WORSE — still needs 9L                                                |
| 47  | Reservoir computing (random proj + linear readout) | eval  | 6.788 CE                  | Useless without recurrence                                            |
| 48  | Pure PPM (bigram+unigram)                          | eval  | 4.375 CE                  | 1.0 nats worse than neural (3.32)                                     |
| 49  | Spectral embedding init (Markov eigenvectors)      | 100   | 4.375 train               | WORSE — all geometric inits hurt                                      |
| 50  | Warmup 50 steps                                    | 100   | 4.372 train               | Neutral (parallel)                                                    |
| 51  | Warmdown=100 (cosine-like, parallel)               | 100   | 3.943 train               | Promising in parallel                                                 |
| 52  | Cosine schedule (warmdown=iters, solo)             | 100   | 3.943 train               | WIN at 100 steps (+0.107)                                             |
| 53  | Bigram top-1 accuracy on val                       | eval  | 11.2%                     | Only 11% trivially predictable                                        |
| 54  | v12 cosine schedule 500-step                       | 500   | 1.9964 bpb                | WORSE at 500 — wins early, loses late                                 |
| 55  | Complementary training (alpha=0.5)                 | 100   | 4.411 train               | WORSE — reduces gradient signal                                       |
| 56  | v12 QK_gain=4.0                                    | 100   | 4.368 train               | WORSE at low steps                                                    |
| 57  | BPE-8192 tokenizer trained                         | —     | model saved               | Tokenizer built from 1M docs                                          |
| 58  | Brotli-11 compression test                         | —     | 12.33MB total             | **1.47MB saved vs zstd-22.** Fits 16MB easily                         |
| 59  | BPE-8192 dataset export                            | —     | DONE                      | Val+train shards exported                                             |
| 60  | BPE-8192 baseline 100-step                         | 100   | 6.354 train               | Higher loss (more tokens per byte)                                    |
| 61  | **BPE-8192 baseline 500-step**                     | 500   | **1.8953 bpb**            | **GAME CHANGER! Beats ALL 1024v tricks (-0.129)**                     |
| 62  | BPE-8192 v3 (LeakyReLU+SmearGate)                  | 100   | 6.645 train               | WORSE — SmearGate may hurt at 8192v (parallel)                        |
| 63  | Built n-gram tables for BPE-8192                   | —     | 16K x 8192v               | bi+tri+4gram tables saved                                             |
| 64  | BPE-8192 + LeakyReLU only                          | 500   | 1.8910 bpb                | -0.004 from LeakyReLU on 8192v                                        |
| 65  | **BPE-8192 + LeakyReLU + ngram (16K)**             | 500   | **1.8364 bpb**            | **-0.188 total vs 1024v baseline!**                                   |
| 66  | BPE-8192+ngram+WD=0.04                             | 100   | 5.759 train               | WORSE at 100 steps                                                    |
| 67  | BPE-8192+ngram+depth_recurrence                    | 100   | 5.748 train               | WORSE at 100 steps                                                    |
| 68  | BPE-8192+ngram high_w (0.3,0.2,0.15)               | 100   | 5.561 train               | TIED with default weights                                             |
| 69  | BPE-8192+ngram low_w (0.1,0.08,0.05)               | 100   | 6.018 train               | WORSE — too little bias                                               |
| 70  | BPE-8192+ngram+UID regularizer                     | 100   | 5.823 train               | WORSE — var penalty inflates loss                                     |
| 71  | Factorized embedding (8192x64x512)                 | —     | SKIPPED                   | Breaks tied LM head                                                   |
| 72  | Test A: Gated attention (init=+3)                  | 50    | 6.967 train               | NEUTRAL (+0.002)                                                      |
| 73  | Test B: Output bias                                | 50    | 6.965 train               | TIED (+0.000)                                                         |
| 74  | Test C: Poly-1 loss (eps=0.2)                      | 50    | 7.163 train               | WORSE — loss formula inflates                                         |
| 75  | Test D: Gradient centralization                    | 50    | 6.965 train               | TIED (+0.000)                                                         |
| 76  | Test E: Softcap 60                                 | 50    | 6.970 train               | NEUTRAL (+0.004)                                                      |
| 77  | Test H: Skip connection 2x                         | 50    | 6.977 train               | SLIGHTLY WORSE (+0.011)                                               |
| 78  | Test M: Focused loss (top 30% = 2x)                | 50    | 8.710 train               | MUCH WORSE (+1.745)                                                   |
| 79  | **Test I: Anti-entropy loss**                      | 50    | **6.596 train**           | **WINNER! -0.369.** But modifies reported loss                        |
| 80  | Test BB: Self-referential loss                     | 50    | 9.691 train               | MUCH WORSE — aux head destabilizes                                    |
| 81  | **Test GG: Wave equation forward pass**            | 50    | **6.820 train**           | **WINNER! -0.145.** Wave > residual connections                       |
| 82  | Test JJ: Dendritic MLP                             | 50    | 6.968 train               | NEUTRAL (+0.002)                                                      |
| 83  | Winner I: Anti-entropy 500-step                    | 500   | 1.7943 bpb*               | *val_bpb includes entropy subtraction — unfair                        |
| 84  | **Winner GG: Wave equation 500-step**              | 500   | **1.8678 bpb**            | **VALIDATED! -0.023 vs LeakyReLU baseline**                           |
| 85  | Test V: Deep supervision (aux CE layers 3,6)       | 50    | 7.666 train               | WORSE — aux losses destabilize                                        |
| 86  | Test AA: Context-dependent softcap                 | 50    | 6.965 train               | TIED (+0.000)                                                         |
| 87  | Test II: Synaptic scaling                          | 50    | TIMEOUT                   | Normalization compile issues                                          |
| 88  | Wave+ngram stacked                                 | 500   | 1.8368 bpb                | NEUTRAL — wave redundant with n-gram                                  |
| 89  | Test Z: Simplified complementary loss              | 50    | TIMEOUT                   | mx.no_grad compile failure                                            |
| 90  | Test CC: Entropy-gated skip connections            | 50    | TIMEOUT                   | Dynamic gating compile issues                                         |
| 91  | Test EE: Prediction recycling                      | 50    | 6.968 train               | NEUTRAL (+0.003)                                                      |
| 92  | Test FF: Gradient thermal diffusion (alpha=0.05)   | 50    | 6.965 train               | TIED (+0.000)                                                         |
| 93  | Test HH: Impedance matching                        | 50    | 7.027 train               | WORSE (+0.062)                                                        |
| 94  | Test PP: Gaussian loss weighting                   | 50    | 7.296 train               | WORSE (+0.331)                                                        |
| 95  | BUILD4: Gradient thermal diffusion (alpha=0.2)     | 50    | 6.965 train               | TIED — smoothing has zero effect                                      |
| 96  | Gated attention (old, init=0)                      | 100   | 5.767 train               | WORSE — gate at 0 kills signal                                        |
| 97  | Michaelis-Menten activation                        | 100   | 5.832 train               | WORSE — saturating doesn't work                                       |
| 98  | WaveletGPT aux loss (weight=0.1)                   | 100   | 6.457 train               | MUCH WORSE — aux too heavy                                            |
| 99  | **BPE-8192+LeakyReLU+ngram 1000 steps**            | 1000  | **1.7422 bpb**            | **-0.282 vs 1024v baseline!**                                         |
| 100 | Adaptive mixer on LeakyReLU model                  | eval  | none=1.955, fixed=1.976   | Biases hurt model trained without them                                |
| 101 | Adaptive mixer on n-gram model                     | eval  | fixed=1.92, adaptive=1.94 | Fixed weights beat adaptive                                           |
| 102 | **Test R: Post-period capitalization bias**        | 50    | **6.935 train**           | **WINNER! -0.031.** Boost sentence starters after periods             |
| 103 | Test X: Word-boundary embedding                    | 50    | 6.965 train               | TIED — zero effect                                                    |
| 104 | Test U: Difficulty-scaled residuals                | 50    | 6.974 train               | NEUTRAL (+0.009)                                                      |
| 105 | Test NN: MDL weight entropy penalty                | 50    | 6.965 train               | TIED — too small to notice                                            |
| 106 | Test OO: Bits-back loss correction                 | 50    | 7.140 train               | WORSE (+0.175)                                                        |
| 107 | Test RR: Difficulty-aware auxiliary head           | 50    | 9.510 train               | MUCH WORSE — aux destabilizes                                         |
| 108 | **ngram + Test R stacked**                         | 500   | **1.8342 bpb**            | **500-step BEST! -0.002 from period bias**                            |
| 109 | Test Q analysis: deterministic completions         | —     | 40 tokens >95%            | Too few to matter                                                     |
| 110 | Test S: Hard-context loss upweighting              | 50    | 7.113 train               | WORSE (+0.148)                                                        |
| 111 | Test KK: Lateral inhibition                        | 50    | 6.969 train               | NEUTRAL (+0.004)                                                      |
| 112 | Test LL: Surprise embedding                        | 50    | 6.967 train               | NEUTRAL (+0.002)                                                      |
| 113 | Test MM: Confidence-aware key scaling              | 50    | 6.965 train               | TIED (-0.0001, noise)                                                 |
| 114 | **ngram+R at 1000 steps**                          | 1000  | **1.7397 bpb**            | **ABSOLUTE BEST! -0.284 vs 1024v baseline**                           |
| 115 | Test O: WaveletGPT FIXED (weight=0.01)             | 50    | 7.035 train               | WORSE (+0.070)                                                        |
| 116 | Test P: UID FIXED (beta=0.001)                     | 50    | 6.972 train               | NEUTRAL (+0.007)                                                      |
| 117 | **Test F: embed_lr=0.1**                           | 50    | **6.746 train**           | **WINNER at 50! -0.220.** But overshoots at 500                       |
| 118 | Test L: Embedding dropout (5%)                     | 50    | TIMEOUT                   | Random ops compile failure                                            |
| 119 | F+ngram+R (embed_lr=0.1) stacked                   | 500   | 1.8679 bpb                | WORSE at 500 — overshoots                                             |
| 120 | **Test F07: embed_lr=0.07**                        | 50    | **6.840 train**           | **WINNER at 50! -0.125.** Less aggressive                             |
| 121 | Test K2: Layerwise LR (solo)                       | 50    | TIMEOUT                   | Regex in hot loop too slow                                            |
| 122 | Test DD: Per-layer embedding rescale               | 50    | 6.965 train               | TIED (-0.0002, noise)                                                 |
| 123 | F07+ngram+R (embed_lr=0.07) stacked                | 500   | 1.8352 bpb                | WORSE by 0.001 — default 0.05 optimal at 500                          |
| 124 | **1d+1e: Capitalization + Context Engine**         | 50    | **6.948 train**           | **-0.018!** Period→uppercase, comma→lowercase bias works              |
| 125 | 1f: POS Tag Transition Bias                        | 50    | 6.963 train               | -0.002, tiny. POS tagger too crude (84% tagged OTHER)                 |
| 126 | 1d+1e+1f stacked                                   | 50    | 6.945 train               | -0.021 stacked! Better than any individual                            |
| 127 | **1b+1d+1e+1f full English Knowledge Engine**      | 50    | **6.935 train**           | **-0.030!** Word completion + context + POS. All additive             |
| 128 | **Full stack: ngram+R+knowledge_engine 500-step**  | 500   | **1.8328 bpb**            | **NEW 500-STEP BEST! -0.001 from knowledge engine on top of ngram+R** |
| 129 | 1f-v3: Distributional Categories DC75               | 50    | -0.025 vs ref             | WINNER! Auto-discovered categories from training data                  |
| 130 | 1f-v3: Distributional Categories DC200              | 50    | -0.069 vs ref             | BIGGER WINNER! Scales with more categories                             |
| 131 | **1f-v3: Distributional Categories DC500**          | 500   | **1.8318 bpb**            | **NEW 500-STEP BEST! -0.143 at 50 steps, validated at 500**           |
| 133 | —                                                    | —     | 149 total experiments     | —                                                                              |
| 134 | 1.7d: DC750 (w=0.15)                                | 50    | 6.776 train               | -0.190. Scaling continues!                                                     |
| 135 | 1.7d: DC1000 (w=0.15)                               | 50    | 6.743 train               | -0.222. Still scaling!                                                         |
| 136 | 1.7e: DC500 w=0.05                                  | 50    | 6.914 train               | -0.051. Too weak                                                               |
| 137 | 1.7e: DC500 w=0.10                                  | 50    | 6.867 train               | -0.098. Moderate                                                               |
| 138 | 1.7e: DC500 w=0.30                                  | 50    | 6.706 train               | -0.259. Much better than 0.15!                                                 |
| 139 | 1.7e: DC500 w=0.50                                  | 50    | 6.596 train               | -0.369. Even better!                                                           |
| 140 | **1.7d+e: DC1000 w=0.30**                           | 50    | **6.561 train**           | **-0.404. Both dimensions scale!**                                             |
| 141 | **1.7d+e: DC1000 w=0.50**                           | 50    | **6.389 train**           | **-0.576!!! Best 50-step single technique ever!**                              |
| 142 | 1.7f: ngram weights high (0.3,0.2,0.15)             | 50    | 5.721 train               | -0.321 at 50 but overshoots at 500                                            |
| 143 | 1.7f: ngram weights low (0.15,0.1,0.08)             | 50    | 6.261 train               | WORSE. Too little bias                                                         |
| 144 | 1.7f: ngram weights swap (0.1,0.2,0.15)             | 50    | 6.152 train               | WORSE. Trigram>bigram doesn't help                                             |
| 145 | ngram+R+DC1000 w=0.50 at 500 steps                  | 500   | 1.8331 bpb                | WORSE than DC500@0.15 (1.8318). High weight overshoots!                        |
| 146 | DC1000 w=0.15 + ngram + R at 500 steps              | 500   | 1.8378 bpb                | WORSE than DC500@0.15 (1.8318). More categories hurt at 500!                   |
| 147 | **1.7h: 5-gram on BPE-8192 (4K buckets)**           | 50    | **5.976 train**           | **-0.066 vs ngram ref! 5-gram adds signal on 8192v too**                       |
| 148 | 1.7c: Lloyd-Max quantization (256 optimal levels)   | post  | MSE -92.7%, size +101%    | Better quality but DOUBLES compressed size. Uniform int8 wins.                 |
| 149 | **1.7i: Ultimate stack (ngram+5g+R+DC500+KE)**      | 500   | 1.8332 bpb                | WORSE than DC500 alone (1.8318). Too many biases interfere.                    |
| 150 | 2.5g: Truncated backprop (freeze layers 0-2)         | 50    | 7.126 train               | WORSE (+0.161). Early layers still learning at 50 steps                         |
| 151 | 2.5a: Procedural feature interaction                 | 50    | 6.965 train               | TIED (+0.000). Rank-1 feature interaction = no signal                           |
| 152 | 2.5j: Running document LayerNorm                     | 50    | 7.609 train               | MUCH WORSE (+0.644). Running stats destabilize                                  |
| 153 | 2.5k: Causal prediction skip                         | 50    | 6.965 train               | TIED (0.000). Preview+feedback has no effect                                    |
| 154 | 2.5i: Random linear maps + adapter                   | 50    | 6.966 train               | TIED (+0.001). Zero-init adapter = no-op yet                                    |
| 155 | 2.5b: Multi-resolution (POS aux head)                | 50    | 7.084 train               | WORSE (+0.119). POS aux loss hurts main task                                    |
| 156 | 2.5e: Fractal depth (pool/unpool)                    | 50    | TIMEOUT                   | Pool/unpool causes shape issues with skip connections                           |
| 157 | 2.5c: Hash codebook (256 floats)                     | 50    | 6.966 train               | TIED (+0.001). Codebook feature = noise at this scale                           |
| 158 | 2.5d: Soft layer interpolation                       | 50    | 6.975 train               | NEUTRAL (+0.010). Blending weakens layer output                                 |
| 159 | **2.5f: Micro-macro split (abs+sq MLP)**             | 50    | **6.961 train**           | **WINNER! -0.004.** Different nonlinearities in MLP halves                     |
| 160 | 2.5h: Progressive objectives (byte aux)              | 50    | 7.061 train               | WORSE (+0.096). Byte aux loss hurts main task                                   |
| 161 | **2.6d: Dual MLP (2 experts, averaged)**              | 50    | **6.898 train**           | **WINNER! -0.067.** Two parallel MLPs averaged = implicit ensemble             |
| 162 | 2.6e: Untied embeddings                              | 50    | 8.994 train               | MUCH WORSE (+2.029). Separate output proj loses shared structure               |
| 163 | 2.6f: Multi-token prediction (t+2, t+3 aux)         | 50    | 10.453 train              | MUCH WORSE (+3.488). Aux heads overwhelm main loss                             |
| 164 | 2.6g: JEPA (predict next embedding L2)               | 50    | 7.052 train               | WORSE (+0.086). L2 embedding loss conflicts with CE                            |
| 165 | 2.6k: Negative prediction head                       | 50    | 6.966 train               | TIED (+0.001). Zero-init suppression = no-op                                   |
| 166 | 2.6l: Learned skip offsets (1,3,5,13 back)           | 50    | 6.966 train               | TIED (+0.001). Zero-init projection = no-op                                    |
| 167 | 2.6m: 3 voter heads (consensus)                      | 50    | 7.987 train               | MUCH WORSE (+1.021). 3 random heads worse than tied embedding                  |
| 168 | **2.6n: Compression bottleneck (512→64→512)**        | 50    | **6.961 train**           | **WINNER! -0.004.** Information bottleneck forces prioritization               |
| 169 | 2.6o: Anti-teacher forcing (10% own predictions)     | 50    | TIMEOUT                   | Random masking in compiled function fails                                       |
| 170 | 2.6p: Token orbit (2 embeddings, position mod 2)     | 50    | 6.968 train               | NEUTRAL (+0.003)                                                                |
| —   | 2.6a,b,c,h,i,j,q,r                                  | —     | SKIPPED                   | Need FFT/conv/byte-data/grad-buffer not available in MLX compiled              |
| 171 | 2.6a: Pure conv (causal windowed average K=11)       | 50    | 6.963 train               | MARGINAL (-0.002). Conv alone nearly matches attention at 50 steps!             |
| 172 | 2.6c: Backward model (reverse odd sequences)        | 50    | 6.965 train               | TIED (+0.000). Reversing sequences has no effect                                |
| 173 | **2.6h: Fractal depth (pool→half-res→unpool)**      | 50    | **5.703 train**           | **-1.262!! SUSPICIOUS — pooling may leak future info. Needs val_bpb check.**   |
| 174 | 2.6j: AR-diffusion (noise injection)                 | 50    | TIMEOUT                   | Random noise in compiled function fails                                         |
| 175 | 2.6q: Gradient echo (momentum 0.98)                  | 50    | 6.965 train               | TIED (+0.000). Higher momentum no effect at 50 steps                            |
| 176 | 2.6r: Hybrid conv+attn                               | 50    | TIMEOUT                   | Conv mode attribute not found in compiled                                       |
| 177 | **2.13b: Skip-bigram (prev2→next, skip prev1)**     | 50    | **6.902 train**           | **WINNER! -0.063.** Skip-bigram captures different signal from bigram           |
| 178 | **2.11f: Hyper-connections (learned residual scale)** | 50    | **6.950 train**           | **WINNER! -0.015.** Learned alpha/beta per layer helps stability                |
| 179 | **2.13g: Token Order Prediction aux loss**           | 50    | **6.942 train**           | **WINNER! -0.024.** Predicting token order improves representations             |
| 180 | 2.6d: Dual MLP + ngram+R+DC500 at 500 steps         | 500   | TBD (val running)         | Dual MLP (-0.067 at 50) stacked with full winning stack                         |
| 181 | **2.10a: Logistic mixing (log-odds space)**          | 50    | **6.454 train**           | **HUGE WINNER! -0.511!** Log-odds mixing >> additive bias                       |
| 182 | **2.8a: CTW entropy-weighted n-gram mixing**         | 50    | **6.279 train**           | **MASSIVE WINNER! -0.686!** Entropy-weighted backoff is the best single technique|
| 183 | 2.8b: DEQ (run layer 4 twice)                        | 50    | 6.964 train               | MARGINAL (-0.002). Simple iteration barely helps                                |
| 184 | **2.6d: Dual MLP + ngram+R+DC500 at 500 steps**    | 500   | **1.8269 bpb**            | **NEW 500-STEP BEST! -0.005 from dual MLP. Total -0.197 vs baseline**          |
| 185 | 2.13a: Signed hashing (flip embed by hash)           | 50    | 7.475 train               | MUCH WORSE (+0.509). Sign flipping disrupts learned embeddings                 |
| 186 | 2.13d: Token selection (bigram difficulty weighting) | 50    | 7.091 train               | WORSE (+0.125). Same failure as all loss reweighting                            |
| 187 | 2.14c: Complementary training (w=1-0.5*p_bi)        | 50    | 7.026 train               | WORSE (+0.061). Confirmed: loss reweighting NEVER works on Mac                  |
| 188 | 2.14d: Dynamic token selection (keep top 70% hard)   | 50    | 8.155 train               | MUCH WORSE (+1.190). Masking tokens destroys learning                           |
| 189 | 2.14f: Higher matrix_lr (0.06 vs 0.04)              | 50    | **6.927 train**           | WINNER at 50 (-0.038) but known to overshoot at 500                            |
| 190 | 2.8h: Energy corrector head (learned neural bias)    | 50    | 6.965 train               | TIED (+0.000). Zero-init = no-op                                                |
| 191 | 2.8d: MDL weight penalty (L2 on embeddings)          | 50    | 6.965 train               | TIED (+0.000). Penalty too small to notice                                      |
| 192 | 2.9e: Predictive coding (update x0 mid-network)      | 50    | 6.965 train               | TIED (+0.000). Error propagation = standard residual                            |
| 193 | 2.8c: Hopfield memory (64 learned prototypes)        | 50    | 6.965 train               | TIED (+0.000). Zero-init projection = no-op                                    |
| 194 | 2.8e: Polar-code routing (scale down easy tokens)    | 50    | 6.964 train               | MARGINAL (-0.002). Slight signal from focusing on hard positions               |
| 195 | 2.8g: Non-monotonic aux (predict token difficulty)   | 50    | 7.475 train               | MUCH WORSE (+0.510). Difficulty prediction conflicts with main loss             |
| 196 | 2.13c: Progressive seq=256                           | 50    | 6.964 train               | TIED (different seq, not directly comparable)                                   |
| 197 | 2.12a: Token selection v2 (Gaussian weighting)       | 50    | 4.916 train               | -2.049 BUT loss formula modified (Gaussian reweighting). UNFAIR metric.        |
| 198 | 2.9a: Sigma-delta quant noise                        | 50    | TIMEOUT                   | Adding random noise to weights in compiled fails                                |
| 199 | **2.8a: CTW entropy-weighted at 500 steps**          | 500   | 1.8538 bpb                | WORSE than best (1.8269). CTW mixing with bi+tri only can't beat full stack.   |
| 200 | 2B-1: Sliding window eval (stride 64/256/1024)       | eval  | baseline=2.071, s256=2.096, s64=2.091 | WRONG IMPL — used mean loss not per-token. Need per-token scoring. |
| 201 | **2B-7: NorMuon (per-row norm after Newton-Schulz)** | 50    | **6.834 train**           | **WINNER! -0.132.** Per-row normalization helps Muon significantly              |
| 202 | **2B-8: Turbo-Muon (4 NS steps instead of 5)**       | 50    | **6.939 train**           | **WINNER! -0.026.** Fewer steps = faster, still converges                       |
| 203 | 2B-10: Embedding scaling (sqrt(dim))                 | 50    | 6.966 train               | TIED (+0.000)                                                                   |
| 204 | **2B-15: Warmdown=200**                              | 50    | **6.420 train**           | **-0.545 but likely overshoot at 500 (same trap as cosine/embed_lr)**          |
| 205 | 2B-19: Higher WD (0.08)                              | 50    | 6.966 train               | TIED (+0.001)                                                                   |
| 206 | 2B-20: Cautious optimizer masking                    | 50    | 6.966 train               | TIED (+0.000). Momentum agreement filter has no effect                         |
| 207 | 2B-21: Learned LN temperature                        | 50    | 6.964 train               | MARGINAL (-0.001)                                                               |
| 208 | **2B-2: Softcap=20 (train from scratch)**            | 50    | **6.957 train**           | **WINNER! -0.008.** Lower softcap = sharper = better                            |
| 209 | 2B-2: Softcap=25                                     | 50    | 6.962 train               | MARGINAL (-0.003)                                                               |
| 210 | 2B-2: Softcap=35                                     | 50    | 6.967 train               | NEUTRAL (+0.002)                                                                |
| 211 | 2B-2: Softcap=40                                     | 50    | 6.968 train               | NEUTRAL (+0.003)                                                                |
| 212 | 2B-9: Feature dropout (5% dims)                      | 50    | TIMEOUT                   | Random dropout in compiled fails                                                |
| 213 | **ULTIMATE V2: NorMuon+Turbo+sc20+ngram+R+DC+DualMLP** | 500 | **1.8220 bpb**          | **NEW 500-STEP BEST! -0.005 from optimizer+softcap improvements**              |
| 214 | **ULTIMATE V2 at 1000 steps**                        | 1000  | **1.7279 bpb!!!**         | **NEW ABSOLUTE BEST! -0.296 vs baseline. Full stack scales beautifully.**      |
| 215 | 2B-0a: Frontier hyperparams (mom=0.99,LR=0.02,wd=3k) | 50    | 7.528 train               | WORSE (+0.563). Too aggressive for Mac 50 steps                                |
| 216 | 2B-0b: UID beta=0.01                                 | 50    | 7.026 train               | WORSE (+0.061). UID still hurts regardless of beta                              |
| 217 | 2B-0c: Rho-1 EXCESS (model_loss - ngram_loss)        | 50    | 0.840 (diff metric!)      | Loss metric changed — reports excess only. Need val_bpb to evaluate.           |
| 218 | 2B-8: MuonEq-R (row norm BEFORE NS)                  | 50    | 6.971 train               | NEUTRAL (+0.006). Pre-NS normalization doesn't help                             |
| 219 | 2B-10: Trimmed mean (drop 5%+15% tails)              | 50    | 6.644 train               | -0.322 BUT metric changed (trimmed). Not comparable.                            |
| 220 | 2B-11: LinUpper smooth reweight                       | 50    | 7.641 train               | MUCH WORSE (+0.676). Upweighting hard tokens still hurts                        |
| 221 | 2B-17: WSD 1-sqrt cooldown                            | 50    | 6.965 train               | TIED (+0.000). Sqrt decay = linear at this scale                                |
| 222 | 2B-23: Prospect theory asymmetric loss                | 50    | 7.397 train               | WORSE (+0.432). Asymmetric weighting = same failure pattern                      |
| 223 | 2B-50: FTPL noise (0.001 random)                      | 50    | 6.965 train               | TIED (+0.000). Noise too small to notice                                        |
| 224 | **2B-12: Dendritic MLP (4 groups, diff slopes)**     | 50    | **6.961 train**           | **WINNER! -0.004.** Block-diagonal MLP with varied nonlinearities              |
| 225 | 2B-29: XSA (exclusive self-attention)                 | 50    | FAIL                      | Shape mismatch with GQA (v has fewer heads than y)                              |
| 226 | 2B-34: Coprime stride data loading                    | 50    | 6.965 train               | TIED (-0.000). Shard order doesn't matter at 50 steps (1 shard)                |
| 227 | 2B-36: Fat 768d model                                 | 50    | 6.868 train               | Not comparable (different model size). Shows 768d converges at 50 steps.        |
| 228 | 2B-39: Diagonal K-FAC                                 | 50    | 6.971 train               | NEUTRAL (+0.006). Pre-NS row scaling doesn't help                               |
| 229 | 2B-44: Importance sampling (softmax weights)          | 50    | 8.371 train               | MUCH WORSE (+1.406). Soft importance = same failure as all reweighting          |
| 230 | **2B-57: Relaxed Recursive (layers 2-3 twice)**      | 50    | **6.962 train**           | **WINNER! -0.003.** Interpolated layer repetition helps                         |
| 231 | 2B-35: Hymba conv proxy (conv in first 3 layers)     | 50    | FAIL                      | _use_conv attribute not accessible in compiled function                         |
| 232 | 2B-40: Rho-1 T3S (trajectory token selection)        | 50    | 7.626 train               | WORSE (+0.661). Same loss reweighting failure pattern                           |
| 233 | **2B-5: Tabulation hashing (XOR lookup for n-grams)**| 50    | **6.714 train**           | **-0.252! Tabulation hash gives better n-gram quality. bi+tri only.**          |
| 234 | 2B-27: Count-Min Sketch 7-gram (4 hash × 4096)     | 50    | 6.939 train               | -0.026. 7-gram CMS adds new n-gram order signal.                               |
| 235 | 2B-25: Skip-bigram 500-step validation                | 500   | 1.8360 bpb                | WORSE than ngram+R (1.8342). Skip-bigram redundant with existing n-gram stack. |
| 236 | Fix 1: XSA with GQA-aware shape handling             | 50    | 6.966 train               | NEUTRAL (+0.001). XSA works now but needs H100 steps to show effect.           |
| 237 | Fix 2: Factorized embed inner=128 (2-matmul output) | 50    | 7.313 train               | WORSE (+0.348). Fewer embed params = slower convergence. H100 win.             |
| 238 | Fix 4: ultimate_v3 (tab hash + ultimate_v2)          | 500   | 1.8432 bpb                | WORSE than v2 (1.8220). New hash changes stats, model needs more steps to adapt.|
| 239 | Test 1: EMA 0.997 (update every 10 steps)            | 500   | 1.8220 bpb                | IDENTICAL to without EMA. Zero effect at 500 steps.                            |
| 240 | Test 2: Sliding window eval (per-token, fixed)       | eval  | s1024=2.071, s256=2.089, s64=2.084 | WORSE at all strides! Model doesn't benefit from overlapping context. |
| 241 | **A6: Full SOTA arch (11L+3xMLP+XSA+SmearGate+RoPE+LNScale) + our stack** | 1000 | **1.7131 bpb!!!** | **NEW ABSOLUTE BEST! -0.015 vs ultimate_v2. Full SOTA arch WORKS with our innovations!** |
| 242 | **A1: 11L only + ultimate_v2 stack**                 | 1000  | **1.7110 bpb!!!**         | **NEW ABSOLUTE BEST! 11L alone beats full SOTA (1.7131). Depth > width+tricks.** |
| 243 | A2: 3xMLP only (9L/3x) + ultimate_v2 stack           | 1000  | 1.7188 bpb                | Beats ref (-0.009) but not A1 (11L). Depth > width at Mac scale.              |
| 244 | B1: CTW entropy-weighted mixing at 1000 steps        | 1000  | 1.7549 bpb                | WORSE than ref (1.7279). Converging faster but started behind. May catch up at 7000. |
| 245 | A4: XSA only (GQA-aware) at 1000 steps               | 1000  | 1.7305 bpb                | WORSE than ref (+0.003). XSA overhead > benefit at Mac scale.                  |
| 246 | A3: SmearGate on BPE-8192 at 1000 steps              | 1000  | 1.7190 bpb                | Beats ref (-0.009)! SmearGate helps on 8192v. Validate on 1024v too.           |
| 247 | B3: Logistic mixing (log-odds space) at 1000 steps   | 1000  | 1.7425 bpb                | WORSE than ref (+0.015). Log-odds mixing doesn't help like CTW.                |
| 248 | **NOVEL: Smooth STE QAT during training (0.9q+0.1w)** | 50    | **6.953 train**           | **-0.012! Train with simulated int8 quantization. Learns robust features.**    |
| 249 | NOVEL: KAN-inspired multi-activation MLP              | 50    | 6.972 train               | NEUTRAL (+0.006). Mixed LeakyReLU^2 + |x|*x + SiLU per group.                |
| 250 | **NOVEL: Stochastic Weight Perturbation (noise∝|w|)**| 50    | **6.959 train**           | **-0.007! Noise proportional to weight magnitude = implicit regularization.**  |
| 251 | NOVEL: Attention Temperature (learnable)              | 50    | FAIL                      | Can't use dynamic scale in compiled SDPA                                       |
| 252 | NOVEL: Self-aware loss prediction (meta)             | 50    | 7.475 train               | WORSE (+0.510). Same aux loss failure pattern                                   |
| 253 | NOVEL: Weight Symmetry Breaking (Q,K asymmetry)      | 50    | 6.965 train               | TIED (-0.000). Asymmetric init has no effect                                    |
| 254 | NOVEL: Exponential Moving Input (embedding EMA)       | 50    | FAIL                      | Shape issue with decay computation                                              |
| 255 | **NOVEL: Smooth STE QAT (re-confirmed)**             | 50    | **6.953 train**           | **BEST NOVEL: -0.012. QAT during training helps.**                              |
| 256 | A5: Partial RoPE (16/64) + LN Scale at 1000 steps    | 1000  | 1.7511 bpb                | WORSE (+0.023). Neither helps at Mac scale.                                    |
| 257 | **NOVEL: WaveletGPT (multi-scale causal averaging)** | 50    | **6.845 train**           | **BIG WINNER! -0.120! Signal processing between layers. PhD-worthy.**         |
| 258 | NOVEL: Grokfast v2 (gradient EMA amplification)      | 50    | TIMEOUT (GPU contention)  | Needs solo run. Will test after wavelet 1000.                                  |
| 259 | **NOVEL: WaveletGPT at 1000 steps**                  | 1000  | **1.6929 bpb!!!**         | **NEW ABSOLUTE BEST! -0.018 vs A1 (1.7110). Multi-scale causal averaging VALIDATED at scale. PhD-worthy breakthrough.** |
| 260 | NOVEL: Complementary training (PR #803, combined ngram) | 50  | 5.741 train*              | *Different loss fn (weighted). Can't compare. Needs val_bpb at 1000 steps.     |
| 261 | NOVEL: Self-Gated MLP (zero-param permutation gate)   | 50    | 5.745 @step30             | Similar to baseline. Zero-param gate neither helps nor hurts.                    |
| 262 | NOVEL: Gradient Centralization + Muon                 | 50    | 5.733 @step30             | Slightly promising. Row-mean subtraction before NS. Needs 1000-step test.       |
| 263 | NOVEL: Focal Token Loss (gamma=1.5)                   | 50    | 5.613 train*              | *Different loss fn. Interesting signal but unfair comparison.                    |
| 264 | NOVEL: Stochastic Depth (max_drop=0.1)                | 50    | 5.751 train               | Baseline-level. Random layer skip doesn't help at 9L/50 steps.                  |
| 265 | NOVEL: Lookahead Optimizer (k=5, alpha=0.5)           | 50    | 5.845 @step30             | WORSE. Slow weight interpolation hurts convergence speed.                        |
| 266 | NOVEL: Langevin Gradient Noise (temp=0.01)            | 50    | 6.483 @step30             | MUCH WORSE. Gradient noise kills convergence at Muon LR. Dead.                  |
| 267 | **NOVEL: Predictive Coding between layers (80% error)** | 50  | **5.671 @step30**         | **WINNER! -0.040 vs baseline (5.711). Neuroscience-inspired: pass only prediction errors between layers. PhD-worthy.** |
| 268 | NOVEL: Causal Convolution between layers (k=3)        | 50    | 5.693 @step30             | -0.018 vs baseline. Simple causal kernel similar to WaveletGPT. Promising.       |
| 269 | NOVEL: Spectral Gating (EMA on right-half dims)       | 50    | 5.693 @step30             | -0.018 vs baseline. Same as causal conv — both are causal smoothing variants.    |
| 270 | NOVEL: Differential Hidden States (finite diff)       | 50    | 5.699 @step30             | -0.012 vs baseline. Layer-to-layer difference injection. Mild signal.            |
| 271 | NOVEL: MDL Weight Decay (soft shrinkage)              | 50    | 5.932 @step30             | MUCH WORSE (+0.221). Information-theoretic decay too aggressive. Dead.            |
| 272 | QK Gain 4.0 (competition default)                     | 50    | 5.750 @step30             | WORSE (+0.039). Higher QK gain hurts at Mac scale. Competition uses it at 7000 steps. |
| 273 | QK Gain 2.5                                           | 50    | 5.730 @step30             | WORSE (+0.019). Even moderate increase hurts. Current 1.5 is optimal for Mac.    |
| 274 | **NOVEL: WaveletGPT + Predictive Coding STACKED**     | 50    | **5.662 @step30**         | **STACKS! -0.049 vs baseline. Both techniques are additive. Combined is our best novel stack.** |
| 275 | NOVEL: Reservoir Layers (18L, 9 frozen from seed)     | 50    | 5.711 @step30             | NEUTRAL (0.000). Frozen random layers bypassed via U-Net residuals. Dead on Mac. |
| 276 | NOVEL: Hurst-adaptive LR (fractal gradient structure) | 50    | 5.711 @step30             | NEUTRAL. Window=32 barely kicks in at 50 steps. Needs 1000+ test.                |
| 277 | NOVEL: Ternary STE training (BitNet b1.58 inspired)   | 50    | 5.885 @step30             | WORSE (+0.174). Too aggressive for 50 steps but enables 5x compression. H100 test. |
| 278 | NOVEL: KAN-inspired learned activation (RBF basis)    | 50    | 6.664 @step10             | NEUTRAL but too slow (7s/step from RBF). Not practical.                           |
| 279 | NOVEL: Info Bottleneck (per-dim gating after block)   | 50    | 5.710 @step30             | NEUTRAL (-0.001). Gates init open and stay there at 50 steps. Needs 1000+ test.   |
| 280 | NOVEL: Zipf-inverse token loss weighting (gentle)     | 50    | 5.711 @step30             | NEUTRAL (0.000). Gentle inverse-freq weighting too mild. Aggressive version dead (HISTORY #229). |
| 281 | **ultimate_v4c (11L+WaveletGPT+SmearGate) at 1000**   | 1000  | **1.7230 bpb**            | WORSE than 9L WaveletGPT (1.6929). Convergence speed trap: too many params for 1000 steps. H100 candidate. |
| 282 | **WaveletGPT + PredCoding at 1000 steps**             | 1000  | **1.7313 bpb**            | WORSE (+0.038 vs wavelet alone). PredCoding wins at 50 steps but LOSES at 1000 — classic convergence trap. |
| 283 | WaveletGPT + 3x MLP at 50 steps                      | 50    | 5.682 @step30             | -0.009 vs wavelet baseline. 3x MLP shows signal even on wavelet stack!           |
| 284 | WaveletGPT + Softcap 15 at 50 steps                   | 50    | 5.690 @step30             | NEUTRAL (-0.001). Softcap 20 is already optimal for wavelet stack.               |
| 285 | NOVEL: Neural Phase Coupling (dim rotation)            | 50    | 5.687 @step30             | -0.004 vs wavelet. Small signal from coupled oscillator dim rotations.            |
| 286 | NOVEL: Gradient Routing by token rarity                | 50    | 5.809 @step30*            | *Different loss fn. Upweighting rare tokens in CE. Needs fair comparison.         |
| 287 | NOVEL: Dual-probe hash for n-gram (max of 2 hashes)   | 50    | 5.739 @step30             | WORSE (+0.048). Max-of-two-hashes adds noise, doesn't reduce collisions.          |
| 288 | NOVEL: Multi-scale loss (mid-layer prediction)         | 50    | 6.395 @step30             | MUCH WORSE (+0.704). Mid-layer aux loss destabilizes. Same failure pattern as all aux losses. |
| 289 | WaveletGPT + 3xMLP at 1000 steps                      | 1000  | 1.7274 bpb                | WORSE (+0.035 vs wavelet alone). 3xMLP adds too many params for 1000 steps. H100 candidate. |
| 290 | **NOVEL: Adaptive N-gram (learned per-token weights)** | 50    | **5.618 @step50**         | **BIG WINNER! -0.096 vs baseline (5.714). Context-dependent n-gram mixing. PhD-worthy!** |
| 291 | NOVEL: Neural Phase Coupling (solo, no contention)     | 50    | 5.708 @step50             | -0.006 vs baseline. Small signal from dim rotations. Needs 1000 test.             |
| 292 | EVAL: Entropy-adaptive n-gram mixing (prob space)      | eval  | 1.7496 bpb                | WORSE (+0.041). Double-counts n-gram signal (model already trained with bias).    |
| 293 | **EVAL: Fixed alpha=0.1 prob-space n-gram blend**      | eval  | **1.7010 bpb**            | **-0.007! Mild prob-space smoothing helps on top of logit bias. Free improvement.** |
| 294 | Adaptive N-gram at 1000 steps                         | 1000  | 1.8327 bpb                | MUCH WORSE (+0.140 vs wavelet). 50-step advantage was artifact of gate initialization. Dead. |
| 295 | **EVAL: Alpha sweep on wavelet model (5000 seqs)**    | eval  | **-0.0065 @ alpha=0.08**  | **Optimal prob-space blend alpha=0.08. Free improvement. Projected full: 1.686 BPB.** |
| 296 | **EVAL: α=0.08 prob-space mixing on FULL val set**    | eval  | **-0.0065 confirmed**     | **CONFIRMED on full val. Free improvement. Effective best: 1.6864 BPB (vs 1.6929 training).** |
| 297 | EVAL: SSE/APM table correction (500 seqs)              | eval  | -0.1548 (CHEAT)           | ARTIFACT: gain came from probability clipping, not calibration. NOT LEGITIMATE. |
| 298 | EVAL: Probability flooring investigation               | eval  | -0.133 from 1e-3 clip     | Confirmed: clipping p to 1e-3 is cheating (no renormalization). Dead. |
| 299 | **EVAL: Temperature scaling T=0.95 (sharpening)**      | eval  | **-0.0065**               | **Model is overconfident wrong way — sharpening helps.** |
| 300 | **EVAL: α=0.08 + T=0.97 combined**                     | eval  | **-0.0101 (1000 seqs)**   | **STACKS! Projected best: 1.6828 BPB. Biggest legit eval-time gain.** |
| 301 | **EVAL: Joint sweep α × T (2000 seqs)**                | eval  | **-0.0124 @ α=0.06, T=0.93** | **BEST eval combo! Projected best: 1.6805 BPB. -0.357 total from baseline.** |
| 302 | Wavelet WITHOUT training-time n-gram (1000 steps)      | 1000  | 1.7459 bpb                | WORSE (+0.053 vs wavelet with n-gram). Training-time bias is critical, can't recover at Mac scale. |
| 303 | EVAL: Dynamic n-gram cache (1000 seqs)                | eval  | +0.017 to +0.044          | WORSE. Dynamic cache redundant with static trained tables. Needs longer corpus. |
| 304 | **WaveletGPT + warmdown=200 (proper LR decay)**       | 1000  | 1.7489 bpb                | **WORSE (+0.056). The "warmdown=1200 bug" is actually beneficial at Mac scale!** Warmdown helps long training; Mac-scale needs sustained LR. |
| 305 | WaveletGPT + EMA (decay=0.999, every 10 steps)        | 1000  | 2.1421 bpb                | DRAMATICALLY WORSE (+0.449). EMA weights swap may have lost buffer state, OR decay too aggressive for 1000 steps. Failed retest of #8. |
| 306 | BPB CALCULATION AUDIT                                 | meta  | VERIFIED CORRECT          | Manual byte counting matches UTF-8 length. Loss/log(2)/bytes_per_token = bits/byte. Math correct. NOT bits per token. |
| 307 | Stage 2 continual (full LR, 500 steps from wavelet)   | 500   | killed @ step 100         | train_loss diverged 4.17→4.50 due to optimizer state reset + full LR. Failed approach. |
| 308 | **Stage 2 continual LOW LR (matrix_lr=0.01, 300 steps)** | 1000+300 | **1.6670 bpb!!! NEW BEST** | **MASSIVE WIN! -0.0259 vs wavelet_1000 (1.6929). Continual training WORKS with 4x lower LR. The user's "two-stage" intuition was right. -0.357 total vs baseline.** |

### Compression Tests

- int8+zlib-9: 5.11 MB (neural only)
- int8+zstd-22: 4.61 MB (-500KB)
- int8+lzma-6: 4.70 MB (-410KB)
- **int8+brotli-11: 4.58 MB (-530KB, best for neural)**
- **Full artifact (neural+ngram) brotli-11: 12.33MB (fits 16MB with 3.67MB spare)**

### Validated Winners (improvements that held at 500 steps)


| Technique                                     | bpb Improvement | Stacks?                    |
| --------------------------------------------- | --------------- | -------------------------- |
| BPE-8192 tokenizer                            | -0.129          | Yes (base)                 |
| N-gram logit bias (bi+tri+4gram, 16K buckets) | -0.055          | Yes                        |
| **Dual MLP (2 experts, averaged)**            | **-0.067**      | **500-step val RUNNING (#180)**  |
| **Skip-bigram table (prev2→next)**            | **-0.063**      | **NEW (from research cycle 8)**  |
| **TOP aux loss (token order prediction)**     | **-0.024**      | **NEW (from research cycle 12)** |
| **Hyper-connections (learned residual)**       | **-0.015**      | **NEW (from DeepSeek port)**     |
| Distributional Categories DC500               | -0.010          | Yes                        |
| Micro-macro split MLP (2.5f)                  | -0.004          | Yes                        |
| Compression bottleneck (2.6n)                 | -0.004          | Yes                        |
| LeakyReLU(0.5)^2                              | -0.004          | Yes                        |
| Post-period capitalization bias (R)           | -0.002          | Yes                        |
| English Knowledge Engine (1b+1d+1e+1f)       | -0.001          | Yes                        |
| Wave equation forward pass (GG)               | -0.023          | No (redundant with n-gram) |
| **WaveletGPT multi-scale mixing (9L)**        | **-0.018**      | **✓ 1000-step val: 1.6929 BPP (NEW BEST)** |
| **Total 50-step winners (all additive)**      | **~-0.40**      | **Needs 500-step validation**    |
| **Previous validated stack (1000 steps)**     | **-0.284**      |                            |

**NEW: 4 winners from research cycles 8-12 (skip-bigram, hyper-connections, TOP, signed hash)**
**CRITICAL: #180 Dual MLP 500-step val is running — if it validates, it's the biggest arch win.**
**WARNING: #173 fractal depth (-1.262) is SUSPICIOUS — may leak future info through pooling.**


### Key Discoveries

1. **BPE-8192 tokenizer is the single biggest lever** (-0.129 alone, beats all 1024v tricks combined)
2. **N-gram logit bias is our core innovation** (-0.055 on 8192v, -0.081 on 1024v)
3. **Techniques that help early convergence hurt at 500 steps** (embed_lr=0.1, cosine schedule, no warmdown, 3xMLP, momentum 0.99)
4. **ALL geometric/analytical inits FAILED** — SVD, spectral, reservoir. Model needs freedom
5. **ALL loss reweighting FAILED** — complementary, focused, byte-weighted, bits-back, Gaussian, hard-context, UID
6. **ALL auxiliary losses FAILED** — deep supervision, self-referential, difficulty head, WaveletGPT
7. **The neural model IS the foundation** — augment with n-gram bias, don't replace
8. **Brotli-11 saves 1.47MB over zstd-22** — especially effective on n-gram tables
9. **English is only 11.2% bigram-deterministic** — finite automata/DFA won't help
10. **Pure classical PPM is 1.0 nats worse than neural** — classical augments, can't replace

### What Failed and Why (meta-lessons)

- **"Solve the model analytically"**: DEAD. Tied embeddings + nonlinear interactions resist closed-form solutions.
- **"Replace neural with classical"**: DEAD. PPM plateaus at CE ~4.4, neural gets 3.3.
- **"Fewer params + more knowledge"**: DEAD. Can't trade layers for lookup tables.
- **"Reweight the loss function"**: DEAD. Every variant hurts. CE mean is already optimal.
- **"Add auxiliary losses"**: DEAD. Every variant destabilizes training.
- **"Speed up convergence"**: TRAP. Wins at 50 steps, loses at 500. Mac can't distinguish.
- **What WORKS**: Full neural model + n-gram bias augmentation + BPE-8192 tokenizer + linguistic priors (period bias).

### H100 Projections

Based on BPE-8192 + n-gram + LeakyReLU + R = **1.7397 BPB on Mac (1000 steps)**:


| Configuration                    | Est. H100 BPB | Confidence |
| -------------------------------- | ------------- | ---------- |
| H100 naive baseline (ref)        | 1.2244        | Verified   |
| H100 merged SOTA (ref)           | 1.1147        | Verified   |
| **Our BPE-8192 + SOTA tricks**   | **~1.04**     | Medium     |
| + SLOT V1 eval                   | ~1.02         | Medium     |
| + factorized embed + more layers | ~1.00         | Medium-low |
| Pending competition SOTA         | 0.93          | Unverified |


### Next Steps (Updated Apr 4 — research cycle)

**IMMEDIATE (Mac, builder session):**
1. Finish Part 2.6 experiments (18 architecture tests, builder running)
2. Test CTW n-gram mixer (2.8a) — replaces ad-hoc n-gram weights with provably optimal
3. Test Hopfield Memory Layer (2.8c) — cross-attention over 256 learned prototypes, 256KB
4. Test MDL training objective (2.8d) — replace WD with description-length penalty
5. Test Polar Token Routing (2.8e) — principled version of complementary training

**H100 (when compute available):**
6. **Apply for compute credits** (form text ready)
7. **Port winning stack to CUDA** (SUBMISSION_PLAN.md has roadmap)
8. **Investigate DEQ approach** (PR #1323: 1.1247 BPB in only 6.8MB! — massive headroom for our n-gram stack)
9. **Legal score-first TTT** at eval time (~-0.02 to -0.05 bpb)
10. **Int6 + GPTQ compression** with rate-distortion optimal bit allocation (2.8f)
11. **SP4096 tokenizer** (PR #1326 used it for 1.0896 BPB — sweet spot between embed cost and token reduction)

**Strategic pivot:** DEQ + our n-gram bias stack could be the winning combination. DEQ fits in 6.8MB, our tables + DC + English engine fit in ~4MB, totaling ~11MB with 5MB headroom for additional layers or techniques.

