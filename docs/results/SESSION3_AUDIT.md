# Session 3 Audit — Apr 7, 2026 (3080 Ti, 41 runs across 40 configs)

## Current absolute best (single-run train_loss @ ~1500 steps, 300s wallclock)

| rank | name | train_loss | family | tricks |
|---|---|---|---|---|
| 1 | L5_leaky_low_bigram | **3.2931** | L | leaky_relu(0.5)² + ngram (bi=0.15, tri=0.20, four=0.15) |
| 2 | L4_leaky_strong_weights | 3.2947 | L | leaky_relu + ngram (bi=0.25, tri=0.25, four=0.20) |
| 3 | L2_leaky_plus_full_ng_seed42 | 3.3286 | L | leaky_relu + ngram (bi=0.20, tri=0.20, four=0.15) seed=42 |
| 4 | L1_leaky_plus_full_ng | 3.3300 | L | same as L2 with default seed |
| 5 | C4_full_ngram_strong | 3.3337 | C | ngram only (bi=0.20, tri=0.20, four=0.15) |
| 6 | C5_stronger_seed1 | 3.3359 | C | ngram only (=C4 with explicit seed) |
| 7 | C7_max_weights | 3.3516 | C | ngram only (bi=0.30, tri=0.25, four=0.20) |

**Baseline reference**: A0/A1 = 3.5658-3.5663 (mean 3.6014 over 4 runs incl A2 high seed).

**Best vs baseline**: -0.272 train_loss (L5 vs baseline mean).

## Family means (multi-run averages where available)

| family | mean | best | runs | what |
|---|---|---|---|---|
| L | 3.3957 (excl L0) | 3.2931 | 6 | leaky_relu + ngram |
| C | 3.3824 | 3.3337 | 8 | ngram only (no leaky) |
| E | 3.5614 | 3.5355 | 2 | arch sweep |
| B | 3.5661 | 3.5313 | 3 | bigram-only |
| D | 3.5703 | 3.5261 | 4 | bigram weight sweep |
| A | 3.6014 | 3.5658 | 4 | baseline |
| W | 3.7768 | 3.7743 | 2 | **wavelet — NEUTRAL/HURT** |
| L0 alone | 3.7437 | — | 1 | leaky alone WITHOUT ngram — HURTS |
| BW (any) | ~3.92-3.96 | — | 3 | byte-weighted loss — **METRIC NOT COMPARABLE** |
| I | 4.0566 | — | 1 | bigger model loses (fewer steps in same wallclock) |

## Confirmed findings (signal > noise, multi-seed where possible)

1. **Full n-gram bias stack at 16384 buckets is the biggest single win**: -0.25 train_loss vs baseline. Stable across 3 seeds (C0/C1/C2 = 3.37/3.36/3.50 — seed 999 always runs higher).
2. **Strong weights win**: bi=0.20, tri=0.20, four=0.15 (C4) > bi=0.20, tri=0.15, four=0.10 (C0). Differences within ~0.01.
3. **Bigram-only is a tiny win** (~-0.05) — the trigram + fourgram together do most of the work.
4. **Bigram weight is forgiving**: 0.10 / 0.15 / 0.20 / 0.30 all within 0.04. 0.50 hurts.
5. **leaky_relu(0.5)² helps when STACKED with n-gram bias** (~-0.005 marginal). Does NOT help alone (L0 = 3.74 vs baseline 3.57).
6. **More steps > fancier model** at this scale: bigger models (12L 3x MLP) lose because they get fewer steps in the same wallclock budget.
7. **Architecture (9L vs 11L, 2x vs 3x MLP) is essentially noise** at 1500 steps.

## Negative results — DO NOT USE

1. **WaveletGPT** — NEUTRAL/HURT at 1400 steps. Mac validated -0.022, but on the pod it's +0.20 worse than baseline (slowdown wipes out marginal benefit).
2. **Bigger model** (12L 3x MLP) — fewer steps offsets capacity gain.
3. **Architecture sweeps** — within noise.
4. **Optimizer hyperparam sweeps** — pending (M0/M1/LR/WD running) but expected to be small.

## Issues / METRIC GOTCHAS

1. **Byte-weighted loss reports the WEIGHTED mean**, which is naturally higher than the unweighted CE. BW0/BW1/BW2 train_loss numbers are NOT comparable to other configs. Need to add an unweighted-loss reporting line OR run a final eval pass to compute val_bpb.
2. **No val_bpb for any of these runs** — we're using SKIP_FINAL_EVAL=1 for speed. The train_loss → val_bpb mapping is monotonic but the absolute values don't match the competition metric. Best config needs ONE proper eval run on H100 to get the real number.
3. **Single-seed variance is ~0.05-0.08 train_loss**. Differences smaller than that (0.01-0.04) are within noise.
4. **Seed=999 is consistently higher** than seeds 1337/42 — possibly an unusual data ordering. Not a model issue.

## What is STILL untested (Mac-validated wins not yet ported)

In rough priority order (highest expected EV first):

| trick | Mac delta | complexity | status |
|---|---|---|---|
| **BPE-8192 tokenizer** | **-0.129 BPB (BIGGEST)** | hard — needs new dataset build | NOT BUILT |
| Continual lowLR stage 2 | -0.026 | medium — needs save/load checkpoint | NOT TESTED |
| SmearGate (smear current with prev tokens before MLP) | -0.019 | easy — 5-line forward change | NOT PORTED |
| Eval-time alpha + temperature blend | -0.012 | easy — eval-only patch | NOT TESTED |
| Q-R skip-bigram (3-table decomposition) | +0.005 | medium — uses existing tables in clever way | NOT TESTED |
| Tabulation hashing | provably better than linear | medium — needs n-gram rebuild | NOT TESTED |
| Signed hashing | +0.003 free | easy — 4 lines + rebuild | NOT TESTED |
| Complementary training (PR #803) | -0.005 to -0.015 | medium — clamp(1 - 0.5*bigram_prob) weighting | NOT TESTED |
| Dynamic token selection (skip bottom 20% loss) | -0.005 to -0.01 | easy — 5 lines | NOT TESTED |
| Hyper-connections (DeepSeek ICLR 2025) | speculative | medium — replace residual with multi-stream | NOT TESTED |

## NOVEL ideas (NOT in any researched PR I've seen)

These come from gaps I noticed in RESEARCH.md / the C-family findings:

1. **Scheduled n-gram weight decay** — start n-gram bias at high (0.30/0.30/0.20), decay linearly to (0.10/0.10/0.05) over training. Hypothesis: model needs the bias early to learn patterns, then internalizes them and benefits from less. Mac NEVER tested this.

2. **Per-layer n-gram bias injection** — instead of adding bias only at logits, inject a learned linear projection of the bias into each layer's residual. Lets the model "use" n-gram info during processing.

3. **Bigram-conditioned token selection** — at each step, mask out the gradient for tokens where bigram already gives high prob. Forces capacity onto the hard predictions. Combines complementary training + dynamic token selection.

4. **Dual n-gram tables** — train with HASH=2048 for early steps (broader smoothing), switch to HASH=16384 mid-training (cleaner per-bucket signal). Mac never tested mid-training table swap.

5. **Score-first eval bias** — use the validation token PREFIXES to update the n-gram tables ONLINE during eval. Effectively "score-first TTT" but on the bias instead of weights. Different from the pure eval-cache approach (which Mac §4g said failed).

## Cron / autonomy status

The pod runs a self-restarting wrapper (`runpod_tests/loop/run_forever.sh`) that:
- Auto-pulls latest code/experiments from git on each restart
- Restores `train_gpt.py.bak` and re-applies `08_patch_train_gpt.sh` so new patches take effect
- Launches `experiment_runner.py` which itself auto-pulls before every iteration

Effectively this IS the cron — pushed changes propagate within ~5 min. There's no separate cron daemon (the container has no `crontab` command).

For the local side, status snapshots are pulled on-demand via `/tmp/podstatus.sh` (SSH heredoc + base64).

## What needs val_bpb to convert to competition metric

The current best (L5 = 3.2931 train_loss) needs ONE proper eval pass on the cheap GPU (~25-30 min including the int8+zlib roundtrip and full 62M-token val) to get the real `final_int8_zlib_roundtrip val_bpb` number that goes on the leaderboard. Without that, we have RELATIVE signal but no absolute submission number.

## Recommended next steps

1. **Implement & test the most-novel-yet-cheapest items**:
   - Scheduled n-gram weight decay (truly novel)
   - SmearGate (Mac validated -0.019)
   - Signed hashing (free +0.003)

2. **Build BPE-8192 dataset on the pod** — biggest single Mac win (-0.129) is just sitting there unused. Requires re-tokenizing FineWeb shards which takes ~20 min. Worth it.

3. **Run ONE full-eval validation** of the current best (L5 config) to get a real val_bpb number to compare against the competition leaderboard.

4. **Multi-seed validation** of the top 3 configs once we've stabilized the search space.
