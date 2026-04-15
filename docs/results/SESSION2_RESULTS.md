# Session 2 Results — Autonomous RunPod Loop (Apr 7, 2026)

## Key finding

**Full n-gram bias stack (bi=0.20 + tri=0.15 + four=0.10) gives -0.21 train_loss vs baseline at 1500 steps**, validated across 3 seeds.

| family | mean train_loss | best | runs | config |
|---|---|---|---|---|
| **C — full n-gram** | **3.4044** | **3.3572** | 4 | `USE_NGRAM_BIAS=1 NGRAM_W_BIGRAM=0.20 NGRAM_W_TRIGRAM=0.15 NGRAM_W_FOURGRAM=0.10` |
| E — arch (11L/9L/3x) | 3.5614 | 3.5355 | 2 | mostly noise vs 9L 2x |
| B — bigram=0.20 only | 3.5661 | 3.5313 | 3 | bigram alone is small win |
| D — bigram weight sweep | 3.5703 | 3.5261 | 4 | 0.10 ≈ 0.15 ≈ 0.20 (all close) |
| **A — baseline (no n-gram)** | 3.6133 | 3.5663 | 3 | reference |

All experiments: 9L 512d 2x MLP, 1500 steps (300s wallclock on RTX 3080 Ti), SP-1024 tokenizer, n-gram tables at HASH_BUCKETS=16384.

## What we learned

1. **N-gram bias helps a lot** — but only the FULL bi+tri+four stack. Bigram alone gives -0.05; bigram+trigram is *worse* than baseline; bigram+trigram+fourgram is -0.21.
2. **N-gram needs enough training** — at 1200 steps the full n-gram looked broken. At 1500 steps it dominates. Mac LESSONS.md hinted at this (n-gram diminishing-returns at higher step counts) but we observed the opposite at this regime: n-gram needs the model to "settle" before the bias becomes useful.
3. **Hash buckets matter** — 2048 buckets caused trigram noise (collision pressure ~500x at trigram). 16384 fixes it. LESSONS.md §4f said 8K is the sweet spot for Mac; 16K worked for us.
4. **Bigram weight is forgiving** — 0.10 / 0.15 / 0.20 / 0.30 are all within 0.02 of each other. 0.50 hurts.
5. **Architecture (9L vs 11L, 2x vs 3x MLP) is essentially noise at this regime** — matches LESSONS.md §1 ("architectural changes don't help at low step counts").
6. **The cheap GPU is compute-bound** — going from 1200→1500 steps drops baseline from 3.94 to 3.61 (-0.33). Step count matters more than any single config knob at this scale.

## What's still unknown

- **Actual val_bpb** — we measured train_loss only (SKIP_FINAL_EVAL=1). Need a full eval pass on the winning config to get the real submission number.
- **Whether the win transfers to H100** — the full n-gram needs ~1500 steps on cheap GPU to "kick in". On H100 with 7000+ steps, we expect it transfers but should verify.
- **Optimizer hyperparam effect** — F sweep was cut for time. Mac LESSONS.md says momentum=0.99, lr=0.02, warmdown=3000 are the frontier defaults; we still use 0.95/0.04/1200.
- **Bigger model effect** — I0_big_12L_3x_full was running at session end. Single result.

## Recommended H100 config

```bash
USE_NGRAM_BIAS=1
NGRAM_W_BIGRAM=0.20
NGRAM_W_TRIGRAM=0.15
NGRAM_W_FOURGRAM=0.10
NGRAM_HASH_BUCKETS=16384

# These transfer from Mac LESSONS.md but we couldn't validate on cheap GPU:
NUM_LAYERS=11           # Mac says 11L > 9L at H100 scale
MLP_MULT=3              # Mac says 3x > 2x at H100 scale
MUON_MOMENTUM=0.99      # frontier default per LESSONS.md §35
MATRIX_LR=0.02          # frontier default per LESSONS.md §35
WARMDOWN_ITERS=3000     # frontier default per LESSONS.md §35

# Still need:
# - Build BPE-8192 tokenizer (Mac says this is the biggest single win, -0.13 BPB)
# - Build n-gram tables for the 8192 vocab (rebuild 04_build_ngrams.py with VOCAB=8192)
# - Compress n-gram tables with int8+brotli to fit in 16MB artifact
# - Run a SINGLE seed with SKIP_FINAL_EVAL=0 to get actual val_bpb
```

## How to validate before the 3-seed final

```bash
# On 1xH100, single seed, full eval (~25 min, ~$2):
NUM_LAYERS=11 MLP_MULT=3 MODEL_DIM=512 \
USE_NGRAM_BIAS=1 NGRAM_W_BIGRAM=0.20 NGRAM_W_TRIGRAM=0.15 NGRAM_W_FOURGRAM=0.10 \
NGRAM_HASH_BUCKETS=16384 \
MUON_MOMENTUM=0.99 MATRIX_LR=0.02 WARMDOWN_ITERS=3000 \
SKIP_FINAL_EVAL=0 \
ITERATIONS=1000000 MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024 \
python3 train_gpt.py
```

Then read `final_int8_zlib_roundtrip val_bpb:N.NNNN` from the log.

If val_bpb < 1.10, queue the 3-seed final on 8xH100 via `runpod_tests/submission/3seed_final.sh` (after rewriting it with the validated config).

## Infrastructure built (reusable)

- `runpod_tests/loop/experiment_runner.py` — round-robin runner with auto-pull, leaderboard, jsonl results
- `runpod_tests/loop/run_forever.sh` — self-restarting wrapper, also auto-pulls
- `runpod_tests/loop/experiments.json` — declarative config queue
- `runpod_tests/loop/analyze.py` — multi-seed aggregation + family grouping
- `runpod_tests/chore/08_patch_train_gpt.sh` — 8 idempotent patches:
  1. PyTorch 2.4 GQA workaround
  2. torch.compile disable
  3. Progressive seq init
  4. SKIP_FINAL_EVAL early-exit
  5. Phase transition seq clamp
  6. NGRAM_BIAS load + apply (NEW this session)
  7. SKIP_LAST_VAL skip the wallclock-cap val pass (NEW this session)
  8. SKIP_POST_LOOP bail before GPTQ + zlib (NEW this session)
- `/tmp/podrun.sh`, `/tmp/podpull.sh`, `/tmp/podstatus.sh` — local helpers for SSH heredoc remote control (RunPod's SSH proxy blocks scp/sftp)
