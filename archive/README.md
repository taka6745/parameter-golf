# archive/ — historical iterations

Retained for reference + reproducibility. Not referenced by live pod infra.

## MLX training evolutions

- `train_gpt_mlx_v2.py` through `train_gpt_mlx_v17.py` — per-iteration MLX training scripts.
  - Active MLX baselines stay at repo root: `train_gpt_mlx.py`.
  - `v13` is cited in `ngram_logit_bias.py` comments as the n-gram hash source-of-truth.
  - `v17` was the last iteration before the pod-centric workflow took over.

## GPU test harnesses (superseded by runpod_tests/)

- `gpu_quick_test.py` — baseline speed test
- `gpu_speed_test.py` — speed experiments across configs
- `gpu_progressive_test.py` — fixed-step quality test (1000 steps each)
- `gpu_timed_test.py` — ⭐ the "definitive" timed quality test (120s per config) — #7 winner lives here
- `gpu_timed_test2.py` — follow-up round

## One-off analysis / eval scripts

- `analyze_competition.py` — competition submissions analyzer
- `eval_adaptive_mix.py` — adaptive mixture eval experiment
- `peek_shards.py` — peek at `data/datasets` shards
- `progressive_seq_patch.py` — reference patch file (the idea landed in `train_gpt.py` via `runpod_tests/chore/08_patch_train_gpt.sh`)
