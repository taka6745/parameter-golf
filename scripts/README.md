# scripts/ — operator shell helpers

Run all scripts **from the repo root** (they use relative paths for `train_gpt.py`, `data/`, etc.).

- `ab_shard_test.sh` — A/B shard-size test using `train_gpt_mlx.py` + `generate.py`
- `quick_test.sh` — N-step quick train loss probe. Default script: `archive/train_gpt_mlx_v2.py`. Override: `bash scripts/quick_test.sh <script.py> <run_id> [steps]`
- `run_gpu_batch.sh` — batch run on a RunPod GPU (rsyncs `train_gpt.py` + data)
- `run_gpu_test.sh` — single-config run on a RunPod GPU (rsyncs `train_gpt.py`, `ngram_logit_bias.py`, etc.)
