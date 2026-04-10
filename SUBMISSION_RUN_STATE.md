# SUBMISSION_RUN_STATE.md — Final Submission Run Tracker

This file tracks the **REAL submission run** for openai/parameter-golf,
not Phase 2 experiments. Phase 2 work is documented in `PHASE2_AUTOMATION_STATE.md`.

## Run

**Pod**: `aklt7paqnjwhal` (paramgolf-c-8xh100-sxm)
**Hardware**: 8x NVIDIA H100 80GB HBM3 SXM, 224 vCPU, 2 TB RAM
**Cost rate**: $21.52/hr
**SSH**: `aklt7paqnjwhal-6441217f@ssh.runpod.io`
**Monitor cron**: `be912385` (3,13,23,33,43,53 — every 10 min)

## Configuration (Option C)

Stack from commit 91d7777 (`033c60d` + get_data.sh mkdir fix):

```
NUM_LAYERS=11 MLP_MULT=4              # match PR #1493 architecture
NUM_LOOPS=2 LOOP_START=3 LOOP_END=5  # 3-layer recurrence
ENABLE_LOOPING_AT=0.35
QK_GAIN_INIT=5.25
PARALLEL_RESIDUAL_START=7             # per-block parallel residuals (matches PR #1493)
EMA_DECAY=0.9965 WARMDOWN_FRAC=0.72 MUON_WD=0.095 MATRIX_LR=0.022
MATRIX_BITS=8                         # int8 weight quant (our edge over PR #1493's int6)
USE_PARALLEL_MUON=1                   # batched Newton-Schulz
USE_CMP_QUANT_VALUE_DEDUP=1           # NIGHT_MODE L10 alphabet snap (compresses for 16 MB fit)
TORCH_COMPILE_DISABLE=0               # CRITICAL re-enable, was silently disabled
TORCHDYNAMO_DISABLE=0
TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
USE_CUDNN_BENCHMARK=1
USE_GATED_ATTENTION=1 USE_NORMUON=1 USE_NORM_PCT_DROPOUT=1
USE_PREFETCH_LOADER=1
TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3   # legal score-first TTT
PREQUANT_TTT_ENABLED=0                # rule violation, kept off
USE_NGRAM_BIAS=0 USE_NGRAM_BACKOFF=0   # Issue #1017 grey area, kept off per user policy
USE_NGR_LOG_FREQ_INV=0 USE_CTX_PARTITIONED_TAB=0
SEEDS=42,314,999                       # 3-seed real submission
```

## Targets

- **PR #1493 leaderboard #1**: val_bpb 1.0810 (3-seed mean, std 0.0002, on 8xH100 SXM)
- **Record threshold**: must beat 1.0810 by ≥ 0.005 nats at p < 0.01 → **val_bpb ≤ 1.0760**
- **Our projection**: ~1.072–1.078 (Option C with int8 edge + torch.compile fix)
- **Record probability**: ~30–40% based on the math

## Output

Records folder will be assembled at:
`records/track_10min_16mb/2026-04-10_SP8192_NL11_MLP4_int8_ParMuon_PR7_LegalTTT/`
With files: `train_seed{42,314,999}.log`, `final_model_seed{42,314,999}.int6.ptz`,
`README.md`, `submission.json`, `train_gpt.py` (LZMA-wrapped).

The cron will pull this folder back to the local repo when all 3 seeds complete.

## Fire Log

| time (Z) | phase | notes |
|---|---|---|
| 01:42:01 | spinup | Pod created, runpodctl, paid for first hour |
| 01:43 | bootstrap fail | get_data.sh Step 0 cp failed (mkdir bug). Submission process died. Cost ~$1.40. |
| 01:46:27 | retry 2 launch | Fixed via commit 91d7777, relaunched as PID 1467 |
| 01:57 | bootstrap-data | Tokenize **21% done** (3.2M / 15.4M docs, 35 train shards written). SP cached ✓, 48 GB docs hardlinked ✓. ETA tokenize complete ~02:35-02:45Z. Spend ~$3.94. |
