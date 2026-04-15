---
id: EXP-2026-04-16-001
slug: drop-gated-attention-seed42
idea: IDEA-001
pod: paramgolf-h100
gpu: 1xH100SXM
seed: 42
git_sha: e5121f3
config:
  SKIP_GATES_ENABLED: 0
  SEED: 42
  MAX_WALLCLOCK_SECONDS: 600
started: 2026-04-16T16:55Z
finished: null
status: running
cost_usd: null
val_bpb: null
outcome: pending
---

# EXP-2026-04-16-001: IDEA-001 drop gated attention — seed 42

## What I'm running

Disabling gated attention per IDEA-001 evidence (P15 ablation: zeroing `attn.gate_proj.weight` improves val NLL by 1.02%; P11 cross-seed: rel_std 1.14–1.25 = pure noise).

Single env-var flip: `SKIP_GATES_ENABLED=0`. Otherwise default submission/run.sh config with seed 42, 600s wallclock.

Idea: [IDEA-001](../ideas/IDEA_001_drop-gated-attention.md)

## Command

```bash
# On H100 pod:
cd /workspace/paramgolf
git fetch origin && git reset --hard origin/main
SKIP_GATES_ENABLED=0 SEED=42 MAX_WALLCLOCK_SECONDS=600 \
  RUN_ID=exp_20260416_001_drop_gates_s42 \
  bash submission/run.sh 2>&1 | tee logs/exp_20260416_001.log
```

## Expected (from IDEA-001)

- Metric: val_bpb (quantized_sliding_window)
- Threshold: ≤ 1.074 (Δ = −0.008)
- Falsification: ≥ 1.079

## Results

| metric | value | notes |
|---|---:|---|
| `val_bpb` | _pending_ | |
| `artifact_bytes` | _pending_ | target 16,000,000 |
| `step_time_ms` | _pending_ | |
| total steps | _pending_ | |
| wallclock s | _pending_ | |

## Log excerpts

_pending — will be pulled when training completes_

## Full logs

- Pod path: `/workspace/paramgolf/logs/exp_20260416_001.log`
- Homelab cache: _pending_
- Local copy: _pending_

## Comparison

| Baseline | val_bpb | Source |
|---|---:|---|
| Our 1.082 submission | 1.082 | records/track_10min_16mb/2026-04-10_*/submission.json |
| This experiment | _pending_ | this doc |

## Next action

_pending — determined by result_

## Cost ledger

- Pod: paramgolf-h100 @ $2.99/hr
- Wallclock: ~10 min train + ~5 min eval/quant = ~15 min projected
- Cost: ~$0.75 projected

## Notes

First autonomous-loop experiment. Validates Loop B's ability to dispatch on idle pod, write an EXP doc, launch training, and (in a later fire) pull results.
