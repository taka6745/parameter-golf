---
id: EXP-YYYY-MM-DD-NNN
slug: short-kebab-case
idea: IDEA-NNN
pod: paramgolf-h100                   # matches POD_HOSTS.env label
gpu: 1xH100SXM                        # e.g. 1xH100SXM, 8xH100SXM, 1xRTX3090
seed: 42
git_sha: d8735a0                      # commit used for the run
config:                               # env vars overridden vs default submission/run.sh
  SKIP_GATES_ENABLED: 0
  QK_GAIN_INIT: 5.25
  EMA_DECAY: 0.9965
started: YYYY-MM-DDTHH:MMZ            # UTC
finished: YYYY-MM-DDTHH:MMZ
status: pending                       # pending | running | complete | failed | crashed | aborted
cost_usd: 0.00                        # filled in at completion
---

# EXP-YYYY-MM-DD-NNN: <one-line summary>

## What I'm running

<One paragraph. What's the specific thing we're testing? How does it differ from the IDEA's baseline?>

Link to idea: [IDEA-NNN](../ideas/IDEA_NNN_<slug>.md)

## Command

```bash
source POD_HOSTS.env
# (exact command or script that was run on the pod)
SKIP_GATES_ENABLED=0 QK_GAIN_INIT=5.25 EMA_DECAY=0.9965 \
  bash submission/run.sh
```

## Expected (from IDEA)

Per IDEA-NNN:
- **Metric**: val_bpb
- **Threshold**: ≤ X.XXXX
- **Falsification**: ≥ X.XXXX

## Results

| metric | value | notes |
|---|---:|---|
| `val_bpb` (quantized_sliding_window) | | |
| `val_loss` | | |
| `artifact_bytes` | | 16,000,000 cap |
| `step_time_ms` | | |
| total training steps | | |
| wallclock s | | |

## Log excerpts

<Key lines from the pod training log. Focus on:>
<- Final val_bpb line>
<- Any anomalies during training>
<- OOM / crashes if applicable>

```
(paste the last 20-30 lines of the training log here, or the relevant snippet)
```

## Full logs

- Pod path: `/workspace/paramgolf/logs/<run_id>.log`
- Homelab cache: `https://paramgolf.koda-software.com/logs/<run_id>.log` (if pushed)
- Local copy: `data/experiment_logs/EXP-YYYY-MM-DD-NNN.log`

## Comparison

| Baseline | val_bpb | Source |
|---|---:|---|
| Our 1.082 submission | 1.082 | records/track_10min_16mb/2026-04-10_*/submission.json |
| This experiment | X.XXXX | this doc |
| Δ vs baseline | +/− X.XXXX | |

## Next action

<One of:>
<- PASS expected, run N more seeds at same config (create EXP-...-NNN+1)>
<- PARTIAL, adjust config to <what> (create EXP-...-NNN+1 at new config)>
<- FAIL expected — IDEA-NNN moves to killed; update STACK_NOVELTY_TRACKER_v2 row>
<- WARRANTS FINDING — create FINDING-YYYY-MM-DD-<slug>.md>

## Cost ledger

- Pod: <pod name + $/hr>
- Wallclock: <HH:MM>
- Cost: $ X.XX

## Notes

<Free-form. Include any surprises, config drift, log anomalies, debugging notes.>
