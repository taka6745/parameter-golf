---
id: IDEA-001
slug: drop-gated-attention
created: 2026-04-16
updated: 2026-04-16
status: in-experiment
layer: L03
novelty_class: CN
expected_bpb: [-0.012, -0.008]
cost_hours: 0.5
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l03-gated-attention-per-head-sigmoid
prior_art_checked: 2026-04-16
next_step: retrain-seed42-then-seeds-314-999
---

# IDEA-001: Drop gated attention entirely (SKIP_GATES_ENABLED=0)

> **Hypothesis**: Removing the learned per-head sigmoid gate on attention outputs reduces val_bpb by 0.008-0.012 at 3-seed mean, because P15 probe (gate_proj ablation) showed +1.02% NLL when weights are zeroed and P11 showed all 11 `attn.gate_proj` tensors have rel_std 1.14-1.25 across seeds 42/314/999 (pure lottery-ticket noise).

## Method

Our current 1.082 submission enables a per-head sigmoid gate on the attention output (`SKIP_GATES_ENABLED=1` in `submission/train.py`). Hard probe evidence (§11.5 of `STACK_UTILISATION_RESULTS.md`) shows zeroing the 11 `attn.gate_proj.weight` tensors improves val NLL by 1.02% — the gate is actively hurting. Cross-seed analysis (§12) confirms these weights are noise: std > mean across three independent seeds.

Full ablation: retrain with `SKIP_GATES_ENABLED=0`, remove the parameters entirely so they don't waste budget. Reclaim 45 KB of model-artifact space (either leave empty → brotli shrinks faster, or re-spend on widened MLP / extra n-gram table / embedding precision).

```python
# submission/train.py: the change is a single env-var flip, but add a sanity check
assert int(os.environ.get("SKIP_GATES_ENABLED", "1")) == 0, \
    "gated attention disabled per IDEA-001"
# → h.skip_gates_enabled is already wired to None when False, so no further code change needed
```

**Integration point**: env var `SKIP_GATES_ENABLED=0` in `submission/run.sh`. No code changes. Reduction in `skip_gates` parameter tensors = 45 KB uncompressed.

## Expected BPB

- **Range**: [-0.012, -0.008]
- **Mechanism**: P15 ablation zeroed the gate weights and observed −1.02% NLL (from 7.533 to 7.457). That's a *direct* measurement of the effect. Converting 1% nat change to BPB at ~2.35 B/tok: roughly 0.010 BPB.
- **Lower bound**: −0.005 BPB (if the ablation-test delta doesn't fully transfer to a from-scratch retrain)
- **Upper bound**: −0.015 BPB (if retraining without gates lets other params absorb what the gates were doing, compounding)

## Testable prediction

- **Metric**: val_bpb (quantized_sliding_window)
- **Threshold**: ≤ 1.074 (from baseline 1.082, Δ = −0.008)
- **After**: 600 s wallclock, 3 seeds (42, 314, 999)
- **On**: 1×H100 SXM first, then 8×H100 if positive

## Falsification criterion

Kill this idea if 3-seed mean val_bpb ≥ 1.079 (less than 0.003 BPB improvement — below comp's 0.005-nat significance bar).

If 1.074 ≤ mean ≤ 1.079 → partial; consider re-spending the reclaimed 45 KB on a different component (wider MLP hidden 2048 → 2176).

## Stacking plan

- **Composes with**: all other L03/L04/L05 ideas (IDEA-002 QK-Gain 5.25, IDEA-003 EMA 0.9965, IDEA-004 Pre-Quant TTT). No conflict.
- **Conflicts with**: nothing documented.
- **Blocks**: IDEA-014 (wider MLP) which needs the reclaimed 45 KB.
- **Budget footprint**: reclaims 45 KB uncompressed (~10-20 KB after brotli). Either left fallow (small artifact shrink) or re-spent.

## Prior-art audit

- **Arxiv (2023-2026)**: gated attention variants are well-studied. Our finding that it *hurts* at byte-level 16MB scale is specific to this config.
- **Comp PRs**: comp SOTA shipped skip-gates as a positive (our own submission does too). Nobody has PR'd removal based on per-seed instability.
- **Verdict**: re-test of own-stack component; novelty is the removal justified by cross-seed probe evidence.
- **Checked by**: claude 2026-04-16

## Lineage

- **Where this came from**: `STACK_UTILISATION_RESULTS.md §11.5 P15` (gate ablation) + `§11.6 P11` (cross-seed noise on attn.gate_proj)
- **Supersedes**: n/a (this is a removal, not a replacement)

## Risks

- Gated attention may be doing something at inference time that the zeroing ablation didn't capture (e.g. stabilizing layer norm distributions). Mitigation: check per-layer loss curves during training.
- The 45 KB reclaim may be absorbed by brotli without changing real BPB capacity — confirm artifact_bytes before and after.

## Notes

Revision 0 — 2026-04-16: initial approved candidate, queued for first experiment fire.
