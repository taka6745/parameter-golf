---
id: IDEA-010
slug: wavelet-gpt-verify
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L03
novelty_class: WN
expected_bpb: [-0.005, 0.005]
cost_hours: 0.5
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l03-wavelet-gpt-multi-scale-embedding-mixing
prior_art_checked: 2026-04-16
next_step: ablation-with-and-without-wavelet
---

# IDEA-010: Verify Wavelet GPT contribution in current stack

> **Hypothesis**: Wavelet GPT shipped with a measured −0.018 BPB at 500 steps but may have regressed after stacking with gated attention, parallel residuals, and depth recurrence. A controlled A/B (with vs without) on the current stack tells us whether it still contributes positively.

## Method

Toggle wavelet mixing via env var (if plumbed) or patch the model class to bypass. Run both configs at identical seed + wallclock.

```bash
# A: WAVELET_GPT_ENABLED=1 (current default)
# B: WAVELET_GPT_ENABLED=0
# Compare val_bpb over 2 seeds each.
```

## Expected BPB

- **Range**: [-0.005, +0.005] — either it still helps (~-0.005) or it no longer contributes (~0).
- Possibility it actually hurts: priors say unlikely but cross-component interactions exist.

## Testable prediction

- If WAVELET_GPT_ENABLED=0 delta ≤ -0.003 BPB vs ENABLED=1 at 2 seeds, Wavelet is dead — remove it and reclaim budget.
- If delta ≥ +0.003 BPB, Wavelet is still earning its keep, keep it.

## Falsification criterion

- If delta is in [-0.003, +0.003], call it neutral — consider removing for artifact simplicity.

## Stacking plan

- Solo verification experiment. Independent of other IDEAs.

## Prior-art audit

- Our own shipped technique. Re-verifying.

## Risks

- Nothing besides losing a few seeds of compute.

## Notes

Cheap diagnostic; informs whether Wavelet's budget can be re-spent.
