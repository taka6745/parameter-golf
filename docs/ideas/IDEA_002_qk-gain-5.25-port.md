---
id: IDEA-002
slug: qk-gain-5.25-port
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L03
novelty_class: CP
expected_bpb: [-0.005, -0.002]
cost_hours: 0.5
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l03-qk-gain-per-head-scalar-init40
prior_art_checked: 2026-04-16
next_step: retrain-QK-5.25-seed42
---

# IDEA-002: Port QK-Gain 5.25 from comp frontier (we still at 4.0)

> **Hypothesis**: Increasing the per-head QK-Gain init from 4.0 → 5.25 reduces val_bpb by 0.002-0.005, matching the delta the comp frontier shipped in records from 2026-03-22 onwards.

## Method

`QK_GAIN_INIT` is a per-head scalar multiplied into the QK dot product before softmax (sharpens attention). Our 1.082 uses 4.0; comp SOTA (PR #1394, #1493) moved to 5.0-5.25 with measured −0.001 BPB per 0.5 gain step.

```python
# env var change only
QK_GAIN_INIT=5.25  # was 4.0
```

**Integration point**: env var in `submission/run.sh`. Zero code changes.

## Expected BPB

- **Range**: [-0.005, -0.002]
- **Mechanism**: sharper attention lets the model commit more confidently to high-information tokens. Probe P7 shows rare tokens carry 2.3× the loss — sharper attention should help the model attend to the rare-relevant context positions.
- **Lower bound**: −0.002 (comp's own measurement was −0.001/step × 2.5 steps)
- **Upper bound**: −0.005 (if rare-token context attention benefits disproportionately)

## Testable prediction

- **Metric**: val_bpb
- **Threshold**: ≤ 1.080 (−0.002 from 1.082 baseline)
- **After**: 600 s wallclock, seed 42 first, confirm at 314/999 if positive

## Falsification criterion

Kill if mean val_bpb ≥ 1.081 at 2 seeds.

## Stacking plan

- **Composes with**: IDEA-001 (gated attn removal), IDEA-003 (EMA 0.9965). All independent knobs.
- **Conflicts with**: nothing.
- **Blocks**: nothing.
- **Budget footprint**: 0 bytes (just a scalar tweak).

## Prior-art audit

- **Comp PRs**: shipped in comp SOTA frontier #1394 onwards. Known port.
- **Verdict**: comp-port (CP), not world-novel.
- **Checked by**: claude 2026-04-16

## Lineage

- `STACK_NOVELTY_TRACKER_v2.md §L03 QK-Gain` row marked as "Actionable bump to 5.25"

## Risks

- May over-sharpen at some layer/head combos and destabilize. Mitigation: watch loss curves for first 100 steps.

## Notes

Revision 0 — 2026-04-16: initial approved candidate.
