---
id: IDEA-003
slug: ema-0.9965-port
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L05
novelty_class: CP
expected_bpb: [-0.003, -0.001]
cost_hours: 0.5
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l05-ema-weight-averaging-decay0997
prior_art_checked: 2026-04-16
next_step: retrain-EMA-0.9965-seed42
---

# IDEA-003: Port EMA decay 0.9965 from comp frontier (we still at 0.997)

> **Hypothesis**: Bumping EMA decay from 0.997 → 0.9965 reduces val_bpb by 0.001-0.003 by smoothing over more of training history, matching comp PRs #1421 and #1471.

## Method

Single env var. EMA averages weights across training steps; higher decay = longer averaging window.

```bash
EMA_DECAY=0.9965  # was 0.997
```

## Expected BPB

- **Range**: [-0.003, -0.001]
- **Mechanism**: longer averaging reduces seed variance, typically worth 0.001-0.002 BPB per comp SOTA reports

## Testable prediction

- val_bpb ≤ 1.081 at seed 42; confirm at seeds 314/999 if positive

## Falsification criterion

Kill if mean ≥ 1.0815 at 2 seeds.

## Stacking plan

- Composes with everything. Budget: 0 bytes.

## Prior-art audit

- Comp PRs #1421, #1471. CP port, not novel.

## Lineage

- `STACK_NOVELTY_TRACKER_v2.md §L05` "Actionable port: bump EMA_DECAY"

## Risks

- Minimal; well-understood knob.

## Notes

Revision 0 — 2026-04-16.
