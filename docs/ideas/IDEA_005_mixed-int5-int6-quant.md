---
id: IDEA-005
slug: mixed-int5-int6-quant
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L07
novelty_class: CP
expected_bpb: [-0.003, -0.001]
cost_hours: 1.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l07-mixed-int5int6-per-layer
prior_art_checked: 2026-04-16
next_step: port-Hessian-weighted-int5int6-allocator
---

# IDEA-005: Port Hessian-weighted mixed int5/int6 allocation

> **Hypothesis**: Allocating int5 to low-variance layers and int6 to high-variance layers (per GPTQ Hessian) reduces val_bpb by 0.001-0.003 while shrinking the compressed artifact, matching comp PRs #1429 and #1438.

## Method

Current quant: all matrices int6, embedding int8. Comp frontier: Hessian-driven per-layer bit allocation — lowest-variance layers get int5, highest get int6, embedding stays int8.

```python
# Rough algorithm: sort layers by Hessian trace / Frobenius norm; bottom 30% → int5, rest → int6
# Plumb MATRIX_BITS_PER_LAYER=int6,int6,int5,int5,int6,... (comma list by layer)
# or: add a threshold env var MATRIX_BITS_SPLIT_PCT=0.30 (bottom 30% get int5)
```

Implementation: extend `dequantize_mixed` + `gptq_mixed_quantize` in records/.../train_gpt.py to accept a per-layer bits config. Not a single env var; needs ~30 LOC.

## Expected BPB

- **Range**: [-0.003, -0.001]
- **Mechanism**: int5 loses precision but most layers can tolerate it; moving 30% of matrix budget int6→int5 saves ~2 MB compressed, which re-enables a wider MLP or extra layer.
- **Caveat**: the BPB delta may come mainly from re-spending the reclaimed budget, not from the quant itself. In that case, couple this idea with "add 1 layer" in the same experiment.

## Testable prediction

- artifact_bytes reduction ~1-2 MB, val_bpb ≤ 1.081. If coupled with +1 layer, val_bpb ≤ 1.078.

## Falsification criterion

Kill if mean val_bpb ≥ 1.083 (regression) at 2 seeds.

## Stacking plan

- Composes with IDEA-001/002/003/004 (all upstream changes).
- Blocks: widening MLP / adding layers (those want the reclaimed budget).

## Prior-art audit

- Comp PRs #1429, #1438 shipped this. CP port.

## Lineage

- `STACK_NOVELTY_TRACKER_v2.md §L07 Mixed int5/int6` row marked as actionable port.

## Risks

- Naive int5 on attention may damage quality disproportionately. Mitigation: start with MLP-only int5, keep attn at int6.

## Notes

Revision 0 — 2026-04-16.
