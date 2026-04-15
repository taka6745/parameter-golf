---
id: IDEA-011
slug: embed-int8-to-int6
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L07
novelty_class: WN
expected_bpb: [-0.010, -0.002]
cost_hours: 1.5
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l07-embedding-int8-int6
prior_art_checked: 2026-04-16
next_step: retrain-embed-int6-and-measure-artifact-shrink
---

# IDEA-011: Embedding int8 → int6 (reclaim ~1 MB per P2 Shannon entropy)

> **Hypothesis**: P2 probe showed token embedding has 4.72 bits of Shannon entropy in an int8 container (69 distinct values / 256 possible). Re-quantizing to int6 (64 levels) with GPTQ calibration should NOT hurt val_bpb and will reclaim ~1.05 MB. Re-spend the reclaimed MB on a wider MLP hidden_dim (2048 → 2560) for +0.005-0.010 BPB improvement.

## Method

Two changes, must ship together:
1. `EMBED_BITS=6` (was 8)
2. `MLP_MULT=5.0` (was 4.0, consumes reclaimed budget)

```bash
EMBED_BITS=6
MLP_MULT=5.0
MATRIX_CLIP_SIGMAS=12.85  # same
```

Alternative path: keep MLP_MULT at 4.0 and add a 12th transformer layer with reclaimed budget.

## Expected BPB

- **Range**: [-0.010, -0.002]
- **Mechanism**: embed has slack per P2; wider MLP uses the reclaimed bytes for capacity. The model doesn't lose expressiveness from int6 embed (same 64 effective levels), and gains from wider MLP.

## Testable prediction

- artifact_bytes decreases by ~1 MB before MLP widening, increases back to ~16 MB after.
- val_bpb ≤ 1.077 at seed 42 with both changes.

## Falsification criterion

- Kill the embed int6 half if val_bpb ≥ 1.082 WITHOUT MLP widening (means int6 embed itself hurts).
- Kill the MLP widen half if widening doesn't push val_bpb below 1.079.

## Stacking plan

- Strongly composes with L03 wider-MLP direction.
- Composes with all L05/L06/L08 changes.
- Blocks: IDEA-007 (DEQ) — they use the same budget-reclaim argument differently.

## Prior-art audit

- Nobody in comp has shipped embed int6 + wider MLP stack together. Our probe data (P2, P5) is the evidence base.
- Verdict: WN (probe-informed novel allocation).

## Risks

- GPTQ calibration on embed at int6 may have unusual artifacts; the 138 unseen vocab rows (P5) may clip awkwardly.
- Mitigation: use EMBED_CLIP_SIGMAS=20.0 (current) — generous clip; verify by spot-checking unseen rows' embedding vectors before vs after.

## Notes

Highest-value probe-informed IDEA. Two-step: embed shrink first (verify artifact size), then add MLP widening in a follow-up experiment.
