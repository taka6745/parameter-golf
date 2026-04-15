---
id: IDEA-008
slug: hymba-hybrid-retest
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L03
novelty_class: CN
expected_bpb: [-0.02, -0.005]
cost_hours: 3.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l03-hymba-hybrid-parallel-mambaattention
prior_art_checked: 2026-04-16
next_step: retest-hymba-with-current-stack
---

# IDEA-008: Re-test Hymba hybrid (parallel Mamba + Attention)

> **Hypothesis**: Hymba (parallel Mamba + Attention heads with sigmoid-mix) landed at 1.1189 in comp PR #852 — 0.004 from SOTA, never shipped. Re-running on our current n-gram/TTT/QK-gain stack should land below 1.082.

## Method

Each block gets: traditional attention heads + parallel Mamba-2 path; outputs are sigmoid-gated blended. Start from comp PR #852 architecture and graft onto our current training loop + bias stack.

```python
# Block forward pseudo:
def forward(x):
    attn_out = self.attn(self.ln1(x))
    mamba_out = self.mamba(self.ln1(x))  # parallel, not sequential
    g = torch.sigmoid(self.gate)
    fused = g * attn_out + (1 - g) * mamba_out
    x = x + fused
    return x + self.mlp(self.ln2(x))
```

**Integration point**: new attention block class; ~100 LOC. `pip install mamba-ssm`.

## Expected BPB

- **Range**: [-0.02, -0.005]
- **Mechanism**: Mamba path captures long-range linear-time context that attention struggles with at 2048 tokens; attention keeps short-range sharpness. P7 showed rare-token loss dominates — Mamba may help by capturing long-range semantic context that disambiguates rare tokens.

## Testable prediction

- val_bpb ≤ 1.077 at seed 42.

## Falsification criterion

Kill if val_bpb ≥ 1.082 at seed 42 (no improvement).

## Stacking plan

- Conflicts with current attention architecture (replaces, not adds).
- Composes with everything downstream (TTT, n-gram bias, DC500).

## Prior-art audit

- Comp PR #852 at 1.1189. CN (Hymba itself is published: NVIDIA 2024).
- Our angle: stacking Hymba onto our modern bias + TTT + compression stack; the comp PR was on an older stack.

## Risks

- Mamba training dynamics fragile at small scale; may need hyperparameter tuning.
- Kernel fusion between attn and Mamba at comp budget (600s) is hard.

## Notes

Medium-effort. Parallel candidate to IDEA-007 (DEQ) — both are "replace architecture" bets.
