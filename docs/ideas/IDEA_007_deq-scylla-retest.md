---
id: IDEA-007
slug: deq-scylla-retest
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L03
novelty_class: CN
expected_bpb: [-0.05, -0.01]
cost_hours: 4.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l03-deq-deep-equilibrium-with-scylla
prior_art_checked: 2026-04-16
next_step: retest-DEQ-with-current-best-ngram-stack
---

# IDEA-007: Re-test DEQ (Deep Equilibrium) + Scylla in our current environment

> **Hypothesis**: DEQ (1-layer with 22 implicit-function iterations) used only ~6.8 MB of model budget in comp PR #1323 at 1.1247 BPB. That leaves ~9.2 MB of budget for n-gram tables + bias stacks. Re-applying this with our current n-gram + TTT + bias stack should land below 1.082.

## Method

Replace the 11-layer transformer with a 1-layer DEQ that runs 22 iterations to equilibrium. This is a massive re-architecture. Use comp PR #1323's `train_gpt.py` as the starting point, strip out its (naive) bias tables and graft our current n-gram engine + DC500 + Context Engine + Legal TTT.

Budget accounting:
- DEQ model: 6.8 MB (comp baseline)
- Our bias stack: ~1 MB + DC500 1 MB + Context Engine 0.2 MB + TTT overhead
- Remaining: 7-8 MB for LARGER / ADDITIONAL n-gram tables — the real win

```python
# Fork records/track_10min_16mb/PR1323/train_gpt.py (if exists; else reconstruct from comp repo)
# Graft our NLFI bigram/trigram/fourgram + DC500 embeddings
# Increase DEQ depth iters or add side-expert
```

**Integration point**: a fork of our current `train.py` + DEQ substitution. ~150-300 LOC.

## Expected BPB

- **Range**: [-0.05, -0.01] — wide because we don't know how much of PR #1323's 1.1247 came from DEQ vs how much from their bias stack
- **Mechanism**: DEQ gives recurrent-depth for free (no extra params); reclaimed 9.2 MB can host substantially larger n-gram tables

## Testable prediction

- val_bpb ≤ 1.072 at seed 42 (cheap-pod POC)
- artifact_bytes ≤ 16 MB

## Falsification criterion

Kill if val_bpb ≥ 1.085 at seed 42 (regression vs our 1.082 baseline) — DEQ may not converge well on byte-level data in 10-min budget.

## Stacking plan

- **Composes with**: n-gram tables (the whole point), context engine, TTT
- **Conflicts with**: current 11-layer transformer architecture (this replaces it)
- **Blocks**: all other L03 architecture changes (major fork)

## Prior-art audit

- Comp PR #1323 prototyped at 1.1247. Not currently in SOTA stack.
- Arxiv: DEQ (Bai et al. 2019) is well-studied; the novelty is the budget-reclamation angle for parameter-constrained LMs.
- Verdict: CN (comp-novel stacking, known architecture).

## Risks

- DEQ convergence during training is fragile; may need many wallclock seconds per step.
- May not fit 600s training budget. Fallback: non-record track submission.

## Notes

**High-effort, high-payoff**. Only run if IDEA-001..006 together haven't closed the 0.012 BPB gap to SOTA — because DEQ is a much bigger experiment.
