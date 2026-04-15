---
id: IDEA-024
slug: contrastive-pretraining-phase
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L05
novelty_class: WN
expected_bpb: [-0.010, -0.003]
cost_hours: 3.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l05-contrastive-pretraining-phase
prior_art_checked: 2026-04-16
next_step: prior-art-audit-then-prototype
---

# IDEA-024: Short contrastive pretraining phase before CE training

> **Hypothesis**: Spending the first ~30s of the 600s training budget on a contrastive-objective phase (pull positive byte-context pairs together in embedding space; push unrelated pairs apart) gives the model an information-dense starting representation that standard cross-entropy training builds on more efficiently. Net val_bpb improvement: 0.003-0.010.

## Method

Phase 1 (30s): contrastive pretraining using an InfoNCE-style loss on the token embedding + first few transformer layers:

```python
# For a batch of sequences of shape [B, T]:
# - Pick 2 random positions in each sequence: (pos_i, pos_j) with |i-j| ∈ [4, 32]
# - Compute embeddings: e_i, e_j via the current model's first K layers + projection head
# - Positive pair: (e_i, e_j) from the same sequence
# - Negative pairs: (e_i, e_j') from different sequences in the batch
# Loss: InfoNCE(e_i, e_j, {e_j' negatives}, temperature=0.1)

# The projection head is disposable (discarded after phase 1)
```

Phase 2 (570s): standard cross-entropy training as usual, starting from the weights produced by phase 1.

**Integration point**: add a phase-1 block to `submission/train.py` at train-start. ~80 LOC + a small projection-head module (2-layer MLP, 100K params — temporary, discarded before Phase 2). Env vars: `CONTRASTIVE_PRETRAIN_ENABLED=1`, `CONTRASTIVE_STEPS=150` (30s at ~5 steps/s on H100).

## Expected BPB

- **Range**: [-0.010, -0.003]
- **Mechanism**: contrastive pretraining on byte-level sequences encourages the model to develop discriminative embeddings early. Standard CE starts from these well-structured embeddings rather than near-random. At 10-min training budgets, small warmup-style wins of 0.003-0.010 are plausible.
- **Lower bound**: -0.003 if the embedding-space structure doesn't survive the CE phase (CE quickly overwrites contrastive init)
- **Upper bound**: -0.010 if contrastive structure in embeddings persists through CE and gives better gradients for the whole 570s

## Testable prediction

- **Metric**: val_bpb at seed 42
- **Threshold**: ≤ 1.079 (−0.003 from 1.082)
- **Secondary diagnostic**: phase-1 contrastive accuracy (fraction of times positive pair has higher similarity than all negatives) should climb >70% by end of phase-1

## Falsification criterion

Kill if:
- val_bpb ≥ 1.082 at 2 seeds (no improvement — contrastive pretraining didn't transfer)
- OR val_bpb ≥ 1.090 at seed 42 (contrastive pretraining ACTIVELY HURT; regression)

## Stacking plan

- **Composes with**: all downstream changes (L03 arch, L07 quant, L06 eval). Phase 1 is a 30s upstream modification.
- **Composes strongly with**: IDEA-017 (MAML TTT init) — both are "better init before the main training"
- **Conflicts with**: nothing structural
- **Blocks**: nothing
- **Budget footprint**: 0 bytes in artifact (projection head discarded before Phase 2)

## Prior-art audit

Audited 2026-04-16 by Loop A fire 27 (Explore subagent).

- **Arxiv (2023-2026)**:
  - "A Statistical Theory of Contrastive Pre-training and Multimodal Generative AI" (Jan 2025, arxiv 2501.04641) — theoretical framework for multi-modal; doesn't address byte-level LMs or short init phases
  - "Learning Transformer-based World Models with Contrastive Predictive Coding" (Mar 2025, arxiv 2503.04416) — CPC for temporal; no short-pretraining warmup connection
  - SimCSE (arxiv 2104.08821) — sentence embeddings, not byte-level
- **Comp PRs** (openai/parameter-golf): **none** found for `contrastive|infonce|simclr|simcse`.
- **Key gap**: prior work applies contrastive learning to sentences, vision-language, or multi-modal pretraining — NOT byte-level token initialization as a <1-min warmup before CE in sub-10-min training budgets. The contrastive-init→CE transfer at parameter-constrained byte-LM scale appears unexplored.
- **Verdict**: **world-novel**.
- **Checked by**: claude 2026-04-16

## Lineage

- `STACK_NOVELTY_TRACKER_v2.md §L05` row "Contrastive pretraining phase"
- RESEARCH_PROTOCOL.md §1 grid cell (L05 × contrastive pretraining)

## Risks

- 30s budget taken from main training — if contrastive doesn't give ≥30s worth of CE wins, net loss. Mitigation: tight phase-1 budget tuning; start with 15s.
- Contrastive signal may collapse to trivial solutions (all embeddings similar) on byte-level. Mitigation: careful temperature (0.1), normalize embeddings, use hard negatives.
- Projection head params count if not properly discarded. Mitigation: make discard explicit in code.

## Notes

Simple, classical idea. Cheap to test. If it lands, a nice stacking win with everything else.
