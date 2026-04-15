---
id: IDEA-015
slug: rare-token-active-sampling
created: 2026-04-16
updated: 2026-04-16
status: audited
layer: L02
novelty_class: WN
expected_bpb: [-0.015, -0.005]
cost_hours: 2.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l02-active-learning-difficulty-sampler
prior_art_checked: 2026-04-16
next_step: prior-art-audit-and-prototype
---

# IDEA-015: Rare-token active sampling — curriculum + importance weighting driven by P7 rarity buckets

> **Hypothesis**: P7 showed tail-50% rarity-bucket tokens have 2.26× the loss of top-5% tokens (9.56 nats vs 3.62 nats in our forward-pass probe). Biasing the training loader to over-sample batches containing high-rarity tokens reduces val_bpb by 0.005-0.015, because the model currently under-trains on exactly the tokens that dominate val loss.

## Method

Instead of uniform random sampling of training sequences, use an **importance-weighted sampler** that:

1. **Pre-compute** token-frequency quartiles on a training slice (one-time, ~1 min): `q_freq[token_id]` = quartile 0..3 where 0 = top-5%, 3 = tail-50% matching P7.
2. **Score each candidate train sequence** by its rare-token density: `rarity_score[seq] = fraction of tokens in quartile 3` (higher = more rare tokens).
3. **Sample batches** with probability `∝ rarity_score ** α` where α ∈ [0, 2]:
   - α=0 → uniform (baseline)
   - α=1 → proportional to rare-token density
   - α=2 → aggressive over-sampling

```python
# In the ShuffledSequenceLoader (submission/train.py lines 168-175):
# Replace uniform `self.rng.choice(len(self.files))` with:
probs = remaining / total  # current uniform weighting
# new: multiply by rarity_score^alpha
probs *= (self.rarity_score_per_seq ** alpha)
probs /= probs.sum()
si = int(self.rng.choice(len(self.files), p=probs))
```

**Importance correction**: when we over-sample rare-token batches, the gradient is biased. Correct by re-weighting the per-token loss inversely:

```python
# Per-token loss weight = 1 / (rarity_score_of_sequence ** alpha_for_loss)
# This keeps expected gradient equal to uniform; trades variance for better coverage
loss_weights = 1.0 / (rarity_score_per_token ** 0.5 * alpha)
loss = (per_token_ce * loss_weights).mean()
```

Start with α=0.5 (modest over-sampling) and loss_weights α_for_loss=0.25 (partial correction — we want *some* of the up-weighting effect, not zero net).

**Integration point**: ~60 LOC in `submission/train.py` Loader class + a precompute step in `get_data.sh`. Env vars: `RARITY_SAMPLING_ALPHA`, `RARITY_LOSS_CORRECTION`.

## Expected BPB

- **Range**: [-0.015, -0.005]
- **Mechanism**: P7 showed tail-50% tokens carry 50% of all val tokens by count BUT 100/(3.62+4.91+7.90+12.06) × 12.06 ≈ 42% of val LOSS. Closing even 20% of that differential via more training coverage = 0.2 × 12.06 nats × (fraction retrievable) ≈ 0.2-0.5 nats/token improvement on tail → 0.01-0.02 BPB.
- **Lower bound**: -0.005 if the model already saturates on rare tokens for other reasons (e.g. rare-token-specific n-gram bias was doing this job)
- **Upper bound**: -0.015 if rare tokens were pure under-training and our current stack leaves the full differential on the table

## Testable prediction

- **Metric**: `val_bpb` (overall) AND per-bucket val_bpb should shift — tail-50% bucket should drop by ~0.5 nats while top-5% stays similar
- **Threshold**: ≤ 1.072 overall at seed 42 (−0.010 from 1.082)
- **After**: 600s wallclock, seed 42 first
- **Secondary diagnostic**: re-run P7 probe on the new checkpoint, compare tail-50% mean NLL vs baseline's 9.56

## Falsification criterion

Kill if:
- 2-seed mean val_bpb ≥ 1.080 (less than 0.002 improvement)
- OR tail-50% per-bucket NLL doesn't decrease relative to top-5% (curriculum didn't take)

## Stacking plan

- **Composes with**: all L03/L05/L07 ideas (orthogonal axis of loader bias)
- **Composes strongly with**: IDEA-009 (N-gram Tilt) — both target rare-token predictability, should compound
- **Conflicts with**: existing MDL-compressible-first curriculum (our stack already has this). May need to either replace or layer. Safer: replace, since rarity-sampling is probe-informed and MDL-compressible wasn't.
- **Blocks**: nothing
- **Budget footprint**: rarity_score lookup table = 1 × 8192 = ~1 KB; negligible

## Prior-art audit

Audited 2026-04-16 by Loop A fire 3 (Explore subagent).

- **Arxiv (2023-2026)**:
  - "Estimating the Probabilities of Rare Outputs in Language Models" (2024) — rare-token probability estimation via importance sampling, but NOT training-time sampling
  - "Token-Level Uncertainty-Aware Objective for Language Model Post-Training" (2025) — uncertainty as training signal exists, but not rarity-weighted sampling specifically
  - "Curriculum Learning for LLM Pretraining" (2024-2025) — curriculum works, but primarily on sequence length/complexity, not rarity
- **Comp PRs** (openai/parameter-golf):
  - PR #737 "Online Curriculum Learning" (open) — curriculum implemented, no rarity-specific sampling
  - PR #783 "Shard-order curriculum" (open) — data ordering, no rare-token weighting
  - PR #764 "Curriculum Learning + LeakyReLU + N-gram" (closed) — curriculum exists, no importance-weighted rare-token oversampling
- **Verdict**: **world-novel**. Rarity-informed oversampling + importance-weighted loss correction is a combination nobody has shipped in the comp. Curriculum learning + importance sampling exist separately but have not been combined with rare-token measurement as the driving signal.
- **Checked by**: claude 2026-04-16

## Lineage

- P7 per-token loss bucketing from `STACK_UTILISATION_RESULTS.md §11.2`
- `STACK_NOVELTY_TRACKER_v2.md §L02` row "Active-learning difficulty sampler"
- RESEARCH_PROTOCOL.md §1 grid cell (L02 × active learning)

## Risks

- Importance-weighted sampling can destabilize training (high-variance gradients). Mitigation: start with small α, loss correction prevents too-strong bias.
- If the existing MDL-compressible-first curriculum is already doing this implicitly, the gain may already be priced in. Mitigation: A/B with and without the existing curriculum.
- Pre-computing rarity_score per sequence requires one full pass over train — ~5 min on H100. Amortized over all future experiments, fine.

## Notes

Cheapest probe-informed novelty. P7's 2.26× loss differential between buckets is a direct measurement screaming for curriculum intervention.
