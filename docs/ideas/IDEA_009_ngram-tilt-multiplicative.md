---
id: IDEA-009
slug: ngram-tilt-multiplicative
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L06
novelty_class: WN
expected_bpb: [-0.006, -0.003]
cost_hours: 1.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l06-n-gram-tilt-multiplicative-boost
prior_art_checked: 2026-04-16
next_step: implement-multiplicative-ngram-tilt
---

# IDEA-009: N-gram "Tilt" (multiplicative bias, not additive)

> **Hypothesis**: Replacing our additive n-gram logit bias with a multiplicative boost `exp(β·𝟙[t==hint])/Z` where β≈1.0, reduces val_bpb by 0.003-0.006. World-novel (no published work combines multiplicative n-gram boost at byte scale with our stack).

## Method

Current bias: `logits[t] += α · bigram_score[t]` (additive on logits).

Proposed (Tilt): `p(t) = (1-λ)·p_lm(t) + λ·𝟙[t ∈ ngram_topk]` implemented as `logits[t] += β` for tokens where the n-gram cache hints with high confidence, zero elsewhere. The multiplicative (in probability space) shape is different from additive-in-log-space and should compose better with Score-First TTT.

```python
# Hint selection: for each ctx, pick top-k bigram/trigram successors by score
# Boost logits[t] += β only for those tokens; β~1.0 after tuning
for t in ngram_topk[ctx]:
    logits[..., t] += beta
```

Start with β = 1.0, k = 8 topk per order.

## Expected BPB

- **Range**: [-0.006, -0.003]
- **Mechanism**: additive bias is linear; hinted tokens get roughly the same boost regardless of LM's prior confidence. Tilt is effectively multiplicative on p (= additive on log p for hinted tokens only), which lets hints amplify high-confidence LM picks rather than overriding them.

## Testable prediction

- val_bpb ≤ 1.079 at seed 42.

## Falsification criterion

Kill if val_bpb ≥ 1.082 (neutral or worse).

## Stacking plan

- Composes with L08 tables (our existing) and L06 mixer.
- Conflicts with additive bias (replaces). Must gate via env var to allow A/B.

## Prior-art audit

- Arxiv: "multiplicative bias" for LM logits is used in ranking / re-ranking literature (e.g. classifier-free guidance) but not at byte-LM comp scale.
- Comp PRs: none found that use multiplicative n-gram boost.
- Verdict: WN world-novel-candidate.

## Risks

- β tuning sensitivity. Mitigation: sweep β ∈ {0.5, 1.0, 1.5} in parallel.

## Notes

Cheapest WN novelty here — if it lands, it's paper-worthy on its own.
