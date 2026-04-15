---
id: IDEA-018
slug: cma-es-rare-token-params
created: 2026-04-16
updated: 2026-04-16
status: draft
layer: L04
novelty_class: WN
expected_bpb: [-0.015, -0.005]
cost_hours: 4.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l04-cma-es-for-rare-token-param-subset
prior_art_checked: null
next_step: prior-art-audit-then-prototype
---

# IDEA-018: CMA-ES fine-tuning of the rare-token-correlated param subset

> **Hypothesis**: After main training completes, applying CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to a small subset of parameters (≤~2000) identified as most correlated with rare-token loss reduces val_bpb by 0.005-0.015. Reason: the P7 tail-50% rare-token bucket carries 2.26× per-token loss vs top-5% top-bucket, and those high-loss tokens may live in the specific params the gradient-based Muon optimizer under-fits due to rarity of their gradient signal.

## Method

Two-phase:

**Phase A — identify the rare-token-correlated param subset.**

1. Load our 1.082-baseline checkpoint
2. On a val sample, compute per-token loss (already done as P7 probe data)
3. Split val tokens into (rare = tail-50% bucket) and (common = top-5% bucket)
4. Compute per-parameter attribution: `A[θ_i] = Σ_rare-tokens grad(loss_rare_tok, θ_i) · θ_i - Σ_common-tokens grad(loss_common_tok, θ_i) · θ_i`
5. Rank `A[θ_i]` by magnitude; pick top-k (k=1000-2000) params that disproportionately influence rare-token loss
6. These become the **rare-token subset** — the free variables for CMA-ES

**Phase B — CMA-ES optimize the subset.**

Standard CMA-ES with:
- Dimension d = |subset| (≤2000)
- Population λ = 4 + 3·log(d) ≈ 26
- Objective: `rare_token_val_loss(θ_subset_proposal | rest_frozen)` on a held-out rare-token-sequence subset
- Budget: 50 generations = ~1300 evals × 600ms/eval = ~15 min on H100 (fits in non-record track easily)

```python
# Phase A attribution (pseudocode)
model.load_state_dict(checkpoint)
param_tensors = [p for p in model.parameters() if p.requires_grad]
rare_grad = [torch.zeros_like(p) for p in param_tensors]
common_grad = [torch.zeros_like(p) for p in param_tensors]

for batch in val_loader:
    logits = model(batch.x)
    per_token_loss = F.cross_entropy(logits, batch.y, reduction='none')
    for tok_idx in rare_indices(batch.y):
        g = torch.autograd.grad(per_token_loss[tok_idx], param_tensors, retain_graph=True)
        for ac, gi in zip(rare_grad, g): ac += gi.abs()
    for tok_idx in common_indices(batch.y):
        g = torch.autograd.grad(per_token_loss[tok_idx], param_tensors, retain_graph=True)
        for ac, gi in zip(common_grad, g): ac += gi.abs()

attribution = [r - c for r, c in zip(rare_grad, common_grad)]
flat = torch.cat([a.flatten() for a in attribution])
_, top_k_idx = flat.topk(1000)
# Phase B: CMA-ES over flat[top_k_idx], keep rest frozen
```

**Integration point**: a new post-training phase invoked from `submission/run.sh` via `CMAES_RARE_TOKEN_ENABLED=1`, ~200 LOC between Phase A (attribution) and Phase B (pycma or custom CMA-ES). Runs after the main training but before GPTQ quantization.

## Expected BPB

- **Range**: [-0.015, -0.005]
- **Mechanism**: CMA-ES has strong sample efficiency on low-dim search spaces. 1000-param fine-tuning gives local improvement beyond what gradient descent extracts when the gradient signal is sparse (rare tokens). P7 showed rare-tokens are the loss bottleneck; if even 10% of their loss is recoverable via targeted param nudges, that's ~0.3 nats × tail-50%-fraction = ~0.04 total NLL = ~0.015 BPB at 2.35 B/tok.
- **Lower bound**: -0.005 (if rare-token loss is mostly structural/architectural and not fine-tunable via param nudges)
- **Upper bound**: -0.015 (if CMA-ES finds multiple compounding rare-token wins)

## Testable prediction

- **Primary**: val_bpb ≤ 1.077 at seed 42 (−0.005 from 1.082), with **per-bucket measurement**: rare-token bucket NLL should drop while common-token NLL should stay flat
- **After**: full CMA-ES run (~15 min), seed 42 first; if positive, confirm at 314/999

## Falsification criterion

Kill if:
- val_bpb ≥ 1.081 at 2 seeds (not enough improvement)
- OR rare-token NLL doesn't drop more than common-token NLL (CMA-ES didn't target what we intended)

## Stacking plan

- **Composes strongly with**: IDEA-015 (rare-token active sampling; IDEA-015 gives better main-training rare-token fit, then IDEA-018 polishes it)
- **Composes with**: IDEA-009 (n-gram Tilt) — multiplicative rare-token bias at eval, CMA-ES at train
- **Conflicts with**: nothing structural
- **Blocks**: nothing
- **Budget footprint**: 0 bytes (post-training polish on existing params; no new parameters)

## Prior-art audit

_To be filled by next Loop A fire with Explore subagent._

- **Arxiv (2023-2026)**: search "CMA-ES language model fine-tuning", "evolutionary optimization transformer parameters", "rare token optimization LLM"
- **Comp PRs**: grep for `cma`, `evolutionary`, `rare-token-optim` in comp PR titles
- **Verdict**: TBD; CMA-ES on neural nets is known (ES / NE community) but the rare-token-correlated subset selection is novel
- **Checked by**: _pending_

## Lineage

- P7 probe data (STACK_UTILISATION_RESULTS.md §11.2) — the rare-token loss dominance is what motivates this
- `STACK_NOVELTY_TRACKER_v2.md §L04` row "CMA-ES for rare-token param subset"
- RESEARCH_PROTOCOL.md §1 grid cell (L04 × evolutionary strategies)

## Risks

- CMA-ES on 2000-dim is borderline; convergence may be slow. Mitigation: start with k=500, ramp to k=2000 only if convergence is OK.
- Attribution via per-token gradients is O(n_tokens × n_params) — needs careful batching to fit on H100. Mitigation: use layer-wise attribution if per-param is too expensive.
- Gradient attribution may not identify the right subset — the params that ARE under-fit in training may not be the ones with the highest attribution. Mitigation: also try "Hessian-diagonal × gradient" as an alternative subset selection.

## Notes

This is a **probe-informed novelty** — only possible because P7 measured per-bucket loss. Would not have been obvious as a direction without probe data. Strong stacking potential with IDEA-015 rare-token sampling.
