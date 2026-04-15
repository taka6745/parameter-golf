---
id: IDEA-022
slug: bayesian-ttt-averaging
created: 2026-04-16
updated: 2026-04-16
status: audited
layer: L06
novelty_class: WN
expected_bpb: [-0.012, -0.003]
cost_hours: 2.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l06-bayesian-model-averaging-over-ttt-snapshots
prior_art_checked: 2026-04-16
next_step: prior-art-audit-then-prototype
---

# IDEA-022: Bayesian model averaging over Score-First TTT snapshots

> **Hypothesis**: Instead of using ONLY the final TTT-adapted weights at each val chunk, snapshot intermediate weights after each inner SGD step and average their output logprobs with Bayesian weights proportional to the (held-out) training-data likelihood per snapshot. Reduces val_bpb by 0.003-0.012 by marginalizing over the TTT trajectory's uncertainty, not just its endpoint.

## Method

Current Score-First TTT: for each val chunk, run 3 SGD steps on the chunk (after scoring its tokens causally). Use the final weights to score the next chunk.

Proposed: snapshot weights after each of K inner SGD steps: `θ_0, θ_1, θ_2, ..., θ_K`. Compute per-snapshot weight `w_k ∝ p(heldout | θ_k)` where heldout is a small slice of the current chunk held aside specifically for BMA weight estimation. Final prediction for next-chunk token:

```python
# TTT with BMA
theta_snapshots = [theta_init]
for step in range(K):
    grad = compute_grad(theta_snapshots[-1], current_val_chunk_seen_tokens)
    theta_new = theta_snapshots[-1] - lr * grad
    theta_snapshots.append(theta_new)

# Weight each snapshot by held-out log-likelihood
w = [heldout_loglik(theta_k) for theta_k in theta_snapshots]
w = softmax(w)

# Score next chunk: ensemble over snapshots
for token_idx in next_chunk:
    p_per_snapshot = [forward(theta_k, ctx)[token_idx] for theta_k in theta_snapshots]
    p_ensemble = sum(w_k * p_k for w_k, p_k in zip(w, p_per_snapshot))
    score += -log(p_ensemble)
```

Cost: K forward passes per val chunk instead of 1, so eval wallclock scales by K. With K=3 (matching current TTT_EPOCHS), that's 3× eval time — within the 10-min eval budget if our current eval is <3 min.

**Integration point**: wraps the existing Score-First TTT eval loop. ~80 LOC in `submission/train.py`. Env var `TTT_BMA_ENABLED=1`.

## Expected BPB

- **Range**: [-0.012, -0.003]
- **Mechanism**: the TTT trajectory passes through different effective models. Early snapshots capture "less-specialized" behavior; late snapshots capture "more-specialized to current chunk". BMA marginalizes over this uncertainty. Classic Bayesian ML shows ensembling over the posterior typically improves calibration → BPB.
- **Lower bound**: -0.003 if the snapshots are too similar (TTT hasn't meaningfully diverged the trajectory)
- **Upper bound**: -0.012 if the trajectory has real variance and BMA captures the optimal mixture

## Testable prediction

- **Metric**: val_bpb at seed 42 with TTT_BMA_ENABLED=1 vs baseline TTT
- **Threshold**: ≤ 1.076 (−0.006 from 1.082)
- **Secondary**: eval wallclock should stay under 10 min (budget check)

## Falsification criterion

Kill if:
- val_bpb ≥ 1.082 at 2 seeds (no improvement from ensembling)
- OR eval wallclock > 10 min (blows eval budget on 8×H100 comp-spec)

## Stacking plan

- **Composes with**: all training-time ideas
- **Composes with**: IDEA-017 (MAML-pretrained TTT init) — MAML gives better snapshots, BMA averages them more usefully
- **Conflicts with**: nothing structural
- **Blocks**: nothing
- **Budget footprint**: 0 bytes (eval-time logic only; no new model params)

## Prior-art audit

Audited 2026-04-16 by Loop A fire 21 (Explore subagent).

- **Arxiv (2023-2026)**:
  - "Test-Time Learning for Large Language Models" (2025, arxiv 2505.20633) — dynamic LLM adaptation at test time, doesn't address snapshot ensembling
  - "When, Where and Why to Average Weights?" (Feb 2025, arxiv 2502.06761) — checkpoint averaging strategies, **training-time trajectory**, not inner TTT SGD snapshots
  - Classical: Snapshot Ensembles (Huang 2017), BMA (Hoeting 1999) — foundational but not applied to TTT inner steps
- **Comp PRs** (openai/parameter-golf): **none found** for `snapshot`, `bma`, `bayesian_average`.
- **Verdict**: **partial-overlap-with-trajectory-ensembles / world-novel-for-byte-LM-TTT**. Applying BMA to TTT's inner SGD trajectory with held-out likelihood weighting at byte-LM scale is not documented. Snapshot Ensembles + BMA exist classically, but the specific combination is novel.
- **Checked by**: claude 2026-04-16

## Lineage

- `STACK_NOVELTY_TRACKER_v2.md §L06` row "Bayesian model averaging over TTT snapshots"
- RESEARCH_PROTOCOL.md §1 grid cell (L06 × Bayesian model averaging)
- Motivated by P7 calibration signal: if the model is over-confident on rare tokens, ensembling should improve calibration

## Risks

- Holdout slice for BMA weighting might bias the eval (taking some of the chunk away from scoring). Mitigation: use the first few tokens of EACH chunk as the BMA calibration set (score them normally, then use their logliks).
- Eval wallclock 3× on 8×H100 might blow the 10-min budget. Mitigation: verify with a smoke run at K=3 on 1×H100 first; only ship K that fits 8×H100 budget.
- Snapshots may be near-duplicates (small SGD steps); ensemble gives no benefit. Mitigation: diagnose via checking weight-diff-norms between snapshots before committing.

## Notes

Lowest-cost L06 novelty in the queue. If the overhead fits the eval budget, it's almost "free BPB".
