---
id: IDEA-014
slug: arithmetic-coding-loss
created: 2026-04-16
updated: 2026-04-16
status: audited
layer: L06
novelty_class: WN
expected_bpb: [-0.025, -0.010]
cost_hours: 3.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l06-arithmetic-coding-loss-train-for-ac-rate-directly
prior_art_checked: 2026-04-16
next_step: prior-art-audit-and-prototype
---

# IDEA-014: Arithmetic-coding loss — train the model for AC bit rate directly, not cross-entropy

> **Hypothesis**: Replacing cross-entropy (which minimizes -E[log p(token)] averaged over tokens) with a direct arithmetic-coding rate loss that accounts for the quantization + entropy-coding pipeline reduces val_bpb by 0.010-0.025, because it aligns the training objective with the actual scoring metric rather than a proxy.

## Method

Current training: `loss = -log p(token | context)` per token, averaged over the batch. This is correct asymptotically for a perfect entropy coder but it ignores:
- Post-quantization rounding (int6 GPTQ changes effective probabilities)
- Brotli's block-level context model (some probability mass structures compress better than others)
- The 16 MB artifact constraint (code bytes + compressed model bytes)

Proposed: augment the training loss with:

```python
L_final = (1 - λ) * L_ce                              # standard cross-entropy
       + λ     * L_ac_rate                            # arithmetic-coding surrogate
       + μ     * L_compressibility                    # weight-entropy regularizer

# L_ac_rate: use a STE-style surrogate that passes gradients through an approximate
# arithmetic coder. For a block of target tokens, simulate:
#     total_bits = sum over tokens of bits_to_encode(target | cumulative_p_under_quantized_model)
# The "quantized model" is an on-the-fly int6 STE version of current weights.
#
# L_compressibility: add a soft penalty on the weight histogram's entropy
# (minimizing -Σ p_bin log p_bin encourages peaky weight distributions → better brotli
# ratio). This is where we cash in the P2 Shannon-entropy slack we measured.
```

Start with λ=0.1, μ=0.01. Schedule: increase λ over training as the model stabilizes (early on, CE alone is enough; late, optimize directly for AC rate).

**Integration point**: new loss terms in the training loop. ~80 LOC in `submission/train.py` around line 547 (the existing `forward` → `F.cross_entropy` call).

## Expected BPB

- **Range**: [-0.025, -0.010]
- **Mechanism**: current LM loss optimizes for an idealized code length that ignores the 2-stage quantize+brotli pipeline. A direct AC-rate loss should close 0.01-0.02 BPB of the "quantization tax" + "compressor mismatch" gap. P2 showed embed at 4.72 bits/8-bit-alloc = 41% wasted Shannon slack; compressibility regularizer should convert some of that to real BPB.
- **Lower bound**: -0.010 if the loss surrogate is too rough (just matches current training)
- **Upper bound**: -0.025 if the surrogate is informative AND the compressibility regularizer materially boosts post-brotli ratio

## Testable prediction

- **Metric**: `val_bpb` (quantized_sliding_window)
- **Threshold**: ≤ 1.072 at seed 42 (−0.010 from 1.082)
- **After**: 600 s wallclock comp budget, standard 3-seed confirmation
- **Secondary**: `artifact_bytes` should DECREASE by 0.3-0.8 MB because weights become more brotli-friendly

## Falsification criterion

Kill if 2-seed mean val_bpb ≥ 1.080 AND artifact_bytes doesn't change materially (the surrogate didn't help either objective).

## Stacking plan

- **Composes with**: everything (it's just an additional training loss)
- **Conflicts with**: nothing documented; may interact with QAT's STE gradients — test carefully
- **Blocks**: nothing
- **Budget footprint**: ~40 KB of loss-term code (negligible); model params unchanged

## Prior-art audit

Audited 2026-04-16 by Loop A fire 3 (Explore subagent).

- **Arxiv (2023-2026)**:
  - "Language Modeling is Compression" (ICLR 2024) — establishes LM training ≈ AC bits in the limit, but does NOT propose explicit AC loss during training
  - 2025 papers on neural weight compression + rate-distortion theory — focused on POST-training compression, not training-time objectives
  - MDL training exists theoretically but lacks a direct neural LM + differentiable AC implementation
- **Comp PRs** (openai/parameter-golf):
  - PR #538 "FP8 + Arithmetic Coding + SWA" (open 3/23/26) — uses AC at eval/compression stage, NOT as training loss
  - PR #459 "Weight Entropy Regularization" (open) — weight entropy regularizer, but via standard reg, not AC proxy
  - PR #1385 "Compressor-Aware Training (CAT)" (open) — differentiable compression proxies for LZ-family compressors, NOT for AC. **Closest match but targets LZ, not AC.**
- **Verdict**: **partial-overlap-with-PR-1385** on the broad "compression-aware training" theme, BUT arithmetic-coding-specific loss + compressibility regularizer combination is novel and targets our specific GPTQ→brotli pipeline.
- **Checked by**: claude 2026-04-16

## Lineage

- `STACK_NOVELTY_TRACKER_v2.md §L06` row "Arithmetic-coding loss"
- Backlog: RESEARCH_PROTOCOL.md §1 grid cell (L06 × MDL + L06 × arithmetic coding)
- Motivated by: P2 weight-entropy slack (embed has 41% Shannon waste) + the cross-entropy-vs-compression gap in standard training

## Risks

- STE through an arithmetic coder is unusual; gradient noise may destabilize training. Mitigation: start with small λ, ramp up.
- May interact badly with EMA / pre-quant TTT (IDEA-003/004). Run as pure-baseline test first before stacking.
- "Compressibility regularizer" could collapse weights to near-zero trivially. Need a constraint that prevents this (e.g. require weight-tensor Frobenius norm ≥ τ).

## Notes

Probe-informed. If this works it's paper-worthy on its own (the "train for the real metric" story is compelling). A prior-art audit next fire is critical — this kind of idea may be in recent compression-aware training literature.
