---
id: IDEA-021
slug: tensor-train-mixed-cores
created: 2026-04-16
updated: 2026-04-16
status: audited
layer: L07
novelty_class: WN
expected_bpb: [-0.020, -0.005]
cost_hours: 6.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l07-tensor-train-decomposition-int4-cores
prior_art_checked: 2026-04-16
next_step: prior-art-audit-then-prototype
---

# IDEA-021: Tensor-Train decomposition of weight matrices with mixed int4/int5 cores

> **Hypothesis**: Post-training tensor-train (TT) decomposition of the MLP and attention weight matrices into rank-k core tensors, each quantized at int4 or int5 per-core based on Hessian sensitivity, reduces the compressed artifact by 1.5-3 MB (to be re-spent on an extra transformer block) while keeping val_bpb within 0.003 of baseline. Net gain: 0.005-0.020 BPB from the re-spent budget on additional capacity.

## Method

For each target weight matrix `W ∈ R^(m × n)` (e.g., MLP fc 2048 × 512, attention c_q 512 × 512):

1. **Reshape** to a high-order tensor of shape `(m1 × m2 × ... × n1 × n2)` via prime factorization and permutation. E.g., 2048×512 → (8, 16, 16, 16, 16, 2) or similar 6-mode tensor.
2. **TT decomposition** via SVD on contracted modes: yields TT cores `G_1, G_2, ..., G_D` each of shape `(r_{k-1}, n_k, r_k)` where `r_k` is the TT-rank.
3. **Per-core Hessian sensitivity**: compute `|| ∂L/∂G_k ||²` over a calibration batch. Lowest-sensitivity cores → int4; highest → int5 or int6.
4. **Quantize each core** at its assigned precision via standard GPTQ-style rounding.
5. **Reconstruction at forward-pass time**: contract the TT cores back on-GPU (cheap if ranks are small).

```python
# Sketch
W_tt_cores, W_orig_shape, tt_ranks = tensor_train(W, max_rank=16)
hessian_per_core = compute_core_sensitivity(W_tt_cores, calibration_batch)
sorted_cores = sorted(W_tt_cores, key=lambda c: hessian_per_core[c])
for i, core in enumerate(sorted_cores):
    bits = 4 if i < len(sorted_cores) * 0.3 else 5  # bottom 30% → int4
    W_tt_cores_q[i] = quantize_gptq(core, bits)
# Store only the quantized cores; the full W is reconstructed on demand
```

TT-rank `max_rank=16` on a 2048×512 matrix: original 1.05 M params → TT storage ≈ 16 × (8+16+16+16+16+2) × 16 = 18,432 params per core × 6 cores ≈ **110 K params total = 10× reduction**. At int4, that's 55 KB per MLP matrix vs current 1 MB int6 packed = **94% size reduction**.

Realistic accuracy: TT-rank 16 typically retains 90-95% of expressivity for MLP-scale matrices. Compensating with the reclaimed budget (spent on a deeper / wider stack) should net positive.

**Integration point**: forward-pass path modification. Store `W_tt_cores` instead of `W`; reconstruct W as needed (cache reconstructed W for duration of one forward+backward). ~250 LOC for TT decomp/recomp + quantization hooks. Non-record track first.

## Expected BPB

- **Range**: [-0.020, -0.005]
- **Mechanism**: two-step. Step 1 — raw TT quant at int4/5 mixed costs ~0.003-0.010 BPB of reconstruction error (pure loss, hopefully small via rank-16 + Hessian-aware bit assignment). Step 2 — the reclaimed 2-3 MB of budget spent on +1 layer or wider MLP buys back 0.010-0.020 BPB (per Chinchilla-ish scaling at our size). Net: -0.005 to -0.020.
- **Lower bound**: -0.005 if reconstruction loss is close to the gain
- **Upper bound**: -0.020 if TT captures most of the weight info at rank-16 AND the reclaimed budget buys a substantial capacity bump

## Testable prediction

- **Primary**: val_bpb ≤ 1.077 at seed 42 when TT-quant is stacked with +1 layer (12L total)
- **Ablation 1**: val_bpb delta from TT-quant ALONE (no capacity re-spend) should be ≤ +0.010 BPB (i.e., reconstruction loss bounded)
- **After**: 600s training (stock 1.082 config + TT quant + 12L)

## Falsification criterion

Kill if:
- TT-quant alone causes val_bpb > +0.015 BPB regression (reconstruction too lossy)
- OR TT-quant + 12L val_bpb ≥ 1.080 (less than 0.002 net improvement after all compounding)

## Stacking plan

- **Composes with**: almost everything. Orthogonal to L03/L05/L06 changes; affects only how L07 stores weights.
- **Conflicts partially with**: IDEA-005 mixed int5/int6 (both try to reclaim weight budget; we'd pick one or the other, not both)
- **Blocks**: future +2 layers (we'd need to re-spend the budget once, not twice)
- **Budget footprint**: reclaims 1.5-3 MB after brotli. Re-spend on +1L transformer (1 MB) or +33% MLP hidden dim (1.5 MB) — plan to spend in a companion experiment.

## Prior-art audit

Audited 2026-04-16 by Loop A fire 18 (Explore subagent).

- **Arxiv (2023-2026)**:
  - **TensorGPT** (2307.00526) — TT-decomposes embedding layer in GPT-scale LLMs, 46-65× compression on embeddings. But uses UNIFORM quantization, NOT per-core Hessian-guided bit allocation.
  - **Tender** (ISCA 2024, 2406.12930) — post-training TT decomp + runtime requantization co-design. Focus: activation tensor outliers. Does NOT mix int4/int5 per TT core.
  - **HAWQ-V2** (NeurIPS 2020, 1911.03852) — Hessian-aware mixed-precision at LAYER granularity. Doesn't apply to TT cores.
- **Comp PRs** (openai/parameter-golf): PRs #1429, #1438 do mixed int5/int6 per-layer Hessian-weighted (standard matrix format, not TT). No TT+per-core mixed-precision found.
- **Verdict**: **partial-overlap with HAWQ-V2 (Hessian allocation) and TensorGPT (TT decomp) — novel combination**. Core innovation: transferring per-layer Hessian sensitivity down to TT-core granularity, exploiting fine-grained rank structure.
- **Checked by**: claude 2026-04-16

## Lineage

- `STACK_NOVELTY_TRACKER_v2.md §L07` open-novelty "Tensor-train decomposition (int4 cores)"
- RESEARCH_PROTOCOL.md §1 grid cell (L07 × tensor-train decomposition / neural compression)

## Risks

- Reshape choice (factorization) affects TT-rank requirements drastically. Mitigation: try multiple factorizations, pick the lowest TT-rank for ≤3% reconstruction error.
- Forward-pass reconstruction overhead: TT contract is O(sum of rank products). For rank-16 on 6 modes, each forward pass adds ~1-2ms per matrix. Budget: acceptable if we save training wallclock elsewhere (via fewer quantized bytes → less memory pressure).
- Backprop through TT: the gradient w.r.t. cores is standard (tensor network calculus), but may interact awkwardly with QAT. Mitigation: apply TT only AFTER standard training, as a post-hoc compression step.

## Notes

The reclaim+re-spend pattern is the key insight: TT alone LOSES BPB (reconstruction error); combined with +1 layer it GAINS. Must design the experiment to test BOTH phases separately before claiming a win.
