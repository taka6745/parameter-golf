---
id: IDEA-023
slug: sigma-delta-quantization
created: 2026-04-16
updated: 2026-04-16
status: audited
layer: L07
novelty_class: CP
expected_bpb: [-0.012, -0.002]
cost_hours: 4.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l07-sigma-delta-quantization-residual-feedback
prior_art_checked: 2026-04-16
next_step: prior-art-audit-then-prototype
---

# IDEA-023: Sigma-delta (residual feedback) quantization of weight tensors

> **Hypothesis**: Treating weight quantization as a sigma-delta modulator (feed the rounding error from weight `i` forward as a correction to weight `i+1` in the tensor traversal) achieves effective int8-quality reconstruction with physical int4 storage, reducing the compressed artifact by 1.5-2 MB while keeping val_bpb within 0.005 of baseline. Re-spending the reclaimed budget on added capacity nets 0.002-0.012 BPB improvement.

## Method

Audio sigma-delta DACs achieve high effective bit-depth via oversampling + noise-shaping: quantize to low resolution, but feed the per-sample rounding error back as a correction to subsequent samples. Applied to weight tensors:

```python
def sigma_delta_quantize(W_flat, n_bits=4):
    """Quantize each weight to n_bits, but propagate the rounding error forward."""
    step = (W_flat.max() - W_flat.min()) / (2**n_bits - 1)
    bias = W_flat.min()
    q_out = torch.zeros_like(W_flat, dtype=torch.int8)
    residual = 0.0
    for i in range(W_flat.numel()):
        corrected = W_flat[i] + residual  # add accumulated error
        q_level = round((corrected - bias) / step)
        q_level = clamp(q_level, 0, 2**n_bits - 1)
        reconstructed = q_level * step + bias
        residual = corrected - reconstructed  # new error, propagate to next
        q_out[i] = q_level
    return q_out, bias, step
```

Implementation: vectorize via scan (cumulative residual with per-element correction); GPU-friendly via torch.cumsum trick or custom Triton kernel. Per-tensor sigma-delta produces a bitstream that's statistically white (noise-shaped to high frequencies in the weight's traversal order), which makes brotli-after-shuffle compress marginally worse than naive int8 BUT the stored bits-per-weight is 4, not 8 — so net win.

Key nuance: **traversal order matters**. Row-major, column-major, or Z-order all give different error statistics. Default row-major; test other orders.

**Integration point**: replace GPTQ int8→int6 path with sigma-delta int4 for MLP matrices (keep embedding int8 for now). ~150 LOC. `SIGMA_DELTA_ENABLED=1` env var.

## Expected BPB

- **Range**: [-0.012, -0.002]
- **Mechanism**: 4-bit sigma-delta has been shown in audio DSP to achieve ~7-8 effective bits of SNR at the original sample rate. Applied to weights, expect similar: ~7 effective bits at int4 storage = equivalent to int6 naive but 33% fewer stored bits. Reclaimed budget 1.5-2 MB re-spent on capacity.
- **Lower bound**: -0.002 (marginal if error propagation interacts badly with transformer's matrix math)
- **Upper bound**: -0.012 (if the effective-bits improvement transfers cleanly AND brotli compresses the noise-shaped output efficiently)

## Testable prediction

- **Metric**: post-quant val_bpb vs current int6 GPTQ at seed 42
- **Threshold**: ≤ 1.085 WITHOUT capacity re-spend (bounded reconstruction loss); ≤ 1.075 WITH re-spend (+1 layer or wider MLP)
- **Secondary**: artifact_bytes decrease by 1.5-2 MB before re-spend

## Falsification criterion

Kill if:
- Sigma-delta-quant alone loses > 0.015 BPB vs int6 GPTQ baseline (reconstruction error too high)
- OR sigma-delta + re-spend val_bpb ≥ 1.080 at 2 seeds (less than 0.002 net)

## Stacking plan

- **Composes with**: all non-L07 ideas
- **Conflicts with**: IDEA-005 (mixed int5/int6), IDEA-011 (embed int8→int6), IDEA-021 (tensor-train). All compete for the same L07 slot; pick one.
- **Blocks**: similar as IDEA-005/011/021 — capacity re-spend on +1L / wider MLP uses the same freed budget
- **Budget footprint**: reclaims ~1.5-2 MB compressed

## Prior-art audit

Audited 2026-04-16 by Loop A fire 24 (Explore subagent).

- **Arxiv (2023-2026)**:
  - **"SDQ-LLM: Sigma-Delta Quantization for 1-bit LLMs of any size"** (Sep 2025, arxiv 2510.03275) — **DIRECT prior art**. Implements exactly this concept: sigma-delta quantization for transformer weights with error-feedback propagation, per-layer OSR allocation, ~1.58-bit effective precision. Github: `Dreamlittlecat/LLM-Quant-Factory`.
  - "Residual Quantization for Low Bit-Width Neural Networks" (IEEE 2021) — foundational residual-feedback approach.
- **Comp PRs** (openai/parameter-golf): none ship sigma-delta. Only generic QAT/quantization PRs (#1595, #1562).
- **Verdict**: **already-done-in-SDQ-LLM (arxiv 2510.03275)**. Novelty reclassification from WN → **CP (comp-port)** since no comp PR has shipped SDQ-LLM's approach — porting their 1.58-bit technique to our 16 MB stack IS the comp-port work. Not world-novel but still a valid port-with-evidence per MOONSHOT_RULES hard rule #2.
- **Checked by**: claude 2026-04-16

## Status update

Novelty class was WN → reclassified to **CP** (comp-port) based on direct prior-art hit. Still worth running because:
- No comp PR ships SDQ-LLM's approach
- SDQ-LLM's paper uses ImageNet-class LLMs, not byte-level 16 MB; our port is the scale-transfer port
- Expected BPB range unchanged (still [-0.012, -0.002]); the re-spend argument (reclaim + bigger model) is the main lever

## Lineage

- `STACK_NOVELTY_TRACKER_v2.md §L07` row "Sigma-Delta quantization (residual DAC)"
- RESEARCH_PROTOCOL.md §1 grid cell (L07 × neural compression / audio DSP port)

## Risks

- Error propagation order dependence: if traversal order interacts with how the tensor is used during matmul (e.g., row-major traversal but column-contiguous matmul), reconstruction during forward pass has unexpected error profile. Mitigation: profile error at layer boundaries before stacking.
- GPU efficiency: the scan-like structure is harder to parallelize than naive GPTQ. Mitigation: custom Triton kernel; fall back to CPU offline sigma-delta pass if needed.
- May not compose cleanly with QAT; apply post-training only.

## Notes

Classical audio DSP port. Cheap to prototype (150 LOC for core algorithm), likely world-novel for our setting. If budget savings materialize, IDEA-021 (tensor-train) may be redundant; pick whichever reclaims more.
