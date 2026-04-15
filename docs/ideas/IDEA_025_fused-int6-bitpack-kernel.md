---
id: IDEA-025
slug: fused-int6-bitpack-kernel
created: 2026-04-16
updated: 2026-04-16
status: draft
layer: L11
novelty_class: WN
expected_bpb: [-0.005, -0.001]
cost_hours: 6.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l11-fused-bit-pack-kernels-int6-packunpack
prior_art_checked: null
next_step: prior-art-audit-then-prototype
---

# IDEA-025: Fused int6 bit-pack kernels (unpack-in-matmul, no fp16 intermediate)

> **Hypothesis**: Replacing the current "int6 store → unpack to fp16 → matmul" path with a custom Hopper kernel that unpacks the int6 weights directly in wgmma's A-register tiles (bit-extraction in the matmul epilogue) eliminates a full dequant pass worth of memory bandwidth + compute per layer, yielding 10-20% eval-time speedup. That frees eval wallclock for extra TTT epochs or a bigger hedge mixer. Expected val_bpb: 0.001-0.005.

## Method

Currently: on every forward pass, int6-packed weights get unpacked to fp16 (or bf16) tensors before matmul. This costs memory bandwidth and compute.

Proposed: custom Triton / CUDA kernel that:
1. Loads the int6-packed bytes into shared memory
2. Unpacks via bit-extraction DURING the wgmma load-A path (fuses unpack into the matmul prelude)
3. Stores no intermediate fp16 tensor; matmul consumes the unpacked values directly from registers

```python
# Replace
W_fp16 = unpack_int6(W_packed)  # full pass over W
y = x @ W_fp16                  # matmul

# With
y = fused_int6_matmul(x, W_packed)  # one fused kernel
```

Implementation: ~300 LOC of Triton. The tricky part is aligning the int6 bit-packing with wgmma's 8×8×16 or 16×8×16 tiles so each thread extracts the right bits for its assigned A-tile element.

**Integration point**: replace `torch.ops.quant.int6_matmul` with custom `paramgolf.fused_int6_matmul` in each block's MLP and attention forward. Zero artifact bytes (library-side kernel, free per MOONSHOT_RULES §1.6).

## Expected BPB

- **Range**: [-0.005, -0.001]
- **Mechanism**: eval-time speedup → more wallclock available for TTT or hedge mixer complexity. Training-time speedup marginal (matmul is not dequant-bottlenecked), but eval-time win is material.
- **Lower bound**: -0.001 (if current PyTorch path is already fused well via torch.compile)
- **Upper bound**: -0.005 (if eval budget was actually tight and we can re-spend the freed 1-2 min on 2-3× TTT epochs)

## Testable prediction

- **Metric**: eval wallclock step time should drop ≥15% vs current path
- **Derived**: val_bpb improves by 0.001-0.005 when the extra budget is spent on more TTT epochs
- **Secondary**: no training-time regression (matmul correctness)

## Falsification criterion

Kill if:
- Speedup <8% (kernel engineering didn't pay off)
- OR numerical drift vs PyTorch reference matmul exceeds 1e-3 relative (bit-packing bug)

## Stacking plan

- **Composes with**: everything. Infrastructure-only, doesn't change model weights / architecture.
- **Composes with**: IDEA-016 (megakernel) — this is the "MLP-shaped" subkernel of IDEA-016; the fusion can be layered or merged.
- **Conflicts with**: nothing structural
- **Blocks**: nothing
- **Budget footprint**: 0 bytes in artifact (kernel library)

## Prior-art audit

_To be filled by next Loop A fire with Explore subagent._

- **Arxiv (2023-2026)**: "fused int4 matmul Hopper", "wgmma int6 bit-pack", "Marlin int4 kernel", "bit-packed matmul CUDA"
- **Comp PRs**: grep for `fused_int6`, `bit_pack`, `wgmma_int` in comp PR titles
- **Verdict**: TBD; Marlin (int4) and BitBLAS exist for int-matmul kernels, but int6-specific Hopper wgmma fusion is likely novel
- **Checked by**: _pending_

## Lineage

- `STACK_NOVELTY_TRACKER_v2.md §L11` row "Fused bit-pack kernels (int6 pack/unpack)"
- RESEARCH_PROTOCOL.md §1 grid cell (L11 × custom CUDA kernels)
- Eval budget is the binding constraint for IDEA-012 (moonshot online cache) and IDEA-022 (BMA ensemble), so eval-speed wins directly enable those

## Risks

- Int6 doesn't naturally align with byte boundaries (6 bits × 4 values = 3 bytes). Bit extraction inside wgmma requires careful indexing. Mitigation: pack 4 int6 into 3 bytes per group, align groups to wgmma tile size.
- Hopper-native kernel → won't run on earlier GPUs. Mitigation: this is an 8×H100 comp; acceptable.
- Kernel debugging cost high. Mitigation: unit-test against PyTorch reference at each tile size.

## Notes

Prerequisite-like for moonshot IDEA-012 if eval wallclock is binding. Smaller scope than IDEA-016 megakernel; ship first.
