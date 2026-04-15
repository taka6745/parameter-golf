---
id: IDEA-016
slug: fused-megakernel
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L11
novelty_class: WN
expected_bpb: [-0.008, -0.002]
cost_hours: 12.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l11-custom-cuda-megakernel-fused-full-block
prior_art_checked: 2026-04-16
next_step: prior-art-audit-then-prototype-nonrecord-track
---

# IDEA-016: Fused megakernel (full transformer block in one CUDA kernel)

> **Hypothesis**: Fusing the full transformer block (RMSNorm + QKV proj + FA3 attention + output proj + residual + MLP pre-norm + MLP fc + MLP gate+proj + residual) into a single hand-written CUDA megakernel gives 20–40% throughput over the current stack (FA3 + Triton TMA MLP), yielding 4–8 more training steps in the 600 s budget, which translates to 0.002–0.008 BPB improvement.

## Method

Current stack runs 5-6 fused kernel boundaries per block (norm, QKV gemm, FA3, out gemm, MLP fc gemm, MLP gate+proj gemm + elementwise). Each boundary has a kernel-launch overhead (~2–4 µs on H100 Hopper) plus L2 cache spill.

A single fused kernel:
- Keeps `x` in shared memory across the whole block (L2 bandwidth saved)
- Fuses elementwise ops (norm, activation, residual add) into the matmul epilogues
- Uses wgmma + TMA for the matmuls, manually tiled for our shape (batch × 2048 seq × 512 dim)
- Re-uses loaded weights across attention-out and MLP-fc (they have contiguous shapes)

Pseudocode at the kernel level:
```cuda
// fused_block_kernel<<<grid=[B, num_ctas_per_seq], block=256>>>
// persistent threadblock; each block processes one (batch, seq_chunk) pair
// registers hold the residual stream; shared memory tiles the weights

__global__ void fused_block(
    const half* __restrict__ x,          // [B, T, D]
    const int8_t* __restrict__ w_qkv,    // [D*(2H+2Hkv), D] packed int6-in-int8
    const int8_t* __restrict__ w_o,      // [D, D]
    const int8_t* __restrict__ w_mlp_fc, // [4D, D]
    const int8_t* __restrict__ w_mlp_pr, // [D, 4D]
    half* __restrict__ out,              // [B, T, D]
    ...
) {
    // 1. Load x slice → registers
    // 2. RMSNorm in-register
    // 3. wgmma QKV proj (int6 unpack in epilogue)
    // 4. FA3 attention (subkernel call, shmem-resident Q/K/V)
    // 5. wgmma out proj + residual add (epilogue fusion)
    // 6. RMSNorm
    // 7. wgmma MLP fc + LeakyReLU² (epilogue)
    // 8. wgmma MLP proj + residual add (epilogue)
    // 9. Store out
}
```

Ship as a pip-installable library (the user already has the budget exemption for kernel libraries per `MOONSHOT_RULES.md §1.6`).

**Integration point**: fork `submission/train.py` with a `TORCH_COMPILE_DISABLE=1` path that uses `torch.ops.paramgolf.fused_block(...)` in each block's forward. Fallback: if the kernel isn't available, use the current split-kernel path. `~300-500 LOC` for kernel + pybind + Python wrapper.

## Expected BPB

- **Range**: [-0.008, -0.002]
- **Mechanism**: each extra training step = extra gradient update on ~786k tokens. 4-8 more steps at ~10k gradient-step BPB sensitivity gives 0.002-0.008. Lower bound if the throughput gain is neutralized by kernel-bug stability / NaN issues in early runs.
- **Lower bound**: -0.002 (conservative if only 10% throughput gain, 4 extra steps)
- **Upper bound**: -0.008 (if 40% throughput gain + those steps land in the most-productive phase of training)

## Testable prediction

- **Primary metric**: step_time_ms drops by ≥15% at comparable config
- **Derived**: val_bpb ≤ 1.078 at seed 42 with identical-LR schedule but more steps in the 600s budget
- **Secondary**: no divergence / NaN across 3 seeds in the first 100 steps

## Falsification criterion

Kill if:
- Step-time gain <8% (kernel engineering didn't pay off)
- OR training diverges at 2+ seeds (stability problem)
- OR val_bpb regresses vs stock stack (bug in the fused computation)

## Stacking plan

- **Composes with**: everything at the training-loop level. Orthogonal axis.
- **Conflicts with**: possibly `TORCH_COMPILE_DISABLE` handling — needs the `paramgolf.fused_block` custom op to be importable even when compile is disabled (trivial to satisfy).
- **Blocks**: nothing documented
- **Budget footprint**: 0 bytes in the 16 MB artifact (kernel library is not counted — it's an import per `MOONSHOT_RULES.md §1.6`)

## Prior-art audit

Audited 2026-04-16 by Loop A fire 6 (Explore subagent).

- **Arxiv (2023-2026)**:
  - "Deep Kernel Fusion for Transformers" (Feb 2026, arxiv 2602.11808) — fuses SwiGLU MLP GEMMs + pointwise ops for 13.2% H100 speedup, but attention stays separate; inference-only, not training
  - "FlashAttention-2 on Hopper via CUTLASS" (2023) — attention fusion only, no full-block fusion
- **Comp PRs** (openai/parameter-golf):
  - PR #1450 — Triton TMA **MLP-only** megakernel (fc → relu² fused), +10.5% throughput, 1.08480 BPB record. We already ship this.
  - PRs #1316, #1192, #1072, #1420, #1561, #1560, #1530 — multiple MLP-only / attention-only megakernels with 3-10% gains
  - PRs #1502, #1501, #1441 — partial full-block fusion explorations, NOT shipped as records
- **Verdict**: **partial-overlap-with-PR-#1450+**. MLP-only and attention-only megakernels are production. **Full transformer block fusion** (RMSNorm + QKV + FA3 + output + residual + MLP-norm + MLP fc + activation + MLP proj + residual all in a single persistent Hopper wgmma+TMA kernel) is **not yet shipped**. Novelty lies in the end-to-end fusion scope.
- **Checked by**: claude 2026-04-16

## Lineage

- OpenAI "requests for PRs" list explicitly includes megakernels (`README.md`)
- `STACK_NOVELTY_TRACKER_v2.md §L11` row "Custom CUDA megakernel (fused full block)"
- RESEARCH_PROTOCOL.md §1 grid cell (L11 × CUDA custom kernels)

## Risks

- CUDA kernel engineering is high-effort and bug-prone. Mitigation: prototype + validate against the split-kernel baseline tensor-by-tensor before doing a real training run.
- H100 architecture quirks (wgmma instruction scheduling, TMA cluster sizes) may require careful tuning. Budget an extra cycle for that.
- Non-record track is the right POC venue — we can't afford to burn the 600 s comp budget on kernel debugging.

## Notes

Highest-effort, highest-speedup-ceiling idea in the queue. Its BPB payoff is modest-ish but the throughput unlock compounds with future experiments (every `step_time_ms` drop buys ALL subsequent experiments more effective compute).
