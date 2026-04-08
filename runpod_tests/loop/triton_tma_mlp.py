"""triton_tma_mlp.py — Triton TMA megakernel for fused leaky_relu(0.5)^2 MLP path.

Ported verbatim from openai/parameter-golf record/tma-megakernel-triple-loop (b27fe93,
@andrewbaggio1, val_bpb=1.08480 SOTA on track_10min_16mb).

Source: SOTA_TMA_KERNEL_INTEL.md in repo root.

Hopper TMA-only — requires `triton.tools.tensor_descriptor.TensorDescriptor`. On
pre-Hopper GPUs (3090, 4070 Ti) the import fails and `HAS_TRITON_TMA` is set False.
Cheap-pod training falls back to the standard `F.leaky_relu(self.fc(x), 0.5).square()
→ self.proj(...)` path. The kernel is hot only on H100 production runs.

Worth: +10.5% throughput → +127 training steps in 600s budget → -0.02 to -0.03 BPB
on the 8xH100 production run.

Marker: KER_TMA_MEGAKERNEL_MARKER (in train_gpt.py via patcher Patch 42).
Env gate: USE_KER_TMA_MEGAKERNEL=1 (default 0 → standard path, bit-exact baseline).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import Tensor

HAS_TRITON_TMA = False
_TritonImportError: str | None = None

try:
    import triton
    import triton.language as tl
    try:
        from triton.tools.tensor_descriptor import TensorDescriptor  # Hopper-only
        HAS_TRITON_TMA = True
    except ImportError as _e:
        _TritonImportError = f"TensorDescriptor unavailable (pre-Hopper or Triton<3.0): {_e}"
except ImportError as _e:
    _TritonImportError = f"triton not installed: {_e}"


if HAS_TRITON_TMA:

    @triton.jit
    def _fused_leaky_relu_sq_tma_kernel(
        a_desc, b_desc, c_desc, aux_desc,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr,
    ):
        """TMA-based fused fc → leaky_relu(0.5) → square.

        Persistent kernel: one program per SM, loops over output tiles.
        Interleaved writes split BLOCK_N into 2 halves for memory throughput.
        """
        dtype = tl.bfloat16
        start_pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        k_tiles = tl.cdiv(K, BLOCK_K)
        num_tiles = num_pid_m * num_pid_n

        for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N

            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for ki in range(k_tiles):
                offs_k = ki * BLOCK_K
                a = a_desc.load([offs_am, offs_k])
                b = b_desc.load([offs_bn, offs_k])
                accumulator = tl.dot(a, b.T, accumulator)

            # Interleaved write: split into two halves for better memory throughput
            acc = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            # First half: fused activation
            c0 = acc0.to(dtype)
            c0_ag = tl.where(c0 > 0, 2.0 * c0, 0.5 * c0)  # leaky_relu(0.5) gradient
            c_desc.store([offs_am, offs_bn], c0_ag)
            c0_post = 0.5 * c0_ag * c0  # leaky_relu(0.5)(h)^2
            aux_desc.store([offs_am, offs_bn], c0_post)
            # Second half
            c1 = acc1.to(dtype)
            c1_ag = tl.where(c1 > 0, 2.0 * c1, 0.5 * c1)
            c_desc.store([offs_am, offs_bn + BLOCK_N // 2], c1_ag)
            c1_post = 0.5 * c1_ag * c1
            aux_desc.store([offs_am, offs_bn + BLOCK_N // 2], c1_post)


    def _triton_fused_leaky_relu_sq(x_flat: Tensor, fc_weight: Tensor) -> tuple[Tensor, Tensor]:
        """TMA wrapper: fused fc -> leaky_relu(0.5) -> square."""
        M, K = x_flat.shape
        N, K2 = fc_weight.shape
        assert K == K2, f"shape mismatch: x_flat K={K} vs fc_weight K={K2}"
        act_grad = torch.empty((M, N), device=x_flat.device, dtype=x_flat.dtype)
        post = torch.empty((M, N), device=x_flat.device, dtype=x_flat.dtype)
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
        a_desc = TensorDescriptor.from_tensor(x_flat, [BLOCK_M, BLOCK_K])
        b_desc = TensorDescriptor.from_tensor(fc_weight, [BLOCK_N, BLOCK_K])
        c_desc = TensorDescriptor.from_tensor(act_grad, [BLOCK_M, BLOCK_N // 2])
        aux_desc = TensorDescriptor.from_tensor(post, [BLOCK_M, BLOCK_N // 2])

        def grid(META):
            return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

        _fused_leaky_relu_sq_tma_kernel[grid](
            a_desc, b_desc, c_desc, aux_desc, M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=1, NUM_SMS=NUM_SMS,
            num_stages=4, num_warps=8,
        )
        return post, act_grad


    class _FusedMLP(torch.autograd.Function):
        """Custom autograd: TMA-fused forward, hand-written backward."""

        @staticmethod
        def forward(ctx, x, fc_w, proj_w):
            x_flat = x.reshape(-1, x.shape[-1])
            post, act_grad = _triton_fused_leaky_relu_sq(x_flat, fc_w)
            out = F.linear(post, proj_w)
            ctx.save_for_backward(x_flat, fc_w, proj_w, act_grad, post)
            ctx.orig_shape = x.shape
            return out.reshape(*x.shape[:-1], out.shape[-1])

        @staticmethod
        def backward(ctx, grad_output):
            x_flat, fc_w, proj_w, act_grad, post = ctx.saved_tensors
            go = grad_output.reshape(-1, grad_output.shape[-1])
            dW_proj = go.T @ post
            dpre = (go @ proj_w) * act_grad
            dW_fc = dpre.T @ x_flat
            dx = dpre @ fc_w
            return dx.reshape(ctx.orig_shape), dW_fc, dW_proj


def is_supported(x: Tensor) -> bool:
    """True iff: triton TMA imports OK + on CUDA + Hopper-class device."""
    if not HAS_TRITON_TMA:
        return False
    if not x.is_cuda:
        return False
    try:
        major, _ = torch.cuda.get_device_capability(x.device)
        return major >= 9  # Hopper = SM90+
    except Exception:
        return False


def fused_mlp(x: Tensor, fc_w: Tensor, proj_w: Tensor) -> Tensor:
    """User-facing entry point. Caller verifies env var; this just runs the kernel."""
    if not is_supported(x):
        raise RuntimeError(f"triton_tma_mlp not supported on this device: {_TritonImportError or 'capability < SM90'}")
    return _FusedMLP.apply(x, fc_w.to(x.dtype), proj_w.to(x.dtype))
