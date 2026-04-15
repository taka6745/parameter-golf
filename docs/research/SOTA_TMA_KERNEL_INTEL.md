# SOTA TMA megakernel intel — record/tma-megakernel-triple-loop b27fe93

Extracted 2026-04-08 0840Z by C30 explore agent. Used to design `KER_tma_megakernel_mlp_port` patch.

**Source**: `records/track_10min_16mb/2026-04-07_TMA_MegaKernel_TripleLoop_ParallelResiduals/train_gpt.py` (open openai/parameter-golf, +1536 lines)
**Author**: @andrewbaggio1
**SOTA val_bpb**: 1.08480 (5-seed mean, std=0.0007)
**Hardware**: 8×H100 80GB SXM (the kernel uses Hopper TMA → H100-only)

## Why this matters

The kernel fuses `fc → leaky_relu(0.5) → square` into a single Triton kernel using Hopper Tensor Memory Accelerator (TMA) descriptors. Eliminates materialization of the ~384 MB intermediate activation buffer per batch. **+10.5% training throughput** → +127 additional steps in 600s budget → ~ -0.02 to -0.03 BPB.

**On 3090 cheap pods**: this kernel CANNOT run (no TMA). Falls back to the standard `F.linear → F.leaky_relu → .square() → F.linear` path. Validation has to wait for the production H100 run.

## The kernel (verbatim from b27fe93)

```python
@triton.jit
def _fused_leaky_relu_sq_tma_kernel(
    a_desc, b_desc, c_desc, aux_desc,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr,
):
    """TMA-based fused fc -> leaky_relu(0.5) -> square."""
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
        c0 = acc0.to(dtype)
        c0_ag = tl.where(c0 > 0, 2.0 * c0, 0.5 * c0)
        c_desc.store([offs_am, offs_bn], c0_ag)
        c0_post = 0.5 * c0_ag * c0
        aux_desc.store([offs_am, offs_bn], c0_post)
        c1 = acc1.to(dtype)
        c1_ag = tl.where(c1 > 0, 2.0 * c1, 0.5 * c1)
        c_desc.store([offs_am, offs_bn + BLOCK_N // 2], c1_ag)
        c1_post = 0.5 * c1_ag * c1
        aux_desc.store([offs_am, offs_bn + BLOCK_N // 2], c1_post)
```

## Wrapper

```python
def _triton_fused_leaky_relu_sq(x_flat: Tensor, fc_weight: Tensor) -> tuple[Tensor, Tensor]:
    M, K = x_flat.shape
    N, K2 = fc_weight.shape
    assert K == K2
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
```

## Custom autograd Function (hand-written backward)

```python
class _FusedMLP(torch.autograd.Function):
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
```

## MLP module integration

```python
class MLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = int(dim * mult)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        if HAS_TRITON and x.is_cuda and self.training:
            return _FusedMLP.apply(x, self.fc.weight.to(x.dtype), self.proj.weight.to(x.dtype))
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())
```

## Launch config

- BLOCK_M=128, BLOCK_N=256, BLOCK_K=64
- num_warps=8, num_stages=4
- Grid: `min(NUM_SMS, cdiv(M,BLOCK_M)*cdiv(N,BLOCK_N))` (persistent kernel)
- Memory pattern: interleaved writes (output split into 2 halves)

## Anchor for our port

- **Our `MLP` class**: lines 606-617 of train_gpt.py
- **Our `LeakyReLU`**: USE_LEAKY_RELU=1 path uses negative_slope=0.5 + squared activation (matches the SOTA's leaky_relu(0.5).square() pattern)
- **Our `MLP_MULT`**: 2 (vs SOTA's likely 4) — kernel block dims may need re-tuning
- **TritonDescriptor import**: from `triton.tools.tensor_descriptor import TensorDescriptor` (Triton 3.0+, Hopper-only feature)
- **Fallback**: `if HAS_TRITON and x.is_cuda and self.training and torch.cuda.get_device_capability()[0] >= 9` — capability ≥ 9 = Hopper. Pre-Hopper falls through to standard path.

## Composes with our existing patches

- LEAKY_RELU_MARKER (P): the SOTA kernel uses leaky_relu(0.5) which is what our LEAKY_RELU patch already enables. Composes.
- USE_PARALLEL_RESIDUALS: the SOTA also uses parallel residuals. Composes.
- DEPTH_RECUR_MARKER: the SOTA name "triple-loop" suggests DEPTH_RECUR_CYCLES=3.

## Validation expectation

- **Cheap pods (3090, no TMA)**: kernel falls back to standard path, no measurable difference. Smoke test passes if HAS_TRITON detection works correctly.
- **8xH100 production run**: +10.5% throughput. Validates against SOTA val_bpb 1.08480.
