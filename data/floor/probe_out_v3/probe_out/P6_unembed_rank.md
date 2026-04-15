# P6: Unembed / Head Low-Rank Analysis

For each candidate head/embed matrix, compute SVD and report cumulative variance.
If rank-k explains 95%, factoring to rank-k saves `(rows + cols - rows×cols/k×k)` params.

## `w.skip_weights`  shape=torch.Size([8, 512])

- Full params: 4,096
- k for 50% var: 1
- k for 90% var: 2
- k for 95% var: 2  →  rank-2 factor = 1,040 params (74.6% savings)
- k for 99% var: 5

## `w.skip_gates`  shape=torch.Size([8, 512])

- Full params: 4,096
- k for 50% var: 1
- k for 90% var: 1
- k for 95% var: 2  →  rank-2 factor = 1,040 params (74.6% savings)
- k for 99% var: 5

## `w.blocks.0.attn.gate_proj.weight`  shape=torch.Size([8, 512])

- Full params: 4,096
- k for 50% var: 2
- k for 90% var: 6
- k for 95% var: 7  →  rank-7 factor = 3,640 params (11.1% savings)
- k for 99% var: 8

