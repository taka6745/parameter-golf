# P8: Per-Layer Block Contribution Ratio

`out_norm / in_norm` per block during forward pass. Ratio ≈ 1 = block barely changes residual stream (possibly redundant). Ratio >> 1 = block amplifies. Ratio << 1 = block collapses / dampens.

| layer | mean in_norm | mean out_norm | out/in ratio |
|---:|---:|---:|---:|
| 0 | 22.627 | 108.449 | 4.793 |
| 1 | 108.449 | 111.561 | 1.029 |
| 2 | 111.561 | 99.260 | 0.890 |
| 3 | 99.260 | 100.515 | 1.013 |
| 4 | 100.515 | 94.363 | 0.939 |
| 5 | 94.363 | 89.947 | 0.953 |
| 6 | 88.839 | 77.942 | 0.877 |
