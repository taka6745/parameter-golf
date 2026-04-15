# P2: Weight-Bit Entropy (on raw quantized ints)

| Module | alloc_bits | H (bits/w) | distinct | numel | wasted (bits) | wasted (MB) |
|---|---:|---:|---:|---:|---:|---:|
| `tok_emb.weight` | 8 | 4.72 | 69 | 4,194,304 | 13,776,390 | 1.72 |
| `blocks.10.mlp.proj.weight` | 8 | 3.33 | 59 | 1,048,576 | 4,898,727 | 0.61 |
| `blocks.5.mlp.proj.weight` | 8 | 3.33 | 46 | 1,048,576 | 4,898,535 | 0.61 |
| `blocks.7.mlp.proj.weight` | 8 | 3.33 | 60 | 1,048,576 | 4,898,239 | 0.61 |
| `blocks.8.mlp.proj.weight` | 8 | 3.33 | 45 | 1,048,576 | 4,897,824 | 0.61 |
| `blocks.1.mlp.proj.weight` | 8 | 3.33 | 51 | 1,048,576 | 4,897,812 | 0.61 |
| `blocks.9.mlp.proj.weight` | 8 | 3.33 | 45 | 1,048,576 | 4,897,759 | 0.61 |
| `blocks.4.mlp.proj.weight` | 8 | 3.33 | 45 | 1,048,576 | 4,897,711 | 0.61 |
| `blocks.3.mlp.proj.weight` | 8 | 3.33 | 40 | 1,048,576 | 4,897,632 | 0.61 |
| `blocks.6.mlp.proj.weight` | 8 | 3.33 | 45 | 1,048,576 | 4,897,593 | 0.61 |
| `blocks.2.mlp.proj.weight` | 8 | 3.33 | 37 | 1,048,576 | 4,897,592 | 0.61 |
| `blocks.3.mlp.fc.weight` | 8 | 3.33 | 34 | 1,048,576 | 4,897,327 | 0.61 |
| `blocks.10.mlp.fc.weight` | 8 | 3.33 | 26 | 1,048,576 | 4,897,204 | 0.61 |
| `blocks.4.mlp.fc.weight` | 8 | 3.33 | 31 | 1,048,576 | 4,897,096 | 0.61 |
| `blocks.0.mlp.proj.weight` | 8 | 3.33 | 41 | 1,048,576 | 4,897,068 | 0.61 |
| `blocks.9.mlp.fc.weight` | 8 | 3.33 | 24 | 1,048,576 | 4,896,482 | 0.61 |
| `blocks.8.mlp.fc.weight` | 8 | 3.33 | 25 | 1,048,576 | 4,896,455 | 0.61 |
| `blocks.5.mlp.fc.weight` | 8 | 3.33 | 27 | 1,048,576 | 4,896,438 | 0.61 |
| `blocks.0.mlp.fc.weight` | 8 | 3.33 | 52 | 1,048,576 | 4,896,431 | 0.61 |
| `blocks.1.mlp.fc.weight` | 8 | 3.33 | 37 | 1,048,576 | 4,896,296 | 0.61 |
| `blocks.2.mlp.fc.weight` | 8 | 3.33 | 27 | 1,048,576 | 4,896,083 | 0.61 |
| `blocks.7.mlp.fc.weight` | 8 | 3.33 | 35 | 1,048,576 | 4,895,738 | 0.61 |
| `blocks.6.mlp.fc.weight` | 8 | 3.33 | 28 | 1,048,576 | 4,895,650 | 0.61 |
| `blocks.9.attn.c_q.weight` | 8 | 3.29 | 37 | 262,144 | 1,233,584 | 0.15 |
| `blocks.10.attn.c_q.weight` | 8 | 3.32 | 35 | 262,144 | 1,227,243 | 0.15 |
| `blocks.8.attn.proj.weight` | 8 | 3.33 | 34 | 262,144 | 1,225,440 | 0.15 |
| `blocks.9.attn.proj.weight` | 8 | 3.33 | 28 | 262,144 | 1,225,382 | 0.15 |
| `blocks.3.attn.c_q.weight` | 8 | 3.33 | 47 | 262,144 | 1,225,273 | 0.15 |
| `blocks.2.attn.c_q.weight` | 8 | 3.33 | 53 | 262,144 | 1,224,925 | 0.15 |
| `blocks.10.attn.proj.weight` | 8 | 3.33 | 34 | 262,144 | 1,224,672 | 0.15 |
