# P11: Cross-Seed Weight Stability

Compared 3 seeds: `final_model_seed42.int6, final_model_seed314.int6, final_model_seed999.int6`.

**High rel_std = weight is noisy across seeds (lottery ticket / under-determined).** Tensors in the bottom of this table have real signal that transfers; top are idiosyncratic.

| Module | shape | numel | mean_abs | std_across_seeds | rel_std |
|---|---|---:|---:|---:|---:|
| `blocks.9.attn.gate_proj.weight` | (8, 512) | 4,096 | 0.05004 | 0.06247 | 1.25 |
| `blocks.8.attn.gate_proj.weight` | (8, 512) | 4,096 | 0.05355 | 0.06626 | 1.24 |
| `blocks.8.attn.c_k.weight` | (256, 512) | 131,072 | 0.07824 | 0.09378 | 1.2 |
| `blocks.9.attn.c_v.weight` | (256, 512) | 131,072 | 0.09709 | 0.1151 | 1.19 |
| `blocks.9.attn.c_k.weight` | (256, 512) | 131,072 | 0.07615 | 0.09023 | 1.18 |
| `blocks.8.attn.c_v.weight` | (256, 512) | 131,072 | 0.09878 | 0.1168 | 1.18 |
| `blocks.10.attn.c_v.weight` | (256, 512) | 131,072 | 0.08573 | 0.1011 | 1.18 |
| `blocks.10.attn.gate_proj.weight` | (8, 512) | 4,096 | 0.06396 | 0.07536 | 1.18 |
| `blocks.0.attn.gate_proj.weight` | (8, 512) | 4,096 | 0.1009 | 0.1184 | 1.17 |
| `blocks.8.attn.c_q.weight` | (512, 512) | 262,144 | 0.07322 | 0.08547 | 1.17 |
| `blocks.9.attn.c_q.weight` | (512, 512) | 262,144 | 0.07362 | 0.08546 | 1.16 |
| `blocks.10.attn.c_k.weight` | (256, 512) | 131,072 | 0.08373 | 0.09713 | 1.16 |
| `blocks.1.attn.gate_proj.weight` | (8, 512) | 4,096 | 0.07101 | 0.0817 | 1.15 |
| `blocks.3.attn.gate_proj.weight` | (8, 512) | 4,096 | 0.05718 | 0.06575 | 1.15 |
| `blocks.6.attn.gate_proj.weight` | (8, 512) | 4,096 | 0.0549 | 0.06311 | 1.15 |
| `blocks.7.attn.gate_proj.weight` | (8, 512) | 4,096 | 0.06342 | 0.07266 | 1.15 |
| `blocks.10.attn.c_q.weight` | (512, 512) | 262,144 | 0.08085 | 0.0926 | 1.15 |
| `blocks.3.attn.c_k.weight` | (256, 512) | 131,072 | 0.08474 | 0.097 | 1.14 |
| `blocks.8.attn.proj.weight` | (512, 512) | 262,144 | 0.09088 | 0.1039 | 1.14 |
| `blocks.2.attn.gate_proj.weight` | (8, 512) | 4,096 | 0.05999 | 0.06858 | 1.14 |
| `blocks.3.attn.c_q.weight` | (512, 512) | 262,144 | 0.07661 | 0.08741 | 1.14 |
| `blocks.6.attn.c_k.weight` | (256, 512) | 131,072 | 0.08644 | 0.09862 | 1.14 |
| `blocks.5.attn.gate_proj.weight` | (8, 512) | 4,096 | 0.05793 | 0.06608 | 1.14 |
| `blocks.9.attn.proj.weight` | (512, 512) | 262,144 | 0.09437 | 0.1076 | 1.14 |
| `blocks.0.attn.proj.weight` | (512, 512) | 262,144 | 0.08221 | 0.09366 | 1.14 |

## Most stable (bottom 10)

| Module | shape | rel_std |
|---|---|---:|
| `blocks.6.attn_scale` | (512,) | 0.17 |
| `blocks.4.attn_scale` | (512,) | 0.158 |
| `blocks.6.mlp_scale` | (512,) | 0.126 |
| `blocks.5.attn_scale` | (512,) | 0.125 |
| `blocks.9.mlp_scale` | (512,) | 0.0805 |
| `blocks.7.mlp_scale` | (512,) | 0.0779 |
| `blocks.8.mlp_scale` | (512,) | 0.0637 |
| `_nlfi_bigram_mult` | (16384,) | 0 |
| `_nlfi_fourgram_mult` | (16384,) | 0 |
| `_nlfi_trigram_mult` | (16384,) | 0 |
