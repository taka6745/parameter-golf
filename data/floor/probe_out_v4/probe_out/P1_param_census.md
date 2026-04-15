# P1: Parameter Census

- Checkpoint: `final_model_seed42.int6.ptz`  decompression: `brotli+unshuffle`
- Modules: 140  Total uncompressed tensor bytes: 36,270,442 (216.2% of 16 MB — brotli shrinks this to fit)

| Module | q-shape | q-dtype | numel | bytes | % of cap |
|---|---|---:|---:|---:|---:|
| `tok_emb.weight` | (8192, 512) | int8 | 4,202,496 | 4,210,688 | 25.10% |
| `blocks.0.mlp.fc.weight` | (2048, 512) | int8 | 1,050,624 | 1,052,672 | 6.27% |
| `blocks.1.mlp.fc.weight` | (2048, 512) | int8 | 1,050,624 | 1,052,672 | 6.27% |
| `blocks.2.mlp.fc.weight` | (2048, 512) | int8 | 1,050,624 | 1,052,672 | 6.27% |
| `blocks.3.mlp.fc.weight` | (2048, 512) | int8 | 1,050,624 | 1,052,672 | 6.27% |
| `blocks.4.mlp.fc.weight` | (2048, 512) | int8 | 1,050,624 | 1,052,672 | 6.27% |
| `blocks.5.mlp.fc.weight` | (2048, 512) | int8 | 1,050,624 | 1,052,672 | 6.27% |
| `blocks.6.mlp.fc.weight` | (2048, 512) | int8 | 1,050,624 | 1,052,672 | 6.27% |
| `blocks.7.mlp.fc.weight` | (2048, 512) | int8 | 1,050,624 | 1,052,672 | 6.27% |
| `blocks.8.mlp.fc.weight` | (2048, 512) | int8 | 1,050,624 | 1,052,672 | 6.27% |
| `blocks.9.mlp.fc.weight` | (2048, 512) | int8 | 1,050,624 | 1,052,672 | 6.27% |
| `blocks.10.mlp.fc.weight` | (2048, 512) | int8 | 1,050,624 | 1,052,672 | 6.27% |
| `blocks.0.mlp.proj.weight` | (512, 2048) | int8 | 1,049,088 | 1,049,600 | 6.26% |
| `blocks.1.mlp.proj.weight` | (512, 2048) | int8 | 1,049,088 | 1,049,600 | 6.26% |
| `blocks.2.mlp.proj.weight` | (512, 2048) | int8 | 1,049,088 | 1,049,600 | 6.26% |
| `blocks.3.mlp.proj.weight` | (512, 2048) | int8 | 1,049,088 | 1,049,600 | 6.26% |
| `blocks.4.mlp.proj.weight` | (512, 2048) | int8 | 1,049,088 | 1,049,600 | 6.26% |
| `blocks.5.mlp.proj.weight` | (512, 2048) | int8 | 1,049,088 | 1,049,600 | 6.26% |
| `blocks.6.mlp.proj.weight` | (512, 2048) | int8 | 1,049,088 | 1,049,600 | 6.26% |
| `blocks.7.mlp.proj.weight` | (512, 2048) | int8 | 1,049,088 | 1,049,600 | 6.26% |
| `blocks.8.mlp.proj.weight` | (512, 2048) | int8 | 1,049,088 | 1,049,600 | 6.26% |
| `blocks.9.mlp.proj.weight` | (512, 2048) | int8 | 1,049,088 | 1,049,600 | 6.26% |
| `blocks.10.mlp.proj.weight` | (512, 2048) | int8 | 1,049,088 | 1,049,600 | 6.26% |
| `blocks.0.attn.c_q.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.0.attn.proj.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.1.attn.c_q.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.1.attn.proj.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.2.attn.c_q.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.2.attn.proj.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.3.attn.c_q.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.3.attn.proj.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.4.attn.c_q.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.4.attn.proj.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.5.attn.c_q.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.5.attn.proj.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.6.attn.c_q.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.6.attn.proj.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.7.attn.c_q.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.7.attn.proj.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
| `blocks.8.attn.c_q.weight` | (512, 512) | int8 | 262,656 | 263,168 | 1.57% |
