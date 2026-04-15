# P16: Llama-3.2-1B BPB Ceiling

Scored 81,237 bytes of FineWeb val text (first 20 chunks of 4000 chars).

| Model | Params | Val BPB |
|---|---:|---:|
| **Llama-3.2-1B** | 1.2B | **0.9532** |
| Our 1.082 submission | 30M (16 MB int6) | 1.082 |
| xz -9e | (compressor) | 2.211 |

**Interpretation**: Llama-3.2-1B represents what a ~40× larger model with ~1000× the training compute achieves on the same text. If our 1.082 is above Llama-1B's 0.9532, the headroom to <1.0 BPB is real. The <1.0 moonshot is theoretically supported by a small frontier LM.
