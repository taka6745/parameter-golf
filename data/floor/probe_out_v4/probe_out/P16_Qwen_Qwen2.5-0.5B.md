# P16: Llama-3.2-1B BPB Ceiling

Scored 81,237 bytes of FineWeb val text (first 20 chunks of 4000 chars).

| Model | Params | Val BPB |
|---|---:|---:|
| **Llama-3.2-1B** | 1.2B | **1.0582** |
| Our 1.082 submission | 30M (16 MB int6) | 1.082 |
| xz -9e | (compressor) | 2.211 |

**Interpretation**: Llama-3.2-1B represents what a ~40× larger model with ~1000× the training compute achieves on the same text. If our 1.082 is above Llama-1B's 1.0582, the headroom to <1.0 BPB is limited. The <1.0 moonshot would require beating a 1.2B-param model — likely impossible at 16 MB without structural novelty beyond scaling.
