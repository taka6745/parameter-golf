# P15: Gated-Attention Ablation

Zeroed all 11× `attn.gate_proj.weight` tensors (45 KB of params that P11 flagged as lottery-ticket noise). Ran 30×2048=61,440 val tokens.

| | mean NLL/token |
|---|---:|
| baseline | 7.5334 |
| gates zeroed | 7.4568 |
| Δ | -0.0766 nats (-1.02%) |

**Interpretation**: gates contribute marginally (-0.077 nats). Worth ablating in a full retrain to confirm.
