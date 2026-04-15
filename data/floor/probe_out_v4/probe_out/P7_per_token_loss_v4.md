# P7: Per-Token Loss (fix-calibrated)

Sample: 100 × 2048 = 204,800 tokens. Mean NLL/token = 7.5334 nats ≈ 10.8683 bits/token.
Approx BPB (at 2.35 B/tok): **4.6248**. Compare to submission val_bpb = 1.082.

## By position

| range | tokens | mean NLL |
|---|---:|---:|
| [0, 128) | 12,800 | 8.1268 |
| [128, 512) | 38,400 | 7.6193 |
| [512, 1024) | 51,200 | 7.5184 |
| [1024, 2048) | 102,400 | 7.4345 |

## By token rarity

| bucket | tokens | mean NLL |
|---|---:|---:|
| top5pct | 6,344 | 3.6228 |
| top5-20pct | 33,439 | 3.9547 |
| top20-50pct | 62,582 | 6.5326 |
| tail50pct | 102,435 | 9.5552 |
