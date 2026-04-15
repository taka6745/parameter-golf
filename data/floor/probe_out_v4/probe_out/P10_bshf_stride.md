# P10: BSHF Stride Sweep
Brotli quality=11 over the dequantized torch.save payload (36,337,057 bytes) at various byte-shuffle strides.

Baseline (current submission uses stride=2): 16,051,299 bytes.

| stride | compressed bytes | delta vs stride=2 | compress time |
|---:|---:|---:|---:|
| 1 | 16,058,493 | +7,194 bytes (+0.04%) | 63.1s |
| 2 | 16,051,299 | +0 bytes (+0.00%) | 67.9s |
| 3 | 16,077,476 | +26,177 bytes (+0.16%) | 74.2s |
| 4 | 16,060,143 | +8,844 bytes (+0.06%) | 70.3s |
| 5 | 16,086,273 | +34,974 bytes (+0.22%) | 78.1s |
| 8 | 16,070,740 | +19,441 bytes (+0.12%) | 77.1s |
