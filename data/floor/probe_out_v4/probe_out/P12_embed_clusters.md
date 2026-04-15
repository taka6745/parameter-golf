# P12: Embedding Cosine-Similarity Clusters

Vocab size 8192, embedding dim 512. Cosine sim computed between row-normalized vectors.

## Top-1 similarity distribution

| threshold | rows with top-1 > threshold | pct |
|---|---:|---:|
| > 0.95 | 0 | 0.0% |
| > 0.9 | 97 | 1.2% |
| > 0.8 | 160 | 2.0% |
| > 0.7 | 575 | 7.0% |
| > 0.5 | 4830 | 59.0% |

## Greedy clustering by cos-sim threshold

| threshold | clusters | max cluster size | mean |
|---|---:|---:|---:|
| 0.95 | 8192 | 1 | 1 |
| 0.9 | 8134 | 11 | 1 |
| 0.8 | 8051 | 42 | 1 |

## Top-20 most similar vocab pairs

| row_i | row_j | cosine |
|---:|---:|---:|
| 29 | 74 | 0.9452 |
| 74 | 29 | 0.9452 |
| 27 | 81 | 0.9419 |
| 81 | 27 | 0.9419 |
| 128 | 29 | 0.9393 |
| 61 | 77 | 0.9376 |
| 77 | 61 | 0.9376 |
| 53 | 55 | 0.9348 |
| 55 | 53 | 0.9348 |
| 17 | 110 | 0.9337 |
| 110 | 17 | 0.9337 |
| 59 | 90 | 0.9316 |
| 90 | 59 | 0.9316 |
| 50 | 259 | 0.9314 |
| 259 | 50 | 0.9314 |
| 48 | 54 | 0.9302 |
| 54 | 48 | 0.9302 |
| 13 | 42 | 0.9301 |
| 42 | 13 | 0.9301 |
| 35 | 42 | 0.9296 |
