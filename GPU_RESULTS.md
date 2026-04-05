# GPU Test Results

## Hardware: RTX 3080 Ti (12GB, Ampere, RunPod $0.18/hr)

## Baseline Speed (gpu_quick_test.py, Apr 5)

| Config | ms/step | Steps/10min | VRAM |
|--------|---------|-------------|------|
| 9L/512d | 42.4 | 14,139 | 1.13GB |
| 11L/512d | 52.2 | 11,484 | 1.48GB |

## Speed Experiments (gpu_speed_test.py, 30 steps each)

| Config | ms/step | Steps/10min | Notes |
|--------|---------|-------------|-------|
| 9L baseline | 41.6 | 14,434 | |
| 11L baseline | 51.4 | 11,678 | |
| 11L + layer drop 50% | 39.9 | 15,054 | 1.3x faster |
| 11L + layer drop 80% | 32.6 | 18,407 | 1.6x faster |
| 11L + seq=256 | 13.2 | 45,289 | 3.9x faster |
| 11L + seq=128 | 8.4 | 71,768 | 6.1x faster |
| Tiny 4L/256d | 8.3 | 72,298 | 6.2x faster |
| Hybrid conv+attn | 46.7 | 12,844 | 1.1x (not worth it) |
| Lossy 50% mask | 52.7 | 11,376 | NO speedup |

## Fixed-Step Quality (gpu_progressive_test.py, 1000 steps each)

| Strategy | Eval Loss | Time | vs Reference |
|----------|-----------|------|--------------|
| **Prog grow 500@4L + 500@11L** | **9.5625** | **39.8s** | **-0.18 BETTER, 1.4x faster** |
| Layer drop (6L proxy) | 9.6750 | 32.6s | -0.07 better, 1.7x faster |
| Standard 11L/1024 (REF) | 9.7438 | 54.7s | baseline |
| Prog ALL 4L/128→11L/256→11L/1024 | 9.7812 | 24.1s | +0.04 same, 2.3x faster |
| Prog seq 700@128 + 300@1024 | 9.9437 | 22.7s | +0.20 worse, 2.4x faster |
| Prog seq 500@128 + 500@1024 | 10.1000 | 32.2s | +0.36 worse, 1.7x faster |

## ⭐ TIMED QUALITY TEST (gpu_timed_test.py, 120s each — THE DEFINITIVE TEST)

**Question: what gets the best quality in a fixed time budget?**

| Strategy | Eval Loss | Steps | vs Ref | Verdict |
|----------|-----------|-------|--------|---------|
| **7. Mostly short 90%@128 + 10%@1024** | **8.8875** | **13,762** | **-0.44** | **⭐ WINNER** |
| 3. Prog seq 70/30 (128→1024) | 9.1687 | 11,185 | -0.16 | Good |
| 6. All seq=128 (max steps) | 9.2063 | 15,066 | -0.12 | Good but no long context |
| 1. Standard 11L/1024 (REF) | 9.3250 | 2,114 | 0.00 | Baseline |
| 2. Prog seq 50/50 (128→1024) | 9.3656 | 8,573 | +0.04 | Neutral |
| 4. 3-phase 128→256→1024 | 9.3844 | 7,960 | +0.06 | Neutral |
| 5. Grow+seq 4L/128→11L/256→11L/1024 | 9.4594 | 14,405 | +0.13 | Worse despite more steps |

### Key Finding

**Train at seq=128 for 90% of time, switch to seq=1024 for last 10%.**
- 6.5x more steps than standard (13,762 vs 2,114)
- 0.44 BETTER eval loss than standard
- Short sequences train local patterns fast, final 10% learns long-range

### H100 Projection
```
Phase 1 (9 min): seq=128 → ~60,000 steps at ~9ms/step
Phase 2 (1 min): seq=1024 → ~700 steps at ~85ms/step
TOTAL: ~60,700 steps vs ~7,000 baseline = 8.7x more training
```

## Dead Speed Tricks
- Lossy token mask: NO speedup (backward same cost regardless)
- Hybrid conv/attn: only 1.1x (not worth complexity)
- Progressive grow + seq combo: WORSE quality despite more steps
