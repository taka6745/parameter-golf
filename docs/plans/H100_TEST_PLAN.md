# GPU Testing Plan

## Budget: ~$28 (8 GPU-hours)

## Phase 1: Pipeline Validation (A40/3080Ti spot, ~$0.05)

**Goal:** Does our code run on CUDA? Pass/fail. Fix bugs cheap before H100.

| # | Test | Steps | Measures | Status |
|---|------|-------|----------|--------|
| P1 | Competition baseline (SP-1024) | 10 | Runs? ms/step? | |
| P2 | Our BPE-8192 tokenizer | 10 | Runs? | |
| P3 | 11L architecture | 10 | Runs? | |
| P4 | torch.compile enabled | 10 | Compiles? | |
| P5 | N-gram bias module (PyTorch port) | 10 | Runs? | |
| P6 | NorMuon optimizer | 10 | Runs? | |

## Phase 2: Speed Shootout (same pod, ~$0.10)

**Goal:** Compare ms/step across configs. Steps 20-50 only (skip compile warmup).

| # | Test | Steps | Measures | Status |
|---|------|-------|----------|--------|
| S1 | Transformer 9L/512d baseline | 50 | ms/step reference | |
| S2 | Transformer 11L/512d | 50 | ms/step vs S1 | |
| S3 | Transformer 11L/3xMLP | 50 | ms/step vs S1 | |
| S4 | torch.compile reduce-overhead | 50 | Speedup vs S1 | |
| S5 | seq=256 (progressive seq test) | 50 | Speedup vs S1 | |
| S6 | GLA (flash-linear-attention) | 50 | Speedup vs S1 (THE BIG TEST) | |
| S7 | RWKV-7 (via fla library) | 50 | Speedup vs S1 | |

## Phase 3: Quality + Compression (same pod, ~$0.15)

**Goal:** Does GPTQ work? Does sliding window eval give correct BPB?

| # | Test | Steps | Measures | Status |
|---|------|-------|----------|--------|
| Q1 | Full 500-step run (best config from S-tests) | 500 | val_bpb on CUDA | |
| Q2 | GPTQ int6 on trained model | post | Compressed size, quant gap | |
| Q3 | Sliding window eval (stride=64) | eval | val_bpb delta vs stride=1024 | |
| Q4 | Temperature scaling (T=0.85) | eval | val_bpb delta | |

## Phase 4: Speed Multiplication (same pod, ~$0.10)

**Goal:** Test 2C speed experiments on CUDA.

| # | Test | Steps | Measures | Status |
|---|------|-------|----------|--------|
| M1 | Progressive grow (4L to 11L) | 50+50 | Quality after growth | |
| M2 | Progressive layer dropping | 50 | ms/step + quality | |
| M3 | Random layer subset (stochastic depth) | 50 | ms/step + quality | |
| M4 | Lossy token mask (50%) | 50 | ms/step + quality | |

## Phase 5: Full H100 Run (8xH100 SXM, ~$21.50)

**Goal:** Final submission. Only after P1-M4 pass.

| # | Test | Time | Measures | Status |
|---|------|------|----------|--------|
| H1 | Best config from all phases, seed 42 | 10+10 min | val_bpb | |
| H2 | Same config, seed 314 | 10+10 min | val_bpb | |
| H3 | Same config, seed 999 | 10+10 min | val_bpb | |
| H4 | Package submission.json + PR | - | Submit! | |

## Estimated Total Cost

| Phase | GPU | Time | Cost |
|-------|-----|------|------|
| P1-P6 | 1x spot ($0.18/hr) | ~10 min | $0.05 |
| S1-S7 | same pod | ~15 min | $0.05 |
| Q1-Q4 | same pod | ~30 min | $0.10 |
| M1-M4 | same pod | ~15 min | $0.05 |
| H1-H4 | 8x H100 SXM ($2.69/hr each) | ~60 min | $21.50 |
| **Total** | | | **~$21.75** |
| **Buffer** | | | **~$6.25** |

## What to Port Before Running

Our Mac stack into CUDA train_gpt.py:
1. N-gram logit bias (ngram_logit_bias.py exists in PyTorch)
2. NorMuon (15 lines in optimizer)
3. Turbo-Muon coefficients (drop-in)
4. 11L config (env var NUM_LAYERS=11)
5. Signed hashing on n-gram tables (2 lines in table build)
6. DC500 categories (load + apply in forward)
7. Softcap=20 (env var LOGIT_SOFTCAP=20)
8. Dual MLP (if we include arch winners)

## Decision Points

- After S6 (GLA speed): if GLA is >30% faster, switch architecture
- After Q2 (GPTQ): if model doesnt fit 16MB, adjust table sizes
- After M1-M4: if any speed trick works, include in H100 config

## Scripts

- `run_gpu_batch.sh` - automated: create pod, upload, run all tests, download, stop
- `train_gpt.py` - competition baseline CUDA (needs our modifications)
- `ngram_logit_bias.py` - our n-gram bias module (already PyTorch)
