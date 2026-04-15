# RunPod Test Plan — Parameter Golf Submission

**Goal:** Validate our complete stack on H100 and produce a 3-seed submission.
**Budget:** ~$45-55 (covers 5 tests + 3-seed final + buffer).
**Current best:** 1.6929 BPP (Mac, 9L wavelet, 1000 steps).
**Target:** sub-1.00 BPP on H100. Stretch: 0.75-0.85 BPP.

---

## Executive Summary

We have ~25 validated Mac wins. The 3 biggest unknowns to validate on H100 are:
1. **Progressive seq + high LR** (-0.15 BPP projected, GPU-validated on 3080Ti)
2. **Multi-order n-gram eval cache** (-0.10 BPP projected, 8/10 top PRs use it — our biggest gap)
3. **Architecture defaults at scale** (11L, 3x MLP, seq=2048 — never tested at scale)

Tests are designed to **isolate each variable** so we can attribute gains. Each test = 1 H100 hour ≈ $5.

---

## What's Validated vs What Mac Hasn't Tested

### Validated on Mac (just port these in)
| Component | Setting | Source |
|---|---|---|
| Tokenizer | BPE-8192 | -0.129 BPP, biggest single lever |
| Activation | LeakyReLU(0.5)² | -0.014 BPP |
| Attention softcap | 20 (was 30) | -0.008 BPP |
| Bigram bias | w=0.20, 16K hash, signed | -0.040 BPP |
| Trigram bias | w=0.15, 16K hash | -0.021 BPP |
| 4-gram bias | w=0.10, 16K hash | -0.019 BPP |
| DC500 categories | w=0.15 | -0.010 BPP |
| Period/sentence bias | English orthographic | -0.002 BPP |
| Optimizer | NorMuon (per-row norm) | -0.132 BPP |
| Turbo-Muon | 4 NS steps (was 5) | free speedup |
| WaveletGPT | multi-scale mixing | -0.018 BPP |
| SmearGate | gate prev+curr token | -0.009 BPP |
| Predictive coding | 80% error propagation | -0.040 BPP (stacks with wavelet) |
| Compression | Brotli-11 (vs zlib-9) | -1.47 MB saved |

### H100-only / never tested at scale (the unknowns)
| Component | Setting | Why test |
|---|---|---|
| **Layers** | 9 → **11** | All top PRs use 11L; Mac 1000-step test inconclusive |
| **MLP expansion** | 2x → **3x** | Top 8/10 PRs use 3x |
| **Sequence length** | 1024 → **2048** | Universal in H100 entries |
| **QK gain** | 1.5 → **4.0** | PR #1176 uses 4.0 |
| **XSA** | off → **last 4 layers** | Universal in H100 (had GQA bug on Mac) |
| **Quantization** | int8 → **int6 GPTQ** | Better quality per byte |
| **Per-layer mixed precision** | uniform int6 → **int7/6/5 by layer** | From our entanglement entropy finding |
| **Lloyd-Max codebook** | linear quant → **non-uniform** | 86% less quant error (256B overhead) |
| **EMA(0.997)** | off → **on** | -0.005 BPP, 4/10 top PRs |
| **Score-First TTT** | off → **LoRA on val tokens** | -0.05 BPP, 6/10 top PRs |

### Discovered overnight (highest priority unknowns)
| Component | Source | Confidence |
|---|---|---|
| **Progressive seq + high LR** | RTX 3080Ti GPU test, eval_loss 6.97 vs 9.33 | High (GPU validated) |
| **Cosine LR Phase 2** | Simulation, 8x better than constant | Medium (simulation only) |
| **Multi-order n-gram eval cache** | Channel capacity analysis, 8/10 top PRs | High (theory + competition) |
| **Hedge mixer for combination** | Beats additive logits in simulation | Medium |
| **2-pass eval (fast → TTT → sliding)** | Designed from constraints | Medium |

---

## Critical Theoretical Findings (Why These Tests Matter)

1. **Phase 2 is 8x under-resourced** (channel capacity analysis). Without the eval cache, our model can only learn 12% of long-range patterns. **The eval cache is not optional** — it's essential for progressive seq to work.

2. **Kelly criterion predicts our LR ratio to 101% accuracy** (33.6x predicted vs 33.3x empirical). This validates that Phase 1 lr=1e-3 and Phase 2 lr=3e-5 are theoretically optimal.

3. **Gradients are 91.6° orthogonal between phases** — short-range and long-range patterns fill different weight subspaces. Progressive training has zero interference.

4. **Token co-occurrence is small-world** (diameter=4) → 4 layers propagate everything, layers 5-10 are refinement. This justifies aggressive late-layer quantization.

5. **Weights are log-normal (σ=1.6)** → Lloyd-Max quantization saves 86% of int6 error for 256 bytes of codebook.

6. **Wavelet's `k = min(2^(i+1), seq_len)` naturally adapts to progressive seq** — layers 7+ are dormant during Phase 1 and activate in Phase 2. Zero extra code needed.

---

## Pre-Test Setup (DO ONCE)

### 1. Create RunPod 1xH100 dev pod
```bash
# From local machine
runpodctl create pod --name paramgolf-dev --gpuType "NVIDIA H100 80GB HBM3" \
  --gpuCount 1 --containerDiskSize 50 --volumeSize 100 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --communityCloud --ports "22/tcp"

# Get pod ID and SSH host from runpod.io
```

### 2. SSH in and clone repo
```bash
ssh PODHOSTID@ssh.runpod.io -i ~/.ssh/id_ed25519
cd /workspace
git clone https://github.com/taka6745/paramgolf.git
cd paramgolf
pip install -q sentencepiece numpy huggingface_hub datasets tqdm zstandard brotli
```

### 3. Prepare BPE-8192 data on H100
```bash
# This takes ~10 min, do it ONCE at the start
.venv/bin/python3 data/cached_challenge_fineweb.py --variant bpe8192 --train-shards 10
# Creates: data/datasets/fineweb10B_bpe8192/fineweb_train_*.bin
```

### 4. Copy precomputed n-gram tables
```bash
# From local Mac (or rebuild on H100)
scp data/bigram_tab_8192v.npy data/trigram_logprobs_8192v.npy \
    data/fourgram_logprobs_8192v.npy data/dist_cats_500_8192.npz \
    data/lloyd_max_codebook_64.npy \
    PODHOSTID@ssh.runpod.io:/workspace/paramgolf/data/
```

---

## Code Patches Required

The competition's `train_gpt.py` needs the following patches before running tests. Apply these BEFORE Test 0.

### Patch A: Architecture changes (Test 1+)
Modify `train_gpt.py` model class:

```python
# 1. LeakyReLU(0.5)² activation (1 line)
# Find: self.act = lambda x: torch.relu(x) ** 2
# Replace with:
self.act = lambda x: torch.nn.functional.leaky_relu(x, 0.5) ** 2

# 2. Softcap 20 (was 30) — change one constant
# Find: SOFTCAP = 30
SOFTCAP = 20

# 3. Add wavelet_mix function near top of file
def wavelet_mix(x, layer_idx, mix_ratio=0.2):
    """WaveletGPT: causal multi-scale averaging on half dims.
    Validated: -0.018 BPP on Mac (9L, 1000 steps).
    Naturally adapts to progressive seq via k = min(2^(i+1), T)."""
    B, T, D = x.shape
    half = D // 2
    left = x[..., :half]
    right = x[..., half:]
    k = min(2 ** (layer_idx + 1), T)
    cs = torch.cumsum(right, dim=1)
    shifted = torch.nn.functional.pad(cs[:, :-k], (0, 0, k, 0))
    counts = torch.arange(1, T + 1, device=x.device, dtype=right.dtype)
    counts = counts.clamp(max=k).unsqueeze(0).unsqueeze(-1)
    right_avg = (cs - shifted) / counts
    right_mixed = (1 - mix_ratio) * right + mix_ratio * right_avg
    return torch.cat([left, right_mixed], dim=-1)

# 4. In TransformerBlock.forward(), add wavelet between attention and MLP:
#    x = x + self.attention(self.norm1(x))
#    x = wavelet_mix(x, self.layer_idx)  # ADD THIS
#    x = x + self.mlp(self.norm2(x))
```

### Patch B: N-gram bias loading and application (Test 1+)
```python
# Near top of train_gpt.py, after model creation:
import numpy as np

# Load precomputed n-gram tables
bigram_table = torch.from_numpy(np.load("data/bigram_tab_8192v.npy")).to(device)
trigram_table = torch.from_numpy(np.load("data/trigram_logprobs_8192v.npy")).to(device)

# Hash function (signed polynomial)
def hash_bigram(prev, vocab_size=8192):
    return (prev * 36313) % 16384

def hash_trigram(prev2, prev1, vocab_size=8192):
    return ((prev2 * 36313 + prev1 * 27191) % 16384)

# In forward pass, ADD to logits before softmax:
# Get bigram/trigram contexts from input tokens
bigram_logits = bigram_table[hash_bigram(prev_tokens)]  # (B, T, V)
trigram_logits = trigram_table[hash_trigram(prev2_tokens, prev_tokens)]
ngram_weight = current_ngram_weight  # 0.40 in Phase 1, 0.05 in Phase 2
logits = logits + ngram_weight * (0.20 * bigram_logits + 0.15 * trigram_logits)
```

### Patch C: Progressive seq + LR scheduling (Test 2+)
```python
# In Hyperparameters class:
progressive_seq = bool(int(os.environ.get("PROGRESSIVE_SEQ", 1)))
phase1_seq_len = int(os.environ.get("PHASE1_SEQ_LEN", 128))
phase1_lr_mult = float(os.environ.get("PHASE1_LR_MULT", 25.0))
phase1_fraction = float(os.environ.get("PHASE1_FRACTION", 0.85))
phase2_seq_len = int(os.environ.get("PHASE2_SEQ_LEN", 1024))
phase1_ngram_weight = float(os.environ.get("PHASE1_NGRAM_WEIGHT", 0.40))
phase2_ngram_weight = float(os.environ.get("PHASE2_NGRAM_WEIGHT", 0.05))

# After optimizer creation, multiply initial LR:
if args.progressive_seq:
    for group in optimizer.param_groups:
        group['lr'] *= args.phase1_lr_mult

# In training loop, replace step iteration with:
import math
phase1_end_time = args.max_wallclock_seconds * args.phase1_fraction
current_phase = 1 if args.progressive_seq else 2
base_lr_p1 = optimizer.param_groups[0]['lr']

for step in range(args.iterations):
    elapsed = time.perf_counter() - t0
    if elapsed >= args.max_wallclock_seconds:
        break

    # Phase transition
    if args.progressive_seq and current_phase == 1 and elapsed >= phase1_end_time:
        current_phase = 2
        # Drop LR to 1e-4 (start of cosine in Phase 2)
        for group in optimizer.param_groups:
            group['lr'] = 1e-4
        log0(f"PHASE TRANSITION at step {step}: seq {args.phase1_seq_len} -> {args.phase2_seq_len}")

    # Phase 2: cosine decay from 1e-4 to 3e-5
    if current_phase == 2:
        phase2_progress = (elapsed - phase1_end_time) / max(1, args.max_wallclock_seconds - phase1_end_time)
        lr_phase2 = 3e-5 + 0.5 * (1e-4 - 3e-5) * (1 + math.cos(math.pi * phase2_progress))
        for group in optimizer.param_groups:
            group['lr'] = lr_phase2

    # Current seq and ngram weight
    if current_phase == 1:
        seq_len = args.phase1_seq_len
        ngram_weight = args.phase1_ngram_weight
    else:
        seq_len = args.phase2_seq_len
        ngram_weight = args.phase2_ngram_weight

    # Get batch with current seq_len
    x, y = loader.next_batch(args.train_batch_tokens, seq_len, grad_accum_steps)
    # ... rest of training step
```

### Patch D: EMA weight averaging (Test 1+)
```python
import copy

# After model creation:
ema_model = copy.deepcopy(model)
ema_decay = 0.997

# After each optimizer.step():
with torch.no_grad():
    for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(ema_decay).add_(p_model.data, alpha=1-ema_decay)

# At eval time, swap to EMA model:
model_for_eval = ema_model  # use this for final val_bpb computation
```

### Patch E: Eval n-gram cache + hedge mixer (Test 3+)
Add `eval_ngram_cache.py` to the workspace (already exists), then in the eval function:

```python
from eval_ngram_cache import NgramEvalCache, HedgeMixer

def eval_with_cache(model, val_tokens, vocab_size=8192):
    cache = NgramEvalCache(max_order=7, min_count=2, vocab_size=vocab_size)
    mixer = HedgeMixer(n_experts=2, eta=0.1)
    
    total_loss = 0
    n_tokens = 0
    
    for doc_tokens in val_iter(val_tokens):
        # Run model on document, get per-position logits
        with torch.no_grad():
            logits = model(doc_tokens)
            model_probs_all = torch.softmax(logits, dim=-1).cpu().numpy()
        
        for i in range(1, len(doc_tokens)):
            actual = doc_tokens[i].item()
            model_p = model_probs_all[i-1]
            cache_p = cache.predict(doc_tokens[:i].tolist())
            mixed = mixer.mix([model_p, cache_p])
            
            total_loss += -math.log(mixed[actual] + 1e-15)
            n_tokens += 1
            
            mixer.update([model_p[actual], cache_p[actual]])
            cache.update(doc_tokens[:i+1].tolist())
    
    return total_loss / n_tokens
```

### Patch F: Sliding window eval (Test 3+)
```python
# Replace standard eval loop:
def sliding_window_eval(model, val_tokens, window=1024, stride=512):
    """Score with overlapping windows. Each token is scored at its 
    deepest-context position in any window."""
    losses = {}
    for start in range(0, len(val_tokens) - window, stride):
        chunk = val_tokens[start:start+window]
        with torch.no_grad():
            logits = model(chunk.unsqueeze(0))[0]
        # Only score positions [stride:window] (discard low-context start)
        # except for the very first window
        score_start = 0 if start == 0 else stride
        for i in range(score_start, window - 1):
            global_pos = start + i + 1
            actual = chunk[i+1].item()
            loss = -torch.log_softmax(logits[i], dim=-1)[actual].item()
            losses[global_pos] = loss
    return sum(losses.values()) / len(losses)
```

### Patch G: Per-layer mixed precision GPTQ (Test 4+)
```python
# After training, replace uniform int6 with mixed precision:
def per_layer_quantize(model, codebook):
    """Layer 0-3: int7, Layer 4-6: int6, Layer 7-10: int5.
    All use Lloyd-Max codebook for non-uniform quantization."""
    
    # Load Lloyd-Max codebooks (one per bit width)
    cb6 = torch.from_numpy(np.load("data/lloyd_max_codebook_64.npy"))
    # Generate cb5 (32 levels) and cb7 (128 levels) from log-normal fit
    
    for layer_idx, layer in enumerate(model.layers):
        if layer_idx < 4:
            bits = 7
        elif layer_idx < 7:
            bits = 6
        else:
            bits = 5
        # Quantize this layer with the appropriate codebook
        # ... (use GPTQ algorithm with the chosen bit width and codebook)
```

---

## Test Sequence

### Test 0: Smoke test (50 steps, ~$1)
**Goal:** Verify port compiles and trains without crashes.

```bash
# Apply Patches A, B, D
# Run 50 steps with our base stack
ITERATIONS=50 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=10 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
MLP_EXPANSION=3 \
TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model" \
DATA_PATH="./data/datasets/fineweb10B_bpe8192" \
.venv/bin/python3 train_gpt.py 2>&1 | tee logs/test0_smoke.txt
```

**Validation criteria:**
- Compiles without errors
- Loss decreases (final < initial)
- ms/step is in expected range (~80-120ms for 11L/3x at seq=1024)

If this fails, fix bugs before proceeding.

---

### Test 1: H100 baseline (full stack, NO progressive seq) — 15 min, ~$5
**Goal:** Get a clean baseline number with all our Mac wins ported. This is the floor.

```bash
# Apply Patches A, B, D
# NOT progressive seq yet
ITERATIONS=10000 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
MLP_EXPANSION=3 \
SEQ_LEN=2048 \
QK_GAIN=4.0 \
USE_XSA=1 \
XSA_LAYERS=4 \
USE_EMA=1 \
EMA_DECAY=0.997 \
TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model" \
DATA_PATH="./data/datasets/fineweb10B_bpe8192" \
.venv/bin/python3 train_gpt.py 2>&1 | tee logs/test1_baseline_h100.txt
```

**Validation criteria:**
- Hits ~7,000 steps in 10 min
- Final val_bpb in range 1.05-1.15 (similar to top non-progressive submissions)
- This is our **baseline** for measuring progressive seq gain

**What we learn:**
- Whether 11L + 3x MLP + seq=2048 + QK 4.0 + XSA actually help at scale
- Real H100 step time
- Real eval pipeline timing

---

### Test 2: + Progressive seq + cosine LR Phase 2 — 15 min, ~$5
**Goal:** Validate the -0.15 BPP from progressive seq + high LR.

```bash
# Same as Test 1 + Patch C
PROGRESSIVE_SEQ=1 \
PHASE1_SEQ_LEN=128 \
PHASE1_LR_MULT=25.0 \
PHASE1_FRACTION=0.85 \
PHASE2_SEQ_LEN=1024 \
PHASE1_NGRAM_WEIGHT=0.40 \
PHASE2_NGRAM_WEIGHT=0.05 \
\
ITERATIONS=100000 \
MAX_WALLCLOCK_SECONDS=600 \
NUM_LAYERS=11 MODEL_DIM=512 MLP_EXPANSION=3 \
QK_GAIN=4.0 USE_XSA=1 XSA_LAYERS=4 USE_EMA=1 \
TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model" \
DATA_PATH="./data/datasets/fineweb10B_bpe8192" \
.venv/bin/python3 train_gpt.py 2>&1 | tee logs/test2_progressive.txt
```

**Validation criteria:**
- Phase 1 reaches ~50,000+ steps at seq=128
- Phase 2 reaches ~1,000+ steps at seq=1024
- Final val_bpb is **lower** than Test 1 by 0.05-0.15 BPP
- No NaN/diverge during phase transition

**Decision point:**
- If improvement < 0.05 BPP: progressive seq doesn't transfer well to H100. Fall back to Test 1 config for submission.
- If improvement 0.05-0.15 BPP: proceed to Test 3 with confidence
- If improvement > 0.15 BPP: jackpot, we're well into top-10 territory

---

### Test 3: + Eval cache + hedge mixer + sliding window — 15 min, ~$5
**Goal:** Validate the -0.10 BPP from the eval cache. **This is our biggest gap.**

```bash
# Same training as Test 2, just changed eval strategy
# Apply Patches E and F
USE_EVAL_CACHE=1 \
EVAL_CACHE_MAX_ORDER=7 \
USE_HEDGE_MIXER=1 \
USE_SLIDING_WINDOW=1 \
SLIDING_STRIDE=512 \
\
PROGRESSIVE_SEQ=1 PHASE1_SEQ_LEN=128 PHASE1_LR_MULT=25.0 \
PHASE1_FRACTION=0.85 PHASE2_SEQ_LEN=1024 \
PHASE1_NGRAM_WEIGHT=0.40 PHASE2_NGRAM_WEIGHT=0.05 \
\
NUM_LAYERS=11 MODEL_DIM=512 MLP_EXPANSION=3 \
QK_GAIN=4.0 USE_XSA=1 XSA_LAYERS=4 USE_EMA=1 \
MAX_WALLCLOCK_SECONDS=600 \
TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model" \
DATA_PATH="./data/datasets/fineweb10B_bpe8192" \
.venv/bin/python3 train_gpt.py 2>&1 | tee logs/test3_evalcache.txt
```

**Validation criteria:**
- Same model as Test 2 (just different eval) → val_bpb should be **lower** by 0.05-0.15 BPP
- Eval time increases from ~10s to ~30s (still well under 10 min cap)
- Cache memory usage < 2 GB

**Decision point:**
- If improvement < 0.03 BPP: eval cache implementation might be wrong. Debug.
- If improvement 0.05-0.15 BPP: confirms our biggest gap is closed
- This is the test most likely to surprise us

---

### Test 4: Full stack with Score-First TTT + per-layer GPTQ — 15 min, ~$5
**Goal:** Maximum-effort run with everything stacked.

```bash
# Apply Patches A through G
USE_SCORE_FIRST_TTT=1 \
TTT_LR=1e-4 \
TTT_LAYERS=last2 \
TTT_EPOCHS=1 \
\
USE_MIXED_PRECISION=1 \
QUANT_LAYERS_LOW="0,1,2,3:int7" \
QUANT_LAYERS_MID="4,5,6:int6" \
QUANT_LAYERS_HIGH="7,8,9,10:int5" \
USE_LLOYD_MAX=1 \
\
USE_EVAL_CACHE=1 EVAL_CACHE_MAX_ORDER=7 \
USE_HEDGE_MIXER=1 USE_SLIDING_WINDOW=1 SLIDING_STRIDE=512 \
\
PROGRESSIVE_SEQ=1 PHASE1_SEQ_LEN=128 PHASE1_LR_MULT=25.0 \
PHASE1_FRACTION=0.85 PHASE2_SEQ_LEN=1024 \
PHASE1_NGRAM_WEIGHT=0.40 PHASE2_NGRAM_WEIGHT=0.05 \
\
NUM_LAYERS=11 MODEL_DIM=512 MLP_EXPANSION=3 \
QK_GAIN=4.0 USE_XSA=1 XSA_LAYERS=4 USE_EMA=1 \
MAX_WALLCLOCK_SECONDS=600 \
TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model" \
DATA_PATH="./data/datasets/fineweb10B_bpe8192" \
.venv/bin/python3 train_gpt.py 2>&1 | tee logs/test4_full_stack.txt
```

**Validation criteria:**
- Lower val_bpb than Test 3 by 0.03-0.08 BPP
- Quantized model fits in 16MB after Brotli
- TTT doesn't crash or diverge
- This is our **expected best** single-seed result

---

### Test 5: 3-seed validation on 8xH100 — 45 min, ~$15
**Goal:** Get the variance estimate for final submission. **Only run if Test 4 is good.**

Switch to 8xH100 pod for this:
```bash
runpodctl create pod --name paramgolf-final --gpuType "NVIDIA H100 80GB HBM3" \
  --gpuCount 8 --containerDiskSize 100 --volumeSize 200 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
```

```bash
# Run the EXACT config from Test 4, but 3 seeds
for SEED in 42 314 999; do
    SEED=$SEED \
    [all the env vars from Test 4] \
    torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/test5_seed${SEED}.txt
done

# Compute mean and std
.venv/bin/python3 -c "
import re
scores = []
for seed in [42, 314, 999]:
    with open(f'logs/test5_seed{seed}.txt') as f:
        for line in f:
            m = re.search(r'final_int8_zlib_roundtrip val_loss:[\d.]+ val_bpb:([\d.]+)', line)
            if m:
                scores.append(float(m.group(1)))
                break
print(f'Seeds: {scores}')
print(f'Mean: {sum(scores)/len(scores):.4f}')
import statistics
print(f'Std: {statistics.stdev(scores):.4f}')
"
```

**Validation criteria:**
- All 3 seeds complete successfully
- Std < 0.01 BPP (otherwise we have a stability problem)
- Mean is the submission score

---

## Cost Summary

| Test | GPUs | Time | Cost | Cumulative |
|---|---|---|---|---|
| Test 0 (smoke) | 1xH100 | 5 min | $1 | $1 |
| Test 1 (baseline) | 1xH100 | 15 min | $5 | $6 |
| Test 2 (progressive) | 1xH100 | 15 min | $5 | $11 |
| Test 3 (eval cache) | 1xH100 | 15 min | $5 | $16 |
| Test 4 (full stack) | 1xH100 | 15 min | $5 | $21 |
| Test 5 (3-seed final) | 8xH100 | 45 min | $15 | $36 |
| **Buffer** | — | — | $10 | **$46** |

---

## Decision Tree

```
Test 0 fails → debug locally, don't proceed
Test 0 passes → Test 1
  ↓
Test 1 val_bpb < 1.10 → Test 2
Test 1 val_bpb 1.10-1.20 → still proceed (baseline is meant to be modest)
Test 1 val_bpb > 1.20 → something is broken, debug
  ↓
Test 2 improvement vs Test 1:
  > 0.15 BPP → CELEBRATE, proceed to Test 3
  0.05-0.15 → proceed to Test 3
  0.00-0.05 → progressive seq is weak on H100, still proceed but expect smaller win
  < 0.00 → progressive seq HURTS, fall back to Test 1 config for Test 5
  ↓
Test 3 improvement vs Test 2:
  > 0.10 BPP → eval cache is huge, proceed to Test 4
  0.05-0.10 → confirmed our biggest gap, proceed to Test 4
  < 0.05 → eval cache impl might be broken, debug or skip TTT in Test 4
  ↓
Test 4 improvement vs Test 3:
  > 0.05 BPP → all techniques stacking, proceed to Test 5
  0.00-0.05 → diminishing returns but still good, proceed
  < 0.00 → TTT is destabilizing, drop it for Test 5
  ↓
Test 5 → submit best mean
```

---

## Submission Steps (after Test 5)

1. Take the seed with the median score (not best, not worst)
2. Verify final artifact is < 16 MB
3. Create `records/track_10min_16mb/2026-04-XX_OURNAME/` directory
4. Copy `train_gpt.py`, `submission.json`, `README.md`
5. `submission.json` format:
```json
{
  "val_bpb_mean": 0.XXX,
  "val_bpb_seeds": [s1, s2, s3],
  "val_bpb_std": 0.XXX,
  "training_time_seconds": 600,
  "artifact_size_bytes": XXXXXXX,
  "techniques": [
    "BPE-8192", "WaveletGPT", "NorMuon", "Progressive seq + high LR",
    "Cosine LR Phase 2", "Multi-order n-gram eval cache", "Hedge mixer",
    "Score-First TTT", "Per-layer mixed int7/6/5 GPTQ + Lloyd-Max codebook",
    "n-gram logit bias (bigram+trigram+4gram)", "DC500 categories",
    "DualMLP", "LeakyReLU(0.5)²", "EMA(0.997)", "11L 3x MLP XSA"
  ]
}
```
6. Push branch, open PR
7. Wait for OpenAI to verify on 8xH100s

---

## Things Mac Hasn't Tested (Reference)

These are documented for completeness — most should be tried only if Tests 1-5 leave time/budget.

### Worth trying if Test 4 leaves time
- **Partial RoPE (16/64 dims)** — universal in top H100 entries
- **Higher MLP expansion (3.5x or 4x)** — might compound with int5 quant on late layers
- **Order-adaptive entropy gating** — dynamic n-gram order selection per token
- **Skip-bigram eval cache** — distance-2 patterns
- **2:4 structured sparsity** — Phase 2 only, 2x more steps

### Documented dead ends (don't waste H100 time)
- SVD compression of bigram table (kills predictions, 28% top-1 match at rank 100)
- Vocabulary curriculum (CE loss already does this implicitly)
- Cyclic LR (no resonance found in simulation)
- Reservoir computing layers (too risky for competition timeline)
- Cross-tokenizer prediction (too complex, small expected gain)
- Gradient compression for DDP (irrelevant — comm is < 0.04% overhead)
- DCT transform for weight compression (worse than direct int6+zlib)
- Per-row / block-wise quantization scales (overhead > savings)
- PMI-SVD embedding init (59% worse than random)
- Tabulation hashing (-0.252 at 50 steps but hits a convergence trap at 500+)
- Layer drop in Phase 1 (worse than no drop)
- Noise annealing (catastrophic — 2.7M loss)
- Grokfast (timed out, GPU contention prevented clean test)

---

## Quick Reference: Critical Findings

| Finding | Impact | Source |
|---|---|---|
| Phase 2 is **8x under-resourced** | Eval cache fills 88% of gap | Channel capacity (Cycle 80) |
| Kelly criterion predicts LR ratio **101%** | 33.6x predicted vs 33.3x actual | Cycle 31 |
| Gradients are **91.6° orthogonal** between phases | Zero interference | Cycle 65 |
| Fractal analysis: **monofractal Δh=0.03** | 2-phase optimal, 3-phase useless | Cycle 13 |
| Small-world (diameter=4) | 4 layers propagate, 5-10 refine | Cycle 32 |
| Stochastic resonance: high LR Phase 1 = optimal noise | Why our schedule works | Cycle 14 |
| Weights log-normal σ=1.6 → Lloyd-Max saves **86%** | Free 0.008 BPP | Cycle 19 |
| Wavelet's k = min(2^i, T) **naturally adapts** to progressive seq | No code changes needed for stacking | Cycle 61 |
| Memory bandwidth, not compute, is bottleneck | Why seq=128 is fast | Cycle 35 |
| 99.7% cache hit rate at byte level on real docs | Confirms cache value | Cycle 55 |

---

## Monte Carlo Projection

Based on 10K simulations with 15 techniques + confidence intervals:

| Percentile | BPP | Outcome |
|---|---|---|
| P5 (everything works) | 0.553 | Top 5 |
| P25 (most works) | 0.639 | Top 10-15 |
| **P50 (expected)** | **0.715** | **Top 25-30** |
| P75 (some fail) | 0.793 | Top 50 |
| P95 (worst case) | 0.912 | Still beats SOTA (1.08) |

- **100% chance of beating merged SOTA** (1.08)
- **99% chance of sub-1.0 BPP**
- **79% chance of beating open best** (0.81)

---

## Files Referenced

- `train_gpt.py` — competition's main script (apply patches A-G)
- `eval_ngram_cache.py` — production-ready cache + hedge mixer module
- `progressive_seq_patch.py` — code patches reference document
- `data/bigram_tab_8192v.npy` — precomputed bigram (2.94 MB)
- `data/trigram_logprobs_8192v.npy` — precomputed trigram (0.64 MB)
- `data/dist_cats_500_8192.npz` — DC500 categories (0.18 MB)
- `data/lloyd_max_codebook_64.npy` — Lloyd-Max int6 codebook (256 B)
- `GPU_RESULTS.md` — full overnight research findings (52 sections)
- `SETUP.md` — current Mac stack with what's at default
- `HISTORY.md` — 266 experiments tracked
- `COMPETITION_SCOPE.md` — analysis of 1300 competition PRs
