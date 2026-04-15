# Revival Candidates — Mac-Failed Techniques to Reconsider Later

> **UPDATE Apr 6 23:48 BNE**: One technique from Tier 3 has been REVIVED: **Continual training from checkpoint** now works at Mac scale with 4x lower LR. **NEW BEST: 1.6670 BPB** (#308). The previous failure was due to optimizer state reset + full LR, not a fundamental flaw. Lesson: when reviving "dead" techniques, always question whether the failure mode was the IMPLEMENTATION or the IDEA.


Of the ~280 failed/neutral experiments across all sessions, this file evaluates which could be **revived under different conditions** (mainly H100 with longer training, PyTorch instead of MLX, or alternative architectures). For each candidate, we record:

- **Original experiments** (HISTORY.md numbers)
- **Why it failed on Mac**
- **Revival conditions** (what would need to change)
- **Revival priority** (1=critical, 5=skip)
- **Estimated benefit** when revived

**Best Mac result**: WaveletGPT 1.6929 BPB + eval-time α=0.06 + T=0.93 = **1.6805 effective BPB**.
**Competition SOTA**: 1.1147 BPB merged, 0.93-0.40 pending. We need to close ~0.55 BPB gap.

---

## TIER 1 — Convergence Speed Trap (HIGH REVIVAL — H100 LIKELY WINS)

These won at 50 steps but lost at 500/1000. The "trap" is that Mac can't train long enough for them to converge. **Every top H100 entry uses many of these.** Mark them all as MUST-RUN at H100 scale.

| Technique | Exp #s | Mac Result | Why It Failed Here | Revival Conditions | Priority | Est. H100 Benefit |
|-----------|--------|-----------|-----|-----|-----|-----|
| **11 layers** | #2, 27, 242, 281 | -0.018 at 1000 (#242 1.7110) but #281 lost when stacked | Already wins on Mac alone, fails when combined with other techniques | H100 + 7000 steps | **1** | **-0.02 to -0.04 BPB** |
| **3x MLP expansion** | #27, 243, 283, 289 | +0.035 at 1000 (#289), -0.009 at 50 with wavelet | Too many params for 1000 steps | H100 + 7000 steps. Universal in top entries. | **1** | **-0.01 to -0.02 BPB** |
| **XSA (exclusive self-attention)** | #225, 236, 245 | +0.003 at 1000 (#245) | Universal in H100 entries. Mac scale doesn't show benefit. | H100 + 4-6 layers + 7000 steps | **1** | **-0.005 to -0.015 BPB** |
| **Complementary training** (PR #803) | #55, 187, 260 | +0.061 at 50, different loss fn at 50 | All loss reweighting fails on Mac. **PROVEN at 0.4416 BPB on H100.** | H100 + entropy-adaptive eval-time alpha | **1** | **-0.01 to -0.03 BPB** |
| **CTW entropy mixing** | #182, 199, 244 | -0.686 at 50 (#182), +0.025 at 500, +0.027 at 1000 (#244) | Best 50-step single tech ever, but overshoots | H100 + 7000+ steps. May converge to optimum. | **1** | **-0.05 to -0.15 BPB** |
| **Logistic mixing** | #181, 247 | -0.511 at 50, +0.015 at 1000 | Same convergence trap as CTW | H100 + 7000+ steps | **2** | **-0.01 to -0.05 BPB** |
| **Adaptive N-gram** (learned per-token weights) | #290, 294 | -0.096 at 50, +0.140 at 1000 | Gate network needs much longer training to learn proper weights | H100 + 7000+ steps. Genuinely novel. | **2** | **-0.02 to -0.05 BPB** |
| **Predictive Coding** between layers | #267, 282 | -0.040 at 50, +0.038 at 1000 | Neuroscience-inspired. Adds layer-projection params. | H100 + 7000+ steps + smaller error gain (0.5 not 0.8) | **2** | **-0.01 to -0.03 BPB** |
| **Higher QK gain (4.0)** | #56, 272 | +0.039 at 50 | PR #1176: 1.1015 BPB with this | H100 + 7000+ steps. Documented in top entry. | **2** | **-0.005 to -0.015 BPB** |
| **Higher matrix LR (0.06+)** | #189 | -0.038 at 50 (overshoots at 500) | Standard convergence trap | H100 + warmdown decay + 7000 steps | **2** | **-0.01 to -0.02 BPB** |
| **Cosine LR schedule** | #52, 54 | -0.107 at 100, worse at 500 | Wins early, loses late at Mac. Standard at H100. | H100 + 7000 steps | **2** | **-0.005 to -0.01 BPB** |
| **Warmdown=200 (proper LR decay)** | #36, 204, 304 | +0.056 at 1000 (#304) | "Bug" of warmdown=1200 is beneficial at Mac. Standard at H100. | H100 with iterations matching warmdown | **2** | **-0.005 to -0.01 BPB** |
| **embed_lr=0.07-0.1** | #117, 119, 120, 123 | -0.220 at 50 (#117), worse at 500 | Aggressive embed LR overshoots at Mac scale | H100 + warmdown | **3** | **-0.005 BPB** |
| **WD=0.04-0.08** | #11, 205 | Tied or modest at 50 | Top entries use 0.04-0.08 | H100 + longer training | **2** | **-0.005 to -0.01 BPB** |
| **Truncated backprop / freeze early layers** | #150 | +0.161 at 50 | Early layers still learning at 50 steps | H100 + freeze AFTER step 3500 | **3** | **-0.005 BPB** |
| **DC1000 (more categories)** | #135, 145, 146 | -0.222 at 50, +0.013 at 500 | Too many categories overshoot at Mac scale | H100 + 7000 steps | **3** | **-0.005 to -0.015 BPB** |
| **5-gram / higher order n-grams** | #147 | -0.066 at 50 only | Diminishing returns at Mac scale. Top entries use orders 2-12. | H100 + 4M hash buckets | **2** | **-0.01 to -0.04 BPB** |
| **7-gram Count-Min Sketch** | #234 | -0.026 at 50 | Same higher-order story | H100 | **3** | **-0.005 to -0.015 BPB** |
| **Skip-bigram (prev2→next)** | #177, 235 | -0.063 at 50, neutral at 500 (1.8360) | Static n-gram redundancy with our existing tables | H100 + complementary training context | **3** | **-0.005 BPB** |
| **TOP loss (Token Order Prediction)** | #179 | -0.024 at 50, never validated | Aux loss with positive signal | H100 + 7000 steps | **3** | **-0.005 to -0.015 BPB** |
| **Hyper-connections** (DeepSeek learned residual) | #178 | -0.015 at 50, never validated | DeepSeek paper validates at scale | H100 + 7000 steps | **3** | **-0.005 to -0.015 BPB** |
| **Multi-token prediction** (t+2, t+3 aux heads) | #163 | +3.488 at 50 (overwhelms loss) | Modern LM technique. Standard since 2024. | H100 + tiny aux weight (0.05) + 7000 steps | **3** | **-0.01 to -0.02 BPB** |
| **Stacking 11L + WaveletGPT + SmearGate** | #281 | +0.030 at 1000 (1.7230) | Too many params at Mac. The "right" combination of our wins. | H100 + 7000 steps | **1** | **-0.03 to -0.05 BPB** |
| **WaveletGPT + 3xMLP** | #289 | +0.035 at 1000 (1.7274) | Same overhead trap | H100 + 7000 steps | **2** | **-0.01 BPB on top** |
| **Hurst-adaptive LR** | #276 | Neutral at 50 | Window=32 doesn't kick in by step 50 | H100 + 7000 steps + larger window | **4** | **-0.005 BPB if it works** |
| **Phase Coupling** (learned dim rotations) | #285, 291 | -0.006 at 50 (small) | Coupled oscillators, novel | H100 + 7000 steps | **4** | **-0.005 to -0.015 BPB** |
| **Causal Conv between layers** | #268 | -0.018 at 50 | Similar mechanism to WaveletGPT | H100 to test | **4** | **-0.005 BPB if not redundant** |
| **Differential Hidden States** | #270 | -0.012 at 50 | Layer-to-layer finite difference | H100 to test | **4** | **-0.005 BPB if not redundant** |
| **Neural Phase Coupling** | #285 | -0.004 at 50 with wavelet | Tiny signal, may compound | H100 + 7000 steps | **4** | **-0.005 BPB** |

**Tier 1 total estimated H100 benefit if all successfully revived: -0.10 to -0.30 BPB**

---

## TIER 2 — MLX/Framework Limitations (MEDIUM REVIVAL — PYTORCH WINS)

These failed because MLX's compiled graph doesn't support certain operations (random ops, dynamic shapes, custom attention masks). They would likely work in PyTorch on H100.

| Technique | Exp #s | Why It Failed | Revival Conditions | Priority |
|-----------|--------|---|---|---|
| **Random dropout / feature dropout** | #118, 212 | MLX compile fails on `mx.random` in compiled functions | PyTorch native dropout | **2** |
| **Anti-teacher forcing** (10% own predictions) | #169 | Random masking compile failure | PyTorch | **3** |
| **AR-diffusion noise injection** | #174 | Random noise compile failure | PyTorch | **4** |
| **Sigma-delta quantization noise** | #198 | Compile failure on noisy weights | PyTorch | **4** |
| **Learnable attention temperature** | #251 | `mx.fast.scaled_dot_product_attention` doesn't support dynamic scale | PyTorch SDPA or manual attention | **3** |
| **Causal Sinkhorn attention** | (researched, not built) | Incompatible with FlashAttention | Manual attention in PyTorch | **3** |
| **Conv+attention hybrid (Hymba)** | #176, 231 | `_use_conv` attribute not accessible in compiled function | PyTorch | **3** |
| **Embedding EMA (Exponential Moving Input)** | #254 | Shape issue with decay computation | PyTorch | **4** |
| **Layerwise LR scaling** | #121 | Regex parameter matching too slow in hot loop | PyTorch optimizer param groups | **3** |
| **Synaptic scaling** | #87 | Normalization compile issues | PyTorch | **4** |
| **Entropy-gated skip connections** | #90 | Dynamic gating compile issues | PyTorch | **4** |
| **Z: Simplified complementary loss** | #89 | `mx.no_grad` compile failure | PyTorch | **4** |
| **Embedding dropout** | #118 | Random ops in compiled fail | PyTorch native | **3** |

**Tier 2 strategy**: When porting to H100 in PyTorch, batch-test all of these in one PR sweep. Cheap to try.

---

## TIER 3 — Need Careful Re-Implementation (MEDIUM REVIVAL)

These failed due to implementation issues, not fundamental flaws. Worth careful retesting.

| Technique | Exp #s | Why It Failed | Fix Required | Priority |
|-----------|--------|---|---|---|
| **EMA weight averaging** | #8, 239, 305 | #8 had bugs (memory leak), #239 zero effect at 500, #305 +0.45 catastrophic | Buffer state preservation, correct shadow weight handling, decay tuning. Top entries use this. | **1** |
| **Sliding window eval** | #22, 200, 240 | First impl was wrong (mean loss not per-token), fixed impl WORSE | Model needs to be TRAINED with sliding context, not just eval'd | **3** |
| **Dynamic n-gram cache at eval** | #22, 303 | Hurts model trained with bias (double-counting) | Apply only to model trained WITHOUT n-gram bias | **3** |
| **Sequence packing / document boundaries** | (zero exps) | Never tested | Add BOS-aware attention masking, reset position embedding at BOS, custom loss masking | **2** |
| **Information bottleneck (per-dim gating)** | #279 | Gates init at 1.0, never close at 50 steps | Init gates lower, longer training | **4** |
| **MDL weight decay** | #271, 191 | Too aggressive at 50 steps (#271 +0.221) or too mild (#191 tied) | Re-implement with annealed schedule | **4** |
| **Compression-aware training** (joint MDL) | (built, not run) | Quantization penalty per step too expensive | Compute periodically, not every step | **3** |
| **Multi-resolution tokenization** (char + BPE) | (researched, not built) | Needs custom data loader | Major refactor, but truly novel | **4** |
| **DFA/FSM-augmented attention** | (researched, not built) | Needs differentiable quantization | Complex but novel | **4** |
| **Compression bottleneck (512→64→512)** | #168 | Validated -0.004 at 50, never scaled | Test at 1000+ to confirm | **3** |
| **Micro-macro split MLP** | #159 | Validated -0.004 at 50, never scaled | Test at 1000+ to confirm | **3** |
| **Dendritic MLP** (4 groups, varied slopes) | #224 | -0.004 at 50, never scaled | Test at 1000+ | **3** |
| **Relaxed Recursive layers** | #230 | -0.003 at 50, never scaled | Test at 1000+ | **3** |
| **Attention output position gating** | (built as #attn_reweight, not run) | Wasn't tested at 1000 | Quick test on H100 | **4** |

---

## TIER 4 — Need Different Conditions (LOW-MEDIUM REVIVAL)

These need fundamentally different training setups (different tokenizer, compression pipeline, or training data).

| Technique | Exp #s | Required Conditions | Priority |
|-----------|--------|---|---|
| **SP-4096 tokenizer** | (zero exps) | Build tokenizer + retrain n-gram tables. PR #1326 used it for **1.0896 BPB**. | **2** |
| **int6 GPTQ compression** | (zero on Mac) | Implement GPTQ algorithm. Could free 2-4MB of artifact for more params. | **2** |
| **Mixed-precision per-layer** | (analysis only) | Layer sensitivity scoring + bit allocation knapsack | **2** |
| **Fisher-weighted bit allocation** | (zero) | Track gradient² as Fisher proxy during training | **3** |
| **Ternary STE / BitNet b1.58** | #277 | +0.174 at 50. Needs warmup phase + much longer training. **5x compression** = 80M params in 16MB! | **2** |
| **Factorized embedding** (8192×64×512) | #71, 237 | +0.348 at 50 (slower convergence). H100 + 7000 steps. Saves 2.3MB. | **3** |
| **DEQ (deep equilibrium)** | (zero) | PR #1323: 1.1247 BPB in 6.8MB! Needs iterative solver. | **2** |
| **MoE with proper routing** | (rejected) | At 16MB capacity splitting hurts. Maybe 2-expert with gating works. | **4** |
| **Knowledge distillation** | (no teacher) | Need to train large teacher first. Multi-day project. | **4** |
| **Reservoir layers** (frozen from seed) | #275 | Failed on U-Net architecture (residuals bypass). May work without U-Net. **0 bytes for frozen layers!** | **3** |
| **Lloyd-Max quantization** | #148 | Better quality but 2x size. Could work with sparser activations. | **5** |

---

## TIER 5 — Genuinely Dead (DON'T REVIVE)

These have fundamental flaws that won't be fixed by better hardware or implementation. Listed for reference.

### Loss reweighting (always fails)
- Focal loss / focused loss / Gaussian / hard-context / byte-weighted / Zipf-inverse / Gradient routing / importance sampling / prospect theory / dynamic token selection
- **Why dead**: CE mean is already optimal. Reweighting either down-weights training signal (slower) or up-weights noise (worse).
- Exps: #9, #10, #43, #55, #74, #78, #94, #110, #160, #186, #187, #188, #220, #222, #229, #232, #263, #280, #286

### Most aux losses (always destabilize)
- Self-referential, deep supervision, multi-token, JEPA, multi-scale loss, difficulty-aware aux, byte aux, POS aux
- **Why dead**: Aux losses fight the main CE objective. Only TOP (#179, +signal) and CPC variants survive.
- Exps: #80, #85, #98, #107, #155, #163, #164, #195, #252, #288

### Geometric/analytical inits
- SVD embedding, spectral embedding, reservoir computing, Markov eigenvectors
- **Why dead**: Tied embedding architecture rejects all closed-form initializations. Model needs to learn freely.
- Exps: #44, #47, #49

### Pure classical replacements
- Pure PPM, pure Mamba, pure conv, MoR/depth recursion
- **Why dead**: Classical methods plateau ~1.0 nat above neural. They augment, not replace.
- Exps: #45, #46, #48, #171

### Sigma-delta / Floyd-Steinberg quantization
- Spatial dithering for weights
- **Why dead**: Weights lack the spatial correlation that makes spatial dithering work.

### Other confirmed dead
- Untied embeddings (#162, +2.029) — loses shared structure
- Backward model (#172) — no signal from sequence reversal
- Hash codebook (#157) — zero-init no-op
- Negative prediction head (#165) — zero-init no-op
- Polar token routing (#194) — marginal noise
- Bits-back loss (#106) — formula broken
- Gradient thermal diffusion (#92, #95) — smoothing has zero effect
- Lateral inhibition (#111) — marginal
- Surprise embedding (#112) — marginal
- Prediction recycling (#91) — neutral
- Confidence-aware key scaling (#113) — noise
- Soft layer interpolation (#158) — weakens layer output
- Context-dependent softcap (#86) — tied
- Energy corrector head (#190) — zero-init
- Hopfield memory zero-init (#193) — needs proper init
- Lookahead optimizer + Muon (#265) — slow weights hurt convergence
- Langevin gradient noise (#266) — kills convergence with Muon LR
- Importance sampling (#229) — same loss reweighting failure
- Gradient centralization standalone (#75, #262) — marginal
- Smooth layer interpolation (#158) — weakens output
- Word-boundary embedding (#103) — tied
- Procedural feature interaction (#151) — rank-1 = no signal
- Running document LayerNorm (#152) — destabilizes
- POS aux head (#155) — POS too crude (84% OTHER)
- Anti-entropy loss (#79, #83) — modifies metric, unfair

---

## Revival Roadmap

### Phase 1: Port to H100 (PyTorch, week 1)
1. **MUST**: Port WaveletGPT, n-gram bias stack, DC, period bias to PyTorch
2. **MUST**: 11L + 3xMLP + XSA + WaveletGPT + n-gram stack (Tier 1 priority 1)
3. **MUST**: Complementary training + entropy-adaptive eval-time alpha
4. **MUST**: EMA weight averaging (proper buffer handling)
5. **TEST**: All Tier 2 framework-blocked techniques in one sweep

### Phase 2: Compression pipeline (week 2)
6. **int6 GPTQ** with calibration data
7. **Fisher-weighted mixed precision** per layer
8. **Try BitNet ternary** (5x density potential)
9. **DEQ** if it fits in 6.8MB (PR #1323 baseline)

### Phase 3: Tokenizer change (week 3)
10. **SP-4096 tokenizer** (rebuild n-gram tables, retrain)
11. Compare with BPE-8192 + our techniques

### Phase 4: Long-tail (if time permits)
12. Sequence packing with BOS-aware masking
13. Multi-resolution tokenization (char + BPE)
14. DFA-augmented attention heads

### Estimated H100 best with full revival:
```
Mac best (effective):       1.6805 BPB
+ 11L + 3xMLP + 7000 steps: ~1.10  (-0.58)
+ XSA + complementary:       ~1.05  (-0.05)
+ Higher orders n-gram:      ~1.02  (-0.03)
+ EMA + warmdown:            ~1.00  (-0.02)
+ int6 GPTQ + BitNet:        ~0.98  (-0.02)
+ Eval-time α + T:           ~0.97  (-0.01)
TARGET (sub-1.00):           ~0.97 BPB ← Beats merged SOTA, competitive with pending
```

## Summary by Priority

**Priority 1 (must revive on H100)**: 11L, 3x MLP, XSA, complementary training, CTW mixing, EMA, full stack combinations
**Priority 2 (likely worth it)**: Higher LR/QK gain, all higher-order n-grams, SP-4096 tokenizer, int6 GPTQ, BitNet ternary, MDL/sequence packing
**Priority 3 (worth a quick test)**: Predictive coding, factorized embedding, KAN-style activations, dynamic n-gram cache
**Priority 4 (low expected value)**: Phase coupling, Hurst LR, multi-resolution tokenization, DFA attention
**Priority 5 (skip)**: All loss reweighting, all aux losses, geometric inits, classical replacements

## Notes on Methodology

The Mac scale (1000 steps) systematically penalizes techniques that:
1. **Add parameters** (need more steps to converge them)
2. **Slow per-step convergence** (LR schedules, weight averaging, regularization)
3. **Add auxiliary objectives** (compete with main loss for capacity)
4. **Require longer context** (sequence length, doc boundaries)

The Mac scale REWARDS techniques that:
1. **Provide direct inductive bias** (n-gram tables, period bias, DC categories)
2. **Modify forward pass cheaply** (WaveletGPT)
3. **Improve optimizer** (NorMuon, Turbo-Muon)
4. **Sharpen the existing distribution** (low softcap, eval-time temperature)

This explains why our Mac best (1.6929) consists entirely of techniques in the second category. H100 (7000 steps) reverses these biases — convergence-trap techniques become winners.
