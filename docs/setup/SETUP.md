# Parameter Golf — Full Stack Setup

## Section 1: Current Mac (MLX) Optimal Stack

**Best training result: 1.6670 BPB** (WaveletGPT 1000 steps + Stage 2 continual lowLR 300 steps, exp #308)
**Best effective: ~1.6546 BPB** (with eval-time α=0.06 + T=0.93 stacked on top)
**Total improvement vs baseline: -0.382 BPB**

### NEW BREAKTHROUGH: Two-Stage Training Works
The key insight: continual training from a converged checkpoint **does work**, but ONLY with significantly reduced LR (4x lower: matrix_lr=0.04 → 0.01). Previous attempts failed because optimizer state reset + full LR pushed the model AWAY from its converged minimum. Stage 2 with low LR keeps the model near the minimum and explores it more carefully, gaining -0.0259 BPB in just 300 additional steps.

| Component | Our Setup | Baseline Default | Changed? | Novel? | Seen in Comp? | Est. BPB Benefit | Experiments Run | Notes |
|-----------|-----------|-----------------|----------|--------|---------------|-----------------|-----------------|-------|
| **ARCHITECTURE** | | | | | | | | |
| Layers | 9 (tested 11) | 9 | Tested | No | Yes (all top use 11) | -0.017 (11L@1k) | ~30 | 11L wins at 1000 steps but not in v2 stack yet |
| Dimensions | 512 | 512 | No | No | Yes | — | ~5 | 768 tested, converges at 50 but untested at scale |
| Attention heads | 8H / 4KV (GQA) | 8H / 4KV | No | No | Yes | — | ~8 | GQA is baseline default |
| U-Net skip connections | Yes (learned weights) | Yes | No | No | Yes (PR #289) | baseline | ~5 | Sigmoid-gated variant exists (PR #1302) |
| Depth recurrence | Not used | No | Tested | No | No | +0.001 (neutral) | ~4 | Relaxed recursive -0.003 at 50 steps |
| **MLP** | | | | | | | | |
| Activation | LeakyReLU(0.5)^2 | ReLU^2 | **Yes** | No | Yes (PR #493) | -0.014 | ~12 | Competition uses slope 0.9+, we use 0.5 |
| Dual MLP | 2 parallel MLPs averaged | Single MLP | **Yes** | **Somewhat** | **No** | -0.067 | ~8 | Not seen in any competition PR |
| MLP expansion | 2x | 2x | No | No | No (comp uses 3x) | — | ~7 | 3x wins at 1000 steps (-0.009) but more params |
| **ATTENTION** | | | | | | | | |
| XSA | Not used | No | Tested | No | Yes (PR #1019) | +0.003 (worse) | ~6 | Doesn't help at Mac scale, universal in H100 entries |
| RoPE config | Standard (full, base=10k) | Standard | Tested | No | Yes | +0.023 (worse) | ~5 | Partial RoPE (16/64) worse at Mac scale |
| QK gain | 1.5 | 1.5 | No | No | Yes (PR #1176: 4.0) | — | ~3 | Higher values untested on Mac |
| Softcap | **20.0** | 30.0 | **Yes** | No | Yes (baseline) | **-0.008** | ~4 | Lower = sharper predictions |
| **EMBEDDING / TOKENIZER** | | | | | | | | |
| Tokenizer | **BPE-8192** | SP-1024 | **Yes** | No | No (comp uses 4096) | **-0.129** | ~45 | SINGLE BIGGEST LEVER. 8192 not seen; 4096 in PRs #1323, #1326 |
| SmearGate | **Yes** (gate prev+curr token) | No | **Yes** | **Somewhat** | Yes (PR #65) | **-0.009** | ~10 | Validated at 1000 steps on 8192v |
| Tied embeddings | Yes | Yes | No | No | Yes | baseline | ~3 | Factorized tested, worse at Mac scale |
| Embed init std | 0.005 | 0.005 | No | No | Yes | — | ~2 | |
| **N-GRAM TABLES** | | | | | | | | |
| Bigram logit bias | **Yes** (w=0.2, 16K buckets) | No | **Yes** | **Yes** | Partially (PR #1302) | **-0.040** | ~32 | Our framing as "logit bias" is novel; n-gram mixing is known |
| Trigram logit bias | **Yes** (w=0.15, 16K hash) | No | **Yes** | **Yes** | Partially | **-0.021** | incl. above | Stacks additively with bigram |
| 4-gram logit bias | **Yes** (w=0.10, 16K hash) | No | **Yes** | **Yes** | Partially | **-0.019** | incl. above | Diminishing returns per order |
| 5-gram / higher | Not used | No | Tested | No | Yes (orders 2-12 in comp) | -0.066 @50 | ~4 | Competition uses up to 12-gram with 4M buckets |
| Hash function | Polynomial (36313/27191/51497) | N/A | **Yes** | **Somewhat** | No (tabulation in comp) | — | ~5 | Tabulation hash -0.252 but breaks retrain |
| **DISTRIBUTIONAL / STATISTICAL BIAS** | | | | | | | | |
| DC500 categories | **Yes** (500 auto-clusters, w=0.15) | No | **Yes** | **Yes** | **No** | **-0.010** | ~24 | Token cluster transition bias — genuinely novel |
| Period/sentence bias | **Yes** (caps after period) | No | **Yes** | **Yes** | **No** | **-0.002** | ~8 | English-specific orthographic signal |
| **LOSS FUNCTION** | | | | | | | | |
| Base loss | Cross-entropy (mean) | Cross-entropy | No | No | Yes | baseline | ~45 | |
| Complementary training | Not used (tested) | No | Tested | No | Yes (PR #803) | Needs val | ~3 | **HIGH PRIORITY** — proven on H100, down-weights easy tokens |
| Focal / reweighting | Not used | No | Tested | No | No | Dead | ~15 | ALL loss reweighting fails except entropy mixing |
| Aux losses | Not used | No | Tested | No | No | Dead | ~10 | ALL aux losses fail except TOP (-0.024) |
| **OPTIMIZER** | | | | | | | | |
| Base optimizer | Muon (matrix) + Adam (embed/scalar) | Muon + Adam | No | No | Yes | baseline | ~37 | |
| NorMuon | **Yes** (per-row normalization) | No | **Yes** | **Yes** | **No** | **-0.132** | ~5 | Per-row norm after Newton-Schulz — not in any PR |
| Turbo-Muon | **Yes** (4 NS steps vs 5) | 5 steps | **Yes** | **Somewhat** | **No** | **-0.026** | ~3 | Free speedup, no quality loss |
| Momentum | 0.95 | 0.95 | No | No | Yes | — | ~5 | |
| Embed LR | 0.05 | 0.05 | Tested | No | Yes | Overshoots | ~8 | Wins at 50, loses at 500 (convergence trap) |
| Matrix LR | 0.04 | 0.04 | No | No | Yes | — | ~3 | |
| Warmdown | 1200 iters | 1200 | No | No | Yes | — | ~4 | |
| **NOVEL TECHNIQUES** | | | | | | | | |
| WaveletGPT | **Yes** (multi-scale causal avg) | No | **Yes** | **YES — PhD-worthy** | **No** | **-0.018** | ~5 | Signal processing between layers. Best single novel technique. |
| **Predictive Coding** | **Yes** (80% error propagation) | **Yes** | **YES — PhD-worthy** | **No** | **-0.040** | ~3 | Neuroscience: only prediction errors between layers. STACKS with WaveletGPT (-0.049). |
| Smooth STE QAT | Tested (not in stack) | No | Tested | **Somewhat** | No | -0.012 | ~3 | Simulated int8 during training |
| Stochastic Weight Perturbation | Tested | No | Tested | **Somewhat** | No | -0.007 | ~2 | Noise proportional to weight magnitude |
| Grokfast | Not tested (timeout) | No | Failed | No | No | Unknown | ~2 | GPU contention prevented clean test |
| **TRAINING EFFICIENCY** | | | | | | | | |
| Batch tokens | 8,192 | 524,288 | **Yes** (Mac) | No | No | — | ~13 | Mac-limited. H100 uses full 524K |
| Sequence length | 1024 | 1024 | No | No | No (comp uses 2048) | — | ~4 | 2048 untested at scale |
| Grad accum steps | 8 | 8 | No | No | Yes | — | ~2 | |
| Token selection | Not used | No | Tested | No | Yes (Rho-1) | Dead on Mac | ~8 | ALL selection schemes worse |
| **COMPRESSION / ARTIFACT** | | | | | | | | |
| Weight quantization | int8 | int8 | No | No | Yes (comp uses int6 GPTQ) | — | ~15 | GPTQ gives better quality per byte |
| Compression format | **Brotli-11** | zlib-9 | **Yes** | No | Yes (PR #1302) | **-1.47MB saved** | ~5 | Best compression ratio for neural weights |
| N-gram table format | int8, 16K buckets, brotli | N/A | **Yes** | **Yes** | No | ~8MB artifact | ~8 | Novel: n-gram tables as part of 16MB artifact |
| Total artifact size | **~12.3MB** (3.7MB spare) | ~4.6MB | **Yes** | **Yes** | No | — | — | Neural + n-gram + DC + bias tables |
| **EVAL TIME** | | | | | | | | |
| Eval strategy | Standard forward pass | Standard | No | No | Yes | — | ~7 | Sliding window, adaptive mix, cache ALL worse |
| N-gram at eval | Fixed additive bias | N/A | **Yes** | **Yes** | Partially | included above | — | Applied as logit bias, not a separate mixer |

---

## Section 2: Projected CUDA (H100) Optimal Stack

**Target: sub-1.00 BPB** | Competition SOTA: 1.1147 merged, 0.93-0.40 pending

| Component | Projected Setup | Changed from Mac? | Novel? | Seen in Comp? | Est. BPB Benefit | Confidence | Notes |
|-----------|----------------|-------------------|--------|---------------|-----------------|------------|-------|
| **ARCHITECTURE** | | | | | | | |
| Layers | **11** | Yes (was 9) | No | Yes (universal) | -0.02 to -0.04 | **High** | ALL top entries use 11L |
| Dimensions | 512 | No | No | Yes | — | High | |
| MLP expansion | **3x** | Yes (was 2x) | No | Yes (universal) | -0.01 to -0.02 | **High** | Top 8/10 entries use 3x |
| XSA | **Yes (4-6 layers)** | Yes | No | Yes (PR #1019) | -0.005 to -0.015 | Medium | Universal in H100, neutral on Mac |
| WaveletGPT | **Yes** | No | **YES** | **No** | -0.01 to -0.03 | **Medium** | Needs H100 validation — might scale differently |
| SmearGate | **Yes** | No | Somewhat | Yes | -0.005 to -0.01 | High | Validated on Mac |
| **OPTIMIZER** | | | | | | | |
| NorMuon + Turbo4 | **Yes** | No | **Yes** | **No** | -0.05 to -0.10 | **High** | Our optimizer innovations are unseen |
| Split LR (early/late) | **Yes** | Yes | No | Yes (PR #1172) | -0.005 to -0.01 | Medium | 1.1015 BPB with this technique |
| Complementary training | **Yes** | Yes | No | Yes (PR #803) | -0.01 to -0.03 | **High** | Proven — down-weight easy tokens, boost n-gram alpha |
| QK gain | **4.0-5.0** | Yes (was 1.5) | No | Yes (PR #1176) | -0.005 to -0.01 | Medium | Higher QK gain helps at scale |
| **TOKENIZER** | | | | | | | |
| Vocab size | **BPE-4096 or 8192** | Maybe | No | 4096 seen | -0.05 to -0.10 | Medium | 4096 sweet spot per PRs #1323, #1326 (less embed cost) |
| **N-GRAM TABLES** | | | | | | | |
| Orders | **2-gram to 8-gram** | Yes (was 2-4) | Somewhat | Yes (2-12 in comp) | -0.02 to -0.05 | **High** | More orders = more signal at higher steps |
| Buckets | **4M hash buckets** | Yes (was 16K) | No | Yes | -0.01 to -0.02 | Medium | More buckets = fewer collisions |
| Mixing strategy | **Entropy-adaptive alpha** | Yes | No | Yes (PR #803) | -0.02 to -0.05 | **High** | Dynamic mixing based on context predictability |
| Complementary weighting | **Yes** | Yes | No | Yes | -0.01 to -0.03 | **High** | Neural focuses on what n-grams can't predict |
| DC categories | **DC500-1000** | No | **Yes** | **No** | -0.005 to -0.01 | Medium | May compound at longer training |
| **TRAINING** | | | | | | | |
| Steps | **7,000+** | Yes (was 1000) | No | Yes | -0.30 to -0.50 | **High** | Scaling gives most of the gap |
| Sequence length | **2048** | Yes (was 1024) | No | Yes (universal) | -0.01 to -0.02 | High | More context = better predictions |
| Batch tokens | **524,288** | Yes (was 8192) | No | Yes | — | High | Full utilization of H100 |
| Token selection (Rho-1) | **Yes** | Yes | No | Yes | -0.02 to -0.04 | Medium | May work at H100 scale (dead on Mac) |
| Late QAT | **Yes (final 4%)** | Yes | Somewhat | Yes | -0.005 to -0.01 | Medium | Quantization-aware finetuning at end |
| **COMPRESSION** | | | | | | | |
| Quantization | **int6 GPTQ** | Yes (was int8) | No | Yes (universal) | -0.02 to -0.04 | **High** | Better quality per byte than int8 |
| Compression | **Brotli-11 / LZMA** | No | No | Yes | — | High | |
| Fisher-weighted bits | **Maybe** | Yes | Somewhat | No | -0.005 to -0.01 | Low | Allocate bits by parameter importance |
| **EVAL TIME** | | | | | | | |
| N-gram mixing | **Backoff mixer (orders 2-8+)** | Yes | Partially | Yes | -0.03 to -0.10 | **High** | Enabled by complementary training |
| Sliding window | **Maybe** | Yes | No | Yes | -0.005 to -0.01 | Low | Needs proper implementation |

---

## Stack Coverage Heatmap

How thoroughly have we explored each area?

```
WELL EXPLORED (20+ exps, clear signal):
  ████████████████████  N-gram tables (32 exps, CORE INNOVATION)
  ████████████████████  Embedding/tokenizer (45 exps, BPE-8192 GAME CHANGER)
  ████████████████████  Loss function (45 exps, MOSTLY DEAD except mixing)
  ████████████████████  Architecture (30 exps, depth > width)
  ████████████████████  Optimizer (37 exps, NorMuon WINNER)

MODERATELY EXPLORED (10-20 exps):
  ████████████████      MLP design (27 exps, DualMLP validated)
  ████████████████      Attention mods (21 exps, mostly dead on Mac)
  ████████████████      DC/statistical bias (24 exps, small wins)
  ████████████████      Regularization (20 exps, ALL DEAD)
  ████████████████      Compression (15 exps, QAT promising)

UNDER-EXPLORED (< 10 exps, potential upside):
  ████████              Novel/cross-domain (15 exps, WAVELET BREAKTHROUGH)
  ████████              Data efficiency (13 exps, dead on Mac but H100?)
  ████████              Eval-time (7 exps, ALL DEAD but maybe wrong approach)
  ████                  Training speed (5 exps, H100-specific)
```

## Key Insight: What's Genuinely Novel (Not Seen in Competition)

| Technique | Status | Paper-worthy? |
|-----------|--------|---------------|
| **WaveletGPT** | Validated, -0.018 BPB | **Yes** — signal processing meets transformers |
| **NorMuon** | Validated, -0.132 BPB | **Yes** — per-row normalization in Muon is novel |
| **Distributional Categories** | Validated, -0.010 BPB | **Somewhat** — auto-discovered token clusters |
| **DualMLP** | Validated, -0.067 BPB | **No** — similar to MoE literature |
| **N-gram logit bias framing** | Core technique | **Somewhat** — novel training-time integration vs eval-time mixing |
| **Period/sentence bias** | Validated, -0.002 BPB | **No** — simple heuristic |
| **Turbo-Muon** | Validated, -0.026 BPB | **No** — minor optimization |

## Total Experiments: 266
## Validated Winners: ~25-30
## Dead Ends: ~180+
## Neutral/Inconclusive: ~55
