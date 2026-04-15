# STACK_NOVELTY_TRACKER_v2.md — canonical L01→L11 inventory

**Date**: 2026-04-16  ·  **Authors**: taka + claude  ·  **Supersedes**: `docs/results/STACK_NOVELTY_TRACKER.md` (older rolling state)

**Purpose**: single source of truth for every technique, direction, idea, or named novelty we've ever considered across the 11-layer stack. Includes shipped wins, prototyped-but-dropped, idea-stage, killed / negative, and open research directions. Covers **quality** (BPB reduction) *and* **speed / infra** (kernel, throughput, cost). Built by mining all prior docs + records + leaderboard + research logs; extended with the §10 novelty-mining protocol of `MOONSHOT_RULES.md`.

**Status legend**:
- 🟢 **shipped** — in current 1.082 submission
- 🔵 **leaderboard-elsewhere** — shipped in a comp PR but not in our stack
- 🟡 **prototyped** — built and tested, not in submission (either below threshold or failed composition)
- ⚪ **idea** — documented, not prototyped
- ❌ **killed** — validated as negative result
- 🎯 **open novelty** — no prior art we can find; moonshot candidate

**Novelty class**:
- **WN** — world-novel (no comp PR, no arxiv prior art we can find)
- **CN** — comp-novel (known in literature, not in any comp PR)
- **CP** — comp-port (shipped in a comp PR, portable to our stack)
- **IN** — infrastructure / speed / no quality impact
- **KR** — known + ruled (rules-forbidden)

## TL;DR

**110+ named directions** across 11 layers. Our 1.082 stack ships 24 of them. The comp leaderboard shipped another 18 we haven't ported. We have 13 prototyped-but-dropped ideas (good candidates for re-ship post-gated-attn cleanup). We have 40+ idea-stage entries that have never been touched, concentrated at L03 (architectures), L06 (scoring rules), L07 (quantization), L09 (n-gram engine), and L11 (infra). We explicitly killed ~15 ideas (don't re-propose).

**The most actionable gaps**:
- **L10 × online adapt (moonshot)**: online n-gram / CTW cache — `MOONSHOT_RULES.md` confirms legal, cmix hit 0.9 BPB on enwik9, no comp PR has shipped it
- **L07 × novel quant**: rANS neural prior, tensor-train cores, VQ residual codebooks — all idea-stage, all world-novel-candidate
- **L01 × Huffman-weighted tokenizer**: P12 found 97 near-dup vocab rows; the tokenizer is a high-EV novelty surface
- **L06 × arithmetic-coding loss**: train the model to minimize arithmetic-coding bit rate directly, not cross-entropy
- **L11 × custom megakernels**: fused QKV+norm, fused LM-head+softmax+arith-coder; the comp's frontier is kernel-level

The tracker below enumerates everything, with links to source docs for every entry.

---

## L01 — Tokenizer

**Current** (🟢 in 1.082): SentencePiece BPE, vocab 8192, bytes/token ≈ 3.73 on FineWeb val. Tokenizer file ~1.4 MB counts against 16 MB cap.

**Slack measured** (from `STACK_UTILISATION_RESULTS.md`):
- 138 vocab rows (1.7%) never fire on val → ~70 KB reclaimable
- 806 rows (9.8%) fire <1/8 of uniform → ~410 KB soft slack
- 97 pairs (1.2%) at cos-sim >0.9 → ~50 KB row-tying opportunity
- Token-frequency entropy 10.4 bits (vs uniform 13 bits) → 80% of max — the distribution is rich but skewed

| Name | Status | Class | Expected | Source | Note |
|---|---|---|---|---|---|
| SP1024 BPE (fineweb_1024_bpe) | 🔵 baseline | CP | baseline | `README.md`, `data/tokenizers/fineweb_1024_bpe.model` | Naive comp baseline |
| **SP8192 BPE** | 🟢 shipped | CP | −0.129 BPB vs SP1024 (34.7% fewer tokens) | `RESEARCH.md`, `PLAN.md`, comp PRs #1423-#1485 | 35.99M params, fewer token calls, larger embedding table cost |
| Scylla tokenizer (prune to 998 vocab) | ⚪ idea | WN | −0.07 to −0.13 BPB est | `RESEARCH.md` L.186-204 | Prune TokenMonster vocab, maximum token efficiency at minimum embedding cost |
| H-Net byte-level (260 vocab) | 🟡 prototyped | CN | 1.207 BPB (4-hr track) | comp PR #1305 non-record, `RESEARCH.md §novel-archi` | Hierarchical byte-level encoding; explicit item on OpenAI "requests for PRs" list |
| Kraft-inequality merge validator | ⚪ idea | WN | unknown | `STACK_NOVELTY_TRACKER.md §C` | Use Kraft inequality to prune invalid merges during BPE training (cross-domain from info theory) |
| Entropy-aware BPE merge (ML vocab gradient) | 🟡 prototyped | CN | −0.003 to −0.008 BPB | `PLAN.md §1.7f` | Rerank BPE merge candidates by impact on cross-entropy, not frequency |
| **Huffman-weighted token IDs** | 🎯 open | WN | −0.01 to −0.02 BPB | `STACK_UTILISATION_RESULTS.md §12.5`, `MOONSHOT_RULES.md §2.3` | Variable-length token IDs: common tokens get short codes, rare get long. Informed by P12 (97 near-dup rows) + P7 (skewed freq) |
| Tied near-duplicate rows (97 pairs, cos-sim >0.9) | ⚪ idea | WN | 0.005-0.01 BPB + 50 KB reclaim | `STACK_UTILISATION_RESULTS.md §11.6` | Row-tie pairs like tokens 29↔74 (cos-sim 0.9452) via shared row + per-variant delta |
| Collapse 138 unseen vocab rows | ⚪ idea | CN | 0.001 BPB + 70 KB reclaim | `STACK_UTILISATION_RESULTS.md §4.5` | Demote never-fired rows to byte-fallback embedding, reclaim slots |
| Arithmetic-coding-aware tokenizer | 🎯 open | WN | −0.01 to −0.05 BPB | protocol §10.2 L01 × coding theory | Tokens whose code length approximates −log₂ P(token) under unigram — directly pays the BPB bill |
| BWT-preprocessed tokenizer (DNA-compression port) | 🎯 open | WN | unknown | protocol §10.2 L01 × DNA compression | Burrows-Wheeler transform of training text before BPE; improves compressibility on repetitive sequences |
| Elias-gamma variable-length codes | 🎯 open | WN | −0.005 to −0.01 BPB | protocol §10.2 L01 × universal codes | Encode token IDs with Elias gamma or Levenshtein codes instead of fixed-width |

---

## L02 — Data loader / sampling

**Current** (🟢 in 1.082): Coprime-stride loader + MDL-compressible-first anti-curriculum.

**Slack measured**: not directly probed. Compute is not a binding constraint given the H100 budget.

| Name | Status | Class | Expected | Source | Note |
|---|---|---|---|---|---|
| **Coprime-stride shard loader** | 🟢 shipped | WN | −0.0037 BPB (S2 n=2) | `STACK_NOVELTY_TRACKER.md §A`, comp PRs #1351, #1184 | Load shards in coprime order for better curriculum diversity |
| **MDL-compressible-first anti-curriculum** | 🟢 shipped | WN | −0.0037 BPB | `PLAN.md`, confirmed-win n=2 | Load harder tokens first (reverse MDL) |
| Heterogeneity-loss weighted sampling | ⚪ idea | CN | −0.01 to −0.02 BPB | `STACK_NOVELTY_TRACKER.md §C` (DAT_heterogeneity_loss_weight) | Weight examples by disagreement with n-gram baseline |
| Active-learning difficulty sampler | 🎯 open | CN | −0.005 to −0.02 BPB | protocol §10.2 L02 × active learning | Sample batches enriched for tokens the model currently gets wrong |
| Importance-weighted ELBO training | 🎯 open | CN | −0.005 to −0.01 BPB | protocol §10.2 L02 × sampling theory | Monte-Carlo importance weights per batch to reduce gradient variance |
| Curriculum by per-token entropy | 🎯 open | WN | −0.005 to −0.015 BPB | protocol §10.2 L02 × CL/SPL | Start on low-entropy "easy" byte contexts, progressively admit high-entropy ones |

---

## L03 — Model architecture

**Current** (🟢 in 1.082): 11-layer transformer, dim=512, 8 heads (4 KV), MLP 4×, RoPE dims=16, gated attention, depth recurrence L3–5 ×2, parallel residuals from L7, XSA on all 11, tied embeddings. 35.99M params.

**Slack measured** (from probes):
- 0 dead heads, 0 dead MLP neurons → no prune slack
- CKA confirms depth recurrence is load-bearing (L2↔L3, L5↔L6 CKA = 0.08, 0.13)
- **Gated attention actively HURTS by 1% val NLL (P15)** — ablation test confirms
- P13 CKA shows 3 structural phases (L0–L2, L3–L5, L6+) — any prune must respect boundaries

| Name | Status | Class | Expected | Source | Note |
|---|---|---|---|---|---|
| **Gated attention (per-head sigmoid)** | 🟢 ABLATE | — | **+0.010 BPB if removed** | `STACK_UTILISATION_RESULTS.md §11.5 P15` | **Actionable — retrain with `SKIP_GATES_ENABLED=0`** |
| **LeakyReLU(0.5)² MLP activation** | 🟢 shipped | WN | −0.003 BPB | `PLAN.md`, comp SOTA #1019 | Square of leaky ReLU with 0.5 negative slope; novel composition |
| **Depth recurrence (3-layer loop L3-5 ×2)** | 🟢 shipped | CP | −0.005 to −0.01 BPB | comp PRs #1437, #1485 | LESSONS.md earlier declared dead but modern quant works; CKA confirms load-bearing |
| **Parallel residuals (GPT-J style, L7+)** | 🟢 shipped | CP | marginal | comp PRs #1437, #1420, #1485 | Parallel attention+MLP branches per block with learned lane_merge |
| XSA (cross-sliding-attention) on all 11 | 🟢 shipped | CP | −0.003 BPB | comp PR #198 | Efficient partial attention, last-N layers |
| Tied embeddings | 🟢 shipped | CP | budget saver | standard modded-nanogpt | No separate `lm_head`; shares with `tok_emb` |
| RoPE partial (rope_dims=16) | 🟢 shipped | CP | −0.002 to −0.005 BPB | comp PR #287 | Partial RoPE on 16/64 head dims |
| QK-Gain per-head scalar (init=4.0) | 🟢 shipped | CP | −0.001 BPB per 0.5 gain step | comp SOTA #1019 | Frontier moved to 5.0-5.25; we still at 4.0 — **actionable bump to 5.25** |
| Hymba hybrid (parallel Mamba+Attention) | 🟡 prototyped | CN | −0.004 to −0.03 BPB | comp PR #852 (1.1189 BPB) | Sigmoid-mixed parallel Mamba + attention; 0.004 from record, never shipped |
| **DEQ (Deep Equilibrium) with Scylla** | 🟡 prototyped | CN | **1.1247 BPB @ 6.8 MB model** | comp PR #1323 | **9.2 MB left for bias tables = massive slack** — re-investigate |
| Universal Transformer + ACT | 🟡 prototyped | CN | 1.2409 BPB | comp PR #1293 | Adaptive computation time; explicit OpenAI wishlist item |
| JEPA (Joint Embedding Predictive Architecture) | 🟡 prototyped | CN | 1.3299 BPB @ 1×H100 | comp PR #1312 | Multi-horizon latent prediction; OpenAI wishlist item |
| BankLinear (cross-layer shared weight bank) | 🟡 prototyped | CN | 1.227 BPB (worse) | comp PR #1315 | Layers share 128 pre-trained linear maps, selected per-position |
| HyperGPT (generative weight synthesis) | ❌ killed | — | 8.218 BPB | comp PR #1288 | Catastrophic failure; hypernetwork can't generate good weights |
| **Wavelet GPT (multi-scale embedding mixing)** | 🟢 shipped | WN | −0.018 BPB at 500 steps | `PLAN.md §2.15`, `RESULTS.md` | Wavelet decomposition of token embeddings, soft mixture of scales per position |
| Dendritic MLP (block-diagonal) | 🟢 shipped | CN | −0.004 BPB | `PLAN.md §2B-12`, Nature 2025 | Block-diagonal FFN instead of dense, sparse interaction patterns |
| Skip-weights + skip-gates (UNet-style) | 🟢 shipped | CP | −0.001 to −0.003 BPB | records | rank-2 structure found in P6; already low-rank efficient |
| Text diffusion | ⚪ idea | CN | unknown | OpenAI "requests for PRs" | Diffusion over token sequence; listed as wanted, no comp PR yet |
| H-Net architecture (hierarchical byte-level) | 🔵 (part of L01) | CN | 1.207 BPB (non-record) | PR #1305 | Architecture+tokenizer combined |
| Megakernel (fused full transformer block) | ⚪ idea | WN | speed: 10-30% faster | OpenAI wishlist, comp PR #1420 | Fuse entire block (attn+norm+MLP+residual) into one CUDA kernel |
| State-space models (Mamba-2, S4, S5) | ⚪ idea | CN | unknown | OpenAI wishlist | Pure SSM or SSM-attention hybrid |
| E2E TTT (architecture aware of its test-time updates) | ⚪ idea | WN | unknown | OpenAI wishlist, protocol §10.2 L03 × meta-learning | Architecture designed with test-time-training as first-class |
| Long-context eval (16K+, full-val context) | ⚪ idea | CN | unknown | OpenAI wishlist | Eval at seq_len far beyond 2048 |
| Random-projection adapters | ⚪ idea | CN | unknown | OpenAI wishlist | Learning adapters on random linear maps |
| Norm-PCT-Dropout (top 1% FFN zeroing) | 🟢 shipped | WN | −0.00025 BPB (key for composition) | `STACK_NOVELTY_TRACKER.md §A` | Zero the top 1% highest L2-norm FFN neurons during training |
| Depth-Separated MLP (DS-MLP) | 🎯 open | WN | −0.005 to −0.01 BPB | protocol §10.2 L03 × neuromorphic | Per-head independent MLP (like DWS conv) instead of full MLP — reduce params, increase depth |
| Equilibrium attention (fixed-point iteration) | 🎯 open | WN | unknown | protocol §10.2 L03 × equilibrium models | Attention as fixed point of a contraction map — 1-layer with many iterations |
| Optical-inspired (structured unitary matmul) | 🎯 open | WN | speed: 2-3×, quality neutral | protocol §10.2 L03 × optical | Butterfly / Hadamard-structured matmul blocks |

---

## L04 — Optimizer

**Current** (🟢 in 1.082): Muon + NorMuon + ParMuon + AdamW for scalars.

**Slack measured**: P11 cross-seed stability shows most trained params converge cleanly (scales have rel_std 0.06–0.17); optimizer is working fine. Slack is in the gated attention (see L03).

| Name | Status | Class | Expected | Source | Note |
|---|---|---|---|---|---|
| **Muon (matrix-wise update)** | 🟢 shipped | CP | −0.005 BPB vs AdamW | modded-nanogpt | Standard frontier optimizer |
| **NorMuon (post-NS row normalization)** | 🟢 shipped | WN | −0.00375 BPB (S2 n=2) | comp PR #1493 | Row normalize after Newton-Schulz, preserves gradient structure |
| **ParMuon (parallel / batched NS)** | 🟢 shipped | WN | +3 free training steps / 3% throughput | `SUBMISSION_RUN_STATE.md` | Batch Newton-Schulz per-shape, not sequential per-param |
| Chebyshev-optimized Newton-Schulz | ❌ killed | CN | demoted (arXiv:2506.10935 prior art) | `STACK_NOVELTY_TRACKER.md §A` | Use Chebyshev polynomials to accelerate NS; not world-novel |
| Riemannian Gram projection (QKV on Stiefel) | ❌ killed | CN | +0.0024 worse | `STACK_NOVELTY_TRACKER.md §A` | Project Q/K/V onto Stiefel manifold; breaks at byte scale |
| Schedule-free momentum | ⚪ idea | CN | −0.003 to −0.008 BPB | `PLAN.md §2B-TIER2` | Remove LR schedule; derive momentum from gradient magnitude |
| Langevin dynamics (physics-inspired noise injection) | 🎯 open | WN | unknown | protocol §10.2 L04 × physics | Add annealed gradient noise → better exploration of the loss surface for tiny models |
| Hamiltonian Monte Carlo step | 🎯 open | WN | unknown | protocol §10.2 L04 × physics | Treat weights as state in a Hamiltonian; integrate with leapfrog |
| CMA-ES for rare-token param subset | 🎯 open | WN | −0.01 to −0.02 BPB | protocol §10.2 L04 × evolutionary | Evolutionary strategies for the ~100 params most correlated with rare-token loss (P7) |
| MAML-style meta-learning for TTT | 🎯 open | WN | TTT efficiency +2× | protocol §10.2 L04 × meta-learning | Meta-train for fast adaptation; uses 2× fewer TTT epochs for same BPB |
| K-FAC (natural gradient with Kronecker) | 🎯 open | CN | −0.003 to −0.01 BPB | protocol §10.2 L04 × second-order | Approximate natural gradient per-layer with Kronecker-factored curvature |
| Shampoo (full-matrix preconditioner) | ⚪ idea | CN | −0.003 to −0.01 BPB | protocol §10.2 L04 × second-order | Comp hasn't shipped this yet; marginal over Muon |

---

## L05 — Training loop / regularization

**Current** (🟢 in 1.082): EMA decay=0.997, warmdown_frac=0.667, warmup=20, QAT with matrix_bits=6, embed_bits=8, GPTQ calibration at end.

| Name | Status | Class | Expected | Source | Note |
|---|---|---|---|---|---|
| EMA weight averaging (decay=0.997) | 🟢 shipped | CP | −0.002 BPB | comp SOTA #1394-#1485 | Standard; frontier uses 0.9965 |
| EMA 0.9965 (higher decay) | 🔵 leaderboard | CP | −0.001 to −0.002 BPB | comp SOTA PRs #1421, #1471 | **Actionable port: bump EMA_DECAY** |
| **Pre-Quant AdamW TTT** | 🔵 leaderboard | CP | **−0.014 BPB (biggest single delta vs us)** | `PHASE1_NOVELTY_AUDIT.md §C1`, comp #1416, #1423, #1485 | Full TTT before GPTQ quantization, bakes adaptation into int6 |
| **AR self-generated GPTQ calibration** | 🔵 leaderboard | CP | compliance hedge | comp PR #1019 | Model writes own 64×2048 calibration set via rejection sampling |
| QAT (int6 quantization-aware training) | 🟢 shipped | CP | −0.002 to −0.005 BPB | comp standard | STE through int6 rounding; tightens quantized weight quality |
| **Compression-aware training** | 🎯 open | WN | −0.01 to −0.03 BPB | `STACK_UTILISATION_RESULTS.md §12.5` | Loss = NLL + λ×f(θ) where f is entropy regularizer or brotli-size surrogate |
| Complementary training (weight by 1 - bigram_P) | ❌ killed | — | regression | `RESEARCH.md`, `PLAN.md §2B-33` | Tested twice, worse both times |
| Mean-teacher self-distillation | 🎯 open | CN | −0.005 to −0.01 BPB | protocol §10.2 L05 × distillation (self only!) | **Legal** if teacher is an earlier-step version of the same run; NOT legal with external pretrained teacher |
| Contrastive pretraining phase | 🎯 open | CN | unknown | protocol §10.2 L05 × contrastive | Short contrastive pretraining before CE training |
| Asymmetric label smoothing | 🎯 open | WN | −0.001 to −0.005 BPB | comp PR #? (likely) | Smooth rare-token labels more than common-token labels — informed by P7 |
| Auxiliary loss: predicted-brotli-size head | 🎯 open | WN | −0.01 to −0.02 BPB | protocol §10.2 L05 × compression-aware | Tiny head predicts post-quant brotli size; add to main loss as regularizer |

---

## L06 — Evaluation / scoring rule

**Current** (🟢 in 1.082): Sliding window eval (stride=64), gated Score-First TTT (3 SGD epochs per chunk), n-gram bias (bigram+trigram+fourgram), DC500 (distributional categories), context engine (16-state FSM), temperature scaling.

**Slack measured**: P7 shows position barely helps (0.25 nat span across 2048), rare tokens carry 2.3× more loss. P15 shows gated attention actually hurts.

| Name | Status | Class | Expected | Source | Note |
|---|---|---|---|---|---|
| **Legal Score-First TTT** | 🟢 shipped | CP | −0.043 BPB (champion alone) | `SUBMISSION_RUN_STATE.md`, comp PR #1242, #1318 | Score chunk, then 3 SGD steps, repeat |
| **Sliding window eval (stride=64)** | 🟢 shipped | CP | baseline standard | comp standard | Causal sliding window with 2048 context |
| **N-gram logit bias (bigram+trigram+fourgram)** | 🟢 shipped | CP | −0.032 BPB cumulative | `PLAN.md §1.7h-i` | Backward-looking n-gram cache, entropy-adaptive alpha |
| **N-gram "Tilt" (multiplicative boost)** | 🟡 prototyped | WN | −0.0029 to −0.0055 BPB | `RESEARCH_LOG.md` subagent C | Multiplicative exp(β·𝟙[t==hint])/Z; no published work |
| QK-Gain sharpening | 🟢 shipped | CP | −0.001 per step | comp SOTA | See L03 row; eval interpretation overlaps |
| Temperature scaling (T~0.9) | 🟢 shipped | CP | −0.001 to −0.005 BPB | comp SOTA | Post-hoc softmax temperature tuning |
| DC500 (500 distributional token categories, w=0.30) | 🟢 shipped | WN | −0.259 BPB at 50 steps (best weight) | `PLAN.md §1f-v3`, `STACK_NOVELTY_TRACKER.md` | Categories by distributional similarity, transitions learnable |
| DC1000, DC2000 scale test | ⚪ idea | CN | −0.20 to −0.30 BPB ceiling | `PLAN.md §1.7d` | Test larger DC tables for theoretical ceiling |
| Context Engine (16-state FSM) | 🟢 shipped | WN | −0.018 BPB combined | `PLAN.md §1e` | Sentence-start / URL / price / number / code contexts |
| POS Tag Transitions (12 crude tags) | 🟢 shipped | CN | −0.002 BPB | `PLAN.md §1f` | Bigram POS constraints; expand to 50-100 for gains |
| Word Completion Trie (10K words) | 🟡 prototyped | CN | −0.030 combined | `PLAN.md §1b` | Boost tokens that continue partial words |
| Semantic Type Bias (color/number/name) | ⚪ idea | WN | −0.005 to −0.010 BPB | `PLAN.md §1g` | Semantic triggers ("color" → boost color tokens) |
| **Entropy-adaptive n-gram mixer (full schedule)** | ⚪ idea | WN | −0.05 to −0.16 BPB | `RESEARCH.md §Complementary Training` | α = 0.05 + 0.55·sigmoid(2·(H_neural − 4)) with multi-order backoff 2-10 |
| LoRA TTT (low-rank adaptation) | 🔵 leaderboard | CP | ~−0.037 BPB est | `RESEARCH_LOG.md` | Low-rank weight updates at eval; comp PR #1928 |
| qTTT (query-only TTT, cache K/V) | ⚪ idea | CN | 2-3× more TTT epochs | `RESEARCH.md §legal-eval-time-techniques` | Freeze K/V, adapt only Q → cheaper eval |
| **Arithmetic-coding loss (train for AC rate directly)** | 🎯 open | WN | −0.01 to −0.03 BPB | `WIN_PLAN.md`, `MOONSHOT_RULES.md §2.4` | Minimize actual arithmetic-coding bit rate instead of cross-entropy proxy |
| **Per-token hedge mixer with learned gating** | 🎯 open | WN | −0.02 to −0.05 BPB | `STACK_UTILISATION_RESULTS.md §12.5` | Learned gate over (LM, n-gram, CTW, PPM) per position |
| Temperature per-token by rarity | 🎯 open | WN | −0.01 to −0.02 BPB | `STACK_UTILISATION_RESULTS.md §12.5` | Rare tokens: higher T + bigger n-gram fallback; informed by P7 |
| Ensemble of seeds at eval (seed 42/314/999 together) | ⚪ idea | CN | −0.003 to −0.01 BPB | protocol §10.2 L06 × Bayesian | Average logits from 3 seeds — but need all 3 in 16 MB |
| Seed-averaged param checkpoint | ⚪ idea | CN | −0.005 to −0.01 BPB | `STACK_UTILISATION_RESULTS.md §11.11` | Average seeds' weights (not logits) — "free BPB if it works" |
| Bayesian model averaging over TTT snapshots | 🎯 open | WN | −0.005 to −0.01 BPB | protocol §10.2 L06 × Bayesian | Weighted average of model states across TTT epochs |
| Solomonoff prior (algorithmic prior over strings) | 🎯 open | WN | unknown | protocol §10.2 L06 × Solomonoff | Use Solomonoff prior as a component of the hedge mix |

---

## L07 — Compression / quantization

**Current** (🟢 in 1.082): int6 GPTQ (full Hessian, no Cholesky) for matrices, int8 for embedding, brotli-11 + BSHF stride=2 for final pack.

**Slack measured**:
- Embedding at 4.72 bits entropy vs 8-bit container → ~1.05 MB reclaimable safely (int8 → int6 on embedding)
- MLP entropy at 3.33 bits in int6 container → already tight
- P10: BSHF stride=2 already optimal — no compression win from stride tuning

| Name | Status | Class | Expected | Source | Note |
|---|---|---|---|---|---|
| **Int6 GPTQ (full Hessian)** | 🟢 shipped | CP | standard baseline | `PLAN.md §3`, comp SOTA | Greedy quantization with full Hessian, per-group scaling |
| **Int8 for embedding** | 🟢 shipped | CP | embedding quality | comp standard | Int8 for token embeddings |
| **Brotli-11 compression** | 🟢 shipped | CP | −0.58 MB vs zstd-22 | comp SOTA | Non-standard compressor; better on bit-packed weights |
| **BSHF byte-shuffle stride=2** | 🟢 shipped | IN | −7 KB | `submission/train.py` | Interleave bytes before brotli; stride 2 is optimal (P10) |
| Mixed int5/int6 per-layer (Hessian-weighted) | 🔵 leaderboard | CP | −0.001 to −0.002 BPB | comp PRs #1429, #1438 | **Actionable port** |
| **Embedding int8 → int6** | 🎯 open | WN | 1.05 MB reclaimable | `STACK_UTILISATION_RESULTS.md §12.7` | P2 evidence: 4.72 bits Shannon entropy fits in int6; re-spend MB on extra layer or wider MLP |
| BitNet 1-bit (binary) | 🟡 prototyped | CN | 1.1239 BPB @ 106 M, 15.67 MB | records/track_non_record, comp PR #1239 | Non-record track (2+ hours), explicit OpenAI wishlist (✅ 1-bit) |
| Ternary ({-1, 0, +1}) 1.6-bit | 🟡 prototyped | CN | 1.15 BPB est | records/track_10min/74M_Ternary | 60% fewer params/MB than binary; explicit OpenAI wishlist |
| Int4 quantization | ❌ killed | — | +0.048-0.060 BPB worse | `RESEARCH.md killed` | Too lossy for byte-level; catastrophic degradation |
| Lloyd-Max quantization (K-means codebook) | ❌ killed | CN | 92.7% lower MSE but 2× artifact size | `PLAN.md §1.7c` | Codebook overhead kills net savings |
| Sigma-Delta quantization (residual feedback) | 🎯 open | WN | 1-2 MB saved | `RESEARCH.md §4` (audio DAC port) | Feed quant error to next weight, like audio ΔΣ modulator |
| Golomb coding for weights | ⚪ idea | WN | 1-3 MB savings | `PLAN.md §2B-46` | Variable-length entropy coding instead of fixed 6-bit |
| **Asymmetric numeric systems (rANS) neural prior** | 🎯 open | WN | unknown | `STACK_NOVELTY_TRACKER.md §C` | rANS with learned layer/position-dependent prior — no published work combines |
| **Tensor-train decomposition (int4 cores)** | 🎯 open | WN | unknown, potentially large | `STACK_NOVELTY_TRACKER.md §C` | TT decomp per-core mixed int4/int5 by importance; post-hoc no retraining |
| **VQ residual codebooks (ERVQ)** | 🎯 open | WN | unknown | `STACK_NOVELTY_TRACKER.md §C` | Multi-stage residual VQ per-layer codebook, learned during training |
| Fractional-bit quantization (log2(3) = 1.585-bit) | 🎯 open | WN | small budget savings | protocol §10.2 L07 × neural compression | Pack 5 ternary values into 8 bits (243 combos in 256 states) |
| Neural compression via INR / implicit representation | 🎯 open | WN | unknown | protocol §10.2 L07 × neural compression | Represent the weight tensor as a small MLP that emits weights from coords |
| JPEG-style transform coding on weights | 🎯 open | WN | 1-2 MB savings | protocol §10.2 L07 × JPEG | DCT on weight matrices, keep top-k coeffs, quantize those |
| Signed-digit representation (NAF) | 🎯 open | WN | small savings | protocol §10.2 L07 × signed-digit | Non-adjacent form reduces bit count by ~1/3 for random weights |
| **Compression-aware training** (see L05) | 🎯 open | WN | −0.01 to −0.03 BPB | `STACK_UTILISATION_RESULTS.md §12.5` | Cross-refs L05 row; training objective aware of final compressor |

---

## L08 — N-gram bias (static tables as logit bias)

**Current** (🟢 in 1.082): Bigram + trigram + fourgram + fivegram, 16 K hash buckets each, tabulation hashing, entropy-adaptive mixer.

**Slack measured**: `FLOOR_RESULTS.md` — n-gram structure floors at ~1.8-2.0 BPB. Adding more orders as a BIAS on top of an LM has <0.02 BPB of headroom, not 0.20 I originally budgeted.

| Name | Status | Class | Expected | Source | Note |
|---|---|---|---|---|---|
| **Bigram + Trigram + Fourgram + Fivegram bias** | 🟢 shipped | CP | −0.032 to −0.066 per order | `PLAN.md §2A` | Static tables, logit-bias addition |
| **Tabulation hashing (3-independent)** | 🟢 shipped | WN | −0.0024 per hash | `MINIPAPER_TABULATION_HASH.md`, our submission | XOR-based 3-independent hash vs polynomial 2-dep; unbiased collision noise |
| **Entropy-adaptive bigram/trigram/4gram** | 🟢 shipped | CP | −0.0045 BPB | comp PR L09_entropy_adaptive | Weight n-gram orders by model entropy |
| BigramHash 3072 × 112 | 🔵 leaderboard | CP | −0.001 to −0.002 BPB | comp SOTA #1019, #1408 | Larger hash table, 112-dim embedding |
| EngramLite (multi-head hash with sigmoid gating) | 🟡 prototyped | CN | −0.005 BPB; 23% less storage | comp PR #1089 (1.1086 BPB) | K=2-4 hash heads per order, sigmoid gating, depthwise conv smoothing |
| Skip-bigram (word-internal + word-start) | 🟡 prototyped | CN | −0.063 @ 50 steps, regresses @ 500 | `RESEARCH_LOG.md` subagent C | Skip within-word boundaries from 8-16 token span; likely stale-data artifact |
| **Sevengram (count-min sketch)** | 🟢 shipped | CP | −0.026 BPB | `PLAN.md §2B-27` | Probabilistic count matrix for 7-gram without full table |
| Skip-4gram / Skip-5gram Hadamard | ⚪ idea | CN | unknown | `STACK_NOVELTY_TRACKER.md §C` | Hadamard product of skip-grams for higher order without table explosion |
| **Adaptive cuckoo hashing (collision-free)** | 🎯 open | WN | unknown; collision-free | `STACK_NOVELTY_TRACKER.md §C` | Cuckoo hash with multiple hash fns → no loss from collisions; infrastructure-level |
| Bloomier filter for sparse tables | 🎯 open | WN | size reduction | protocol §10.2 L08 × Bloomier | Bloomier filter variant for probabilistic count storage |
| Elias-Fano encoded n-gram offsets | 🎯 open | WN | 2-3× table compression | protocol §10.2 L08 × universal codes | Monotonic encoding for sparse n-gram position/count lists |

---

## L09 — N-gram engine / dynamic structures

**Current** (🟢 in 1.082): Static tables built from train; no online adaptation.

| Name | Status | Class | Expected | Source | Note |
|---|---|---|---|---|---|
| **Online growing n-gram cache (moonshot)** | 🎯 open | WN | **−0.05 to −0.15 BPB** | `MOONSHOT_RULES.md §2.1`, `STACK_UTILISATION_RESULTS.md §12.5` | **Primary moonshot.** Score-first update; rules confirm legal. Precedent: cmix hits 0.9 BPB on enwik9 |
| **CTW (Context Tree Weighting, online)** | 🎯 open | WN | −0.05 to −0.10 BPB | `MOONSHOT_RULES.md §2.6`, Willems 1995 | Provably near-optimal universal predictor; online-built; ~200 LOC |
| **PPM-Star / PPM-mixture (online)** | 🎯 open | WN | −0.03 to −0.08 BPB | protocol §10.2 L09 × PPM | Mixture over PPM orders with Bayesian weighting |
| Suffix array / BWT-based lookup | 🎯 open | WN | unknown | protocol §10.2 L09 × BWT | O(log n) lookup into growing causal-val suffix array |
| LZMA-family predictor | 🎯 open | CN | unknown | protocol §10.2 L09 × LZ | LZMA's range coder + context model as a predictor component |
| **Rare-token specialist (MoE-lite)** | 🎯 open | WN | −0.01 to −0.03 BPB | `STACK_UTILISATION_RESULTS.md §12.5`, P7 evidence | Tiny side expert gated by "is this context rare?"; targets P7 tail-50% |

---

## L10 — Test-time training / eval-time adaptation

**Current** (🟢 in 1.082): Score-First TTT, 3 SGD epochs per val chunk.

**Rules-confirmed legal** (per `MOONSHOT_RULES.md §1.4`): **any adaptation on val tokens after they're scored**.

| Name | Status | Class | Expected | Source | Note |
|---|---|---|---|---|---|
| **Score-First TTT (3 SGD epochs, chunked)** | 🟢 shipped | CP | −0.043 BPB alone | `SUBMISSION_RUN_STATE.md` | Adaptive: score chunk, 3 SGD steps, advance chunk |
| LoRA TTT | 🔵 leaderboard | CP | ~−0.037 BPB est | comp PR #1928 | Low-rank updates only; cheaper than full TTT |
| qTTT (query-only) | ⚪ idea | CN | 2-3× more epochs feasible | `RESEARCH.md §legal-eval` | Freeze K/V, adapt Q only |
| Online SGD cache (rolling window) | 🎯 open | WN | unknown | protocol §10.2 L10 × online CO | Per-token gradient step, no chunking; convergence analysis under OGD bounds |
| FTRL (Follow-The-Regularized-Leader) TTT | 🎯 open | CN | unknown | protocol §10.2 L10 × OCO | Per-token convex adaptation with regularization; provable regret |
| MAML pretraining for TTT-ready init | 🎯 open | WN | TTT 2× faster convergence | protocol §10.2 L10 × meta-learning | Meta-train such that 1 SGD step is as good as 3 |
| Legal TTT + logit bias stacking | ❌ killed | — | +0.040 worse when stacked | `STACK_NOVELTY_TRACKER.md §champion` | Score-First TTT composes brittle-ly with other changes; needs isolated retune |
| Per-layer Hessian-aware GPTQ calib | ⚪ idea | CN | −0.002 BPB est | `PHASE1_NOVELTY_AUDIT.md §C10` | Scale clipping per layer by Hessian variance |

---

## L11 — Infrastructure / kernels / speed (IN = infra, may not move BPB directly)

**Current** (🟢 in 1.082): FA3 on Hopper, Triton TMA+megakernel fused MLP, expandable-segments allocator, grad_accum=8, compile disabled (saves first-run compile cost).

**Rationale for including speed**: every 5% throughput gain = ~5% more training steps = potentially measurable BPB via more tokens seen. Also eval-time budget is 10 min; speed matters for complex eval (TTT, hedge mixer, online cache).

| Name | Status | Class | Expected | Source | Note |
|---|---|---|---|---|---|
| **Flash Attention 3** | 🟢 shipped | CP | +30% vs SDPA | comp standard | Native Hopper softmax kernel |
| **Triton TMA megakernel (fused MLP)** | 🟢 shipped | CP | +10.5% throughput | comp PRs #1420, #1450, `PHASE1_NOVELTY_AUDIT.md §C9` | Triton TMA fused MLP forward + CUTLASS EVT backward |
| Dynamic Lyapunov adaptive grad clip | 🟡 prototyped | WN | neutral (−0.00005 BPB, no gain) | `STACK_NOVELTY_TRACKER.md §A` | Compute λ₁ of Jacobian for adaptive clip; mechanism confirmed but no BPB win |
| `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` | 🟢 shipped | IN | +8.8% step time | comp PR #1420, `submission/run.sh` | Disable regressed Inductor pass; zero-risk env var |
| `expandable_segments` allocator | 🟢 shipped | IN | frag reduction | `submission/run.sh` | PyTorch async CUDA allocator |
| Torch compile re-enable (with cache) | ⚪ idea | IN | +25-35% throughput on subsequent runs | `CLAUDE.md §wishlist` | Pre-warm compile cache so first-run cost is amortized |
| **Custom CUDA megakernel (fused full block)** | 🎯 open | WN+IN | +20-40% throughput | OpenAI wishlist | Fuse attn+norm+MLP+residual into one CUDA kernel |
| **Fused QKV + norm + RoPE** | 🎯 open | WN+IN | +10-15% | protocol §10.2 L11 × CUDA | Single kernel for QKV projection + pre-attn RMSNorm + RoPE |
| **Fused LM head + softmax + arith-coder** | 🎯 open | WN+IN | eval speedup | protocol §10.2 L11 × CUDA | Single kernel from hidden state → arithmetic-coding output bits |
| PTX-level hand tuning | 🎯 open | IN | +5-15% | protocol §10.2 L11 × PTX | Manual PTX for the inner loops of the above |
| wgmma (Hopper-native matmul) tiling | 🎯 open | IN | +5-15% | protocol §10.2 L11 × Hopper | Custom wgmma instruction tiling for our specific shapes |
| CPU-side worker pool for data prep | ⚪ idea | IN | +5% train throughput | memory `feedback_max_cpu_too.md` | 128 vCPUs on H100 node — saturate with CPU-side precomputation |
| Fused bit-pack kernels (int6 pack/unpack) | 🎯 open | IN | eval speedup | protocol §10.2 L11 × CUDA | Pack/unpack int6 weights directly in kernel, avoid unpack → fp16 → matmul |
| Mixed-precision scheduling (per-layer fp16/bf16/int6) | 🎯 open | IN | +5-15% + small BPB | protocol §10.2 L11 × mixed-precision | Different layers run at different precisions based on sensitivity |
| Activation recomputation w/ selective checkpointing | ⚪ idea | IN | enables 1.3× bigger batch | standard | Memory budget for bigger batch → better gradient estimates |
| Persistent-threadblock CUDA for small matmuls | 🎯 open | IN | L3/L4 attn speedup | protocol §10.2 L11 × CUDA | Keep small MLP/attn matmuls persistent on SMs across forward passes |

---

## Cross-layer moonshots (span multiple layers)

| Name | Layers | Status | Expected | Source | Note |
|---|---|---|---|---|---|
| **Online n-gram cache + hedge mixer + AC loss** | L06 + L09 + L10 | 🎯 open | −0.05 to −0.15 BPB (cumulative) | `WIN_PLAN.md`, `MOONSHOT_RULES.md` | The primary moonshot: LM + online cache + CTW + hedge, trained end-to-end on arithmetic-coding bit-rate |
| **Compression-aware training + int8→int6 embed + extra layer** | L03 + L05 + L07 | 🎯 open | −0.02 BPB | `STACK_UTILISATION_RESULTS.md §12.5` | Co-train for brotli-friendliness; reclaim 1 MB from embedding; spend on extra transformer block |
| **Huffman tokenizer + per-token temperature + rare specialist** | L01 + L06 + L09 | 🎯 open | −0.03 BPB | `STACK_UTILISATION_RESULTS.md §12.5` | Targets rare tokens (P7); vocab restructure + inference-time scaling + dedicated expert |
| **Drop gated attention + re-spend 45 KB on MLP** | L03 | 🎯 open | −0.01 BPB + capacity | `STACK_UTILISATION_RESULTS.md §12.2 P15` | **Immediate next experiment.** Hard evidence from probes |
| **Megakernel full stack** | L11 | 🎯 open | +20-40% throughput → +4-8 more training steps → small BPB | OpenAI wishlist | Fused attention + MLP + norm + residual + gating as one kernel |

---

## Things explicitly KILLED (don't re-propose without new evidence)

| Name | Layer | Why killed | Source |
|---|---|---|---|
| Int4 matrix quantization | L07 | +0.048-0.060 BPB worse | `RESEARCH.md` |
| Lloyd-Max codebook | L07 | 92.7% lower MSE but 2× artifact size | `PLAN.md §1.7c` |
| HyperGPT | L03 | 8.218 BPB catastrophic | comp PR #1288 |
| Parallel FFN+Attn (L05 variant) | L05 | +0.0098 worse | `STACK_NOVELTY_TRACKER.md §A` |
| Riemannian Stiefel projection (QKV) | L04 | +0.0024 worse | `STACK_NOVELTY_TRACKER.md §A` |
| Chebyshev NS | L04 | demoted to comp-novel (arxiv prior) | `STACK_NOVELTY_TRACKER.md §A` |
| Complementary training (1-P bigram weighting) | L05 | worse in 2 independent tests | `RESEARCH.md`, `PLAN.md §2B-33` |
| Dynamic Lyapunov grad clip | L11 | mechanism confirmed, 0 BPB gain | `STACK_NOVELTY_TRACKER.md §A` |
| NGRAM_GATE (learned per-pos gate, 1500 steps) | L09 | couldn't learn in budget | `CLAUDE.md §what's been tried` |
| Bigger model (12L 3× MLP) | L03 | fewer steps offsets capacity | `CLAUDE.md §what's been tried` |
| Byte-weighted loss | L05 | metric not comparable | `CLAUDE.md §what's been tried` |
| Wavelet GPT on cheap GPU | L03 | hurt on cheap GPU; works at full scale | `CLAUDE.md §what's been tried` |
| Low-rank factoring of MLP / attn | L03/L07 | P6: all MLP/attn near-full-rank | `STACK_UTILISATION_RESULTS.md §4.6` |
| Dead-head pruning | L03 | P3: 0 dead heads | `STACK_UTILISATION_RESULTS.md §4.3` |
| Dead-neuron pruning | L03 | P4: 0 dead neurons | `STACK_UTILISATION_RESULTS.md §4.4` |
| BSHF stride tuning | L07 | P10: stride=2 already optimal | `STACK_UTILISATION_RESULTS.md §11.4` |
| External LM distillation | L05 | `MOONSHOT_RULES.md §3` — spirit-of-rules forbids | `MOONSHOT_RULES.md §3` |

---

## Shipped-in-comp but NOT in our stack (port candidates)

These are wins demonstrably shipped by competitors that our 1.082 doesn't use:

| Name | Layer | Expected | Comp PR |
|---|---|---|---|
| Pre-Quant AdamW TTT | L05 | −0.014 BPB (biggest single delta) | #1416, #1423, #1485 |
| AR self-generated GPTQ calibration | L05 | compliance hedge | #1019, #1446 |
| LoRA TTT | L10 | −0.037 BPB est | #1928 |
| QK-Gain = 5.25 (we still at 4.0) | L03 | −0.003 BPB per 0.5 step | comp frontier |
| EMA decay 0.9965 (we still at 0.997) | L05 | −0.001 to −0.002 | #1421, #1471 |
| Mixed int5/int6 Hessian-weighted | L07 | −0.001 to −0.002 | #1429, #1438 |
| BigramHash 3072 × 112 | L08 | −0.001 to −0.002 | #1019, #1408 |
| TMA fused MLP kernel | L11 | +10.5% throughput | #1420, #1450 |

---

## How to use this tracker

1. **Weekly re-audit**: every Monday, `gh pr list openai/parameter-golf` → compare against this list → mark any world-novel entries that got published (downgrade them from `🎯 WN` to `🔵 CP`).
2. **Before shipping a novelty**: check this list to see if we're inadvertently re-doing a killed idea. If the idea is here as ❌ killed, a new experiment needs NEW evidence (different hyperparams, different stack, or new reasoning) to justify re-testing.
3. **When picking the next experiment**: filter by `🎯` (open novelty), sort by expected BPB, cross-check against `STACK_UTILISATION_RESULTS.md` and `MOONSHOT_RULES.md` for capacity + legality. Use `protocol §10` to expand if the list feels stale.
4. **Prior-art subagents**: for any `🎯 WN` entry you're about to ship, spawn a WebFetch/WebSearch subagent to re-verify it hasn't been published since this tracker was last updated.

## Related docs

- `MOONSHOT_RULES.md` — what's legal (moonshots §2), the novelty-mining protocol (§10)
- `STACK_UTILISATION_RESULTS.md` — what probes measured; the "slack" columns above come from here
- `FLOOR_RESULTS.md` — the floor measurements (classical + n-gram + GPT-class)
- `WIN_PLAN.md` — the <1.0 moonshot plan; the "cross-layer moonshots" section above feeds it
- `docs/research/RESEARCH_LOG.md` — running log of every S2 confirm / fire result
- `docs/results/STACK_NOVELTY_TRACKER.md` — older rolling state that this supersedes

## Changelog

- **2026-04-16**: initial comprehensive tracker (v2). Superseded the older rolling `STACK_NOVELTY_TRACKER.md` in docs/results/.
