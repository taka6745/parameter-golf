# Lossy Weight Compression Techniques for Parameter Golf (16MB Budget)

Research report -- April 3, 2026

## Context

Competition: OpenAI Parameter Golf. Artifact limit: 16,000,000 bytes (code + compressed weights). Training: 10 min on 8xH100. Metric: BPB on FineWeb.

Current SOTA uses: int5/int6 quantization, GPTQ calibration, zstd lossless compression. Nobody is doing sophisticated lossy compression on the weight matrices themselves. The question is: what else is out there, and is any of it practical?

---

## 1. Truncated SVD (Low-Rank Decomposition)

**How it works:** Decompose weight matrix W (m x n) into U * S * V^T, then keep only the top-k singular values. Store U_k (m x k) and V_k (k x n) instead of W. Reconstruct as U_k @ diag(S_k) @ V_k.

**Compression ratio:** For an m x n matrix at rank k, you store m*k + k + k*n instead of m*n parameters. For a 512x1536 MLP weight at rank 128: original = 786,432 params, compressed = 65,536 + 128 + 196,608 = 262,272 params. That is 3x compression. At rank 64, 6x compression but with significant quality loss.

**Quality loss:**
- SVD-LLM (ICLR 2025) reports that at 20% compression ratio (keeping 80% of parameters), perplexity degrades by 29% on WikiText-2 for LLaMA-7B
- At 30% compression, perplexity degrades by 81% -- catastrophic
- Post-training SVD without fine-tuning is destructive; the optimization objective (minimize Frobenius norm) does not align with task loss
- Fisher-Weighted SVD (FWSVD) helps by weighting important parameters, but requires calibration data

**Post-training applicability:** Yes, but quality is poor without fine-tuning. SVD-LLM adds a "truncation-aware data whitening" step and sequential layer-by-layer update that helps, but requires calibration passes.

**Relevance to 16MB competition:** MARGINAL. Our model is already small (~25M params). SVD only helps if some weight matrices are significantly over-parameterized AND we can afford the rank reduction. A 512x1536 MLP matrix at rank 128 saves ~500KB but loses quality. The budget is better spent on more parameters at lower precision (int6) than fewer parameters at higher precision.

**Implementation complexity:** Low. numpy.linalg.svd + truncation. Reconstruction is two matrix multiplies (slightly slower inference).

---

## 2. Lottery Ticket / Magnitude Pruning

**How it works:** Remove weights with smallest magnitudes (set to zero). The sparse matrix is then compressed (only store non-zero values + indices). Lottery ticket hypothesis: a dense network contains a sparse subnetwork that, trained from the right initialization, matches dense performance.

**Compression ratio:** Original paper achieves 10-20% remaining parameters (5-10x compression) on MNIST/CIFAR. For LLMs, SparseGPT achieves 50-60% unstructured sparsity on 7B+ models with minimal perplexity loss (~1 point at 2.7B). At 90% sparsity, quality degrades sharply for small models.

**Quality loss:**
- SparseGPT: ~1 perplexity point at 50% sparsity on OPT-2.7B, near-zero loss on 66B models
- Lottery ticket requires iterative pruning + retraining from init -- not post-training
- SparseLLM improves high-sparsity regimes (>60%) by up to 80% perplexity reduction vs naive pruning

**Post-training applicability:** Partially. SparseGPT is one-shot post-training. However, lottery ticket requires retraining from scratch, which contradicts the competition constraint of training the model once in 10 minutes.

**Relevance to 16MB competition:** LOW. The problem: unstructured sparsity requires storing indices (which costs bits). For a small model, the overhead of a sparse format (CSR/CSC) often exceeds the savings. At 50% sparsity with 6-bit weights, you save 3 bits/weight but spend ~16 bits per nonzero on index storage. Only viable at extreme sparsity (>90%), which kills quality at our model size. Structured pruning (entire rows/columns/heads) avoids index overhead but is coarser and more damaging.

**Implementation complexity:** Medium. SparseGPT requires Hessian inverse computation per layer. Sparse storage formats add code complexity.

---

## 3. Kronecker Product Factorization

**How it works:** Approximate weight matrix W (m x n) as a sum of Kronecker products: W ~ sum_i (A_i kron B_i), where A_i is (p x q) and B_i is (m/p x n/q). Storage is sum of (p*q + (m/p)*(n/q)) per term instead of m*n.

**Compression ratio:** Krony-PT compresses GPT-2 (124M) to 81M parameters (35% reduction) by targeting FFN matrices. With aggressive Kronecker factoring, a single 512x1536 matrix can be stored as e.g. (16x48) kron (32x32) = 768 + 1024 = 1,792 params instead of 786,432 -- a 438x theoretical compression. In practice, you need multiple Kronecker terms to maintain quality, so realistic compression is 5-20x per matrix.

**Quality loss:** Krony-PT (81M from 124M) sees perplexity go from 24.67 to 35.75 on WikiText-2 (~45% worse). Better than DistilGPT2 (36.48) at same size, but the gap is real. Quality improves with more Kronecker terms but storage increases proportionally.

**Post-training applicability:** Partially. Van Loan decomposition gives a post-training initialization, but the paper shows fine-tuning is critical. For competitions, you would need to train WITH Kronecker structure from the start (which means modifying the training loop).

**Relevance to 16MB competition:** INTERESTING BUT HARD. The core idea -- representing large matrices as structured products of small matrices -- is sound. But the competition already uses aggressive parameter tying and low-rank-ish tricks (3x MLP instead of 4x, tied embeddings). Kronecker would require modifying the architecture, and the quality penalty at small scale may not justify the complexity. However, if implemented during training (not post-hoc), Kronecker layers could allow wider or deeper models within the same parameter budget.

**Implementation complexity:** High. Requires custom forward pass for Kronecker layers, custom initialization (modified Van Loan), and integration with quantization.

---

## 4. Product Quantization (PQ)

**How it works:** Split each weight vector into M sub-vectors. For each sub-vector position, learn a codebook of K centroids via k-means. Each sub-vector is replaced by its nearest centroid index. Storage = M * ceil(log2(K)) bits per weight vector + codebook storage.

**Compression ratio:** With M=8 sub-vectors and K=256 (8 bits per sub-vector index), a 768-dim float32 vector (3072 bytes) compresses to 8 bytes + codebook overhead. That is ~384x on the vector, but codebook overhead is 256*768*4 = 768KB for one layer. Net compression depends heavily on matrix sizes.

A recent ICCV 2025 paper achieved "as low as 1 bit per parameter (13x model size reduction)" using PQ with EM calibration on generative models. This is extreme -- 1 bit per weight is on par with ternary quantization but with a smarter codebook.

**Quality loss:** PQ at 1-2 bits per weight with EM recalibration can recover most quality. The paper claims competitive image generation quality at 1-bit. For language models, AQLM (Additive Quantization for LMs) at 2 bits achieves 6.93 perplexity on WikiText-2 for LLaMA-7B (vs 5.12 FP16) -- a 35% degradation. At 3 bits, degradation is <5%.

**Post-training applicability:** YES -- this is the key advantage. PQ is inherently a post-training technique (k-means on trained weights). AQLM adds calibration data for better centroid assignment but does not retrain.

**Relevance to 16MB competition:** MODERATE-HIGH. PQ is conceptually what GPTQ already does (group quantization with per-group scales). The question is whether PQ codebooks can beat uniform int6 quantization. The answer is yes in theory -- a learned codebook adapts to the actual weight distribution rather than assuming uniform spacing. But: (a) the competition already uses GPTQ which is a sophisticated form of this, (b) codebook overhead matters at small scale, (c) the decoding step adds latency. The sweet spot might be PQ on the largest matrices (MLP weights) where codebook overhead is amortized.

**Implementation complexity:** Medium. Sklearn k-means or faiss for codebook learning. Custom dequantization kernel needed for inference speed.

---

## 5. Learned / Neural Weight Compression (NWC)

**How it works:** Train a neural codec (encoder-decoder) on a dataset of pretrained weight matrices. The encoder transforms weights into a latent code, which is entropy-coded. The decoder reconstructs weights from the latent code. The codec is optimized for rate-distortion tradeoff on weight reconstruction.

**Compression ratio:** NWC (2025 paper) achieves SOTA at 4-6 bits per weight, outperforming GPTQ/AWQ/SqueezeLLM. At 4 bits, NWC matches or beats GPTQ quality while being more efficient at entropy-constrained coding. Decoding latency for a 4096x4096 tensor: 1.17ms (vs GPTQ 0.08ms).

**Quality loss:** At 4 bits, near-lossless for large models. The learned transforms reduce outlier impact and adapt to per-layer weight distributions. At 2-3 bits, still competitive with fixed-scheme quantization.

**Post-training applicability:** YES -- designed for it. The codec is trained on weight datasets (potentially from other models) and applied to compress any new model.

**Relevance to 16MB competition:** LOW-MODERATE. The problem: the codec itself takes storage. For a 25M-param model being compressed from ~50MB (fp16) to 16MB, the codec would need to be tiny. Also, decoding latency (1.17ms per tensor) is nontrivial when you have 10+ layers with multiple tensors each. The 10-minute eval budget may not accommodate slow decompression. Most importantly, at our model size, the per-layer statistics are too idiosyncratic for a generic learned codec to beat tuned GPTQ.

**Implementation complexity:** Very High. Requires pretraining the codec, integrating entropy coding (ANS/arithmetic), custom decompression during inference.

---

## 6. Arithmetic / ANS Entropy Coding (on quantized weights)

**How it works:** After quantizing weights to discrete levels (e.g., int6), the weight values have a non-uniform distribution. Arithmetic coding or ANS assigns shorter codes to more frequent values, approaching Shannon entropy. Savings = H(uniform 6-bit) - H(actual distribution) bits per weight.

**Compression ratio:** If int6 weights have entropy of 4.5 bits (instead of 6 bits uniform), arithmetic coding saves 25% over fixed-width encoding. DeepCABAC achieves 63.6x compression on VGG-16 (8.7MB from 553MB) combining quantization + context-adaptive arithmetic coding. The CERWU method achieves 20-40% better compression than standard NNCodec, and can represent weights with fractional bits (even <1 bit per weight for many parameters).

**Quality loss:** Zero additional loss -- this is lossless compression of the quantized weights. The only loss comes from quantization itself.

**Post-training applicability:** YES -- arithmetic coding is applied after training/quantization. It is purely a storage optimization.

**Relevance to 16MB competition:** **HIGH -- THIS IS THE MOST ACTIONABLE TECHNIQUE.** Current competition entries use zstd for lossless compression of quantized weights. But zstd is a general-purpose compressor that does not exploit the specific statistical structure of neural network weight distributions. A custom entropy coder that models the per-layer weight distribution could save 10-25% over zstd. On a 15.9MB artifact, that is 1.5-4MB freed -- enough for 1-2 extra transformer layers or significantly wider model.

Specific opportunity: int6 weights after GPTQ have a peaked distribution (many near-zero values, few large). A per-channel entropy model with ANS coding should beat zstd. ZipNN (2024) already shows 17-34% improvement over zstd on neural network weights by exploiting exponent statistics.

**Implementation complexity:** Medium. Python ANS implementations exist. DeepCABAC has reference code. The challenge is that decompression must be fast (can't spend minutes decompressing at eval time).

---

## 7. Weight Sharing via Hashing (HashedNets)

**How it works:** Use a hash function to map weight indices to a smaller set of shared values. All weights that hash to the same bucket share one learnable parameter. Storage = num_buckets * bits_per_weight + hash_seed. No index overhead -- the hash function is deterministic from the seed.

**Compression ratio:** Compression ratio = num_weights / num_buckets. With 786K weights and 8K buckets, that is ~100x compression per matrix. Original paper shows 8-64x compression with "minimal accuracy loss" on MNIST.

**Quality loss:** The competition itself already validates this: BigramHash (3072 features) is used in the SOTA entry, and your own n-gram bias experiments use hash-based feature tables (8K-65K buckets). The quality depends on bucket count: your v12 (8K buckets) lost 0.024 BPB vs v11 (65K buckets), and v13 (16K buckets) lost only 0.014 BPB.

**Post-training applicability:** NO -- weight sharing must be built into training. The model learns shared values through backprop. Post-hoc weight sharing is just bad quantization.

**Relevance to 16MB competition:** **ALREADY IN USE** for n-gram feature tables. The question is whether to apply it to the transformer weights themselves. Hashed weight matrices would allow much wider/deeper models (e.g., 1024-dim with 10x hash sharing = same storage as 512-dim). But: (a) quality degradation at extreme sharing ratios for attention/MLP weights is worse than for n-gram tables, (b) soft weight sharing (Ullrich 2017) achieved 162x on LeNet-5 but that was a toy model, (c) at int6 quantization, weight sharing and quantization overlap -- both reduce unique values.

**Implementation complexity:** Low for feature tables (you already have it). Medium for transformer weight matrices (need custom layers with hash-indexed parameters).

---

## 8. Soft Weight Sharing / Codebook Quantization

**How it works:** Instead of hashing, learn a codebook of K values and a soft assignment of each weight to codebook entries (using a mixture model prior). During training, weights cluster around codebook values. Post-training, hard-assign each weight to nearest codebook entry and store only the index.

**Compression ratio:** Ullrich 2017 achieved 162x on LeNet-5-Caffe (vs 39x for Deep Compression). With K=16 codebook entries, each weight needs 4 bits. With K=64, 6 bits. The advantage over uniform quantization is that codebook values are non-uniformly spaced, matching the actual weight distribution.

**Quality loss:** Near-zero when codebook size is adequate. K=64 (6-bit) is essentially lossless for small models. K=16 (4-bit) shows 1-3% accuracy loss on image classification. For language models, this is similar to NormalFloat quantization (NF4 in QLoRA).

**Post-training applicability:** The hard assignment step is post-training, but training with the soft prior (Bayesian formulation) is needed for best results. Without it, this degenerates into k-means quantization (which is what GPTQ effectively does).

**Relevance to 16MB competition:** MODERATE. This is conceptually what GPTQ-lite already approximates. The marginal gain over GPTQ would come from non-uniform codebook spacing (which NormalFloat already partially achieves). Probably not worth the complexity.

**Implementation complexity:** Medium. Requires modifying training loss with mixture model prior.

---

## Synthesis: What Should You Actually Do?

### Tier 1: High Impact, Immediately Actionable

**Custom entropy coding on quantized weights (Technique 6)**
- Replace zstd with per-layer ANS/arithmetic coding
- Model the actual int6 weight distribution per channel
- Expected savings: 10-25% over zstd on the weight blob
- This is 1.5-4MB freed, which is HUGE at 16MB budget
- That freed space can fund: wider model, more layers, or larger n-gram tables
- Implementation: 1-2 days. Use existing ANS libraries (e.g., constriction, ryg_rans)
- Risk: decompression speed. Must decompress in <1 second at eval time
- ZipNN approach (separate exponent/mantissa, exploit skew) is simpler and gets 17% over zstd

### Tier 2: Moderate Impact, Requires Training Changes

**Hashed weight matrices for specific layers (Technique 7)**
- Apply hash-based weight sharing to the embedding layer or MLP weights
- You already have hash infrastructure from n-gram bias tables
- Could allow doubling embedding dimension at same parameter cost
- Quality risk is real: needs A/B testing at 500+ step scale
- The embedding layer is the best candidate (large, somewhat redundant)

**Kronecker-structured MLP layers (Technique 3)**
- Replace 512x1536 MLP weights with Kronecker products
- Could allow 4x MLP ratio (2048) instead of 3x (1536) at same storage
- Requires training with Kronecker structure from scratch
- Implementation: 2-3 days, high risk of quality regression

### Tier 3: Interesting but Probably Not Worth It

**Product quantization with learned codebooks (Technique 4)**
- GPTQ already captures most of this benefit
- Codebook overhead is proportionally large at 16MB scale
- Only viable if you can show PQ beats GPTQ-lite on your specific model

**Post-training SVD on specific layers (Technique 1)**
- Might work on attention V/O projections (known to be low-rank)
- Savings are small at 512-dim: rank-256 SVD on 512x512 saves nothing
- Only useful if you have a specific layer that is significantly over-parameterized

### Tier 4: Not Applicable

**Learned neural codec (Technique 5)** -- codec storage overhead too large for 16MB
**Lottery ticket pruning (Technique 2)** -- requires retraining from scratch, sparse storage overhead
**Soft weight sharing (Technique 8)** -- subsumes by GPTQ in practice

---

## The Big Picture: Where Are the Remaining Bits?

Current SOTA: 1.0713 BPB (WARP). Shannon limit for English: ~0.8 BPB. Gap: ~0.27 BPB.

Where the gap lives:
1. **Training data coverage** (~0.05-0.10 BPB): 10 min on H100 is not enough to see all of FineWeb. Better data selection helps.
2. **Model capacity** (~0.05-0.10 BPB): 16MB limits the model to ~25M params at int6. More efficient compression = more effective parameters.
3. **Eval-time adaptation** (~0.05-0.10 BPB): TTT/n-gram cache at eval time (already being exploited).
4. **Tokenizer efficiency** (~0.02-0.05 BPB): BPE-8192 vs BPE-1024 (being explored).

Compression techniques address point 2. The single highest-leverage compression improvement is **replacing zstd with a weight-distribution-aware entropy coder**. If that saves 2MB, you can add 2 more transformer layers (~0.01-0.02 BPB) or double the n-gram table size (~0.01 BPB).

---

## Quick Reference: Compression Ratios at a Glance

| Technique | Typical Ratio | Quality Impact | Post-Training? | Competition Fit |
|-----------|--------------|----------------|----------------|-----------------|
| SVD (rank k/n) | 2-6x per matrix | 30-80% ppl increase | Yes, lossy | Low |
| Pruning (50-90%) | 2-10x | 1-50% ppl increase | Partial (SparseGPT) | Low |
| Kronecker | 5-20x per matrix | 30-50% ppl increase | No (needs training) | Medium |
| Product Quantization | 4-16x | 5-35% ppl increase | Yes | Medium |
| Learned Codec (NWC) | Similar to GPTQ | Near-zero at 4bit | Yes | Low (overhead) |
| Entropy Coding (ANS) | 10-25% over zstd | Zero (lossless) | Yes | **HIGH** |
| HashedNets | 8-100x | Variable | No (needs training) | Medium |
| Soft Weight Sharing | 10-160x | Near-zero to moderate | Partial | Low (subsumed) |

---

## Sources

- [SVD-LLM: Truncation-aware SVD (ICLR 2025)](https://arxiv.org/abs/2403.07378)
- [SVD-LLM GitHub](https://github.com/AIoT-MLSys-Lab/SVD-LLM)
- [Low-Rank Matrix Approximation for NN Compression](https://arxiv.org/abs/2504.20078)
- [Krony-PT: GPT-2 Compressed with Kronecker Products](https://arxiv.org/abs/2412.12351)
- [Kronecker Decomposition for GPT Compression (NeurIPS 2021 Workshop)](https://neurips2021-nlp.github.io/papers/35/CameraReady/KroneckerGPT.pdf)
- [Memory-Efficient Generative Models via Product Quantization (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Shao_Memory-Efficient_Generative_Models_via_Product_Quantization_ICCV_2025_paper.pdf)
- [AQLM: Extreme Compression via Additive Quantization](https://arxiv.org/abs/2401.06118)
- [Neural Weight Compression for Language Models](https://arxiv.org/abs/2510.11234)
- [Reducing Storage via Rate-Constrained Quantization and Entropy Coding](https://arxiv.org/html/2505.18758)
- [DeepCABAC: Context-adaptive Arithmetic Coding for DNNs](https://arxiv.org/abs/1905.08318)
- [Deep Compression: Pruning, Quantization, Huffman (ICLR 2016 Best Paper)](https://arxiv.org/abs/1510.00149)
- [ZipNN: Lossless Compression for AI Models](https://arxiv.org/abs/2411.05239)
- [SparseGPT: One-Shot Pruning](https://arxiv.org/pdf/2301.00774)
- [HashedNets: Compressing NNs with the Hashing Trick (ICML 2015)](https://arxiv.org/abs/1504.04788)
- [Soft Weight-Sharing for NN Compression (ICLR 2017)](https://arxiv.org/abs/1702.04008)
- [Lottery Ticket Hypothesis (Frankle & Carlin 2018)](https://arxiv.org/abs/1803.03635)
- [OpenAI Parameter Golf GitHub](https://github.com/openai/parameter-golf)
