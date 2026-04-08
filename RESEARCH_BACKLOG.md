# Research Backlog — candidate pool for the stack-novelty campaign

**Day-1 seed: 30 candidates (3 per layer).** The C30 research cron appends more on every fire. The C5 monitor cron pops from the top of each layer's table when a pod's queue runs dry. Slot #3 in each layer is the world-novel seed.

See `STACK_NOVELTY_PLAN.md` for the per-layer hypothesis details and the world-novel justifications. This file is the **operational queue**, not the design doc.

Schema for every row:
```
| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc |
```

`novelty_estimate` ∈ `world-novel-candidate` | `comp-novel` | `in-comp` | `unknown`. `code_skeleton_loc` is rough lines-of-code for implementation. `added_utc` is when the row was added (so we can age out stale candidates).

**Minimum backlog floor: every layer must have ≥5 untried candidates.** When a layer drops below 5, the next C30 fire prioritizes filling it (3 WebSearches + 1 GitHub search).

---

## L01 — Tokenizer candidates

| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc |
|---|---|---|---|---|---|---|---|
| 1 | TOK_bpe8192_standard | LESSONS §18c | BPE-8192 with frequency merges; Mac claims -0.129 BPB at 500 steps; tables exist on disk but never built for vocab=8192 | -0.05 to -0.13 BPB | comp-novel | 30 (tokenizer swap + table rebuild script) | 20260408T0000Z |
| 2 | TOK_vocab512_compact | inverted §33 logic | smaller vocab → bigram coverage densifies → more bias signal per byte | -0.01 BPB net (after n-gram tables grow) | comp-novel | 25 | 20260408T0000Z |
| 3 | TOK_entropy_aware_bpe_merge | Plan-A WebSearch + extension of LESSONS §18c | merge pairs whose joint distribution has lowest residual joint entropy after merge → maximize compression of bigram surprise into vocab | -0.05 train_loss vs vanilla BPE-8192 | world-novel-candidate | 80 (custom sentencepiece training loop) | 20260408T0000Z |
| 4 | TOK_dynamic_byte_codepoint_merger | C30#2 — ByteFlow arXiv:2603.03583 extension | compute next-byte entropy per position online; place merge boundaries where prediction entropy is highest (hard-to-predict bytes don't merge) | -0.04 to -0.08 BPB | comp-novel | 90 | 20260408T0312Z |
| 5 | TOK_frequency_variance_BPE | C30#2 novel synthesis | merge pairs by frequency × variance ratio: prioritize merges that REDUCE variance in token-length distribution → more uniform token difficulty across vocab | -0.015 BPB | **world-novel-candidate** | 65 | 20260408T0312Z |
| 6 | TOK_learned_byte_huffman_init | C30#2 novel synthesis (Huffman + SentencePiece weight prior) | build Huffman tree on FineWeb byte frequencies; codeword lengths become initialization signal for SentencePiece merge weights → biases vocab toward info-theoretic optimality | -0.02 to -0.05 BPB | **world-novel-candidate** | 75 | 20260408T0312Z |

---

## L02 — Data pipeline candidates

| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc |
|---|---|---|---|---|---|---|---|
| 1 | DAT_coprime_stride_validate | existing Patch 19 | USE_COPRIME_STRIDE=1; coprime-to-seq_len stride decorrelates batches | -0.005 train_loss | comp-novel | 0 (env var only) | 20260408T0000Z |
| 2 | DAT_in_batch_dedup | standard pretrain hygiene | drop seqs whose first 256B already appeared this batch; FineWeb has duplicate boilerplate | -0.005 train_loss | comp-novel | 40 | 20260408T0000Z |
| 3 | DAT_byte_entropy_curriculum | Plan-A novel | order shards low-to-high zstd-ratio; easy bytes first → curriculum → faster early loss drop → more effective steps in 10 min | -0.015 train_loss | world-novel-candidate | 60 (offline shard ranking + ordered loader) | 20260408T0000Z |

---

## L03 — Embedding candidates

| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc |
|---|---|---|---|---|---|---|---|
| 1 | EMB_dual_codebook_lloyd_max | LESSONS §37 | K-means cluster the 1024 embeddings into 64 prototypes + int4 residual; ~1.7 MB freed | -0.005 BPB indirect | comp-novel | 70 | 20260408T0000Z |
| 2 | EMB_factorized_2matmul | LESSONS §22 | (vocab × 64 + 64 × 512) tied head; bypass tied-head constraint, save 1.87 MB | -0.003 BPB after spending freed bytes | comp-novel | 50 | 20260408T0000Z |
| 3 | EMB_hadamard_rotated_tied | Plan-A WebSearch (PALU/HiRA ICLR 2025 extension) | precompose Walsh-Hadamard rotation R into tok_emb; logits unchanged exactly, but quant noise after rotation is uniformly spread → lower int4/int6 GPTQ error | -0.004 BPB at int6, -0.012 at int4 | world-novel-candidate | 90 (Hadamard buffer + GPTQ-aware rotate) | 20260408T0000Z |
| 4 | EMB_poly_fourier_embed | C30 WebSearch — PETE arXiv:2505.02266 (May 2025) | replace learned tok_emb with Fourier polynomial expansion of token IDs ([cos(2πk/V), sin(2πk/V), ...]) + light MLP projection; saves ~0.5 MB → reallocate to n-gram tables | -0.008 BPB indirect | comp-novel | 45 | 20260408T0245Z |
| 5 | EMB_tied_tensorized_tt | C30 WebSearch — arXiv:1901.10787 + arXiv:2401.12819 | TT-decompose tok_emb (1024×512) into ranks [1,16,16,1] and share factors with output projection (dynamic tying); saves ~1.2 MB | -0.006 BPB indirect | comp-novel | 80 | 20260408T0245Z |
| 6 | EMB_byte_adaptive_projection_mixing | C30 novel synthesis — Bolmo (2025) entropy-driven inversion | learned sigmoid gate ϕ(byte_entropy_bucket) routes between (a) compact 512-d learned embedding for common bytes and (b) 256-d Fourier for rare bytes; allocates capacity by frequency | -0.007 BPB | **world-novel-candidate** | 70 | 20260408T0245Z |
| 7 | EMB_dct_coefficient_energy_truncate | C30#3 signal-processing pollination — alphaXiv 2508.00220 | DCT-II on (1024, 512) tok_emb; truncate to top-K coefficients carrying 85% energy; reconstruct sparse → 30-40% size reduction with preserved logit geometry at int6 | -0.006 to -0.012 BPB indirect | **world-novel-candidate** | 70 | 20260408T0349Z |
| 8 | EMB_wavelet_hard_threshold_dyadic | C30#3 SP — Daubechies-4 wavelet + WaveletGPT distinction | wavelet decomposition + hard thresholding on detail coefficients; multi-resolution: smooth dims compressed, sharp dims preserved. Distinct from WaveletGPT which targets activations | -0.004 to -0.010 BPB indirect | **world-novel-candidate** | 95 | 20260408T0349Z |
| 9 | EMB_polyphase_token_phase_routing | C30#3 SP — Bellanger / Crochiere-Rabiner audio codec polyphase | split 1024 vocab into 4 cohorts (ID % 4 → phase); each phase uses (256, 512) sub-embedding; sum outputs → 62% storage reduction. Audio codec technique transplanted to vocabulary routing | -0.008 to -0.015 BPB indirect | **world-novel-candidate** | 85 | 20260408T0349Z |

---

## L04 — Attention candidates

| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc |
|---|---|---|---|---|---|---|---|
| 1 | ATT_gated_attention_validate | Patch 16 (NeurIPS 2025) | USE_GATED_ATTENTION=1 re-validation under fixed batch (existing patch failed under broken batch) | -0.005 train_loss | comp-novel | 0 (env var) | 20260408T0000Z |
| 2 | ATT_muoneq_mousse_depth_revalidate | Patches 17/18/19 | re-validate prior verdicts which were measured under broken batch | TBD | comp-novel | 0 (env vars) | 20260408T0000Z |
| 3 | ATT_coprime_per_head_rope | Plan-A novel | each head uses a different prime base in its 16 partial-RoPE dims, so heads see slightly different positional spectra; reduces head redundancy | -0.008 train_loss | world-novel-candidate | 25 (extend Rotary class) | 20260408T0000Z |
| 4 | ATT_hymba_mamba2_hybrid | LESSONS §28 + PR #852 | Mamba-2 + attention hybrid; claims 85ms/step at 1.1189 BPB on H100; potentially massive throughput | -0.05 BPB | comp-novel | 110 (HymbaAttention class + mamba-ssm install) | 20260408T0000Z |
| 5 | ATT_triton_gqa_kernel | stretch S3 | hand-tuned Triton kernel for our specific GQA shape (8h, 4kv, head_dim=64); F.scaled_dot_product_attention has overhead at small batches | 20-40% step speedup → more steps in budget | comp-novel | 200 (Triton kernel + bindings) | 20260408T0000Z |

---

## L05 — Feedforward candidates

| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc |
|---|---|---|---|---|---|---|---|
| 1 | FFN_parallel_residuals_revalidate | comp has it merged | USE_PARALLEL_RESIDUALS=1; our prior implementation regressed | -0.005 train_loss | comp-novel | 0 (env var) | 20260408T0000Z |
| 2 | FFN_swish_leakyrelu_mix_gate | Plan-A | combine Swish² and LeakyReLU(0.5)² in parallel halves with learned scalar | -0.004 train_loss | comp-novel | 35 | 20260408T0000Z |
| 3 | FFN_norm_percentile_dropout | Plan-A novel | zero out FFN intermediate features whose row-norm is in the top 1%; targets the rare exploding-activation pathway | -0.006 train_loss | world-novel-candidate | 30 | 20260408T0000Z |
| 4 | FFN_polyglu_state_conditional_routing | C30#2 — arXiv:2603.13347 PolyGLU (Mar 2026) | replace fixed ReLU² with learned input-conditioned softmax gate over [ReLU², Swish², LeakyReLU(0.5)²]; 3 parallel projections + single gate, no expert weight duplication | -0.007 train_loss | comp-novel | 45 | 20260408T0312Z |
| 5 | FFN_squared_activation_sparsity_exploit | C30#2 — arXiv:2503.16672 + custom synthesis | exploit intrinsic 2:4 sparsity in ReLU² outputs: zero-mask the 2-of-4 smallest activations per position with STE gradient; 1.3× faster step → more steps in 10 min budget | -0.004 train_loss + 1.3× throughput | **world-novel-candidate** | 65 | 20260408T0312Z |
| 6 | FFN_per_layer_alpha_learnable_activation | C30#2 — LESSONS §2 + adaptive activation lit | per-layer scalar α∈[0.01,0.5] gates ReLU² intensity: act = α·(ReLU(x))² + (1−α)·x; init α=0.1 shallow / 0.3 deep | -0.005 train_loss | comp-novel | 30 (9 floats + elementwise) | 20260408T0312Z |

---

## L06 — Normalization & residuals candidates

| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc |
|---|---|---|---|---|---|---|---|
| 1 | NRM_ln_scale_validate | Patch (in 1.1147 stack) | USE_LN_SCALE=1 — RMSNorm output × 1/√(layer+1) | -0.003 train_loss | comp-novel | 0 (env var) | 20260408T0000Z |
| 2 | NRM_per_layer_residual_scalar | ReZero variant | per-layer learned residual scalar (init 1.0, ≤9 floats) | -0.004 train_loss | comp-novel | 20 | 20260408T0000Z |
| 3 | NRM_asymmetric_skip_init_half | Plan-A novel | self.skip_weights at line 673 defaults to ones; init at 0.5 instead → explicit info bottleneck | -0.006 train_loss | world-novel-candidate | 5 (init constant) | 20260408T0000Z |

---

## L07 — Loss candidates

| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc |
|---|---|---|---|---|---|---|---|
| 1 | LSS_byte_weight_validate | Patch (LESSONS §3b) | USE_BYTE_WEIGHT=1 at proper scale; never validated on H100 stack | -0.003 BPB direct | comp-novel | 0 (env var) | 20260408T0000Z |
| 2 | LSS_mtp_validate | Patch 21 (DeepSeek-V3) | USE_MTP=1 multi-token prediction aux loss | -0.005 train_loss | comp-novel | 0 (env var) | 20260408T0000Z |
| 3 | LSS_asymmetric_label_smoothing | Plan-A novel | ε=0.01 only for tokens whose unigram log-prob > -3; rare tokens get hard targets | -0.004 train_loss | world-novel-candidate | 40 | 20260408T0000Z |

---

## L08 — Optimizer candidates

| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc |
|---|---|---|---|---|---|---|---|
| 1 | OPT_normuon_validate | Patch 25 (Mac claims -0.132 BPB) | USE_NORMUON=1 per-row norm post-NS | -0.01 train_loss | comp-novel | 0 (env var) | 20260408T0000Z |
| 2 | OPT_muoneq_r_validate | Patch 18 | USE_MUONEQ_R=1 row-only norm post-NS | -0.005 train_loss | comp-novel | 0 (env var) | 20260408T0000Z |
| 3 | OPT_per_projection_lr_split | Plan-A novel | split Muon param group so q.weight, k.weight, v.weight get different LRs (currently they share) | -0.005 train_loss | world-novel-candidate | 60 (param group split) | 20260408T0000Z |

---

## L09 — N-gram engine candidates

| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc |
|---|---|---|---|---|---|---|---|
| 1 | NGR_entropy_adaptive_validate | Patch 14 | USE_ENTROPY_ADAPTIVE_NGRAM=1 — model-entropy-gated bias mixing (prior verdict invalid under broken batch) | -0.005 train_loss | comp-novel | 0 (env var) | 20260408T0000Z |
| 2 | NGR_skipbigram_table | LESSONS §31 | skip-bigram table (Q-R trick); +0.28 bits/tok signal | -0.005 BPB | comp-novel | 80 (build script + bias apply) | 20260408T0000Z |
| 3 | NGR_context_partitioned_tabulation | MINIPAPER_TABULATION_HASH extension | use a different tabulation table per (prev3 mod 16) slice → 16× n-gram capacity in same memory budget by partitioning input space at higher-order modulus | -0.008 train_loss | world-novel-candidate | 60 (16 sub-tables + selector) | 20260408T0000Z |
| 4 | NGR_adaptive_cuckoo_hash_collision_free | C30 WebSearch — Cuckoo hashing (Wikipedia) + TikTok Monolith embedding work + CMU 2024 perfect hashing study | replace tabulation XOR with cuckoo hash (two hash functions + displacement table) → true zero collisions on the fly; bias mixing noise becomes purely stochastic instead of systematic | -0.006 BPB | **world-novel-candidate** | 120 | 20260408T0245Z |
| 5 | NGR_kneser_ney_logit_bias | C30 WebSearch — Chen & Goodman 1998, revisited arXiv:1706.07786, applied to logit space | apply modified Kneser-Ney discount (count_distinct_contexts / count_total) to n-gram bias weight at lookup time → context-dependent bias strength, smoothes long-tail backs-off | -0.004 BPB | comp-novel | 45 | 20260408T0245Z |
| 6 | NGR_counting_bloom_high_freq_suppress | C30 WebSearch — countBF arXiv:2106.04364 + custom synthesis | track n-gram bucket frequencies via 4-bit counting Bloom filter (~0.5 MB) and suppress bias for high-freq contexts (logit *= 1 − freq_rank/B) → avoids swamping the model on common patterns it already predicts confidently | -0.005 BPB | **world-novel-candidate** | 85 | 20260408T0245Z |

---

## L10 — Compression & eval candidates

| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc |
|---|---|---|---|---|---|---|---|
| 1 | CMP_brotli11_wrapper | LESSONS §18 | brotli-11 over LZMA-9 wrapper; saves 1.47 MB on n-gram tables vs zstd-22 | -0.005 BPB indirect | comp-novel | 25 (serializer swap) | 20260408T0000Z |
| 2 | CMP_ar_self_gen_gptq | abaybektursun (top merged 1.1147 record) | autoregressive self-generated calibration data for GPTQ | -0.003 BPB | comp-novel | 100 (calibration loop) | 20260408T0000Z |
| 3 | CMP_per_row_hessian_rans_gptq | Plan-A novel | per-row Hessian-aware rANS coding on GPTQ int6 codes; rANS prior derived from per-row GPTQ Hessian | 0.5-1.0 MB savings → -0.004 BPB indirect | world-novel-candidate | 130 (rANS encoder + Hessian extraction) | 20260408T0000Z |
| 4 | CMP_custom_cuda_int6_dequant_fusion | stretch S1 | custom CUDA kernel for int6 GPTQ dequant + matmul fusion; skips int8 buffer materialization | step time -10 to -25% | comp-novel | 300 (CUDA kernel + bindings) | 20260408T0000Z |
| 5 | CMP_custom_brotli_dictionary | stretch S3 | train a custom Brotli pre-trained dictionary on a corpus of our checkpoints; saves 0.5-1.5 MB | -0.003 BPB indirect | comp-novel | 30 (dict training + serializer flag) | 20260408T0000Z |
| 6 | CMP_mixed_precision_per_layer_alloc | C30#3 — ITERA-LLM arXiv:2505.08981 + Hessian sensitivity | use Hessian variance per-layer to allocate mixed precision: high-variance → int6, low-variance → int4; recompute per GPTQ calibration run | -0.006 to -0.010 BPB (1.2-2.0 MB freed) | **world-novel-candidate** | 95 | 20260408T0349Z |
| 7 | CMP_trellis_coded_quantization_residual | C30#3 — Signal Processing IEEE 1989 + arXiv:2511.04684 (RAS accelerator) | apply trellis-coded quantization (Viterbi-decoded) to GPTQ int6 residuals → reduces per-layer quant noise by ~15-25% via lattice-path optimization | -0.004 to -0.008 BPB (0.8-1.6 MB savings) | **world-novel-candidate** | 120 | 20260408T0349Z |
| 8 | CMP_hadamard_pre_rotation_quant | C30#3 — extension of EMB hadamard rotation to dense weights | pre-compose Walsh-Hadamard rotation into MLP/attention weight matrices BEFORE GPTQ; spreads quant noise uniformly across spectrum | -0.003 to -0.007 BPB | comp-novel | 70 | 20260408T0349Z |

---

---

## Speed/throughput world-novel candidates (cross-layer, added 20260408T0349Z)

**Rule**: each speed candidate is tagged to the layer it touches. Promotion gate = step-time reduction ≥15% AND no train_loss regression > 0.005 (n=2 seeds).

| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc | layer |
|---|---|---|---|---|---|---|---|---|
| 1 | SPD_rmsnorm_fused_into_linear | mathematically equivalent, mirrors LayerNorm-fusion in TensorRT | the next linear after RMSNorm absorbs the RMS scale by precomputing `W_fused = W * (1/||x||)`; eliminates the RMSNorm op entirely on the forward path | -8 to -12% step time, 0 quality cost | **world-novel-candidate** for byte-LM application | 60 | 20260408T0349Z | L06 |
| 2 | SPD_ngram_bias_tile_cache | custom synthesis | precompute the top-K n-gram bias tile (most common contexts) into a fp16 cache; skip the gather entirely on cache hits → memory-bound op becomes register-resident | -10 to -20% step time on n-gram-heavy steps | **world-novel-candidate** | 90 | 20260408T0349Z | L09 |
| 3 | SPD_triton_attention_gqa_8h_4kv | Triton + our specific GQA shape (8h, 4kv, head_dim=64) | hand-tuned Triton kernel skips F.scaled_dot_product_attention's general-case overhead at our small batch; fused softmax + dropout + projection | -15 to -25% attention forward time | comp-novel (Triton GQA exists, our specific shape unknown) | 200 | 20260408T0349Z | L04 |
| 4 | SPD_quantized_intermediate_dequant_fused | NVIDIA W8A8 work + custom synthesis | store FFN intermediate as int8, dequant inside the next matmul kernel (no fp16 buffer materialization); cuts FFN memory traffic in half | -10 to -15% FFN step time | **world-novel-candidate** for byte-LM at this scale | 130 | 20260408T0349Z | L05 |
| 5 | SPD_pinned_prefetch_4x_dataloader | PyTorch dataloader best practice taken to extreme | pin host memory + prefetch_factor=4 + 4 worker threads; eliminates data-loader stalls on the GPU | -3 to -8% step time, mainly on small-batch warmup | comp-novel | 25 | 20260408T0349Z | L02 |
| 6 | SPD_torch_compile_dynamic_partial | PyTorch 2.x + careful subgraph selection | use `torch.compile(dynamic=True, fullgraph=False, mode='reduce-overhead')` ONLY on the FFN forward (skip n-gram + attention which break compile); selective compilation to capture safe wins | -5 to -12% step time | comp-novel (general torch.compile is known, this specific subgraph carve-out for our stack is not) | 40 | 20260408T0349Z | L05 |

**3 of these 6 are world-novel-candidate**: `SPD_rmsnorm_fused_into_linear` (L06), `SPD_ngram_bias_tile_cache` (L09), `SPD_quantized_intermediate_dequant_fused` (L05). The C90 build cron should ship them in that priority order (lowest LOC first).

---

## Backlog hygiene rules

- C30 cron appends to the BOTTOM of each layer's table.
- C5 cron POPS from the TOP when a pod's queue is empty.
- When a candidate transitions to `screened-fail` in `STACK_NOVELTY_TRACKER.md` Section A, its row in this file should be removed (or moved to a `## Failed candidates` section at the bottom of the layer block).
- Manual user injection: just `git push` a row at the desired priority. The next C5 fire picks it up.
