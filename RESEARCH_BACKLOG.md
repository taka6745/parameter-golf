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

---

## L10 — Compression & eval candidates

| priority | name | source | hypothesis | expected_delta | novelty_estimate | code_skeleton_loc | added_utc |
|---|---|---|---|---|---|---|---|
| 1 | CMP_brotli11_wrapper | LESSONS §18 | brotli-11 over LZMA-9 wrapper; saves 1.47 MB on n-gram tables vs zstd-22 | -0.005 BPB indirect | comp-novel | 25 (serializer swap) | 20260408T0000Z |
| 2 | CMP_ar_self_gen_gptq | abaybektursun (top merged 1.1147 record) | autoregressive self-generated calibration data for GPTQ | -0.003 BPB | comp-novel | 100 (calibration loop) | 20260408T0000Z |
| 3 | CMP_per_row_hessian_rans_gptq | Plan-A novel | per-row Hessian-aware rANS coding on GPTQ int6 codes; rANS prior derived from per-row GPTQ Hessian | 0.5-1.0 MB savings → -0.004 BPB indirect | world-novel-candidate | 130 (rANS encoder + Hessian extraction) | 20260408T0000Z |
| 4 | CMP_custom_cuda_int6_dequant_fusion | stretch S1 | custom CUDA kernel for int6 GPTQ dequant + matmul fusion; skips int8 buffer materialization | step time -10 to -25% | comp-novel | 300 (CUDA kernel + bindings) | 20260408T0000Z |
| 5 | CMP_custom_brotli_dictionary | stretch S3 | train a custom Brotli pre-trained dictionary on a corpus of our checkpoints; saves 0.5-1.5 MB | -0.003 BPB indirect | comp-novel | 30 (dict training + serializer flag) | 20260408T0000Z |

---

## Backlog hygiene rules

- C30 cron appends to the BOTTOM of each layer's table.
- C5 cron POPS from the TOP when a pod's queue is empty.
- When a candidate transitions to `screened-fail` in `STACK_NOVELTY_TRACKER.md` Section A, its row in this file should be removed (or moved to a `## Failed candidates` section at the bottom of the layer block).
- Manual user injection: just `git push` a row at the desired priority. The next C5 fire picks it up.
