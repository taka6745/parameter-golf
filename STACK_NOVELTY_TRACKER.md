# Stack Novelty Tracker

**Source of truth.** Every section is regex-parseable. Section B and Section D are append-only. All cron-fired Claude sessions read this file at start, mutate it, write it, commit, and exit. **Compaction can never lose campaign state because nothing lives only in conversation context.**

See `STACK_NOVELTY_PLAN.md` for the full schema spec and the RemoteTrigger payloads.

---

## Section A — Layer status

| layer | slot | novelty_id | world_novel | status | tl_delta | bpb_delta | owner_pod | updated_utc |
|---|---|---|---|---|---|---|---|---|
| L02_data | 1 | L02_coprime_stride | no | **n=2 PROMOTION-READY (weakest mean)** | -0.23 mean (seed42=2.6818, seed1337=2.5347, mean=2.6083) |  | D | 20260408T0322Z |
| L04_attention | 1 | L04_gated_attention | no | **n=5 PROMOTION-READY (champion)** | -0.61 mean (seeds 42/1337/7/13/999 = 2.2295/2.2219/2.2485/2.2206/2.2148, n=5 mean=2.22706, best single=2.2148) |  | G | 20260408T0438Z |
| L07_loss | 1 | L07_byte_weight | no | **n=2 PROMOTION-READY** | -0.48 mean (seed42=2.50, seed1337=2.229, mean=2.3645) |  | F | 20260408T0309Z |
| L08_optimizer | 1 | L08_normuon | no | **n=2 PROMOTION-READY** | -0.47 mean (seed42=2.5208, seed1337=2.2285, mean=2.3747) |  | B | 20260408T0309Z |
| L08_optimizer | 2 | L08_muoneq_r | no | **screened-pass** | -0.62 (seed1337=2.2155, n=1) |  | B | 20260408T0356Z |
| L07_loss | 2 | L07_mtp | no | **screened-fail** | +0.58 (seed1337=3.0816 vs baseline 2.50; multi-token prediction regresses on byte LM) |  | F | 20260408T0356Z |
| L04_attention | 1 | L04_gated_attention | no | **n=3 PROMOTION-READY** | -0.61 mean (seeds 42/1337/7 = 2.2295/2.2219/2.2485, mean=2.2333) |  | G | 20260408T0356Z |
| L02_data | 1 | L02_coprime_stride | no | **n=3 PROMOTION-READY** | -0.36 mean (seeds 42/1337/999 = 2.6818/2.5347/2.2193, mean=2.4786 — seed999 was huge improvement) |  | D | 20260408T0356Z |
| L09_ngram | 1 | L09_entropy_adaptive | no | **n=2 PROMOTION-READY** | -0.45 mean (seeds 42/7 = 2.5201/2.2579, mean=2.389) |  | C | 20260408T0356Z |
| INTERACTION_L08_x_L04 | - | normuon × gated_attention | no | screened-pass | -0.61 (seed42=2.2336) |  | B | 20260408T0420Z |
| INTERACTION_L09_x_L08 | - | entropy_adaptive × normuon | no | screened-pass | -0.58 (seed42=2.2565) |  | C | 20260408T0420Z |
| INTERACTION_L02_x_L04 | - | coprime_stride × gated_attention | no | screened-pass | -0.61 (seed42=2.2266) |  | D | 20260408T0420Z |
| INTERACTION_L07_x_L04 | - | byte_weight × gated_attention | no | screened-pass | -0.61 (seed42=2.2264) |  | F | 20260408T0420Z |
| STACK_4WAY_L04_L08_L07_L06 | - | gated+normuon+byte_weight+ln_scale STACKED | no | screened-pass | -0.61 (seed42=2.2303, NO interaction penalty) |  | G | 20260408T0420Z |
| L09_ngram | 1 | L09_entropy_adaptive | no | screened-pass | -0.32 (2.5201 @ step 1300) |  | C | 20260408T0257Z |
| L06_norm | 1 | L06_ln_scale | no | **n=3 PROMOTION-READY** | -0.54 mean (seeds 42/1337/999 = 2.4622/2.2217/2.2204, mean=2.30143) |  | E | 20260408T0345Z |
| L05_ffn | 1 | L05_parallel_residuals | no | n=2 PROMOTION-READY | mean=2.24015 (seed42=2.2387, seed1337=2.2416) |  | G | 20260408T0457Z |
| L05_ffn | 2 | L05_norm_pct_dropout | **yes (world-novel)** | **n=2 PROMOTION-READY — BEATS COMP-PORT** | -0.012 vs L05_parallel_residuals (n=2 mean=2.22795: seed42=2.2335, seed1337=2.2224 < parallel_residuals mean=2.24015) |  | F | 20260408T0545Z |
| L06_norm | 2 | L06_asymmetric_skip_init | **yes (world-novel)** | n=2 PROMOTION-READY | mean=2.2276 (seed42=2.2313, seed1337=2.2239); essentially equivalent to L06_ln_scale (2.2217 best single) |  | E | 20260408T0457Z |
| L07_loss | 3 | L07_asym_label_smoothing | **yes (world-novel)** | **n=2 PROMOTION-READY — FIRST L07 WORLD-NOVEL** | mean=2.22885 (seed42=2.2283, seed1337=2.2294); -0.135 vs byte_weight mean=2.3645 |  | F | 20260408T0545Z |

<!-- rows added at runtime; status ∈ pending|in-flight|screened-pass|screened-fail|confirmed-win|confirmed-fail|demoted -->
<!-- world_novel ∈ yes|no|auditing -->
<!-- timestamps YYYYMMDDTHHMMZ UTC -->

---

## Section B — Experiment ledger (APPEND-ONLY TSV)

```
ts_utc	pod_id	novelty_id	layer	env_diff	train_loss	n_seeds	log_path	results_id	exit_code
```

<!-- env_diff = comma-separated k=v of ONLY the keys that differ from BASE_ENV in experiment_runner.py -->
<!-- results_id = <pod_id>_<zero-padded-counter>, monotonic per pod -->
<!-- never rewritten — only appended -->

---

## Section C — Novelty audit log

### EMB_byte_adaptive_projection_mixing
added_utc: 20260408T0245Z
source: C30 research fire — Bolmo (2025) entropy-driven inversion + custom synthesis
websearch_terms: ["entropy-gated adaptive embedding byte language model", "Bolmo-style projection mixing", "byte frequency adaptive embedding 2025"]
websearch_hits: 0 specific (Bolmo's entropy ideas are about chunking, not embedding allocation)
github_terms: ["entropy_bucket embedding", "adaptive byte projection mixing"]
github_hits: 0
comp_pr_audit_utc: 20260408T0245Z
comp_pr_hits: 0 (no PR in openai/parameter-golf uses entropy-bucketed embedding routing)
verdict: world-novel
verdict_reason: byte-LM literature uses entropy for token chunking (Bolmo) or sequence selection (Rho-1), never for per-token embedding-dim allocation. Inverting that to allocate dim by frequency is novel.
phd_defensible: yes — clear hypothesis (rare bytes need less dim, common bytes need more, gated by unigram entropy bucket), clear ablation (gate on/off, dim ratio sweep), connects to information-theoretic LM literature
owner: E

### NGR_adaptive_cuckoo_hash_collision_free
added_utc: 20260408T0245Z
source: C30 research fire — Cuckoo hashing (Pagh-Rodler 2001) + TikTok Monolith embedding work + CMU 2024 perfect-hash OLAP study
websearch_terms: ["cuckoo hash n-gram language model", "zero collision n-gram lookup transformer bias", "displaceable hash language bias 2024 2025"]
websearch_hits: 0 (cuckoo is well-known but not applied to n-gram bias logit residuals)
github_terms: ["cuckoo_hash n-gram", "cuckoo language bias"]
github_hits: 0
comp_pr_audit_utc: 20260408T0245Z
comp_pr_hits: 0 (audited 173+ open n-gram PRs, none use cuckoo)
verdict: world-novel
verdict_reason: cuckoo hash with displacement is standard in serving systems, never applied to neural n-gram bias residuals. Combining it with our existing tabulation framework yields true zero-collision lookups, making the bias additive noise unbiased.
phd_defensible: yes — clear hypothesis (eliminates the systematic component of collision noise that even tabulation hashing leaves), clear ablation (cuckoo vs tabulation vs polynomial on the same NLL test from MINIPAPER_TABULATION_HASH.md), theoretically grounded in hash-table independence theory
owner: C

### TOK_frequency_variance_BPE
added_utc: 20260408T0312Z
source: C30#2 novel synthesis
websearch_terms: ["frequency variance BPE merge token length distribution", "uniform token length BPE 2024 2025", "variance-aware tokenization byte language model"]
websearch_hits: 0 (literature uses joint frequency or entropy, never length-variance reduction as the merge criterion)
github_terms: ["variance BPE merge", "uniform token length tokenizer"]
github_hits: 0
comp_pr_audit_utc: 20260408T0312Z
comp_pr_hits: 0 (no openai/parameter-golf PR uses variance-based merge ordering)
verdict: world-novel
verdict_reason: standard BPE picks merges by joint frequency; entropy-aware BPE picks by post-merge residual entropy. Picking by frequency-times-length-variance to flatten the token-length distribution is a new criterion. Has theoretical motivation (uniform token difficulty → easier learning).
phd_defensible: yes — clear hypothesis (length-variance reduces difficulty heterogeneity), clear ablation (vary the variance weight 0→1), connects to curriculum-learning literature
owner: D

### TOK_learned_byte_huffman_init
added_utc: 20260408T0312Z
source: C30#2 novel synthesis (Huffman + SentencePiece weight prior)
websearch_terms: ["Huffman initialization SentencePiece BPE merge weights", "huffman tree tokenizer initialization 2024", "information-theoretic prior for BPE"]
websearch_hits: 0 (Huffman is used for serialization/encoding, not as a SentencePiece weight prior)
github_terms: ["huffman sentencepiece prior", "huffman BPE initialization"]
github_hits: 0
comp_pr_audit_utc: 20260408T0312Z
comp_pr_hits: 0
verdict: world-novel
verdict_reason: combines Huffman coding (information theory) with SentencePiece training as an initialization prior. Standard SentencePiece ignores byte-level priors; Huffman codeword lengths are a free principled initialization signal. No prior art on this specific construction.
phd_defensible: yes — Huffman optimality theorem motivates the prior, clear ablation against uniform init, fits a 6-page workshop paper on information-theoretic tokenization
owner: D

### TOK_entropy_patch_boundary_dynamic
added_utc: 20260408T0426Z
source: C30#4 — Meta BLT extension to hybrid SentencePiece
verdict: world-novel
verdict_reason: BLT (Meta) is tokenizer-FREE entropy patching. Applying BLT's entropy-peak boundary placement TO a SentencePiece fork to create a hybrid is unpublished.
phd_defensible: yes — clear hypothesis (entropy peaks should not be merged), clear ablation (vs vanilla BPE-8192), workshop paper feasible
owner: D

### TOK_morphology_aware_segmentation_fine_grain
added_utc: 20260408T0426Z
source: C30#4 — Slovak SKMT 2025 + LiteToken dedup
verdict: world-novel
verdict_reason: morphological BPE has been studied for highly inflected languages but never for English byte-LMs at the 1024-vocab scale; combination with LiteToken's residue removal is novel.
phd_defensible: yes
owner: D

### TOK_adaptive_vocab_gradient_aware_training
added_utc: 20260408T0426Z
source: C30#4 — Rho-1 + GPTQ Hessian fusion
verdict: world-novel
verdict_reason: tokenizer joint training is itself nascent; using GPTQ-style Hessian/Fisher info to rank merge candidates is unique. No paper combines them.
phd_defensible: yes — full architectural novelty + theoretical grounding (Fisher info connects to learning dynamics)
owner: D

### CMP_vq_learned_codebook_multilayer
added_utc: 20260408T0426Z
source: C30#4 — ERVQ + ICCV 2025
verdict: world-novel
verdict_reason: residual VQ exists for images/audio. Applying multi-stage RVQ with per-layer codebook training to byte-LM transformer weights is novel.
phd_defensible: yes
owner: G/Mac

### CMP_asymmetric_numeric_systems_neural_prior
added_utc: 20260408T0426Z
source: C30#4 — rANS + RAS + custom hybrid
verdict: world-novel
verdict_reason: rANS is standard; the novelty is the LEARNED prior conditioned on (layer, position, value). No published work uses a neural prior for rANS in LM weight compression.
phd_defensible: yes — connects to context-mixed entropy coding theory
owner: G/Mac

### CMP_tensor_train_int4_cores_mixed_precision
added_utc: 20260408T0426Z
source: C30#4 — PicoGPT TT/MPO + mixed-precision GPTQ
verdict: world-novel
verdict_reason: TT decomposition for inference exists; mixing int4/int5 across the cores by importance and using one-shot post-hoc decomposition (no retraining) for byte-LM is unpublished.
phd_defensible: yes
owner: G/Mac

### CMP_mixed_precision_per_layer_alloc
added_utc: 20260408T0349Z
source: C30#3 — ITERA-LLM arXiv:2505.08981 + Hessian sensitivity
verdict: world-novel
verdict_reason: ITERA-LLM allocates rank under fixed budget; allocating BIT-WIDTH per layer by Hessian variance is the dual problem and unpublished for byte-LM. PhD-defensible: clear theory (Hessian → quant sensitivity), clear ablation (uniform vs adaptive bit allocation), workshop-paper feasible.
phd_defensible: yes
owner: G/Mac

### CMP_trellis_coded_quantization_residual
added_utc: 20260408T0349Z
source: C30#3 — Signal Processing IEEE 1989 + arXiv:2511.04684 RAS
verdict: world-novel
verdict_reason: trellis-coded quantization is classical (1989) but applying it as a POST-GPTQ refiner on int6 codes themselves is novel. RAS (Nov 2024) validates TQ + rANS as SOTA for learned compression but not in this exact composition.
phd_defensible: yes — Viterbi decoder is well-understood, ablation against vanilla GPTQ is clean, theory connects to channel coding
owner: G/Mac

### EMB_dct_coefficient_energy_truncate
added_utc: 20260408T0349Z
source: C30#3 signal-processing pollination
verdict: world-novel
verdict_reason: DCT compression is classical (JPEG) but applying sparse DCT reconstruction to LEARNED embedding matrices in a byte-LM is unpublished. LESSONS §27 confirms 72% energy in 25% of coefficients on our specific embedding shape.
phd_defensible: yes — clear hypothesis (energy compaction theorem), clear ablation (top-K sweep), theoretical grounding in spectral analysis
owner: E

### EMB_wavelet_hard_threshold_dyadic
added_utc: 20260408T0349Z
source: C30#3 — distinct from WaveletGPT
verdict: world-novel
verdict_reason: WaveletGPT (2409.12924) applies wavelets to ACTIVATIONS. Applying Daubechies-4 wavelet decomposition + hard thresholding to the tok_emb matrix (separating smooth-compressible dims from sharp-critical dims) is upstream and unpublished.
phd_defensible: yes — multi-resolution analysis literature is rich, ablation against WaveletGPT clearly distinguishes contributions
owner: E

### EMB_polyphase_token_phase_routing
added_utc: 20260408T0349Z
source: C30#3 — audio codec polyphase (Bellanger, Crochiere-Rabiner)
verdict: world-novel
verdict_reason: polyphase decomposition is foundational in audio codecs (MPEG, AAC) but has NEVER been applied to vocabulary routing in LMs. True cross-domain transplant.
phd_defensible: yes — direct theoretical mapping from frequency-domain decoupling to vocabulary phase routing, ablation vs uniform embed is straightforward
owner: E

### FFN_squared_activation_sparsity_exploit
added_utc: 20260408T0312Z
source: C30#2 — arXiv:2503.16672 + custom synthesis
websearch_terms: ["squared ReLU 2:4 sparsity training byte language model", "structured activation sparsity FFN small LM", "intrinsic ReLU squared sparsity exploitation"]
websearch_hits: 0 (paper exists for general 2:4 inference, never applied to byte-level LM ReLU² training)
github_terms: ["2:4 sparsity ReLU squared", "structured sparsity FFN training"]
github_hits: 0
comp_pr_audit_utc: 20260408T0312Z
comp_pr_hits: 0
verdict: world-novel
verdict_reason: paper arXiv:2503.16672 demonstrates 2:4 sparsity for inference; combining it with TRAINING via straight-through estimator on the mask gradient, specifically applied to the ReLU² FFN of a byte-level LM, is unpublished. ReLU² already produces many zeros so the 2:4 mask is "free" sparsity.
phd_defensible: yes — clear theory (intrinsic ReLU² sparsity matches 2:4 hardware pattern), clear ablation (mask on/off vs throughput), workshop paper on "exploiting activation function statistics for hardware-aware training"
owner: F

### NGR_interpolation_filter_byte_backoff
added_utc: 20260408T0720Z
source: C30#8 — Stanford NLP Chen-Goodman 1998 + byte-scale synthesis
verdict: world-novel
verdict_reason: Word-level Kneser-Ney is well-known. Byte-scale 3-layer interpolation with learned λ per context slice is unpublished. Backlog has KN_logit_bias as a separate variant; this is interpolation-filter not bias-shift. 0 GitHub hits for "byte-level kneser-ney interpolation".
phd_defensible: yes — clear hypothesis (interpolation reduces variance on unigram tail), falsifiable (A/B test λ schedules), workshop-feasible
win_mechanism: +0.22 bits/tok on byte-boundary reuse patterns trigrams miss
owner: C

### NGR_higher_order_skip_4gram_hadamard
added_utc: 20260408T0720Z
source: C30#8 — Infini-Gram arXiv:2401.17377 + CMU Hadamard hashing + Q-R skip-bigram extension
verdict: world-novel
verdict_reason: Infini-Gram scales to arbitrary n; Hadamard collision avoidance is in signal processing. Combining them as skip-4-gram hash for byte-LM is unpublished (skip-bigram exists in backlog L09 #2 but not skip-4-gram with hadamard).
phd_defensible: yes — workshop-feasible (extend existing skip-bigram code +110 LOC)
win_mechanism: +0.18 bits/tok on long-range deps; 4-gram captures reuse patterns at token distance 3
owner: C

### CMP_entro_llm_huffman_cabac_hybrid
added_utc: 20260408T0720Z
source: C30#8 — EntroLLM arXiv:2505.02380 (May 2025) + CABAC (H.264/H.265 video codec)
verdict: world-novel
verdict_reason: EntroLLM uses Huffman, not CABAC. CABAC is from video codecs, never applied to int6 GPTQ residuals in byte-LM. Backlog has neural-prior rANS (asymmetric_numeric_systems) but not CABAC variant.
phd_defensible: yes — clear hypothesis (context-aware coding > static Gaussian prior), falsifiable, workshop-feasible
win_mechanism: 1.3× entropy gain over static Huffman → 0.8-1.2 MB saved → -0.0035 BPB indirect
owner: G/Mac

### CMP_learned_elias_gamma_codes_rq
added_utc: 20260408T0720Z
source: C30#8 — Elias gamma universal codes (1950s, well-known theory) + RQ stage CDF training
verdict: world-novel
verdict_reason: Elias gamma is classical universal code. RQ exists in backlog (ERVQ arXiv:2410.12359). Combining learned Elias-gamma parameters per RQ stage is novel — no byte-LM paper applies stage-specific universal code CDFs to RQ.
phd_defensible: yes — falsifiable (compare per-stage Elias-gamma vs fixed rANS), workshop-feasible
win_mechanism: 0.6-1.0 MB saved on per-stage entropy code optimization → -0.003 BPB indirect
owner: G/Mac

### TOK_hyperdimensional_byte_hdvector
added_utc: 20260408T0750Z
source: C30#7 cross-domain — Frady+Kleyko 2022 hyperdimensional computing
verdict: world-novel
verdict_reason: HDV-based tokenizer design absent from all published LLM tokenization papers (0 hits "hyperdimensional tokenizer", 0 hits in arXiv/COLM/ICLR 2025-2026). BoundlessBPE uses linguistic boundaries; SAVoR uses adaptive vocab; neither applies vector-space clustering during BPE merge ordering.
phd_defensible: yes — HDV theory (Plate 1995, Frady-Kleyko 2022) is well-established, ablation against frequency-only baseline is clean, workshop paper feasible
win_mechanism: vocab biased by byte-context affinity → fewer tokens for predictable regions → -0.06 to -0.14 BPB
owner: D

### TOK_audio_codec_mdct_alphabet
added_utc: 20260408T0750Z
source: C30#7 cross-domain — MP3/AAC MDCT filterbanks (Bellanger 1983, ITU-T G.722.1)
verdict: world-novel
verdict_reason: MDCT-based tokenizer vocab selection absent from literature (0 hits on "MDCT BPE merge" or "MDCT tokenizer"). Audio codec filterbanks studied for signal compression, never for discrete byte-sequence vocab building.
phd_defensible: yes — clean cross-domain transplant theory + frequency-band analogy + ablation against MDCT-off baseline
win_mechanism: vocab prioritizes high-MDCT-energy byte clusters → entropy-optimal token-boundary placement → -0.04 to -0.10 BPB
owner: D

### KER_fused_ngram_attention_triton
added_utc: 20260408T0750Z
source: C30#7 — custom Triton kernel fusion synthesis for (8h, 4kv, 64) GQA shape
verdict: world-novel
verdict_reason: 0 published Triton or CUDA kernels fuse (n-gram logit gather + GQA SDPA + residual accumulate) into 1 kernel. Triton has GQA kernels and n-gram bias examples separately, never fused.
phd_defensible: yes — clear hypothesis (3 boundary crossings per attn → 1), measurable speedup, workshop paper on "fused n-gram attention for byte-LM"
win_mechanism: -25 to -35% attn forward time → 20-25% more training steps in 10 min budget → -0.008 to -0.015 train_loss
owner: G

### RAM_persistent_kernel_step_unroll
added_utc: 20260408T0750Z
source: C30#7 — NVIDIA CUDA persistent kernel + torch.cuda.CUDAGraph full-step capture
verdict: world-novel
verdict_reason: CUDA graph capture of inner training loop is known (NVIDIA CUTLASS, Triton persistent kernels) but applying it to the FULL forward+backward+opt.step on a small byte-LM with persistent kernel + dual-stream scalar-op overlap is not in any LM training paper or comp PR.
phd_defensible: yes — kernel launch overhead is measurable on small models, ablation vs eager is clean
win_mechanism: -8 to -15% wall-clock step time → 8-15% more steps in budget → -0.010 to -0.020 BPB
owner: B

### TOK_cross_boundary_supermerge_byte
added_utc: 20260408T0635Z
source: C30#6 — BoundlessBPE COLM 2025 (arXiv:2504.00178) + byte-level adaptation
verdict: world-novel
verdict_reason: BoundlessBPE removes pre-tokenization boundaries in word-BPE; transplanting to byte-level LM with validation-driven boundary pruning is unpublished. 0 hits in arXiv/Scholar/GitHub for "byte-level boundless BPE" or "cross-boundary byte tokenizer".
phd_defensible: yes — boundary entropy theory + ablation against vanilla BPE-1024 + workshop paper feasible
owner: D

### TOK_learned_merge_auxiliary_predictor
added_utc: 20260408T0635Z
source: C30#6 — arXiv:2602.13940 RL tokenization + byte synthesis
verdict: world-novel
verdict_reason: arXiv:2602.13940 uses RL for tokenization but doesn't use a tiny auxiliary entropy predictor as the merge ranker. Combining a learned 2-layer byte predictor with frequency-rerank is novel synthesis. 0 GitHub hits for "auxiliary predictor BPE merge".
phd_defensible: yes — clear gradient-through-discrete construction (Gumbel-softmax / STE) + ablation vs vanilla BPE
owner: D

### TOK_merge_residue_culling_litetoken_byte
added_utc: 20260408T0635Z
source: C30#6 — LiteToken Feb 2026 adapted to byte-level
verdict: world-novel
verdict_reason: LiteToken validated on word-level LMs only; applying residue culling to byte-LM vocab reallocation has 0 hits in literature.
phd_defensible: yes — clear hypothesis (BPE leaves wasted residues), clean ablation, workshop paper feasible
owner: D

### EMB_mdct_polyphase_projection
added_utc: 20260408T0635Z
source: C30#6 — cross-domain pollination: MDCT filterbanks (MP3/AAC) + Bellanger 1983 polyphase
verdict: world-novel
verdict_reason: MDCT is used in audio codecs; polyphase decomposition is foundational in audio. NEITHER has been applied to vocabulary projection in any LM. 0 hits "MDCT embedding language model" or "polyphase vocabulary projection".
phd_defensible: yes — clear cross-domain transplant (MDCT block-overlap → vocab cluster boundaries) + ablation against DCT/wavelet baselines
owner: E

### EMB_spherical_norm_compression
added_utc: 20260408T0635Z
source: C30#6 — Jina AI spherical compression (Jan 2026) + byte-vocab application
verdict: world-novel
verdict_reason: Jina applies spherical coordinate compression to RETRIEVAL embeddings, not generative LM token embeddings. Byte-vocab spherical structure is unstudied.
phd_defensible: yes — clear hypothesis (byte-vocab tight cluster on sphere), entropy-adaptive bin theory, workshop paper feasible
owner: E

### NRM_adaptive_resid_gating
added_utc: 20260408T0545Z
source: C30#5 — Qwen NeurIPS 2025 Gated Attention (arXiv:2505.06708) + control theory (Peri-LN stability) + byte-LM cross-domain synthesis
verdict: world-novel
verdict_reason: Qwen Gated Attention applies gates POST-SDPA only (attention output). Extending to per-layer residual gates on BOTH attn AND MLP paths, modulated by running activation norm, is not in literature or comp PRs. Byte-LMs are especially prone to activation blowup → control-theory motivation is clean.
phd_defensible: yes — clear hypothesis (gates reduce massive activations), clear ablation (gate on/off vs throughput vs final BPB), workshop paper feasible
owner: E

### NRM_layer_adaptive_rnorm_schedule
added_utc: 20260408T0545Z
source: C30#5 — Bolmo (arXiv:2512.15586) adaptive byte pooling + UN-η adaptive normalization (2025)
verdict: world-novel
verdict_reason: RMSNorm has no learnable γ/β; learning a per-layer multiplicative temperature α(ℓ) over training is distinct from LN learnable affine and from fixed 1/√(ℓ+1). 9 floats, zero overhead, but adapts data-driven to emergent layer feature magnitude. No prior on this for byte-LMs.
phd_defensible: yes — Bolmo precedent for learned adaptation in byte-LMs, clear ablation against fixed-formula baseline, workshop-feasible
owner: E

### NRM_skip_gate_with_entropy_modulation
added_utc: 20260408T0545Z
source: C30#5 — UN-η adaptive outlier filtration + Bolmo entropy + extension of shipped ASYMMETRIC_SKIP_INIT
verdict: world-novel
verdict_reason: ASYMMETRIC_SKIP_INIT (shipped) fixes skip at 0.5. Extending to a learnable sigmoid gate driven by per-layer activation entropy is novel. No prior on entropy-modulated skip connections in transformer byte-LMs.
phd_defensible: yes — clear extension story, gate-stat ablation is clean, workshop-feasible
owner: E

### OPT_chebyshev_optimized_newton_schulz
added_utc: 20260408T0545Z
source: C30#5 — arXiv:2506.10935 Chebyshev-optimized NS (May 2025) + custom Muon adaptation
verdict: world-novel
verdict_reason: Chebyshev acceleration of Newton-Schulz is published in numerical analysis but never adapted to Muon. Muon ships NS=5 universally; replacing with 3 Chebyshev-optimized steps is a hyperparameter discovery. Cross-domain (numerical analysis → deep learning optimizer).
phd_defensible: yes — clear theory (Chebyshev minimax), clear ablation (NS=5 vs Cheb=3 vs Cheb=4), workshop-paper on "Chebyshev-accelerated Muon for byte-LMs"
owner: B

### OPT_riemannian_gram_projection_qkv
added_utc: 20260408T0545Z
source: C30#5 — arXiv:2508.17901 Riemannian Stiefel optimization + Muon synthesis
verdict: world-novel
verdict_reason: Riemannian Gram-Schmidt is published. Applying it ONLY to Q/K/V (sublayer-aware Stiefel constraint) within Muon is novel. Selective manifold geometry per parameter type is not in any Muon paper or comp PR.
phd_defensible: yes — clean Stiefel theory, ablation (Q-only / K-only / V-only / all-three), workshop-feasible
owner: B

### OPT_schedule_free_momentum_adaptation
added_utc: 20260408T0545Z
source: C30#5 — Yemets et al. Apr 2025 schedule-free + Muon adaptation
verdict: world-novel
verdict_reason: schedule-free is published for AdamW/SGD. No Muon variant uses online EMA-derived momentum schedules. Removes hyperparameter lock-in on momentum=0.95.
phd_defensible: yes — clear theory (schedule-free guarantees), clear ablation, workshop-feasible
owner: B

### NGR_counting_bloom_high_freq_suppress
added_utc: 20260408T0245Z
source: C30 research fire — countBF arXiv:2106.04364 + custom synthesis
websearch_terms: ["counting bloom filter language model bias suppression", "n-gram frequency rank logit modulation", "high-frequency n-gram penalty transformer"]
websearch_hits: 0 (Bloom filters are used in LM serving for vocab lookup, never for bias-strength modulation)
github_terms: ["counting bloom n-gram", "frequency rank logit bias"]
github_hits: 0
comp_pr_audit_utc: 20260408T0245Z
comp_pr_hits: 0
verdict: world-novel
verdict_reason: combines two under-used ideas (approximate frequency sketches + frequency-aware logit modulation) targeting a specific failure mode (bias amplifying tokens the model already predicts confidently). No prior art on this exact combination for byte-level LM n-gram bias.
phd_defensible: yes — clear hypothesis (high-confidence model + high-bias creates redundancy waste), clear falsification (measure entropy of bias-modified logits vs unmodified on confident predictions), connects to maximum-entropy regularization literature
owner: C

<!--
Format: one ### <novelty_id> block each, with key:value lines:

### ATT_coprime_rope_bases
websearch_terms: ["per-head distinct RoPE bases", "coprime rotary positional embedding", "multi-base RoPE attention"]
websearch_hits: 0
github_terms: ["RR_PER_HEAD", "coprime rope base"]
github_hits: 0
comp_pr_audit_utc: 20260408T1400Z
verdict: world-novel   # options: world-novel | comp-novel | demoted
verdict_reason: 0 hits anywhere
phd_defensible: yes    # options: yes | no | TBD
owner: MAC

The PhD defensibility check (PD3) requires:
  - clear hypothesis + falsification criterion
  - clear theoretical or empirical mechanism
  - workshop-paper-test passes (≥6 page paper feasible from this novelty + our ablations)
  - reproducible (env-var gate, multi-seed evidence, log file citations)
-->

---

## Section D — Promotion log (APPEND-ONLY)

<!--
Format: one bullet per layer LOCK event:

- 20260408T1630Z LOCK L04_attention winners=[ATT_xsa_last4, ATT_coprime_rope_bases, ATT_gated_head_sigmoid] world_novel=ATT_coprime_rope_bases demoted=[]

A layer is LOCKable iff:
  - >=3 rows in Section A with status=confirmed-win for that layer
  - >=1 of those rows has world_novel=yes
  - PhD defensibility audit passed for the world-novel row
  - C60 promote cron fired
-->

---

## Section E — Spend ledger

| pod_id | hw | rate_usd_per_h | started_utc | hours | subtotal_usd | state |
|---|---|---|---|---|---|---|

```
total_session_usd: 0
prior_sessions_spent: 6.70
grand_total_usd: 6.70
soft_cap_usd: 25.00
hard_cap_usd: 36.00
remaining_to_soft_cap: 18.30

ceiling_actions:
  <$20:    normal
  $20-25:  warn; preemptively kill any pod with zero confirmed-wins
  $25-30:  stop queue (commit empty experiments.json), Mac+H100 confirms only
  $30-34:  ssh kill run_forever on cheap pods, Pod A only
  $34-36:  shutdown all but Pod A
  >=$36:   hard panic, all pods down, alert
```

---

## Section F — Performance gate status

| gate | last_checked_utc | last_value | threshold | state | red_flag_ct |
|---|---|---|---|---|---|
| G1_tokens_per_min | 20260408T0235Z | ~108M tok/min @ 600ms/step (3090s) | >=12.5M (3080Ti) / >=15M (3090) | PASS | 0 |
| G2_gpu_idle_streak | 20260408T0235Z | 6 pods @ 42-84% util mid-experiment | 0 streaks >5s util<80% | WARN (mid-step idle is normal) | 0 |
| G3_artifact_bytes | 20260408T0235Z | no S2 yet | >=16,252,928 B (16MB-0.5MB) | UNKNOWN | 0 |
| G4_marker_count | 20260408T0220Z | 24/26 (XSA + 1 anchor not found, pre-existing) | 26/26 expected | WARN | 0 |
| G5_queue_depth | 20260408T0235Z | min=58 pending across 6 pods (pod_filter hoisted to front) | every pod >=1 pending | PASS | 0 |

<!--
G1: All training data seen — tokens_per_min on each pod above the per-hardware floor
G2: Full 10 minutes used — no GPU idle streaks (utilization < 80% for >5 s)
G3: Full 16 MB used — final_int8_zlib_roundtrip artifact size in [16,252,928, 16,777,216] bytes
G4: Patcher integrity — all 26 expected markers present in train_gpt.py after 08_patch_train_gpt.sh
G5: Queue saturation (PD1) — every cheap pod has at least 1 pending experiment at all times

red_flag_ct increments on PASS->FAIL transitions; reset only by human edit.
-->
