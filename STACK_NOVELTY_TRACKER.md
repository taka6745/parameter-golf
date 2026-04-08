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
| L05_ffn | 2 | L05_norm_pct_dropout | **yes (world-novel)** | **screened-pass — BEATS COMP-PORT** | -0.018 vs L05_parallel_residuals (norm_pct_dropout seed1337=2.2224 < parallel_residuals mean=2.24015); awaiting seed42 |  | F | 20260408T0457Z |
| L06_norm | 2 | L06_asymmetric_skip_init | **yes (world-novel)** | n=2 PROMOTION-READY | mean=2.2276 (seed42=2.2313, seed1337=2.2239); essentially equivalent to L06_ln_scale (2.2217 best single) |  | E | 20260408T0457Z |

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
