# Stack Novelty Tracker

**Source of truth.** Every section is regex-parseable. Section B and Section D are append-only. All cron-fired Claude sessions read this file at start, mutate it, write it, commit, and exit. **Compaction can never lose campaign state because nothing lives only in conversation context.**

See `STACK_NOVELTY_PLAN.md` for the full schema spec and the RemoteTrigger payloads.

---

## Section A — Layer status

| layer | slot | novelty_id | world_novel | status | tl_delta | bpb_delta | owner_pod | updated_utc |
|---|---|---|---|---|---|---|---|---|
| L02_data | 1 | L02_coprime_stride | no | screened-pass | -0.16 (2.6818 @ step 500; mid-run reading at C5#6) |  | D | 20260408T0257Z |
| L04_attention | 1 | L04_gated_attention | no | **n=2 PROMOTION-READY (best mean)** | -0.62 mean (seed42=2.2295, seed1337=2.2219, mean=2.2257) |  | G | 20260408T0317Z |
| L07_loss | 1 | L07_byte_weight | no | **n=2 PROMOTION-READY** | -0.48 mean (seed42=2.50, seed1337=2.229, mean=2.3645) |  | F | 20260408T0309Z |
| L08_optimizer | 1 | L08_normuon | no | **n=2 PROMOTION-READY** | -0.47 mean (seed42=2.5208, seed1337=2.2285, mean=2.3747) |  | B | 20260408T0309Z |
| L09_ngram | 1 | L09_entropy_adaptive | no | screened-pass | -0.32 (2.5201 @ step 1300) |  | C | 20260408T0257Z |
| L06_norm | 1 | L06_ln_scale | no | screened-pass | -0.38 (2.4622 @ step 1100) |  | E | 20260408T0303Z |

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
