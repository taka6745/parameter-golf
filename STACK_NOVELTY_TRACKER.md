# Stack Novelty Tracker

**Source of truth.** Every section is regex-parseable. Section B and Section D are append-only. All cron-fired Claude sessions read this file at start, mutate it, write it, commit, and exit. **Compaction can never lose campaign state because nothing lives only in conversation context.**

See `STACK_NOVELTY_PLAN.md` for the full schema spec and the RemoteTrigger payloads.

---

## Section A — Layer status

| layer | slot | novelty_id | world_novel | status | tl_delta | bpb_delta | owner_pod | updated_utc |
|---|---|---|---|---|---|---|---|---|

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
