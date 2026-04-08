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
| G1_tokens_per_min |  |  | >=12.5M (3080Ti) / >=15M (3090) | UNKNOWN | 0 |
| G2_gpu_idle_streak |  |  | 0 streaks >5s util<80% | UNKNOWN | 0 |
| G3_artifact_bytes |  |  | >=16,252,928 B (16MB-0.5MB) | UNKNOWN | 0 |
| G4_marker_count |  |  | 26/26 expected | UNKNOWN | 0 |
| G5_queue_depth |  |  | every pod >=1 pending | UNKNOWN | 0 |

<!--
G1: All training data seen — tokens_per_min on each pod above the per-hardware floor
G2: Full 10 minutes used — no GPU idle streaks (utilization < 80% for >5 s)
G3: Full 16 MB used — final_int8_zlib_roundtrip artifact size in [16,252,928, 16,777,216] bytes
G4: Patcher integrity — all 26 expected markers present in train_gpt.py after 08_patch_train_gpt.sh
G5: Queue saturation (PD1) — every cheap pod has at least 1 pending experiment at all times

red_flag_ct increments on PASS->FAIL transitions; reset only by human edit.
-->
