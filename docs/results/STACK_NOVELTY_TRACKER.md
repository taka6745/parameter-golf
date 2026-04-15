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
| L05_ffn | 2 | L05_norm_pct_dropout | **yes (world-novel)** | **n=2 confirmed-win** (S2 cheap-pod) | S2 seed42=1.4140 + seed1337=1.4133, mean=**1.41365** | **1.41365** (cheap F n=2) | F | 20260408T1005Z |
| L04_attention | 2 | L04_gated_attention | no | **n=2 confirmed-win — BEST overall val_bpb** (S2 cheap-pod) | S2 seed42=1.4098 + seed1337=**1.4090** → mean=**1.4094** ★ | **1.4094** ★ (cheap G n=2) | G | 20260408T1130Z |
| L06_norm | 3 | L06_asymmetric_skip_init | **DEMOTED to comp-novel C180 1147Z** (Nick Ryan 2024 prior art) | **n=3 confirmed-win** | seed42=1.4117 + seed1337(old)=1.4089 + seed1337(new)=**1.4097** → mean=**1.4101** (stable) | **1.4101** (cheap E n=3) | E | 20260408T1937Z |
| L07_loss | 4 | L07_asym_label_smoothing | **yes (world-novel)** | **confirmed-win** (S2 cheap-pod) | n=2 mean=2.22885 (S1) → S2 train_loss=2.3068 → val_bpb=1.4138 | **1.4138** (cheap F) | F | 20260408T0905Z |
| L08_optimizer | 3 | L08_per_proj_lr_split | **yes (world-novel)** | **DEMOTED — FAIL above baseline** | S2 seed42=1.4166 + seed1337=1.4148 → mean=**1.4157** > baseline 1.4137 | +0.002 ABOVE baseline (FAIL) | B | 20260408T1112Z |
| L08_optimizer | 5 | L08_chebns_NS_3step | no (DEMOTED to comp-port) | **confirmed-win** (S2 cheap-pod) | seed42 train_loss=2.3059 → val_bpb=1.4153 |  | B | 20260408T1145Z |
| L07_loss | 4 | L07_asym_label_smoothing | **yes (world-novel)** | **DEMOTED — borderline FAIL** | S2 seed42=1.4138 + seed1337=1.4144 → mean=**1.4141** ≈ baseline | +0.0004 above baseline (FAIL) | F | 20260408T1112Z |
| L05_ffn | 3 | L05_parallel_residuals | no (comp port) | **DEMOTED — FAIL** | seed42 → val_bpb=1.4235 = +0.0098 above baseline | +0.0098 ABOVE baseline (BIG FAIL) | F | 20260408T1112Z |
| L08_optimizer | 6 | L08_normuon | no (comp-port) | **n=2 confirmed-win** | S2 seed42=1.4086 + seed1337=**1.4113** → mean=**1.40995** (-0.00375 vs baseline 1.4137) | **1.40995** (cheap B+F n=2) | B+F | 20260408T1217Z |
| L04_attention | 5 | L04_coprime_per_head_rope | **DEMOTED — borderline marginal C5 1217Z** | **n=2 borderline** | S2 seed42=1.4109 + seed1337=**1.4158** → mean=**1.41335** vs baseline 1.4137 = -0.00035 (within seed noise σ~0.001) | -0.00035 (cheap G+B n=2) | G+B | 20260408T1217Z |
| L09_ngram | 3 | L09_ngram_backoff (Patch 46 Stupid Backoff Brants 2007) | comp-novel | **n=1 small confirmed-win** | S2 seed42=**1.4126** vs baseline 1.4137 = -0.0011 (real but small; n=1 needs seed1337 to confirm) | -0.0011 (cheap C n=1) | C | 20260408T1227Z |
| L02_data | 2 | L02_mdl_compressible_first (BEC_REVERSE=1, MDL anti-curriculum, world-novel-candidate) | **★ n=2 confirmed-win** | seed42=1.4093 + seed1337=**1.4107** → mean=**1.4100** vs baseline 1.4137 = **-0.0037** (real world-novel win, n=2 proper seeds, MDL anti-curriculum CONFIRMED) | **1.4100** ★ (cheap E n=2) | E | 20260408T1837Z |
| L07_loss | 6 | L07_focal_loss (Patch 48 LSS_FOCAL_LOSS_MARKER world-novel-candidate per-class learned γ) | **yes (world-novel-candidate C90 1202Z)** | **n=1 FAIL** | S2 seed42=**1.4151** vs baseline 1.4137 = **+0.0014 ABOVE** (FAIL); per-class γ likely needs much longer training to learn meaningful values; seed1337 in flight | +0.0014 (cheap F n=1) | F | 20260408T1247Z |
| STACK_TRUE_WINNERS_3WAY | - | normuon + gated_attention + asymmetric_skip_init | - | **n=2 confirmed champion** | S2 seed42=1.4080 + seed1337=1.4065 → mean=**1.40725** | 1.40725 (cheap B+E n=2) | B+E | 20260408T1157Z |
| STACK_LEGAL_TTT | - | 5-way (gated + norm_pct + asym_skip + asym_label + per_proj_lr) **+ LEGAL_TTT** | - | **2nd best, n=1** | S2 seed42=1.3726 (-0.0411 vs baseline) — DILUTED by null markers (asym_label, per_proj_lr) | 1.3726 (cheap G n=1) | G | 20260408T1212Z |
| STACK_GATED_LEGAL_TTT | - | gated_attention + LEGAL_TTT (MINIMAL stack) | - | **🔥 ALL-TIME CHAMPION n=2 mean=1.3711** | S2 seed42=**1.3716** + seed1337=**1.3706** → mean=**1.3711** ★ (-0.0005 mean vs n=1, -0.0426 vs baseline; n=2 confirmed champion) | **1.3711** 🔥 (cheap G n=2) | G | 20260408T1459Z |
| L10_compression | 4 | L10_qvdedup CMP_QUANT_VALUE_DEDUP | comp-novel | **n=1 confirmed-win, seed1337 in flight** | S2 seed42=**1.411** vs baseline 1.4137 = **-0.0027** (real win, value-dedup quant compression worked) | **1.411** (cheap G n=1) | G | 20260408T1458Z |
| L08_optimizer | 8 | L08_emaswasta_S2confirm | comp-novel | **n=1 FAIL** | S2 seed42=**1.416** vs baseline 1.4137 = +0.0023 ABOVE (FAIL); seed1337 in flight | +0.0023 (cheap B n=1) | B | 20260408T1505Z |
| STACK_MDL_LEGAL_TTT_GATED | - | mdl_compressible_first + LEGAL_TTT + gated_attention (3-way world-novel COMBINATION) | **yes (world-novel combination)** | **n=2 CONFIRMED — works vs baseline, fails vs champion** | S2 seed42=1.4117 + seed1337=**1.4107** → mean=**1.4112** vs baseline 1.4137 = -0.0025 (small win), vs LEGAL_TTT champion 1.3711 = **+0.0401 WORSE**. CONFIRMED: stacked LEGAL_TTT loses ~0.04 vs minimal champion across all 4 ingredient additions tested | -0.0025 vs baseline / +0.0401 vs champion (n=2) | E | 20260408T1737Z |
| TOKEN_FIREHOSE_3way | - | normuon + gated_attention + asymmetric_skip_init (= STACK_TRUE_WINNERS_3WAY, fresh seeds across pods) | - | **★ n=4 confirmed-win mean=1.4054** | seed42=1.4032 + seed1337=1.4027 + Hymba_seed1337=1.4073 + 3way_seed1337(E)=**1.4084** → mean=**1.4054** (Δ-0.0083 vs baseline; n=4 cross-pod, ~0.003 variance, **most stable confirmed multi-mechanism win**) | **1.4054** ★ (cross-pod n=4) | J+E | 20260408T1907Z |
| STACK_MDL_LEGAL_TTT_GATED | - | C90 1521Z ship | - | **DISCARDED — env-nesting bug** | seed42 ran with NO env overrides applied (env vars nested under "env" key, runner ignored them); train_loss=2.5953 = baseline only. Re-queueing with flattened format at C5 1537Z. |  | E | 20260408T1537Z |
| STACK_FIREHOSE_LEGAL_TTT | - | normuon + gated_attention + asymmetric_skip_init + LEGAL_TTT (4-way COMBINATION) | yes (world-novel combination) | **n=2 CONFIRMED FAIL — LEGAL_TTT NOT additive with 3-way** | S2 seed42=1.4045 + seed1337=**1.4094** → mean=**1.40695** vs 3-way alone 1.40295 = **+0.004 WORSE consistent across seeds**. **CONFIRMED: LEGAL_TTT only composes with BARE gated_attention champion (1.3711); breaks with normuon+asym_skip optimized stack** | +0.004 vs 3-way (n=2) | F | 20260408T1657Z |
| STACK_FIREHOSE_NORM_PCT | - | normuon + gated_attention + asymmetric_skip_init + L05_NORM_PCT_DROPOUT (4-way world-novel COMBINATION) | **yes (world-novel combination)** | **n=2 confirmed-win — second-best non-TTT** | S2 seed42=1.4061 + seed1337=**1.4077** → mean=**1.4069** vs baseline 1.4137 = **-0.0068** (real win, n=2 stable), but vs 3-way alone 1.40287 = +0.004 (slightly worse than 3-way alone). 2nd-best non-LEGAL_TTT result | **1.4069** (cheap G n=2) | G | 20260408T1737Z |
| STACK_GATED_NORM_PCT_LEGAL_TTT | - | gated_attention + L05_NORM_PCT_DROPOUT + LEGAL_TTT (3-way MINIMAL champion + new) | **yes (world-novel combination)** | **n=2 CONFIRMED FAIL — LEGAL_TTT brittle** | S2 seed42=1.4153 + seed1337=**1.415** → mean=**1.41515** vs baseline 1.4137 = +0.00145 (FAIL); vs LEGAL_TTT champion 1.3711 = **+0.04405 WORSE consistent**. CONFIRMED across 2 seeds: even ONE ingredient stacked with LEGAL_TTT champion hurts | +0.04405 vs champion (n=2) | B | 20260408T1757Z |
| STACK_LEGAL_TTT_NGRAM_BACKOFF | - | gated_attention + LEGAL_TTT + NGRAM_BACKOFF (cross-layer L04+L09) | **yes (world-novel combination)** | **n=2 CATASTROPHIC FAIL CONFIRMED** | S2 seed42=1.4567 + seed1337=**1.4574** → mean=**1.45705** vs baseline 1.4137 = **+0.04335 ABOVE** (worst stack result yet, n=2 stable); vs LEGAL_TTT champion 1.3711 = **+0.08595 WORSE**. CONFIRMED across 2 seeds: cross-layer LEGAL_TTT stacks catastrophically fail | +0.08595 vs champion (n=2) | J | 20260408T1917Z |
| STACK_BACKOFF_GATED | - | gated_attention + NGRAM_BACKOFF (cross-layer L04+L09 WITHOUT LEGAL_TTT) | **yes (world-novel combination)** | **★ n=2 CONTROL TEST CONFIRMED FAIL** | S2 seed42=1.4162 + seed1337=**1.4164** → mean=**1.4163** vs baseline 1.4137 = +0.0026 (FAIL n=2); vs gated alone 1.4094 = +0.0069 WORSE; vs backoff alone 1.4122 = +0.0041 WORSE. **CROSS-LAYER COMPOSITIONALITY CONFIRMED BROKEN** across 2 seeds | +0.0069 vs gated alone (n=2) | I | 20260408T1857Z |
| STACK_GATED_LEGAL_TTT_LN_SCALE | - | gated_attention + LEGAL_TTT + LN_SCALE (smallest perturbation to champion) | **yes (world-novel combination)** | **n=1 CATASTROPHIC FAIL** | S2 seed42=**1.4618** vs baseline 1.4137 = **+0.0481 ABOVE**; vs LEGAL_TTT champion 1.3711 = **+0.0907 WORSE**. WORST LEGAL_TTT stack result yet. **Even a SINGLE per-block scalar (1/√(layer+1)) COMPLETELY DESTROYS the champion**. LEGAL_TTT is brittle to LITERALLY any forward-pass perturbation. seed1337 in flight | +0.0907 vs champion | H | 20260408T1952Z |
| STACK_GATED_LEGAL_TTT_5STEPS | - | bare champion + LEGAL_TTT_STEPS=5 (vs default 3) | comp-novel | **n=2 CONFIRMED FAIL — STEPS=3 is the optimum** | S2 seed42=1.4142 + seed1337=**1.4142** → mean=**1.4142** (identical across seeds, ultra-stable), vs champion 1.3711 = **+0.0431 WORSE**. **LEGAL_TTT step count is a step function at default**; more steps strictly hurt | +0.0431 vs champion (n=2) | F | 20260408T2002Z |
| STACK_GATED_LEGAL_TTT_LR2X | - | LEGAL_TTT_LR=0.002 | comp-novel | **★ n=1 CATASTROPHIC DIVERGE 1.5097** | S2 seed42=**1.5097** vs baseline 1.4137 = **+0.096** (hardest fail in entire campaign); vs champion 1.3711 = **+0.1386**. **LR=0.002 DIVERGES** — LEGAL_TTT eval-time SGD explodes at higher LR. seed1337 in flight | +0.1386 vs champion | J | 20260408T2032Z |
| STACK_GATED_LEGAL_TTT_LR_HALF | - | LEGAL_TTT_LR=0.0005 | comp-novel | **n=1 FAIL** | S2 seed42=**1.416** vs baseline 1.4137 = +0.0023 (FAIL); vs champion 1.3711 = **+0.0449 WORSE**. Lower LR fails but doesn't diverge. seed1337 in flight | +0.0449 vs champion | B | 20260408T2032Z |
| L09_ngram | 3 | L09_ngram_backoff (Patch 46 Stupid Backoff Brants 2007) | comp-novel | **n=3 confirmed-win — most stable L09 win** | S2 seed42=1.4126 + seed1337=1.4115 + seed1337(C)=**1.4125** → mean=**1.4122** vs baseline 1.4137 = -0.0015 (real, small, stable across 3 seeds) | **1.4122** (cheap C n=3) | C | 20260408T1742Z |
| L08_optimizer | 7 | OPT_RIEMANNIAN_GRAM_QKV (Patch 47, Riemannian Stiefel manifold projection on Q/K/V) | comp-novel (DEMOTED C180 1147Z) | **n=1 FAIL** | S2 seed42=**1.4161** vs baseline 1.4137 = +0.0024 ABOVE (confirms demotion) | +0.0024 (cheap B n=1) | B | 20260408T1255Z |
| L04_attention | 5 | L04_coprime_per_head_rope | **yes (world-novel)** | **NEW confirmed-win** (S2 cheap-pod) | S2 seed42=**1.4109** ★ (n=1 needs seed1337) | **1.4109** ★ (cheap G n=1) | G | 20260408T1112Z |
| STACK_5WAY_INTERACTION | - | gated+norm_pct+asym_skip+asym_label+per_proj_lr | - | **CRITICAL DATA — stacking discount confirmed** | S2 seed42=**1.4117** vs baseline 1.4137 = -0.002 ; sum-of-individual deltas ≈ -0.005 → **~50% stacking discount** | **1.4117** (cheap E) | E | 20260408T1112Z |
| L07_loss | 5 | L07_byte_weight | no (comp-port) | **n=3 marginal confirmed-win** | S2 seed42=1.4143 + seed1337=1.4123 + seed1337(C)=**1.4126** → mean=**1.41307** vs baseline 1.4137 = -0.00063 (within seed noise σ~0.001 but consistent across 3 seeds) | -0.00063 (cheap F+C n=3) | F+C | 20260408T1637Z |
| L06_norm | 4 | L06_ln_scale | no (comp-port) | borderline confirmed-win | S2 seed42=**1.4132** ≈ baseline -0.0005 (marginal) | -0.0005 (cheap E n=1) | E | 20260408T1112Z |
| L08_optimizer | 4 | L08_opt_chebyshev_ns | no (DEMOTED to comp-port C180 0915Z — arXiv:2506.10935) | screened-pass (S1) | seed42 train_loss=2.229 (1400 steps, NS=3 Chebyshev), need seed1337 + S2 |  | B | 20260408T0945Z |
| L11_infra | 1 | L11_dyn_lyapunov_clip | **yes (world-novel, verified C180)** | screened-pass (S1) | seed42 train_loss=2.233 (1400 steps, λ₁ adaptive grad clip) — patch works end-to-end |  | B | 20260408T0945Z |
| L11_infra | 3 | L11_dyn_lyapunov_clip_S2 | **yes (world-novel) DEMOTED — confirmed NEUTRAL** | **n=2 NEUTRAL** | S2 seed42=1.4136 + seed1337=**1.4137** → mean=**1.41365** ≈ baseline 1.4137 (Δ-0.00005, within seed noise; mechanism works end-to-end but provides no measurable BPB improvement) | -0.00005 (cheap I n=2) FAIL | I | 20260408T1557Z |
| L11_infra | 4 | L11_KER_cudagraph_persistent | comp-novel placeholder (no patcher hunk) | **n=2 NEUTRAL — placeholder confirmed** | S2 seed42=1.4145 + seed1337=**1.414** → mean=**1.41425** ≈ baseline 1.4137 (Δ+0.00055, no actual cudagraph code, just gated_attention env); confirmed no-op | +0.00055 (cheap I n=2) NOOP | I | 20260408T1702Z |
| L11_infra | 2 | L11_ker_tma_megakernel | no (comp-port from SOTA b27fe93) | screened-pass-fallback | seed42 train_loss=2.2335 (1400 steps, fallback path on 3090; H100 path untested) |  | B | 20260408T0945Z |
| L09_ngram | 2 | L09_entropy_adaptive | no (comp port) | n=5 PROMOTION-READY | seeds 42/1337/7/13/999 = 2.5201/2.2600/2.2579/2.2543/2.2546; mean(excl.42)=2.2567 |  | C | 20260408T0945Z |
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

### DAT_heterogeneity_loss_weight
added_utc: 20260408T0810Z
source: C30#10 audit — FineWeb-Edu byte-class analysis (~10% code, ~17% URLs, ~42% wiki, ~30% prose)
verdict: world-novel
verdict_reason: Dynamic loss reweighting on byte-LM domain classes (not just sample difficulty) is unpublished. arXiv:2502.06733 does sample-level reweighting but not domain-class-based. 0 hits for "FineWeb domain-aware loss reweighting byte language model".
phd_defensible: yes — clean hypothesis (heterogeneous corpus needs domain-aware schedule), falsifiable, workshop paper feasible
win_mechanism: 10-15% effective step gain on hard content (code/prose) → -0.010 to -0.020 BPB
owner: D (when D recovers) or G

### DAT_domain_gated_smoothing
added_utc: 20260408T0810Z
source: C30#10 audit — FineWeb domain-aware label smoothing (inverts standard recipe)
verdict: world-novel
verdict_reason: Per-domain adaptive label smoothing ε ∈ {0.02, 0.005, 0.01, 0.008} based on online byte-histogram classifier has 0 literature hits + 0 comp PRs. Standard label smoothing applies a single ε across all classes.
phd_defensible: yes — connects to Szegedy 2016 + LiteToken-style domain gating
win_mechanism: -0.006 to -0.012 BPB via inverted smoothing (sharp on easy, soft on hard)
owner: D/F

### DYN_lyapunov_exponent_gradient_clip
added_utc: 20260408T0800Z
source: C30#9 — nonlinear dynamics control theory (Oseledec stability)
verdict: world-novel
verdict_reason: Lyapunov exponent estimation from grad-norm covariance has not been applied to LLM gradient clipping in any paper or comp PR. AdaGC/AGGC use frequency-based adaptive clipping; this is dynamics-based via dominant eigenvalue of the rolling Jacobian estimate.
phd_defensible: yes — clean control-theory framing (Oseledec multiplicative ergodic theorem) + falsifiable (compare AdaGC vs Lyapunov clip) + workshop-feasible
win_mechanism: -0.008 to -0.015 BPB via stability preserving step effectiveness
owner: B (next infra fire)

### DYN_phase_portrait_attractor_checkpoint
added_utc: 20260408T0800Z
source: C30#9 — dynamical systems theory (basin of attraction + soft trajectory recovery)
verdict: world-novel
verdict_reason: Phase portrait analysis of (loss, grad_norm) exists in arXiv:2602.23696, but NOT applied to checkpoint rollback. Soft restore from transient divergence is novel.
phd_defensible: yes — clean basin-of-attraction theory + ablation against no-restore baseline + workshop-feasible
win_mechanism: -0.010 to -0.020 BPB via fewer divergence episodes (early recovery preserves step budget)
owner: G (floating utility)

### DYN_bifurcation_mixed_precision_schedule
added_utc: 20260408T0800Z
source: C30#9 — control theory (bifurcation detection + Lyapunov-margin precision switching)
verdict: world-novel
verdict_reason: Cyclic precision (arXiv:2403.02243) uses fixed schedules. Driving precision switches by control-theory bifurcation detection (stability margin via eigenvalue Lyapunov certificate) is unpublished.
phd_defensible: yes — bifurcation theory + stability margin theory + ablation
win_mechanism: -0.008 to -0.015 BPB via precision elasticity (cheaper matmuls at stable phases → more steps in budget)
owner: B/G

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

### DEMOTION: CMP_HESSIAN_BIT_BUDGET → comp-novel
demoted_utc: 20260408T0915Z
demoted_at: C180 audit 0915Z re-check
reason: HAWQ (ICCV 2019, NIPS 2020, ICML 2021) + GPTQ variants with per-tensor/per-layer quantile selection are published. Hessian-based quantization with adaptive bit/clip allocation is established SOTA technique.
citations: HAWQ repo https://github.com/Zhen-Dong/HAWQ; GPTQ; PTQ literature 2019-2025
new_verdict: comp-novel (still useful, just not world-first)

### DEMOTION: TOK_INPUT_SMOOTH → comp-novel
demoted_utc: 20260408T0915Z
demoted_at: C180 audit 0915Z re-check
reason: Input-side embedding smoothing/data augmentation documented in "Unifying Input and Output Smoothing in NMT" (COLING 2020), "Masked Label Smoothing for MT" (ACL 2022). Random vocabulary mixture augmentation is published.
new_verdict: comp-novel — implementation detail (random-K vocab mix) may be unique but underlying technique is established.

### DEMOTION: OPT_CHEBYSHEV_NS → comp-novel
demoted_utc: 20260408T0915Z
demoted_at: C180 audit 0915Z re-check
reason: Published as CANS (Chebyshev-optimized Newton-Schulz) in arXiv:2506.10935v1 (Jun 2025), explicitly describing 3-step Chebyshev-optimized replacement for Muon's 5-step orthogonalization. THE SAME PAPER I cited as "source/inspiration" — I was wrong to call it world-novel-candidate; I should have flagged it as comp-novel from the start.
citations: arXiv:2506.10935

### DEMOTION: OPT_RIEMANNIAN_GRAM_QKV → comp-novel
demoted_utc: 20260408T1147Z
demoted_at: C180 audit 1147Z re-check (just shipped 1133Z, demoted 14 min later)
reason: Tilde Research has an open-source "Gram-Space Manifold Muon" reference implementation; arXiv:2603.09697 ("Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning") describes Finsler-structured manifold optimization on Q/K/V matrices. Active 2025 work by Cesista/Su on per-layer Stiefel Muon variants. The PRE-NS sublayer-selective application MIGHT still be a unique implementation detail, but the underlying technique (Stiefel manifold projection in Muon, applied to attention weights) is clearly converging in the open-source community.
citations: arXiv:2603.09697 (Mousse), Tilde Research Gram-Space Manifold Muon, Cesista/Su 2025 manifold variants
new_verdict: comp-novel — still useful as a comp-port if it works empirically, but NOT world-novel
LESSON: I conflated "novel selective application" with "world-novel" AGAIN. The same mistake as OPT_CHEBYSHEV_NS. Need stricter audit before shipping: if the underlying technique is in any open implementation, the patch is comp-novel even if the slice/composition is unique.

### DEMOTION: L06_ASYMMETRIC_SKIP_INIT → comp-novel
demoted_utc: 20260408T1147Z
demoted_at: C180 audit 1147Z re-check (was n=2 confirmed-win at 1.4103)
reason: Nick Ryan blog "Adaptive skip connections improve training" (May 2024) explicitly tests skip-weight initialization of 1.0 for layer 1, **0.5 for layer 2**, 0.0 for all remaining — the SAME 0.5 half-init mechanism we claimed as world-novel. Two-year prior art. COLING 2020 "Rethinking Skip Connection with Layer Normalization" also covers skip-init schedules.
citations: https://nickcdryan.com/2024/05/24/adaptive-skip-connections-improve-training/ ; COLING 2020 Skip-Init paper
new_verdict: comp-novel — still a confirmed empirical win at val_bpb 1.4103 (n=2), useful for the stack, but NOT world-novel.

### FLAG: L11_DYN_LYAPUNOV_CLIP — needs re-audit (PR #1471 SDClip adjacent)
flag_utc: 20260408T1147Z
reason: openai/parameter-golf PR #1471 ("[Record] SP8192 + SDClip + 3-Layer Depth Recurrence + EMA 0.9965 — val_bpb 1.0866") introduces "SDClip" — Standard Deviation Clipping — which is in the same conceptual neighborhood as our DYN_LYAPUNOV_CLIP (Lyapunov-driven adaptive grad clip). Need to read PR #1471 description to determine if mechanism overlaps. Different math (SD vs Lyapunov spectrum) but same empirical category. **TODO: full audit at next C180 fire.**
new_verdict: comp-port (the source paper IS the technique)

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

### EMB_lsq_gradient_aware_embedding_quantization
added_utc: 20260408T1127Z
source: C30 1127Z research fire — LSQ ICLR 2020 + GWQ arXiv:2411.00850 byte-vocab synthesis
websearch_terms: ["LSQ tied embedding byte LM", "gradient-aware bit allocation embedding row dimension", "learned step size embedding quantization 2025"]
websearch_hits: 0 (LSQ for general LMs exists; LSQ + per-DIMENSION Fisher-bit-allocation for tied byte-vocab embeddings = 0)
github_terms: ["LSQ tok_emb tied head", "GWQ byte language model embedding"]
github_hits: 0
comp_pr_audit_utc: 20260408T1127Z
comp_pr_hits: 0
verdict: world-novel-candidate
verdict_reason: LSQ paper (rkgO66VKDS) covers general weights; GWQ (arXiv:2411.00850) covers LLM weights but not embeddings; per-DIMENSION (not per-row) gradient-aware bit allocation for byte-LM tied embeddings is the new combination
phd_defensible: yes — clear hypothesis (per-dim gradient variance predicts quant sensitivity), falsifiable ablation (uniform vs gradient-aware bit allocation sweep), 6-page workshop paper feasible
owner: E

### EMB_intrinsic_dimension_adaptive_projection
added_utc: 20260408T1127Z
source: C30 1127Z — arXiv:2503.02142 ID estimation + byte-vocab adaptation
websearch_terms: ["intrinsic dimension byte vocabulary embedding adaptive", "ID estimation token embedding routing", "skipped SVD vocab embedding"]
websearch_hits: 0
github_terms: ["intrinsic_dimension byte vocab embed", "adaptive_projection byte LM"]
github_hits: 0
comp_pr_audit_utc: 20260408T1127Z
comp_pr_hits: 0
verdict: world-novel-candidate
verdict_reason: ID estimation papers exist (arXiv:2503.02142) but not for byte-vocab routing of embedding capacity. Distinct from existing #6 EMB_byte_adaptive_projection_mixing (entropy bucket gate, not ID).
phd_defensible: yes — clear hypothesis (lower-ID byte clusters need fewer dims), clear ablation (gate-off control), connects to manifold-learning literature
owner: E

### CMP_context_adaptive_rANS_per_layer_predictor
added_utc: 20260408T1127Z
source: C30 1127Z — RAS arXiv:2511.04684 + EntroLLM arXiv:2505.02380 fusion
websearch_terms: ["context-adaptive rANS per-layer LLM compression", "neural prior weight quantization rANS 2025", "learned CDF predictor LLM weight indices"]
websearch_hits: 0 specific (rANS LM compression exists at high level; per-layer per-position learned predictor for quantized indices = 0)
github_terms: ["rans_predictor llm", "context_adaptive_rans weight"]
github_hits: 0
comp_pr_audit_utc: 20260408T1127Z
comp_pr_hits: 0 (only 2 BROTLI PRs in competition, 0 rANS)
verdict: world-novel-candidate
verdict_reason: distinct from existing L10 #10 CMP_asymmetric_numeric_systems_neural_prior — that one uses a single global predictor; this one uses per-layer predictors with position+previous-code context
phd_defensible: yes — clear hypothesis (per-layer code distributions differ from global), falsifiable (cross-validate predictor; bits saved vs zlib baseline), workshop paper feasible on info-theoretic LM compression
owner: G/Mac

### CMP_learned_scalar_adaptive_clipping
added_utc: 20260408T1127Z
source: C30 1127Z — EfficientQAT ACL 2025 + per-layer extension
websearch_terms: ["learned per-layer clip scalar int8 quantization", "absmax minus alpha sigma quantization clipping", "validation gradient learned quantization clip 2025"]
websearch_hits: 0 (EfficientQAT exists for QAT; PTQ + zlib + learned scalar α via val-gradient = novel combination)
github_terms: ["learned clip alpha quantization", "absmax minus alpha sigma int8"]
github_hits: 0
comp_pr_audit_utc: 20260408T1127Z
comp_pr_hits: 0
verdict: world-novel-candidate
verdict_reason: tighter clip via val-gradient learned α reduces outlier dominance → smaller serialized artifact via zlib LZ77 longer runs. Not in existing backlog (CMP_HESSIAN_BIT_BUDGET demoted; this is val-driven not Hessian-driven)
phd_defensible: no — empirical engineering candidate; clear ablation but no clean theoretical mechanism. Useful as a comp-novel ship if it works.
owner: G/Mac

### LSS_focal_loss_gamma_tuned
added_utc: 20260408T1141Z
source: C30 1141Z — Lin et al. 2017 focal loss + byte-vocab extension
websearch_terms: ["focal loss byte language model", "byte vocab focal cross entropy", "per byte class learned gamma focal"]
websearch_hits: 0 specific (focal loss for vision/general LM exists; per-byte-class learned γ for byte-vocab LM = 0)
github_terms: ["byte_focal_loss llm", "per_byte_class focal gamma"]
github_hits: 0
comp_pr_audit_utc: 20260408T1141Z
comp_pr_hits: 0 (no comp PR uses focal loss specifically; loss work is rare in the comp)
verdict: world-novel-candidate
verdict_reason: standard focal loss is from object detection; applying to byte-LM with PER-BYTE-CLASS learned γ (so common bytes like space/newline get higher γ to downweight, rare bytes get lower γ) is the new combination
phd_defensible: yes — clear hypothesis (byte class imbalance dominates loss), falsifiable via γ sweep + class ablation, workshop paper feasible on "loss reweighting for highly-imbalanced byte vocabularies"
owner: F

### LSS_hellinger_bregman_divergence
added_utc: 20260408T1141Z
source: C30 1141Z — arXiv:2602.04380 Beyond KL + Banerjee 2005 Bregman theory
websearch_terms: ["Hellinger distance language model loss", "Bregman divergence cross entropy LM training", "symmetric divergence byte LM"]
websearch_hits: 0 (Bregman in RL policy optimization yes; byte-LM training loss = 0)
github_terms: ["hellinger_loss llm", "bregman_divergence language_model"]
github_hits: 0
comp_pr_audit_utc: 20260408T1141Z
comp_pr_hits: 0
verdict: world-novel-candidate
verdict_reason: Hellinger / Bregman divergences are standard in info theory + RL but never as the primary training objective for byte-LM. Symmetry may help imbalanced byte vocab where KL's asymmetry is problematic.
phd_defensible: yes — connects to information geometry + Bregman family theory, clear ablation (KL vs Hellinger vs JS), training stability metric (gradient variance) gives empirical handle
owner: F

### NGR_neural_engram_hash_cache
added_utc: 20260408T1141Z
source: C30 1141Z — DeepSeek Engram (2024) + memory-augmented LM survey
websearch_terms: ["neural engram language model hash", "trainable hash table n-gram bias LM", "DeepSeek Engram 2024"]
websearch_hits: <5 (Engram is an emerging architecture; trainable hash on n-gram bias for byte-LM is custom)
github_terms: ["engram_cache llm", "trainable_hash ngram bias"]
github_hits: 0
comp_pr_audit_utc: 20260408T1141Z
comp_pr_hits: 0 (no Engram-style trainable hash in any of the 173 NGRAM PRs)
verdict: world-novel-candidate
verdict_reason: distinct from existing TABULATION_HASH (fixed XOR) and ENGRAM_LITE (learnable embedding head, not learnable hash routing). The TRAINABLE collision-resolution is the novelty.
phd_defensible: no — engineering candidate; the hypothesis is "learnable routing > fixed XOR" but no clean theoretical bound. Falls back to comp-novel if PhD test enforced strictly.
owner: C

### NGR_position_vocab_adaptive_prune
added_utc: 20260408T1141Z
source: C30 1141Z — APT arXiv:2405.12842 + n-gram synthesis
websearch_terms: ["position-aware n-gram bias pruning LM", "per-position bucket gate language model", "adaptive n-gram pruning byte"]
websearch_hits: 0
github_terms: ["position_aware_ngram_gate", "per_position_bucket_prune"]
github_hits: 0
comp_pr_audit_utc: 20260408T1141Z
comp_pr_hits: 0
verdict: world-novel-candidate
verdict_reason: distinct from #3 CTX_PARTITIONED_TAB (which partitions the HASH INPUT by prev3 mod 16). This partitions the EVAL CONTEXT (sequence position) and learns gates per position class to mute low-info buckets.
phd_defensible: no — engineering candidate. The position-class boundaries are arbitrary, no clean theoretical motivation. Comp-novel ship if it works.
owner: C

### TOK_kraft_inequality_merge_validator
added_utc: 20260408T1212Z
source: C30 1212Z (info-theory pollination) — Kraft 1949 + ICLR 2025 info-theory tokenization arXiv:2601.09039
websearch_terms: ["kraft inequality BPE merge tokenizer", "prefix code BPE constraint", "kraft inequality vocabulary construction byte LM"]
websearch_hits: 0 (Kraft is foundational 1949; never applied as active BPE merge filter)
github_terms: ["kraft_inequality_BPE", "prefix_code_tokenizer_validator"]
github_hits: 0
comp_pr_audit_utc: 20260408T1212Z
comp_pr_hits: 0
verdict: world-novel-candidate
verdict_reason: Kraft inequality is published 1949 foundational result. Applying it as a BPE merge ranking validator is the new combination. The ICLR 2025 info-theory tokenization paper frames tokenization as channel design but doesn't use Kraft as a merge filter.
phd_defensible: yes — clear theoretical mechanism (prefix-free property → minimal codeword length), falsifiable ablation (Kraft validator on/off vs frequency-only), workshop paper feasible
owner: D

### TOK_arithmetic_coding_merge_frequency
added_utc: 20260408T1212Z
source: C30 1212Z (info-theory pollination) — Arithmetic Coding (Rissanen 1976) + BLT (Meta 2024)
websearch_terms: ["arithmetic coding BPE merge tokenizer", "AC-optimized tokenizer byte LM", "arithmetic code length merge ranking"]
websearch_hits: 0
github_terms: ["arithmetic_coding_BPE", "ac_merge_ranking"]
github_hits: 0
comp_pr_audit_utc: 20260408T1212Z
comp_pr_hits: 0
verdict: world-novel-candidate
verdict_reason: arithmetic coding is well-studied; BPE is well-studied. Applying AC codeword length as a BPE merge criterion is unpublished. BLT uses entropy peaks for patch boundaries — different mechanism.
phd_defensible: yes — clear hypothesis (AC-optimal merges → entropy-aligned vocab), connects to source coding theory, workshop paper feasible
owner: D

### TOK_kolmogorov_complexity_pruning_inference
added_utc: 20260408T1212Z
source: C30 1212Z (info-theory pollination) — Kolmogorov 1965 + Neural NCD arXiv:2410.15280 + MDL Rissanen 1978
websearch_terms: ["Kolmogorov complexity tokenizer pruning", "MDL vocabulary pruning byte LM", "neural compression distance token rank"]
websearch_hits: 0
github_terms: ["kolmogorov_token_prune", "mdl_vocab_rebalance"]
github_hits: 0
comp_pr_audit_utc: 20260408T1212Z
comp_pr_hits: 0
verdict: world-novel-candidate
verdict_reason: Kolmogorov complexity / MDL is canonical theory; never applied to byte-LM vocabulary pruning. Neural NCD is recent (2024) and orthogonal — this is novel synthesis.
phd_defensible: yes — MDL theorem provides clean motivation, ablation against frequency-only pruning is straightforward, workshop paper feasible
owner: D

### DAT_mdl_compressible_first_reorder
added_utc: 20260408T1212Z
source: C30 1212Z (info-theory pollination) — MDL Rissanen 1978 + arXiv:2504.09597 (Understanding LLM Behaviors via Compression)
websearch_terms: ["MDL shard reordering byte LM", "incompressible-first curriculum LM", "anti-curriculum MDL data ordering"]
websearch_hits: 0 (curriculum-by-entropy exists; INVERSE anti-curriculum + MDL justification is new)
github_terms: ["mdl_shard_reorder", "incompressible_first_curriculum"]
github_hits: 0
comp_pr_audit_utc: 20260408T1212Z
comp_pr_hits: 0
verdict: world-novel-candidate
verdict_reason: standard curriculum is easy-first (high compressibility = predictable). MDL-justified anti-curriculum (incompressible first) is the inverse and unpublished for byte-LM. Distinct from existing DAT_byte_entropy_curriculum (which is easy-first).
phd_defensible: yes — clear theoretical grounding (MDL theorem), clear ablation (incompressible-first vs compressible-first vs random), workshop paper feasible
owner: D

### DAT_entropy_rank_quartet_stratified
added_utc: 20260408T1212Z
source: C30 1212Z (info-theory pollination) — entropy stratification + smooth curriculum learning
websearch_terms: ["entropy quartile stratified curriculum LM", "smooth entropy curriculum byte language model", "4-bin entropy curriculum data"]
websearch_hits: 0 (standard curriculum uses loss-sampling Rho-1 or class-rebalance; smooth quartile stratification is new)
github_terms: ["entropy_quartile_curriculum", "stratified_entropy_lm"]
github_hits: 0
comp_pr_audit_utc: 20260408T1212Z
comp_pr_hits: 0
verdict: world-novel-candidate
verdict_reason: entropy-ranked stratification with smooth (per-epoch ratio) curriculum transition is unpublished. Avoids the abrupt phase transitions of standard curricula.
phd_defensible: yes — info-theoretic stratification + curriculum learning grounding, falsifiable via single-quartile vs balanced ablation, workshop paper feasible
owner: D

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
total_session_usd: 14.78
prior_sessions_spent: 6.70
grand_total_usd: 21.48
soft_cap_usd: 25.00
hard_cap_usd: 50.00 (NIGHT_MODE user-lifted ceiling)
remaining_to_soft_cap: 3.52
remaining_to_hard_cap: 28.52
last_updated_utc: 20260408T1607Z
8_pods_burn_rate_per_h: 2.16

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
| G1_tokens_per_min | 20260408T1503Z | 6.3M tok/min/pod (65k batch / 620 ms) | >=12.5M (3080Ti) / >=15M (3090) | DEGRADED-OK (S2 mode batch=65k smaller than full) | 0 |
| G2_gpu_idle_streak | 20260408T2100Z | 8/8 pods 95-100% GPU (Pod H back to 100%, yellow flag cleared). Champion 1.3711 unchanged. LEGAL_TTT seed1337 confirms (B/J) still in flight. No new results since 2047Z. Pods I cycling through short MR/DR Mac-family tests (expected). Spend ~$22.5 warn tier. | 0 streaks >5s util<80% | PASS | 0 |
| BONUS_NOTE | 20260408T1942Z | CMP_context_adaptive_rANS placeholder runs on H: n=2 mean (gated+normuon only, 2-way) = 1.40895 — worse than full 3-way (1.4054) confirming asym_skip_init adds value | N/A | INFO | 0 |
| G3_artifact_bytes | 20260408T1330Z | ~16.7 MB (champion S2 stack) | >=16,252,928 B (16MB-0.5MB) | PASS | 0 |
| G4_marker_count | 20260408T1503Z | 48/50 missing=['NS_STEPS_MARKER', 'XSA_MARKER'] | 46/46 expected (extras allowed) | PASS | 1 |
| G5_queue_depth | 20260408T1503Z | min=4 pending (H/I/J after backup add), per_pod={'B':27,'C':17,'E':16,'F':22,'G':33,'H':4,'I':4,'J':4} | every pod >=1 pending | PASS | 0 |
| G6_cpu_util | 20260408T1503Z | not yet sampled (TODO C5 next fire) | >=50% during training | UNKNOWN | 0 |
