# PHASE1_NOVELTY_AUDIT.md — comp + world novelty audit (2026-04-09)

**Subject**: train_gpt_phase1.py = decoded reproduction of openai/parameter-golf PR #1477 (val_bpb 1.0822, 8×H100 SXM, 3-seed mean).

**Question**: how comp-novel and how world-novel is this stack?

**Method**: two parallel research agents — one with `gh` CLI access for the comp PR landscape, one with WebFetch/WebSearch for the open literature.

---

## TL;DR

- **Comp-novelty: ZERO**. Out of 16 components, **0 are unique to PR #1477**. Every single one appears in ≥2 other top-15 PRs. Shipping the script as-is lands us at **rank ~8-13** of the current leaderboard.
- **Frontier has moved past us**: PR #1485 (open) is at **1.0679 BPB** = our exact stack + (3-layer recurrence) + (Pre-Quant AdamW TTT) + (EMA 0.9965) + (QK_GAIN_INIT 5). The 0.0143 BPB gap is **14× the 0.005 BPB record bar**.
- **World-novelty**: 0 components are unambiguously world-novel as published. **2 are small-twist NOVEL** (LeakyReLU(0.5)² MLP, sigmoid-gated U-Net skip in causal byte LM), **5 are COMP-NOVEL** (XSA per-KV-group on last N, parallel residuals from layer N + lane_merge, eval-time cosine TTT, AR self-gen GPTQ, Brotli-11 packaging), the remaining **9 are STANDARD**.
- **Highest-leverage delta achievable in <100 LOC**: swap score-first TTT for **Pre-Quant AdamW TTT** + bump **QK_GAIN_INIT 4→5** + add **3rd recurrent layer**. That trio is exactly the gap PR #1477 → PR #1485.

---

## Section A: Top 15 PRs by val_bpb (current leaderboard)

| Rank | PR # | val_bpb | Title (short) | Merged? |
|---:|---:|---:|---|---|
| 1 | **1485** | **1.0679** | SP8192 + 3L Depth Recurrence + Parallel Residuals + EMA + QK5 + Pre-Quant AdamW TTT | OPEN |
| 2 | 1482 | 1.0787 | SP8192 + Pre-Quant TTT (QK 5.25, 8ep, freeze-1) | OPEN |
| 3 | 1423 | 1.0791 | SP8192 + Pre-Quant TTT + QK5 + Depth Recurrence + MuonEq-R | OPEN |
| 4 | 1416 | 1.07948 | SP8192 + Pre-Quant TTT (Stukenov bolt-on) | OPEN |
| 5 | 1408 | 1.0800 | dTTT + BigramHash 3072×112 | OPEN |
| 6 | 1351 | 1.0807 | Discriminative pre-quant TTT (per-block adaptive LR) | OPEN |
| 7 | 1437 | 1.08091 | SP8192 + Parallel Residuals + 3L Recurrence + Token-Only N-gram Tilt | OPEN |
| **8** | **1477** | **1.0822** | **(US) SP8192 + Parallel Residuals + Score-First TTT** | OPEN |
| 9 | 1460 | 1.08269 | SP8192 + TTT + Eval-Time Hash Embedding | OPEN |
| 10 | 1413 | 1.08279 | SP8192 + QK5 + Legal Score-First TTT | OPEN |
| 11 | 1420 | 1.08309 | Triple Loop + Fused Triton/CUTLASS Kernels + Parallel Residuals + N-gram Tilt | OPEN |
| 12 | 1412 | 1.08354 | Parallel Residuals + Hessian-Aware SDClip + Progressive Recurrence | OPEN |
| 13 | 1476 | 1.0842 | SP8192 + QK5 + Legal TTT (1-seed) | OPEN |
| 14 | 1450 | 1.08480 | TMA Megakernel + Triple Loop + Parallel Residuals | OPEN |
| 15 | 1394 | 1.08563 | SP8192 + GPTQ Embeddings + Depth Recurrence + MuonEq-R + SDClip (clarkkev base) | OPEN |
| (ref) | 1019 | 1.1147 | **Merged SOTA** — AR Self-Gen GPTQ + XSA-all + BigramHash | merged |

Every entry rank 1-15 is a descendant of @clarkkev's PR #1394. **PR #1477 = PR #1394 + parallel residuals + score-first TTT**, the 8th waypoint along that lineage.

---

## Section B: Component-by-component classification

| # | Component | Comp-novelty class | World-novelty verdict | Where else (comp) |
|---:|---|---|---|---|
| 1 | 11L / d=512 / 8H/4KV / mlp_mult=4 | STANDARD_BASELINE | STANDARD (GQA Llama-2 lineage) | All clarkkev descendants |
| 2 | SP8192 BPE (FineWeb-Edu) | WIDELY_USED | STANDARD | #1394 origin; #1416, #1423, #1482, #1485, ... |
| 3 | Looped layers (loop 4-5, num_loops=2) | WIDELY_USED | STANDARD mechanism (UT 2018), NICHE config | #1394, #1416, #1423, #1482, #1437. Frontier moved to **3-layer (3,4,5)** in #1485, #1471, #1437 |
| 4 | XSA on last 11 layers | WIDELY_USED | NICHE (Zhai 2603.09078, Mar 2026); per-KV-group + last-N is **COMP-NOVEL** | #478 origin; #1019, #1394, #1413, #1416, ... |
| 5 | Parallel residuals from layer 7 | SHARED_2_5 | STANDARD core (GPT-J/PaLM); "from layer N + learned lane_merge" is borderline-COMP-NOVEL | #1412 origin; #1477, #1437, #1467, #1485, #1450, #1420 |
| 6 | Sigmoid skip-gates (U-Net `lerp` skip) | WIDELY_USED | NICHE base (Hourglass, AU-Net); per-channel sigmoid lerp is **NOVEL composition** | #289 origin; every clarkkev descendant |
| 7 | Logit softcap=30 | WIDELY_USED | STANDARD (Gemma 2, 2024) | Universal in clarkkev lineage |
| 8 | `ln_scale = 1/√(layer+1)` | WIDELY_USED | STANDARD as init (Zhang 1908.11365), runtime form is recent NICHE | #315 origin; universal post-#1019 |
| 9 | `qk_gain_init=4` | SHARED_2_5 | NICHE (DA-Transformer, modded-nanogpt); init=4 is unusual | #1217 origin. **Frontier moved to 5.0 (#1413, #1423, #1485) or 5.25 (#1482) — we are below the curve at 4** |
| 10 | LeakyReLU(0.5)² MLP | WIDELY_USED | **NOVEL composition** (no public source for the LeakyReLU+square combo) | #493 origin; universal in #1019, #1394, ... |
| 11 | Muon row-norm + Newton-Schulz + WD=0.085 | WIDELY_USED | STANDARD in modded-nanogpt era | "MuonEq-R" from #1217; universal |
| 12 | TTT (eval-time, 3 epochs, lr=0.005, momentum 0.9) | SHARED_2_5 | **COMP-NOVEL** form (cosine LR across val chunks) | Same hyperparams in #1413, #1460. **#1416/#1423/#1482/#1485 use Pre-Quant AdamW TTT — strictly better, −0.034 vs our −0.002** |
| 13 | EMA decay 0.997 | WIDELY_USED | STANDARD (Polyak 1990) | Stock #1394 value; **#1421/#1466/#1471 already moved to 0.9965** |
| 14 | Full-Hessian GPTQ int6 + int8 tok_emb | WIDELY_USED | STANDARD algorithm (GPTQ Frantar 2022), int6 + full-H is COMP-NOVEL | #535 lineage; universal. **Calibration source differs**: #1019/#1446/#1467 use AR self-gen, our #1394 lineage uses train-batch forward |
| 15 | Brotli-11 compression | WIDELY_USED | **COMP-NOVEL packaging** (no LM compression paper uses Brotli specifically) | #1394 base |
| 16 | Sliding val, stride 64 | STANDARD_BASELINE | STANDARD eval pattern (Shortformer 2020) | Official Track-A eval, every entry |

**Tally**: STANDARD_BASELINE 2 / WIDELY_USED 11 / SHARED_2_5 3 / **PR1477_ONLY 0**.

---

## Section C: What we're MISSING (ranked by leverage)

| # | Technique | Source PR(s) | Best result | Δ vs us | Effort |
|---:|---|---|---:|---:|---|
| **C1** | **Pre-Quant AdamW TTT** (6-8 epochs, lr=4.5e-4–5e-4, freeze 1-2 blocks, cosine, *before* GPTQ) | #1364/#1306/#1416/#1423/**#1485**/#1482 | 1.0679 | **−0.014** | **Single biggest free delta**. Eval-time TTT (us) gives −0.002; pre-quant TTT bakes into the GPTQ-quantized weights for −0.034 |
| C2 | 3-layer depth recurrence (loop 3,4,5 → 14 virtual) | #1331/#1466/#1471/**#1485**/#1437 | 1.0679 | −0.005 to −0.01 | One env var |
| C3 | QK_GAIN_INIT 5.0/5.25 | #1217/#1413/**#1423**/#1351/#1482 | 1.0787 | −0.001 | Single knob |
| C4 | Discriminative TTT with per-block LR 0.3×→1.0× | #1351/**#1408** | 1.0800 | −0.010 vs flat-LR TTT | ~30 LOC |
| C5 | **AR self-generated GPTQ calibration** (model writes its own 64×2048 calibration) | #1019/#1446/#1467 | rule-defensible | compliance hedge | ~50 LOC |
| C6 | BigramHash 3072×112 | #162/#1019/#1408/#1473/#1410 | 1.0800 (#1408 on SOTA stack) | −0.001 to −0.002 | Cheap |
| C7 | Token-only causal n-gram tilt | #1437/#1420/#1145 | 1.08091 | −0.001 | Already in #1420 kernel family |
| C8 | Triple loop NUM_LOOPS=3 | #1420/#1450 | 1.08309 | ~−0.0025 | One env var |
| C9 | **Triton TMA fused MLP fwd + CUTLASS EVT bwd** | #1420/#1450/#1192 | +10.5% throughput | indirect quality + major INFRA novelty | High LOC, but PD3/G6 mandate |
| C10 | Hessian-Aware SDClip (modulate `c = k·σ·(1+λ(r−1))`, λ≈0.175) | #1412 | 1.08354 | −0.002 | ~10 LOC |
| C11 | Eval-Time Hash Embedding (zero-init embedding instantiated AT eval, trained only by TTT) | #1460 | 1.08269 | −0.0004 marginal but world-novel | Genuinely unique; no other PR creates fresh modules at eval |
| C12 | LatentMask TTT (per-channel sigmoid masks + biases, sign-based optimizer) | #1410 | unknown | unknown | Untested probe |
| C13 | Int4 packed MLP (true nibble packing → 13 physical layers) | #1429 | TBD | architecture multiplier | Risky |
| C14 | Coprime-stride loader | #1184/#1351/#726 | −0.003 BPB on #1351 | −0.003 | Cheap |

**Speed/INFRA novelties (PD3/G6 mandate)**: Triton TMA megakernel (#1420, #1450), CUTLASS EVT backward (#1420), Fused Softcap+CE megakernel (#915), Triton KV-cache eval backend (#1153).

---

## Section D: World-novelty cross-cutting observations

**NOVEL combinations even where individual pieces are standards**:
1. **LeakyReLU(0.5)² (#10)** — `F.leaky_relu(x, 0.5).square()` is a one-line variant of Primer ReLU² that I could not source. **Small-twist NOVEL**.
2. **Sigmoid-gated U-Net skip in causal byte LM (#6)** — AU-Net does U-Net skips for byte LMs but uses concat+projection. Per-channel sigmoid gate via `torch.lerp` is a **NOVEL composition** for this domain.
3. **XSA per-KV-group on last 11 layers (#4)** — Zhai 2603.09078 applies XSA uniformly. Per-KV-group + depth-restricted is COMP-NOVEL relative to the public paper.
4. **Eval-time TTT with cosine across val chunks (#12)** — cosine-LR-across-chunks is COMP-NOVEL relative to public TTT papers.

**Specific parameterizations that are notable**:
- `q_gain init=4` — most learned scales init to 1.0; 4 is unusual (and we're below the leaderboard frontier of 5.0/5.25).
- `int6 GPTQ + full Hessian no-Cholesky` — specific bit-width + algorithm variant.
- `Brotli-11` — non-default compressor for ML models.

**Nearly-novel components that could become world-novel with a small twist**:
- **#8 (ln_scale)**: a *learned* per-layer scale starting from `1/sqrt(layer_idx+1)` would be NOVEL — currently unexplored.
- **#9 (q_gain)**: per-head + per-position scale (e.g. learned table of per-token Q gains, leveraging the n-gram bias infrastructure) would be NOVEL.
- **#10 (LeakyReLU²)**: a *learned* negative-slope per-position would be NOVEL and natural.
- **#16 (AR self-gen GPTQ)**: replacing temperature 0.8 sampling with **rejection sampling on highest-Hessian-value tokens** would be NOVEL — implicit importance sampling for calibration. No prior art found.

---

## Section E: Verdicts

**Comp-novelty**: PR #1477 reproduction is a **pure comp-port with zero PR1477_ONLY components**. Re-submitting it identically lands at rank ~8-13. **Cannot be the final submission as-is**.

**World-novelty**: 0 unambiguously world-novel components. 2 small-twist NOVEL (LeakyReLU², sigmoid lerp gate). 5 COMP-NOVEL (XSA per-KV last-N, depth-gated parallel residuals, cosine TTT across val chunks, AR self-gen GPTQ, Brotli-11).

**Honest framing**: this stack is a high-quality assemblage of recent comp-port techniques with two small-twist NOVEL parameterizations. World-novel claims should be confined to those two unless we add a fundamentally new mechanism on top.

---

## Section F: Recommended next moves

**To get to 1.068 quickly (= gap to PR #1485)**: implement the C1+C2+C3 trio in <100 LOC. All three are public, all three are "port-with-evidence" per CLAUDE.md rule 2.

1. **C1 — Pre-Quant AdamW TTT** (the −0.014 BPB lever): replace eval-time SGD TTT with AdamW TTT before GPTQ. 6-8 epochs, lr=4.5e-4-5e-4, freeze 1-2 blocks, cosine schedule.
2. **C2 — 3-layer depth recurrence**: change `loop_start=4 loop_end=5` → `loop_start=3 loop_end=5`. One env var.
3. **C3 — QK_GAIN_INIT 5**: bump default from 4 → 5. One env var.

**To get a compliance hedge** (in case the comp tightens calibration rules):
4. **C5 — AR self-gen GPTQ** from PR #1019: ~50 LOC. Doesn't move val_bpb but defends against rule changes.

**To get genuine world-novel claims** (research output):
5. Bring our unported world-novel patches from `runpod_tests/chore/08_patch_train_gpt.sh` (NGRAM_BIAS, ENTROPY_ADAPTIVE_NGRAM, TABULATION_HASH, GATED_ATTENTION) onto the #1485-class base. The novelty arbitrage is still open because all our previous validations were on the older `train_gpt.py` SP1024 baseline, not on the SP8192 lineage.

---

**Sources** (full reports cached): `/tmp/comp_pr_audit.md`, `/tmp/world_novelty_audit.md`. Both agents available via SendMessage IDs `ae09d9aa73d841c79` (comp) and `a79cc50b23dffc64e` (world) for follow-up questions.
