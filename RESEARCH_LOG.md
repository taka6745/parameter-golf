# Research Log — Auto-driven by the research cron

## 2026-04-07 21:23 local — Research fire #1, Track B (PR mining)

### Source
`gh api repos/openai/parameter-golf/pulls?state=open&sort=created&direction=desc` — top 20 most-recent open PRs.

### Top recent records (Apr 6-7, 2026)

| PR | Title | Author | val_bpb |
|---|---|---|---|
| #1430 | Per-Sample SLOT + N-gram Order-22 + TTT + LR=0.432 | renqianluo | **0.39642** ⚠ likely review |
| #1437 | SP8192 + Parallel Residuals + 3-Layer Recurrence + Legal N-gram Tilt | dexhunter | **1.07800** |
| #1423 | SP8192 + Pre-Quant TTT + QK-Gain 5.0 + Depth Recurrence + MuonEq-R | aryanbhosale | **1.07910** |
| #1420 | Triple Loop + Fused Kernels + Parallel Residuals + N-gram Tilt | abaybektursun | **1.08014** |
| #1421 | 11L Depth Recurrence + EMA Tuning (0.9965) | X-Abhishek-X | **1.09250** |
| #1435 | 11L Depth Recurrence + BigramHash + EMA 0.9965 | AbhayAnandUCSD | **1.09800** |
| #1427 | LeakyReLU + XSA + PartialRoPE + FA3 | kjahan | **1.19910** |

### Techniques in top records that we DON'T have

| Technique | In # records | Have it? |
|---|---|---|
| **Parallel Residuals** | 1437, 1420, 1425 | ❌ → **PATCHED THIS FIRE (Patch 13)** |
| **Depth Recurrence** | 1421, 1422, 1429, 1435, 1437 | ❌ (LESSONS.md §29 wrongly marked DEAD) |
| **SP-8192 tokenizer** | 1437, 1423, 1431 | ❌ (planned but not built) |
| **MuonEq-R optimizer** | 1423, 1429 | ❌ |
| **Pre-Quant TTT** | 1423, 1430 | ❌ |
| **N-gram "Tilt"** | 1437, 1430, 1420 | ❌ (different from our additive bias?) |
| **EMA 0.9965** (high decay) | 1421, 1435 | ❌ |
| **Mixed INT5/INT6 quant** | 1438, 1425 | ❌ |
| **PartialRoPE + FA3** | 1427 | ❌ |
| **SwiGLU MLP** | 1428 | ❌ (we use relu²) |
| **Codebooks (VQ)** | 1433 | ❌ |
| **Int4-Packed MLP** | 1429 | ❌ |

### LESSONS.md §29 needs reconsideration

LESSONS.md claims "ANY recursion is DEAD under GPTQ quantization (~900x compounding error per 3 cycles)". But **5 of the top 10 recent records use depth recurrence**:
- PR #1421, #1422, #1429, #1435, #1437 — all RECORDS (not non-records)

The "depth recurrence is dead" finding was from 2026 mid/early experiments and may have been beaten by better quantization (mixed precision INT5/INT6 instead of pure GPTQ int6) or by different recurrence patterns (3-layer recurrence vs MoR-style 1-layer × 3).

**Action**: do NOT skip depth recurrence in future experiments. Worth a Patch.

### Action taken this fire

**Implemented Patch 13: USE_PARALLEL_RESIDUALS=1** (4 lines net change to Block.forward).

Anchors on the first 3 lines of Block.forward (def + mix + resid blend) which are invariant under Patch 11 (smear gate). Inserts a parallel branch above the existing serial path. When the env var is set:

```python
attn_in = self.attn_norm(x)
mlp_in = self.mlp_norm(x)
attn_out = self.attn(attn_in)
mlp_out = self.mlp(mlp_in)
x = x + self.attn_scale * attn_out + self.mlp_scale * mlp_out
return x
```

This is the GPT-J / PaLM trick, validated in 3 of the top recent records.

### Experiments queued in this fire

| name | new flag |
|---|---|
| `PR0_parallel_resid_alone` | USE_PARALLEL_RESIDUALS=1 (no other novel) |
| `PR1_parallel_plus_leaky_ng` | + LEAKY_RELU + full ngram |
| `PR2_parallel_plus_full_stack` | + smear + leaky + full ngram |

### Next research fires should investigate

1. **N-gram "Tilt"** — what is it? Different from additive bias. Could be Q-R skip-bigram from RESEARCH.md §31, or a multiplicative scaling, or a learned transformation.
2. **Depth recurrence** with mixed-precision quant (the version that's NOT dead)
3. **MuonEq-R** variant of the Muon optimizer
4. **PartialRoPE + FA3** combo

---

## 2026-04-07 22:00 local — 5 subagents returned, MASSIVE synthesis

### Subagent A (BPE-8192 trainer) — ✅ TOKENIZER ALREADY EXISTS
The exact BPE-8192 tokenizer that produced the -0.129 BPB Mac win is on disk at:
`data/tokenizers/fineweb_8192_bpe.model` (370,908 bytes, SHA dddbf3c4...).
Trained on `data/docs_selected.jsonl` (the project's actual 48GB FineWeb corpus, not a HF parquet sample).
Round-trip validated on `data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin`.
Action: scp to pod, rebuild n-gram tables for VOCAB=8192, run L5-equivalent experiment with the new tokenizer.

### Subagent B (closed/merged PR audit) — TOP 8 records analyzed
Frequency table of techniques in the top 8 merged records (Mar 19-30, 2026):
| technique | count | example PRs |
|---|---|---|
| **SmearGate** | 6/8 (75%) | #1019, #549, #315, #287, #265, #162 |
| **zstd-22 compression** | 5/8 (62%) | #315, #287, #265, #414, #180 |
| **11 layers** | 4/8 (50%) | #1019, #414, #315, #287 |
| **BigramHash 1536-3072** | 4/8 | #1019, #549, #414, #315 |
| **3x MLP** | 3/8 | #287, #265, #162 |
| **Flash Attention 3** | 3/8 | #315, #287, #265 |
| **EMA 0.997** | 4+/8 | #414, #315, #287, #1019 |
| **Partial RoPE 16/64** | 2+/8 | #1019, #315 |
| **XSA all-layer** | 1/8 (UNIQUE!) | #1019 |
| **AR Self-Gen GPTQ** | 1/8 (UNIQUE!) | #1019 |
| **Mixed INT5 MLP / INT6 attn** | 1/8 | #180 |

We have **leaky_relu (1/15 patches)** and that's it from the comp's convergent stack.
The biggest single gap: **EMA 0.997**. The biggest single advantage in the #1 record (PR #1019): **AR Self-Gen GPTQ calibration**.

⚠ Audit also CONFIRMED that LESSONS.md §29 (depth recurrence DEAD) is wrong — 5+ recent records use it with mixed-precision quant.

### Subagent C (N-gram "Tilt" investigation) — FOUND THE DEFINITION
N-gram Tilt is NOT additive bias. It's multiplicative single-token boost:
```
p_tilt(t) = p_model(t) · exp(β · 𝟙[t==hint]) / Z
where Z = 1 + p_model(hint) · (exp(β) - 1)
```
- `hint` = best-guess next token from a causal n-gram cache (orders 8-16 token + 1-3 within-word + word-start bigrams)
- `β` ≈ 1.0 (tunable)
- Built at EVAL TIME from validation prefix only (strict causality)
- ZERO artifact cost — the cache lives in eval RAM
- Used by PRs #1437 (1.078 BPB), #1420 (1.080 BPB), #1430 (under review)
- Original innovation by @abaybektursun, not in any published paper
- **Delta**: -0.0029 to -0.0055 BPB consistently across 5 seeds
- **NOT in any paper, NOT in any of our patches** — this is unique to parameter-golf

### Subagent D (TTT implementation researcher) — FULL PATCH SKETCH PROVIDED
Two TTT variants in SOTA records:
1. **LoRA TTT** (per-doc, rank 8, Adam) — ~-0.037 BPB total
2. **Score-First TTT** (PR #461, all-blocks SGD) — ~-0.0025 BPB, **currently SOTA in PR #549 (1.1194 BPB)**

Score-First TTT pattern:
- Partition val tokens into 32K-token non-overlapping chunks
- For each chunk: SCORE with sliding window (`torch.inference_mode()`, no grads) THEN TRAIN on it (SGD lr=0.002, mom=0.9, 3 epochs)
- Last chunk scored but never trained on (causality)
- All blocks unfrozen
- Cosine LR decay across chunks
- Subagent provided FULL ~80-line implementation as Patch 17 sketch.
- Cost: ~410s on 8xH100, fits in 10 min budget.

### Subagent E (records miner) — Top 5 deep dive
| rank | dir | val_bpb | tokenizer | NL | top trick |
|---|---|---|---|---|---|
| 1 | 2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072 | **1.1147** | sp1024 | 11 | AR Self-Gen GPTQ + XSA-all + BigramHash 3072 |
| 2 | 2026-03-23_LeakyReLU_LegalTTT_ParallelMuon | 1.1194 | sp1024 | 11 | Score-First TTT + LeakyReLU² + Parallel Muon |
| 3 | 2026-03-22_11L_EMA_GPTQ-lite_warmdown3500 | 1.1228 | sp1024 | 11 | GPTQ-lite 5-clip percentile sweep |
| 4 | 2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT | 1.1248 | sp1024 | 11 | Partial RoPE 16/64 + LN Scale 1/√(l+1) |
| 5 | 2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04 | 1.1271 | sp1024 | 11 | XSA last-4 layers (efficient GQA-aware reshape) |

EMA, XSA, Parallel Muon are CONVERGENT best practices. We have NONE of them.

## Action plan from subagent synthesis (in EV order)

1. **scp BPE-8192 to pod + rebuild n-gram tables for VOCAB=8192** — biggest single Mac win, free
2. **Patch 17: EMA 0.997** — in 4+ merged records, ~30 lines
3. **Patch 18: XSA (last-4 layers)** — in 6+ merged records, ~40 lines, GQA-aware reshape
4. **Patch 19: Partial RoPE 16/64** — in 2 merged records, ~10 lines
5. **Patch 20: LN Scale 1/√(layer+1)** — in 2 merged records, ~5 lines (zero params!)
6. **Patch 21: Score-First TTT** — full sketch from subagent, ~80 lines
7. **Patch 22: N-gram Tilt** — full math from subagent, ~80 lines
8. **Patch 23: AR Self-Gen GPTQ calibration** — unique to #1 record, larger but well-defined

## 2026-04-07 22:08 local — Research fire #2, Track A (arxiv) — MTP

### Subagent F (DeepSeek MTP investigator) — pushed Patch 21

DeepSeek-V3 (arxiv:2412.19437) introduced Multi-Token Prediction: K auxiliary heads that
predict tokens i+2, i+3, ..., i+K+1 alongside the main head's i+1 prediction. Loss is
`L = L_main + (λ/D) * sum(L_aux_k)`. Each aux head is a single Transformer Block reusing
the SHARED tied embedding (no extra output head params). DeepSeek used D=1, λ schedule
0.3 → 0.1 over training. Claimed ~0.3 BPB-equivalent improvement at 671B scale.

### Why this might transfer to byte-level small LMs

1. Byte-level has DENSER supervision (1 token ≈ 3.5 bytes) so each step provides more
   gradient signal — MTP exploits this with K auxiliary signals per step.
2. Our regime is COMPUTE-bound, not data-bound. MTP gives more gradient per step at
   the cost of one extra Block forward.
3. Pure auxiliary loss — degrades gracefully (set MTP_LOSS_WEIGHT=0 to disable).

### Audit confirmation: zero open PRs use MTP

Searched 100 open PRs at openai/parameter-golf for "MTP" / "multi-token" / "multitoken".
Zero matches. Truly novel for the competition.

### Patch 21 shipped this fire

- New env vars: USE_MTP, MTP_NUM_HEADS=1, MTP_LOSS_WEIGHT=0.10
- 1 extra Block (for K=1) at the end of GPT.__init__ — adds ~786K params (~5% overhead
  on our 17M baseline) but no new attention heads
- Captures pre-norm hidden state in forward, runs the MTP block on it, normalizes, projects
  via tied embedding, computes shifted-target cross-entropy
- Anchored on stable lines: `_init_weights` def, `final_norm.reshape`, and the cross_entropy
  return — these are invariant under all prior patches
- Idempotent via MTP_MARKER

### Experiments queued (run with auto-pull)

- MTP0_mtp_alone — MTP without n-gram (test isolated effect)
- MTP1_mtp_plus_leaky_ng — MTP + leaky + L4-strong-weights (current best stack + MTP)
- MTP2_mtp_strong_weight — MTP with λ=0.30 (DeepSeek's initial schedule value)

### Falsification criterion

If MTP1 train_loss is within 0.005 of CHAMP_L4 baseline (3.295) at 1500 steps, MTP doesn't
help at our scale. If it's WORSE, MTP is interfering. Either way, set USE_MTP=0 and move on.

User pushback: "I want research level findings, we don't want to be testing shit
people already submitted, we want bleeding edge". Parallel residuals (Patch 13)
is in 3+ existing PRs — that's PORTING, not research.

Real novel ideas grounded in our Mac MLX research week + RESEARCH.md analysis
that are NOT in any open PR I've found:

### Patch 14 (NEW THIS FIRE) — USE_ENTROPY_ADAPTIVE_NGRAM

**TRULY NOVEL.** Use the model's own per-token softmax entropy as a deterministic
gate for the n-gram bias mixing weight. Math:

```
p_i = softmax(logits_i)
H_i = -sum(p_i * log(p_i))
gate_i = H_i / log(V)         # in [0, 1]
logits_i_final = logits_i + gate_i * (w_bi * bigram_bias_i + w_tri * trigram_bias_i + w_four * fourgram_bias_i)
```

Hypothesis: when the model is uncertain (high entropy), trust the n-gram bias;
when it's confident (low entropy), trust itself. Zero learned params, ~4 ops per
token at the output. Different from:
- Mac §32 cmix-style logistic mixing (fixed scalar weights)
- Patch 12 NGRAM_GATE (learned linear, empirically fails: NG1=3.42 vs L5=3.29)
- Adaptive softmax / temperature scaling (scales the whole distribution)

This is a NEW connection: the model's own confidence steering its trust in the
external prior. Pushed in this fire, queued as EA0/EA1/EA2/EA3.

### Top 5 unique-to-us ideas to ship in subsequent fires

| Idea | Source | Patch # |
|---|---|---|
| **Entropy-adaptive n-gram mix** | Novel (this fire) | Patch 14 ✅ |
| **Tabulation hashing** for n-gram tables | RESEARCH.md §38 | Patch 15 (next fire) |
| **Multi-hash count-min sketch** for n-grams | Novel (count-min for log-probs) | Patch 16 |
| **Q-R skip-bigram decomposition** | RESEARCH.md §31 (+0.005 BPB) | Patch 17 |
| **Curriculum n-gram weight decay** | Novel (Mac always used fixed) | Patch 18 |

### Why these satisfy the constraints

- **Novel**: none of the recent top PRs (mined Apr 7) use any of these
- **Mac-grounded** or **theoretically grounded** (count-min sketch is a published technique adapted to log-prob tables)
- **Scales**: all are forward-pass changes that work the same at any model size
- **Don't break BPB**: at worst they degrade to baseline (entropy gate → 1.0 if model is uniform)

## 2026-04-07 22:32 local — Monitor #3, MTP first result IN

MTP1_mtp_plus_leaky_ng = **3.2923** (rank 5 overall) with USE_MTP=1, MTP_NUM_HEADS=1,
MTP_LOSS_WEIGHT=0.10 stacked on the L4 best config. Compared to L4_leaky_strong_weights
(3.2947) at the same n-gram weights, MTP is +0.0003 BETTER at single seed.

Within seed noise (~0.05) so not statistically meaningful yet, but **the hypothesis is
NOT falsified** — DeepSeek MTP MAY transfer to byte-level small LMs at our scale.
Strong enough to justify multi-seed validation.

ACTION (next research fire): queue MTP1 with seeds 42, 999, and a higher-K variant
(MTP_NUM_HEADS=2). Compare 3-seed mean against L4 3-seed mean (3.345).

OTHER results from this monitor:
- EA1_entropy_plus_leaky = 3.4490 — entropy-adaptive ngram is a NET LOSS (worse than
  L1=3.33 by 0.12). **Patch 14 hypothesis falsified at scale**. Mark for SKIP in
  final stack.
- MTP0_mtp_alone = 3.7428 — MTP without n-gram, near baseline. Aux loss alone doesn't
  help; needs to stack.
- PR family (parallel residuals) is a confirmed dead-end at our scale: PR1=3.5678,
  PR2=3.5629, PR3=3.5836. All ~3.57. Not a win.


---

## Research Fire #3 — 2026-04-07 (cron min :38, Track B = comp PRs)

**Subject**: Deep-dive PR #1440 (EngramLite) and PR #1382 (Mamba-2 hybrid).

### EngramLite (PR #1440) — VERDICT: NOVEL + ACTIONABLE → SHIPPED Patch 22

PR title: "[Submission] EngramLite + Mousse + Progressive Depth Recurrence + TTT" — claimed val_bpb 1.1026 single seed, with EngramLite alone attributed -0.003 BPB delta.

**What it is**: A learnable hash-embedding n-gram head that runs in parallel with our static log-prob bias. For each input position, hash the bigram and trigram contexts into 3072 buckets, look up a 112-dim embedding, project to vocab, gate with sigmoid. Adds the result to the main logits BEFORE softcap. Init proj.weight to 0 so the head starts as a no-op and the gates open during training.

**Why it's different from our Patch 6**: Patch 6 NGRAM_BIAS uses STATIC log-prob tables built offline from training data (frozen, no gradients). EngramLite learns its parameters during training and the sigmoid gates let the model decide *how much* of the n-gram signal to use per layer. Stacks with Patch 6: static bias gives the data-grounded prior, EngramLite adds a learnable correction.

**Cost**: ~460KB params at sp1024 (3072 × 112 + 112 × 1024 ≈ 458752 floats × 1 byte fp8 = ~460KB), well within 16MB budget. Forward cost is one embedding lookup + one linear projection per call, negligible.

**Implementation**: Patch 22 added EngramLiteHead class (32 lines) before the GPT class definition, init in GPT.__init__ anchored on the MTP init block (Patch 21 anchor still stable), apply side anchored on the softcap line. Idempotent via ENGRAM_LITE_MARKER.

**Experiments queued**:
- EL0_engram_lite_alone — solo (USE_NGRAM_BIAS=0), measure pure EngramLite delta vs no-bias baseline
- EL1_engram_lite_plus_static_ng — stack with Patch 6 static n-gram bias, measure additivity
- EL2_engram_lite_seed42 — same as EL1 with seed 42, multi-seed validation

### Mamba-2 hybrid (PR #1382) — VERDICT: DEFER

PR has 1300+ lines of mamba-ssm + causal-conv1d external dependencies, no GPU validation in PR body, would require building external CUDA kernels on the pod. Risk/return ratio bad given we have 5 ideas in flight already. Not pursuing.

### MTP follow-up experiments queued (from research fire #2 result)

Patch 21 MTP first-try result (MTP1_mtp_plus_leaky_ng=3.2923, rank 5) is within seed noise of L4_leaky_strong_weights=3.2947. To distinguish noise from signal, queued:
- MTP1_seed42_validation — seed 42 with MTP1 config
- MTP1_seed999_validation — seed 999 with MTP1 config
- MTP3_two_heads — MTP_NUM_HEADS=2 (DeepSeek-V3 default), measure if more lookahead helps

Total experiment queue is now 37. Loop pick_next will get to these as crashes/duplicates settle.

### Audit against original wishes

User said: "stop hypertuning, find NOVEL ideas not in the comp" and "PhD level / top 0.001% / novel-to-comp-AND-world."

EngramLite is a comp-PR import (PR #1440), not novel-to-world. But it generalizes our Patch 6 n-gram bias in a way nobody has stacked with our specific static-bias approach. The combination "static log-prob bias (Patch 6) + learnable hash-embedding head (Patch 22) with shared 16384-bucket geometry" is ours. If both stack additively in EL1, that combination is novel-to-comp.

MTP from DeepSeek-V3 (Patch 21) is also a comp-import, not novel-to-world. Same logic — the combo with our n-gram-bias + leaky stack is the novel part.

True "novel-to-world" candidates still in queue: signed-bigram-hash-LSH, Solve-English deterministic completions, n-gram-tilt-decoding (PR #461 framework). Pursuing those next.

---

## Research Fire #4 — 2026-04-07 (cron min :17, Track A = arxiv) — Differential Transformer DEFERRED

**Topic**: Differential Transformer (Microsoft, Oct 2024, arxiv:2410.05258).
Spawned an Explore subagent to deep-dive the paper + check comp PR novelty + assess for our scale.

### Subagent verdict: DEFER

**Mechanism**: `Attn = (softmax(Q1·K1ᵀ/√d) - λ·softmax(Q2·K2ᵀ/√d))·V`. Two parallel softmax attention maps subtracted → noise cancellation. Learnable λ vectors (~6×head_dim params/head, negligible overhead).

**Comp novelty**: ZERO PRs in openai/parameter-golf reference "differential" or "DIFF attention". Genuinely novel-to-comp.

**Why DEFERRED**:
1. **Scale gap**: smallest tested model in paper is 830M (38× our 22M). No empirical validation at our scale. Benefit trajectory MIGHT extrapolate down but it's purely theoretical at 22M.
2. **Crash risk**: Learnable λ uses `exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + λ_init` which has known NaN failure modes (nanoGPT issue #567). Doesn't degrade gracefully — if λ blows up, BPB → ∞.
3. **Implementation cost**: Custom attention forward (~80-120 LOC), can't use F.scaled_dot_product_attention, needs RoPE compat work for our partial-RoPE 16/64.
4. **Violates user constraint**: "DOES NOT BREAK BPB (degrades gracefully if it fails)". DIFF doesn't degrade gracefully.

**Verdict**: DEFER unless we get a successful 100M-param validation first. Not pushing this fire.

### Alternative architectures to investigate next research fire (do NOT ship without validation)

- **Gated Linear Attention (GLA, Yang et al. Dec 2024)**: sublinear complexity, designed for small models, minimal param overhead. Would replace softmax MHA entirely. Worth a separate deep-dive.
- **FusionNet (heterogeneous multi-head attention, Nov 2024)**: mixes low-rank + sparse heads in single attention block. Tested at 70M-800M which is much closer to our 22M scale.
- **YOCO (You Only Cache Once, Microsoft Apr 2024, arxiv:2405.05254)**: shared KV cache across layers. Could free params for the model body. Already noted as merge-record-relevant in our audit.

### What I AM doing instead this fire

Nothing — per user instruction "If you can't find anything novel in this fire, that's OK — just append 'no novel finding' to RESEARCH_LOG.md and let the loop continue. Don't push junk." Not falsifying our existing wins by patching speculatively.

The loop is healthy — 32+ runs, EL family + MTP follow-ups still finishing, top-3 stable. Next research fire (cron min :38, Track B) will scan comp PRs for genuinely new techniques.

---

## Audit Fire #1 — 2026-04-07 ~13:55 UTC — novelty + spend + queue hygiene

### Pod status
Loop alive (PID 123956 runner + GA family running through). Recent runs since last fire:
- GA0_gated_attn_alone (re-run) = 3.5787 (was 3.7840 first time — seed variance)
- GA1_gated_attn_full_ng = 3.3207 (gated attention + full n-gram, marginal)
- GA2_gated_attn_seed42 = 3.3226
- MEGA_stack_all_novel running, step 500 train_loss ~4.14 → BAD, the stack is fighting itself

### Novelty audit (subagent — gh api last 10 open + 10 closed PRs)

**Patches still novel after audit (3/6)**:
- ✓ Patch 15 USE_TABULATION_HASH (Pătraşcu-Thorup 3-independence) — zero comp PRs reference it. Novel-to-comp confirmed.
- ✓ Patch 16 USE_GATED_ATTENTION (NeurIPS 2025 arxiv:2505.06708) — zero comp PRs use it.
- ✓ Patch 21 USE_MTP (DeepSeek-V3 arxiv:2412.19437) — zero comp PRs implement multi-token prediction.

**Patches NO LONGER NOVEL (3/6)** — competitors got there:
- ✗ Patch 22 USE_ENGRAM_LITE — explicitly ported from PR #1440 (already credited)
- ✗ Patch 19 USE_PARTIAL_ROPE_16_64 — also in PR #1440 architecture
- ✗ Patch 20 USE_LN_SCALE — also in PR #1440 architecture

**This means 3/3 of the still-novel patches have FALSIFIED at training-loss level**:
- Tab hash: 3-seed mean 3.396 vs L1 baseline 3.33 (worse by 0.06)
- Gated attention: GA1=3.3207, GA2=3.3226 vs CHAMP_L4=3.276 (worse by 0.045)
- MTP: 2-seed mean (1337+42) = 3.2895, on-par with CHAMP_L4 within noise (NOT a clear win)

**Hard verdict**: We have ZERO genuinely novel patches that empirically beat the CHAMP_L4/L5 baselines at this scale. Our top result is still the same Mac-validated stack: leaky_relu + n-gram bias (CHAMP_L5_seed1337=3.2734).

### Critical threat from competitors

- **PR #1430**: claims val_bpb 0.39642 (65% below SOTA) via "Per-Sample SLOT + N-gram Order-22 + TTT + LR=0.432". The score is suspiciously low — likely illegal under issue #677 rules (the tilt-by-correct-token-at-eval-time class of trick). Worth verifying before trusting.
- **PR #1437**: 1.0780 BPB via SP8192 + Parallel Residuals + 3-Layer Recurrence + Legal N-gram Tilt (3-seed mean). This is the legitimate frontier.
- **PR #1423**: 1.0791 BPB via SP8192 + Pre-Quant TTT + QK-Gain 5.0 + Depth Recurrence + MuonEq-R. Legitimate.
- **PR #1099 (latest merged)**: 1.1133 BPB via Coprime-Stride + Full GPTQ + XSA-all.

### Spend check

Pod uptime since session start ≈ 3h on RTX 3080 Ti @ $0.30/h ≈ **$0.90 spent of $36 budget**. Well under the $25 soft cap. No throttling needed.

### Queue hygiene (NOT pushing — audit only, will push next research fire)

Experiments that should be REMOVED from `runpod_tests/loop/experiments.json` next fire:
- EA0/1/2/3 entropy_adaptive — falsified at scale, already in results
- BG0/BG3 batch-size tweaks — pure parameter tweaks, not novelty
- NG1/NG2/NG3 gate variants — gate experiments already failed in earlier fires
- TH0/1/2 — already ran, falsified, can stay (runner skips them)

The runner's `pick_next` already skips experiments with successful results, so leaving them is non-blocking. But cleaner queue helps the next research fire focus.

### Audit verdict

We need to PIVOT. The "easy port" patches haven't won at our scale. The queue is mostly running through tail experiments now. Top-3 is locked at the leaky+n-gram stack. Next research fire should look for **genuinely orthogonal** wins (not architecture variations): tokenizer changes, eval-time tricks (n-gram cache, score-first, TTT), data ordering (coprime stride from PR #1099), or compression-side wins (GPTQ quant, weight clustering). The current architectural-tweak vector is exhausted.

---

## Research Fire #5 — 2026-04-08 (cron min :08, Track A = arxiv) — EM-INF PASSED

**Topic**: EM-INF (Entropy Minimization at Inference, NeurIPS 2025, arxiv:2505.15134). Spawned a subagent to find ONE eval-time technique we could ship after audit fire #1's "pivot to non-architectural wins" verdict.

### Subagent verdict: SHIP. My override: PASS.

**Why subagent recommended ship**: The technique runs K steps of gradient descent on the *logits* (not the model weights) to minimize the output entropy, with no gold-token access. Causal-only, legal under issue #677, ~30 LOC, zero new params, degrades gracefully. Genuinely novel-to-comp.

**Why I PASSED** (overriding the subagent):

EM-INF is mathematically equivalent to **temperature sharpening** (T < 1) when applied to logits. The loss landscape of `H(softmax(logits/T))` as a function of `1/T` is monotonically decreasing — so K Adam steps move logits in the direction of higher inverse-temperature, i.e., sharpening.

**Decisive argument**: Cross-entropy (BPB) for a calibrated model is **minimized at T = 1**. Proof: the training loss is `-log P(y_true)` where P uses T=1. The trained model's softmax weights are MLE-optimal *with respect to T=1*. Any T ≠ 1 strictly increases the in-distribution validation NLL because:

```
NLL(T) = -log( exp(z_true/T) / sum_v(exp(z_v/T)) )
       = -z_true/T + LSE(z/T)
d(NLL)/dT  at T=1  =  z_true - <z>_{p}  (by Gibbs derivative)
                  =  KL gradient term that ML training already drove to zero
```

So at the trained optimum, dNLL/dT = 0 at T=1. Moving away from T=1 in EITHER direction strictly increases NLL.

**Cross-check with our prior result**: Patch 14 (USE_ENTROPY_ADAPTIVE_NGRAM) was a related "entropy = signal" trap. It falsified at scale (EA0=3.4592, EA1=3.4490, EA2=3.3599, EA3=3.4409 — all worse than baseline). EM-INF is the same class of mistake, just at eval time instead of training time.

**Conclusion**: EM-INF can only help BPB if the model is *miscalibrated and overconfidently wrong on average*, which is the opposite of what we want. For a well-trained model on in-distribution data, it's a pure regression. PASS.

### What this fire produced

Nothing pushed. Audit verdict still stands: pivot to non-architectural wins. Next research fire (Track B at min :38) should look at PR #1437's "Legal N-gram Tilt" since it's the *legal* version of PR #1430's suspicious 0.39642 score, and the term "tilt" implies a multiplicative correction at decode time which IS mathematically distinct from temperature sharpening.

### Better directions for next research fires (logged for handoff)

1. **N-gram Tilt** (PR #1437/#461 framework) — multiplicative reweighting of decode probabilities by an n-gram cache built from the prefix that's already been seen. This is causal and ADDS information rather than just sharpening. Worth a focused subagent dive next fire.
2. **BPE-8192 ngram tables** — task #49 still pending. Would let us A/B test SP1024 vs BPE8192 with the same n-gram-bias stack, which is the single biggest tokenizer-side gap vs the top open PRs.
3. **Coprime-Stride data loader** (PR #1099 merged record) — data ordering trick, could be ported in <50 LOC and is grounded in a merged record, not speculative.

---

## Research Fire #6 — 2026-04-08 (cron min :16, Track B = comp PRs) — Legal N-gram Tilt FORMULA CAPTURED, code patch DEFERRED

**Subject**: Deep-dive PR #1437 + PR #1420 + issue #1017 to extract the canonical "Legal N-gram Tilt" formulation. Subagent got the actual math from PR #1420 (the source — PR #1437 just stacks it).

### Canonical Formula (from PR #1420 line 233)

For each eval position with target token x_t:

```
hint        = ngram_cache.lookup(prefix[:t])    # may be None
has_hint    = (hint is not None)
is_hit      = has_hint and (x_t == hint)

p_hint      = p_model(hint)            # if has_hint, else 0
Z           = 1 + p_hint * (exp(β) - 1)
nll_tilt    = nll_model + has_hint * (log(Z) - β * is_hit)
```

Equivalently:
```
p_tilt(x_t) = p_model(x_t) * exp(β * 1[x_t == hint]) / Z
```

**β ∈ [1.0, 2.0]** (default 1.5) tilt strength
**k ∈ [8, 16]** token-level n-gram order
**+ within-word orders 1-3** (byte-level)
**+ word-start bigrams**

### Why it's LEGAL under issue #1017 (four conditions)

1. **Strict causal**: hint computed from `tokens[< t]` only. Target token never read for hint lookup.
2. **Full normalized**: tilt reweights full vocab via exp+renormalize, gives proper distribution.
3. **Score-before-update**: scoring at position t uses cache state from tokens[< t]; cache then updated with x_t AFTER scoring is locked.
4. **Single left-to-right pass**: no re-scoring, no future leakage.

### Why this is GENUINELY different from EM-INF (last fire's PASS)

EM-INF was equivalent to temperature sharpening — pure information-free entropy reduction that strictly hurts in-distribution NLL. N-gram tilt **uses an external signal** (the prefix-only n-gram hint) to *selectively boost one specific token*. If the hint is reliable, it provides REAL information from the autoregressive eval state that the model didn't capture at train time. Multiplicative reweighting + renormalization is mathematically legitimate.

### Why I'm DEFERRING the code patch despite the formula being clear

1. **Wrong metric for our loop**: Our experiment_runner measures `train_loss` (because `SKIP_FINAL_EVAL=1` saves the 5-min eval pass per run, allowing us to do 8x more experiments per hour). Tilt is **eval-only**. Shipping this patch wouldn't affect ANY of our loop's train_loss measurements, so we couldn't validate it in the loop. Validation requires `SKIP_FINAL_EVAL=0` which we only do for H100 escalation.

2. **Subagent pseudocode has critical bugs**:
   - The "50 LOC sketch" loops `pos` calling `model(prefix[:, :pos+1])` per position → O(L²) forward passes per batch. Untenable for 1024-token blocks. A correct streaming implementation needs to extract per-position logits from the existing single forward pass.
   - `targets.mode()[0]` collapses across batch dimension when updating the cache — not per-sample. Correct version needs per-batch streaming dict updates.
   - Cache key as `tuple(...tobytes())` is slow Python; would want a numpy or torch-native hash.
   - Correct implementation is closer to **150-200 LOC**, NOT the 50 the subagent quoted. That's medium-hard, not easy/medium.

3. **Risk to existing SKIP_FINAL_EVAL=0 pipeline**: modifying the eval/loss path is risky — could break the FINAL int8_zlib_roundtrip val_bpb computation if I get it slightly wrong. We wouldn't notice until H100 escalation time, where each iteration costs $3-5.

### What I AM doing this fire

- **Capturing the formula** in this log so the next H100-escalation fire can implement it from a clean spec instead of re-deriving.
- **Marking as HIGH PRIORITY** for the H100-escalation step. When we have a CHAMPION config ready to escalate, the plan is:
  1. Implement N-gram Tilt as Patch 23 (eval-only, ~150 LOC, gated by `USE_NGRAM_TILT_EVAL=1`)
  2. Test on cheap GPU first with `SKIP_FINAL_EVAL=0` on a SHORT run (200 steps) to confirm it doesn't break the eval pipeline
  3. Then run on H100 with the champion config
  4. Compare `final_int8_zlib_roundtrip val_bpb` with and without tilt

### Estimated value

- PR #1420 reports **+0.003 BPB delta** (single technique contribution at SP8192)
- Discounted for our SP-1024 (smaller vocab, sparser cache, lower hit rate): **+0.0015 to +0.0030 BPB**
- This is on the same order as the largest single-technique gains in any record. WORTH the H100 escalation time.

### Comp novelty

- Used in PR #1437 (1.078), PR #1420 (1.083), and PR #1430 (claimed 0.396 — almost certainly illegal variant). The CANONICAL formulation in PR #1420 is the legal one we want.
- We don't yet have it. Adding it would close one of our biggest known gaps vs the legal frontier.

### Action: NO PUSH this fire. Formula captured for next escalation cycle.

The subagent's verdict was SHIP; I overrode to DEFER for the metric/complexity reasons above. This is consistent with the audit fire #1 verdict "pivot to non-architectural wins" — n-gram tilt IS the right direction, but the right time to implement it is when we have a champion to measure against, not in a research fire that can't validate it.

---

## Audit Fire #2 — 2026-04-08 ~14:45 UTC — re-verify novelty + spend + spot new comp directions

### Pod status
Loop alive (PID 123956 + 125521). Cycle 2 of the experiment queue ~25% through. Recent notable: **EL2_engram_lite_seed42 cycle-2 = 3.2742** (only +0.0008 above CHAMP_L5_seed1337=3.2734) — the previous "EngramLite preliminarily falsified" claim from audit #1 is now SOFT-REVERSED. EngramLite is **tied within noise** with the champion. Not a clear win, not a clear loss. MTP1_seed999_validation cycle-2 = 3.4640 (essentially same as cycle-1 3.4656) — confirms the seed-999 outlier is structural to that family, not random noise.

### Novelty re-verification (subagent — last 25 open PRs scanned)

**Patches 15/16/21 ALL STILL NOVEL** (zero hits in latest 25 PRs):
- ✓ Patch 15 USE_TABULATION_HASH — no comp PR mentions Pătraşcu-Thorup or tabulation hashing for n-gram bias tables
- ✓ Patch 16 USE_GATED_ATTENTION — PR #1410 mentions "Alternating GatedAttention" but that's a different mechanism (parameter-reduction every-other-layer trick, not the NeurIPS 2025 gating arxiv:2505.06708)
- ✓ Patch 21 USE_MTP — zero comp PRs reference "multi-token", "MTP", "DeepSeek", or arxiv:2412.19437

This is the second consecutive audit confirming these 3 are uncontested. They remain our strongest novelty claim, even though only MTP shows marginal training-loss benefit at our scale.

### New PRs since last audit (~1 hour delta)

| PR# | Created | Title | Score | Assessment |
|---|---|---|---|---|
| 1444 | 14:37 | LeakyReLU GPTQ-lite v1 (1xH100) | non-record | Direct competitor on training; LeakyReLU is patch 11 in our stack already |
| 1443 | 13:51 | ByteJEPA — Byte-Level JEPA | 1.3496 BPB | Novel learning objective (joint-embedding predictive arch). Non-competitive score but bounty-driven |
| 1441 | 12:28 | nogakeren System Optimizations (in-dev) | dev | Watch — author tag suggests infrastructure work |
| 1440 | 11:32 | EngramLite + Mousse + ProgDepth | 1.1026 | Already known, source of our Patch 22 |

### Open PR techniques NOT in our 22-patch stack (top 3 most interesting)

1. **PR #1430 Per-Sample SLOT + Causal Backoff N-gram Mixer + entropy-adaptive blend** — claims 0.39642 BPB (suspicious, likely illegal). Even if illegal, the SLOT mechanism (per-sequence learnable [bsz,1,512] hidden + [bsz,1,1024] logit bias = 1536 params) is novel and worth understanding. **Could inform a legal variant.**
2. **PR #1433 EP8 Lattice Codebook VQ + Hadamard transform + Hessian-aware assignment** — only 1.2067 BPB so non-competitive, but the **compression-side infrastructure** is uniquely interesting. We have ZERO compression-side patches.
3. **PR #1443 ByteJEPA** — different learning objective entirely (joint-embedding predictive arch on bytes). Non-competitive at 1.3496 but the 3-stage training pipeline (JEPA pretrain → bridge → CE+SWA) is novel category.

### Spend check

Pod uptime ≈ 3h on RTX 3080 Ti @ $0.30/h ≈ **~$1.40 spent / $36 budget**. Soft cap $25, hard cap $36. **6% utilization, 94% headroom**. No throttling needed.

### Queue hygiene (still deferred per "no patches in audit" rule)

The runner is now cycling through dead families (EA*, BG*, NG*) wasting compute on configs we know are falsified. The audit recommendation is: **next research fire should clean these from experiments.json** to free cycle slots for genuinely interesting next-gen experiments. Specifically remove: EA0/1/2/3, NG1/NG2/NG3, BG0/BG3, MEGA_stack_all_novel. Keep all CHAMP_L5/L4 multi-seed and the EL family (EL2 reversal makes them worth keeping).

### Audit verdict #2

We have THREE genuinely novel-to-comp patches (15, 16, 21). They are **marginal at best** at our 22M scale. The audit fire #1 conclusion still stands: **architectural vector exhausted**. The next genuine progress vector is N-gram Tilt (task #53, deferred to H100 escalation) plus the new directions surfaced this fire:
- Per-sample SLOT (legal variant of PR #1430)
- Compression-side codebook VQ (PR #1433)

Both are PhD-level, both are non-architectural, both fit the "pivot" recommendation. Logging for next research fire (cron min :08 or :38).

---

## Research Fire #7 — 2026-04-08 (cron min :46, Track B = comp PRs) — EMA spec captured + queue cleanup + EL multi-seed expansion

**Track**: B (PRs). Subagent extracted canonical EMA(0.997) implementation pattern from 6 merged records (PR #287, #315, #414, #1019, #1099). Spec captured for next research fire to ship as Patch 17.

### Canonical EMA spec (from 6 merged records, unified pattern)

**Locations**:
- **Init** (after optimizer creation): `ema_state = {n: t.detach().float().clone() for n,t in base_model.state_dict().items()}`
- **Update** (after each `opt.step()`): `ema_state[n].mul_(0.997).add_(t.detach().float(), alpha=0.003)` for n,t in state_dict
- **Swap** (post-training, before final eval): `base_model.load_state_dict({n: t.to(orig_dtype) for n,t in ema_state.items()}, strict=True)`

**Decay**: 0.997 (canonical across all 6 records)
**Memory cost**: 88MB (22M params × 4 bytes fp32 shadow) on 12GB GPU = no risk
**Artifact cost**: 0 (EMA replaces training weights, no extra storage)

**CRITICAL caveat for our metric**: EMA only affects the FINAL eval val_bpb, NOT mid-training train_loss. Same metric problem as N-gram Tilt — our `experiment_runner.py` measures train_loss with `SKIP_FINAL_EVAL=1`, so shipping EMA would NOT show any benefit in our loop. EMA only shines on H100 escalation runs with `SKIP_FINAL_EVAL=0`.

### Why I DEFERRED the actual patch this fire

Same triple-risk pattern as Tilt:
1. **No loop validation possible** — train_loss won't change
2. **Anchor risk** — without reading train_gpt.py training-loop structure (multiple optimizers, opt.step locations), I'd be guessing where to insert. A wrong anchor either silently no-ops OR worse, anchors on a duplicated location causing double-update bugs that we wouldn't catch until H100 time.
3. **Training-loop modifications are higher-risk than other patches** — touching the optimizer step path could introduce subtle bugs that affect ALL runs, not just `USE_EMA=1`.

Logging the spec for next research fire to actually implement after reading train_gpt.py carefully.

### What this fire DID push: queue cleanup + EL multi-seed expansion

Cleaned `runpod_tests/loop/experiments.json` from 37 → **20 experiments**:
- **Removed (15 dead/falsified entries)**: EA0/1/2/3 (entropy adaptive, falsified cycle 1+2), BG0/BG3 (batch tweaks, no novelty), NG1/NG2/NG3 (gate variants, falsified), TH0/1/2 (tabulation hash, falsified at scale), MEGA_stack_all_novel (kitchen-sink stack, fights itself), MTP0_mtp_alone (without n-gram, expected weak), MTP2_mtp_strong_weight (within MTP1 noise band), MTP1_seed999_validation (seed-999 outlier confirmed structural), MTP3_two_heads (worse than 1-head), PR2_parallel_plus_full_stack + PR3_parallel_plus_leaky_seed42 (parallel residuals dead at our scale), EL0_engram_lite_alone (without static n-gram, weak)

- **Added (4 new EL multi-seed validations)**: 
  - EL3_engram_lite_seed1337 — same as EL1 but explicit seed 1337
  - EL4_engram_lite_seed999 — extending the seed-999 outlier check into EL family
  - EL5_engram_lite_seed7 — fresh seed not yet tested
  - EL6_engram_lite_L5weights — EngramLite stacked with L5 weights (0.15/0.20/0.15) instead of L4 weights (0.25/0.25/0.20). NEW combination not yet measured.

**Why expand EL**: Monitor #10 surfaced EL2 cycle-2 = 3.2742 (only +0.0008 above champion). The audit fire #1 "EngramLite preliminarily falsified" verdict is now SOFT-REVERSED — EngramLite is tied within noise. Worth a 5-seed validation to confirm the variance band and decide whether EL is a free addition to the final stack or a coin flip.

### Resulting queue (20 experiments)

| Family | Count | Purpose |
|---|---|---|
| CHAMP_L5 | 5 seeds | Champion validation (already strong) |
| CHAMP_L4 | 3 seeds | Alternative weight ratio (already validated) |
| PR | 2 (PR0/PR1) | Parallel residuals minimal — kept for alternative branching |
| GA | 2 (GA0/GA1) | Gated attention minimal — kept since still novel-to-comp |
| MTP | 2 (MTP1/MTP1_seed42) | MTP marginal validation |
| EL | 6 (1/2/3/4/5/6) | **PRIORITY** — multi-seed expansion of the EL2 reversal |

The runner will pick up the new file on next git pull (~5 min). Queue cycles will be ~2x faster now (20 vs 37 experiments per cycle ≈ 100 min instead of 185 min).

---

## Research Fire #8 — 2026-04-08 (cron min :16, Track A = arxiv → pivot to compression) — INT6 GPTQ-Lite SPEC CAPTURED, code patch DEFERRED

**Subject**: Compression-side win port. Audit fire #1 said "pivot to non-architectural" and we have ZERO compression patches. Spawned subagent to find the simplest shippable compression upgrade for our int8+zlib roundtrip.

### Subagent recommendation: INT6 GPTQ-Lite (percentile-based, no Hessian)

**Source**: PR #1099 (merged, latest), PR #1019 (merged), PR #1444 (open). All use this exact pattern.

**Mechanism**: per-row 99.95th percentile quantization to int6 (clamped to [-31, 31], stored in int8 container), then LZMA-22 compression. No Hessian computation, no AR self-gen, just statistical fallback.

**Pseudocode**:
```python
for name, t in state_dict.items():
    if t.is_floating_point() and t.numel() > 65536:
        s = torch.quantile(t.abs(), 0.9995, dim=1, keepdim=True) / 31.0
        q = torch.round(t.float() / s).clamp(-31, 31).int8()
        quantized[name] = q ; scales[name] = s.half()
buf = io.BytesIO() ; torch.save({"quantized": quantized, "scales": scales}, buf)
final_blob = lzma.compress(buf.getvalue(), preset=9)
```

**Bit budget**: 22M params × 0.75 bytes (int6 packed) + ~1.8MB scales = **~18.4MB raw → ~15.5MB after LZMA-22**. Vs current int8 ≈ 16MB after zlib. Saves ~0.5MB headroom.

**Direct BPB impact**: 1.1141 → 1.1138 = **-0.0003 BPB** (within noise). The real value is the freed size budget, NOT the direct delta.

### Why I DEFERRED the code patch this fire

Same triple-pattern as last 3 research fires (Tilt, EMA, this):

1. **Metric problem**: train_loss in our loop is unaffected by serialization. The patch only changes the FINAL int8_zlib_roundtrip step which we skip via `SKIP_FINAL_EVAL=1` for loop speed. We can't validate it on the cheap GPU without enabling final eval, which slows the loop ~33% per experiment.

2. **Anchor risk**: train_gpt.py's serialization code is the LAST piece I'd want to break. A buggy compression patch could corrupt the int8_zlib_roundtrip path and we wouldn't notice until $3-5 H100 escalation time. The code path is around line ~2100 of train_gpt.py per the subagent — I'd need to read that section carefully before patching.

3. **The +0.0003 BPB direct gain isn't motivating enough** for the validation risk on its own. The size headroom is the real value, but spending it requires re-tuning model capacity which is a separate research problem.

### Spec captured for next H100-escalation fire

When we have a multi-seed champion validated and we're escalating to H100, the plan is:

1. **Patch 23 USE_INT6_GPTQ** — modify the serialization path to swap int8 → int6 packing + LZMA. ~130 LOC, anchored on the existing `final_int8_zlib_roundtrip` block in train_gpt.py.
2. **Test on cheap GPU first** — one short experiment with `SKIP_FINAL_EVAL=0 USE_INT6_GPTQ=1` to verify the roundtrip is lossless and the .lzma file is < 15.9MB.
3. **Then escalate** — measure end-to-end val_bpb delta on H100.

### Backup alternative documented (Lloyd-Max codebook quantization)

We already have `data/lloyd_max_codebook_256.npy` and `data/lloyd_max_codebook_64.npy` on disk from prior work but they're not wired in. Lloyd-Max VQ would be:
```python
codebook = np.load("data/lloyd_max_codebook_256.npy")  # 256 entries
w_flat = weight.float().cpu().numpy().flatten()
indices = ((w_flat[:, None] - codebook[None, :]) ** 2).argmin(axis=1).astype(np.uint8)
# Store: indices (1 byte/param) + codebook (256 floats = 1KB)
# Decompress: float_w = codebook[indices].reshape(orig_shape)
```
- Raw: 22MB + 1KB
- Zlib ratio: similar to int8 → ~16MB final
- Simpler than GPTQ, deterministic, but doesn't save bytes vs naive int8

GPTQ-lite is strictly better. Lloyd-Max is the fallback if GPTQ proves unstable on H100.

### Three deferred eval/compression specs now ready for H100 escalation

| # | Patch | Type | Captured Fire | Estimated gain |
|---|---|---|---|---|
| 53 | USE_NGRAM_TILT_EVAL | eval-time | #6 | +0.0015 to +0.0030 BPB |
| 45 | USE_EMA (decay 0.997) | training | #7 | +0.001 to +0.005 BPB |
| (new) | USE_INT6_GPTQ | serialization | #8 | -0.0003 BPB + 0.5MB headroom |

Combined estimated gain when shipped together at H100 escalation: **+0.003 to +0.008 BPB** without affecting any training-loop metrics.
