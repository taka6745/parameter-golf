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

## 2026-04-07 21:30 local — Mid-fire pivot to TRULY NOVEL

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
