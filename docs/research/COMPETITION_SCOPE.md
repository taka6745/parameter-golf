# Parameter Golf Competition Scope Analysis
## Generated: 2026-04-05 10:25
## Data: 903 open + 397 closed = 1300 total PRs, 25 merged records

## 1. Executive Summary

- **Total PRs**: 1300 (903 open, 397 closed, 34 merged)
- **PRs with parseable scores**: 710 (54%)
- **Merged SOTA**: 1.1147 BPB
- **Best legal open PR**: 0.8104 BPB (PR #1028 by @newjordan)
- **Best ANY open PR**: 0.3212 BPB (PR #850 by @callithyia)
- **Contested (SLOT) PRs**: 0
- **Our position**: 1.7292 BPB (Mac, 1000 steps, int8 quantized)

## 2. Score Leaderboard

### 2a. Merged Records (ground truth)

| # | BPB | Author | Technique Summary | Date |
|---|-----|--------|-------------------|------|
| 1 | 1.11473509 | abaybektursun | 11L XSA-all + Full Hessian GPTQ with autoregressive self-gen | 2026-03-25 |
| 2 | 1.1194 | abaybektursun | LeakyReLU(0.5)² activation (-0.003 BPB vs relu²) + legal sco | 2026-03-23 |
| 3 | 1.12278022 | Tianhao Wu | EMA(0.997) weight averaging + GPTQ-lite optimal clip percent | 2026-03-22 |
| 4 | 1.12484502 | Jack Princz | 11 layers with Partial RoPE (16 of 64 dims), LN Scale (1/sqr | 2026-03-21 |
| 5 | 1.12707468 | Jack Princz | 11 layers with Exclusive Self Attention (XSA) on last 4 laye | 2026-03-21 |
| 6 | 1.13071416 | vadim borisov (tabularis.ai) | 11 layers, int6 quant, zstd-22. Novel contribution: Efficien | 2026-03-20 |
| 7 | 1.14581692 | Raahil Shah | Per-row int6 quantization on MLP/attention weights with zstd | 2026-03-20 |
| 8 | 1.15015359 | aruniyer |  | 2026-03-20 |
| 9 | 1.1556 | unknown |  |  |
| 10 | 1.1574404 | samuellarson | Int6 post-training quantization enables 3x MLP expansion (21 | 2026-03-20 |
| 11 | 1.15861696 | yahya010 | 10-layer 512dim SP-1024, STE int6 QAT (zero quant gap), full | 2026-03-19 |
| 12 | 1.16301431 | aquariouseworkman | 3x MLP expansion with mixed-precision quantization: int6 per | 2026-03-19 |
| 13 | 1.19250007 | Matthew Li | Baseline 9x512 SP-1024 architecture with sliding window eval | 2026-03-19 |
| 14 | 1.1929 | sam | Naive baseline + per-document LoRA test-time training at eva | 2026-03-19 |
| 15 | 1.20143417 | Spokane Way | SP-1024 9x512 KV4 run at TRAIN_SEQ_LEN=4096 with aggressivel | 2026-03-19 |
| 16 | 1.20576485 | Spokane Way | SP-1024 9x512 KV4 run at TRAIN_SEQ_LEN=2048 with tuned seq20 | 2026-03-19 |
| 17 | 1.20737944 | Will DePue | Unlimited compute track: SP-1024 9x512 KV4 run on pgut3 for  | 2026-03-18 |
| 18 | 1.214745 | Nan Liu | 10-layer 512-dim model with lower LR (MATRIX_LR=0.02) and mi | 2026-03-19 |
| 19 | 1.21972502 | Renier Velazco | Keep tok_emb.weight in fp16 during int8 quantization to elim | 2026-03-18 |
| 20 | 1.22296644 | Nan Liu | Same 9x512 SP-1024 KV4 tied-embedding baseline architecture  | 2026-03-18 |
| 21 | 1.2243657 | Baseline | SP-1024 9x512 KV4 run on pgut1 using the published Hugging F | 2026-03-18 |
| 22 | 1.32814313 |  | Non-record 1xRTX5090 submission: systematic 10-experiment ex | 2026-03-19 |
| 23 | 2.3876 | Evangeline Kamin | Depth Recurrence + Mixed-Precision Quantization | 2026-03-21 |
| 24 | N/A | thwu1 | 10 layers with mixed int5/int6 quantization. BigramHash 1024 | 2026-03-20 |
| 25 | N/A | notapplica | Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init | 2026-03-19 |

### 2b. Top 50 Legal Open PRs

| Rank | PR# | BPB | Author | Title (truncated) | Techniques |
|------|-----|-----|--------|-------------------|------------|
| 1 | #1028 | 0.8104 | @newjordan | Medusa: Unstable — DeltaNet Crawler 0.8104 BPB 10m |  |
| 2 | #786 | 0.8128 | @shinegami-2002 | 0.8128 BPB: Classical Compression Eval + N-gram Ba | NGRAM |
| 3 | #642 | 0.8173 | @minh-stakc | Record: 11L + Score-Every-Epoch LoRA TTT 5ep (3-se | TTT |
| 4 | #769 | 0.8495 | @MatoTeziTanka | PROTEUS+STYX — val_bpb 0.8495 (3-seed mean) — Leak | LEAKYRELU |
| 5 | #909 | 0.8609 | @sunnypatneedi | Record: 11-gram Eval Cache + Hedge Mixer (val_bpb: |  |
| 6 | #963 | 0.8609 | @sunnypatneedi | Record: 11-gram Eval Cache + Hedge Mixer (val_bpb: |  |
| 7 | #1047 | 0.8822 | @newjordan | INVALID* (0.8822 BPB mean) Medusa: Unstable S2 — D |  |
| 8 | #795 | 0.8881 | @hypery11 | Record: 11L + order-adaptive 11-gram (mean val_bpb |  |
| 9 | #797 | 0.8960 | @armantsaturian | Record: 7-gram N-gram Cache (0.8960 bpb) | NGRAM |
| 10 | #788 | 0.9059 | @hypery11 | Record: 11L + order-adaptive 9-gram backoff (mean  | NGRAM |
| 11 | #828 | 0.9076 | @bigbag | Record: 0.9076 BPB — 10L + N-gram Backoff + Matrix | NGRAM |
| 12 | #802 | 0.9123 | @Bortlesboat | 10L + Multi-Order N-gram Backoff (0.9123 BPB) | NGRAM |
| 13 | #776 | 0.9258 | @agalimova | Record Submission: 0.9258 BPB — Kitchen Sink (7-gr | TTT, NGRAM |
| 14 | #782 | 0.9362 | @newjordan | Podracing III: Cubric Lite — 0.9362 BPB |  |
| 15 | #774 | 0.9370 | @travispchen | Record: Order-Adaptive Entropy Gating + XSA-All (v | XSA |
| 16 | #187 | 0.9393 | @Idan3011 | [Closed] EMA + Multi-Order N-gram Backoff + PE Con | NGRAM, EMA_SWA |
| 17 | #620 | 0.9443 | @robinojw | Record: LeakyReLU(0.5)² + Per-Document LoRA TTT (m | TTT, LEAKYRELU |
| 18 | #1184 | 0.9485 | @icryo | Record: Scylla + Full GPTQ + XSA-all + FA3 — val_b | GPTQ, XSA, SCYLLA_TOKENMONSTER |
| 19 | #512 | 0.9512 | @MatoTeziTanka | Record: PROTEUS v7 — 11L INT6 + LoRA TTT (mean val | TTT, INT6 |
| 20 | #940 | 0.9581 | @antaloaalonso | Record: Score-First TTT + Multi-Order N-gram Backo | TTT, NGRAM, LEGAL_TTT |
| 21 | #761 | 0.9581 | @Asukabot0 | Record: Score-First TTT + N-gram Backoff (3-seed m | TTT, NGRAM, LEGAL_TTT |
| 22 | #120 | 0.9588 | @andrewgcodes | [Val Only]: MLP 3x + STE int6 QAT + sliding window | SLIDING_WINDOW, INT6, QAT |
| 23 | #777 | 0.9623 | @Robby955 | Record: 0.9623 BPB — 7-Gram Entropy Cache + XSA-al | XSA |
| 24 | #753 | 0.9625 | @newjordan | Podracing II: Electric Bugaloo — 0.9625 BPB (3-see |  |
| 25 | #993 | 0.9631 | @aerosta | Record: 11L XSA + Mixed INT6 + Adaptive N-gram Cac | XSA, NGRAM, INT6 |
| 26 | #764 | 0.9633 | @ndokutovich | Record: Curriculum Learning + LeakyReLU(0.9)² + 7- | NGRAM, LEAKYRELU |
| 27 | #1185 | 0.9641 | @skoustav35 | [10min_16mb] 0.9641 BPB: LeakyReLU² + Score-First  | TTT, NGRAM, LEAKYRELU, LEGAL_TTT |
| 28 | #889 | 0.9642 | @anthony-maio | Record: N-gram Backoff + VRL + LeakyReLU² — val_bp | NGRAM, LEAKYRELU, VRL |
| 29 | #887 | 0.9642 | @anthony-maio | Record: N-gram Backoff + VRL + LeakyReLU² — val_bp | NGRAM, LEAKYRELU, VRL |
| 30 | #376 | 0.9642 | @anthony-maio | Record: N-gram Backoff + VRL + LeakyReLU² — val_bp | NGRAM, LEAKYRELU, VRL |
| 31 | #1246 | 0.9650 | @deborahnelson8788726 | Record: Trinity Ternary GPT — val_bpb 0.9650 (tern | TERNARY_BINARY |
| 32 | #727 | 0.9674 | @Asukabot0 | Record: First Legal Sub-1.0 BPB — Multi-order N-gr | NGRAM |
| 33 | #1055 | 0.9700 | @sanyalsunny111 | SOTA Record: Novel Test-Time Method TARA Val BPB=0 | TTT |
| 34 | #778 | 0.9757 | @raahilshah | Record: 11L Full GPTQ + Multi-Order N-gram Backoff | GPTQ, NGRAM |
| 35 | #517 | 0.9780 | @lukacf | Record*: val_bpb=0.978 BPB — Goldfish ML Autonomou | TTT |
| 36 | #741 | 0.9850 | @andrewbaggio1 | Record: Cosine TTT + Multi-Order N-gram Cache (3-s | TTT, NGRAM |
| 37 | #1241 | 0.9901 | @aiejvn | MDLM Diffusion — val_var_bpb 0.9901, EOS learning  | NOVEL_ARCH |
| 38 | #763 | 0.9917 | @hypery11 | Record: 11L XSA-all + backoff 7-gram (mean val_bpb | XSA, NGRAM |
| 39 | #885 | 0.9958 | @lolrazh | Record: LeakyReLU(0.9)² + N-gram Cache + Entropy-R | NGRAM, LEAKYRELU, QAT |
| 40 | #745 | 1.0222 | @stukenov | Record: XSA-all + Depth Recurrence + Hedge Mixer T | TTT, XSA, DEPTH_RECURRENCE |
| 41 | #875 | 1.0226 | @shalyhinpavel | New Record: Pure Neural GDN 1.0226 BPB (shalyhinpa |  |
| 42 | #168 | 1.0238 | @spokane-way | SOTA Attempt: Paid prefix (val_bpb=1.0238) |  |
| 43 | #702 | 1.0240 | @lukacf | Record: 1.0240 BPB — Multi-Order N-gram Backoff +  | NGRAM |
| 44 | #733 | 1.0278 | @stukenov | Record: XSA-all + Depth Recurrence + Hedge Mixer T | TTT, XSA, DEPTH_RECURRENCE |
| 45 | #755 | 1.0321 | @dcrow85 | Gravity Tokenizer: 1.0321 BPB via ablation leverag |  |
| 46 | #715 | 1.0337 | @Asukabot0 | Record: XSA-all + LeakyReLU² + VR + GA + 7-gram ca | XSA, LEAKYRELU |
| 47 | #792 | 1.0340 | @xexyz | 11L LeakyReLU² + XSA-all + Full GPTQ + 5-gram Back | GPTQ, XSA, NGRAM, LEAKYRELU |
| 48 | #995 | 1.0362 | @dexhunter | Record: 1.0362 BPB — SGD Momentum 0.95 TTT + Hedge | TTT |
| 49 | #278 | 1.0365 | @nicolasdickenmann | Record: 8L Paid Prefix + Sparse Hard Blocks (1.036 |  |
| 50 | #685 | 1.0366 | @andrewbaggio1 | Record: Chained TTT — Cosine Recovery + Multi-Pass | TTT |

### 2c. Top 30 Contested (SLOT) PRs

| Rank | PR# | BPB | Author | Title (truncated) |
|------|-----|-----|--------|-------------------|

## 3. Score Distribution

| BPB Range | Count | % |
|-----------|-------|---|
| <0.7 | 30 | 4% ## |
| 0.7-0.9 | 18 | 2% # |
| 0.9-1.0 | 34 | 4% ## |
| 1.0-1.1 | 78 | 10% ##### |
| 1.1-1.15 | 264 | 37% ################## |
| 1.15-1.2 | 146 | 20% ########## |
| 1.2-1.3 | 63 | 8% #### |
| 1.3+ | 77 | 10% ##### |

## 4. Technique Frequency

| Technique | PRs Using | Best Score | First PR |
|-----------|-----------|------------|----------|
| TTT | 234 | 0.3212 | #77 |
| NGRAM | 173 | 0.3212 | #76 |
| EMA_SWA | 150 | 0.9393 | #69 |
| INT6 | 136 | 0.6430 | #37 |
| GPTQ | 105 | 0.3212 | #64 |
| QAT | 102 | 0.9588 | #37 |
| LEGAL_TTT | 85 | 0.7139 | #430 |
| LEAKYRELU | 80 | 0.6678 | #175 |
| MUON | 78 | 1.0896 | #37 |
| XSA | 73 | 0.6951 | #186 |
| DEPTH_RECURRENCE | 72 | 0.6360 | #30 |
| SLIDING_WINDOW | 72 | 0.6361 | #50 |
| INT5 | 61 | 0.3212 | #76 |
| NOVEL_ARCH | 49 | 0.9901 | #599 |
| SMEARGATE | 42 | 1.0539 | #65 |
| ROPE | 37 | 1.1187 | #181 |
| SLOT | 30 | 0.6361 | #675 |
| VRL | 29 | 0.7271 | #175 |
| TERNARY_BINARY | 15 | 0.9650 | #139 |
| LZMA | 14 | 1.1085 | #160 |
| SP4096 | 13 | 1.0766 | #37 |
| INT4 | 10 | 1.1521 | #305 |
| SCYLLA_TOKENMONSTER | 9 | 0.9485 | #1143 |
| COMPLEMENTARY | 8 | 0.3212 | #803 |
| PARALLEL_RESIDUAL | 7 | 1.0766 | #1204 |
| BROTLI | 2 | 1.1105 | #1179 |
| BPE8192 | 2 | 1.1860 | #78 |
| FP8 | 1 | 1.1511 | #538 |

## 5. Legality Analysis

- **Legal**: 809 PRs (62%)
- **Contested (SLOT)**: 0 PRs (0%)
- **Non-record/negative**: 440 PRs

**The SLOT divide**: All scores below ~1.00 BPB use SLOT. Legal-only frontier is ~0.81 BPB.

## 6. Top Authors

### By PR count

| Author | PRs | Best Score |
|--------|-----|------------|
| @newjordan | 31 | 0.4820 |
| @ibarrajo | 17 | 0.6678 |
| @abaybektursun | 17 | 1.1147 |
| @EthanYangTW | 16 | 1.1145 |
| @dentity007 | 15 | 1.0925 |
| @aamodbhatt | 14 | 1.1156 |
| @dexhunter | 13 | 1.0362 |
| @anthony-maio | 13 | 0.7406 |
| @aryanbhosale | 12 | 1.0766 |
| @Christopher-Lee-McClendon | 12 | 1.0920 |
| @gowtham0992 | 12 | 0.6846 |
| @hypery11 | 11 | 0.5440 |
| @andrewbaggio1 | 10 | 0.9850 |
| @bigbag | 9 | 0.6864 |
| @himanshudongre | 8 | N/A |
| @simon-marcus | 8 | 1.0806 |
| @MatoTeziTanka | 8 | 0.7853 |
| @Bortlesboat | 8 | 0.3461 |
| @Robby955 | 8 | 0.9623 |
| @AnirudhRahul | 8 | 1.1109 |
| @sofiabod | 8 | 0.4405 |
| @resouer | 7 | 0.9300 |
| @mrdavtan | 7 | 1.0970 |
| @JoeProAI | 7 | 1.0672 |
| @RoyiRa | 7 | 1.0541 |
| @andrewmouldon | 6 | N/A |
| @LucasErcolano | 6 | 0.4188 |
| @greqone | 6 | 0.8004 |
| @jfprincz | 6 | 1.1248 |
| @Shuvam-Banerji-Seal | 6 | 1.4078 |

## 7. Novel Approaches

**49 PRs with non-transformer architectures:**

| PR# | BPB | Author | Title |
|-----|-----|--------|-------|
| #1241 | 0.9901 | @aiejvn | MDLM Diffusion — val_var_bpb 0.9901, EOS learning + full dat |
| #1257 | 1.0855 | @BoxiYu | Add: 11L Complement Training + TTT + No-JEPA submission (val |
| #1006 | 1.1085 | @NewyorkDev | 1.1085 BPB: JEPA + AdamW TTT + Full GPTQ + FA3 + LZMA |
| #852 | 1.1189 | @Prush69 | Hymba-11L: SOTA High-Density Takeover (1.1189 BPB) |
| #1124 | 1.1194 | @NewyorkDev | Record: 1.1194 BPB — v9 Batched Muon + Full GPTQ Random Cali |
| #1243 | 1.1230 | @simon-marcus | JEPArdy! Non-Record Submission - JEPA + Leader-Stack - val_b |
| #1106 | 1.1465 | @agalimova | Non-record: MDLM Diffusion — val_var_bpb 1.1465 (first diffu |
| #1100 | 1.1465 | @agalimova | Non-record: LLaDA-MDLM Diffusion — val_var_bpb 1.1465 (first |
| #1245 | 1.1470 | @mkenney2 | [Non-Record] Hymba-8L: Hybrid SSM + Sliding Window Attention |
| #1355 | 1.1526 | @mradassaad | Non-record: Mamba-3 Hybrid + Full Hessian GPTQ + Late QAT —  |
| #599 | 1.1828 | @mkenney2 | [Non-Record] Hymba: Hybrid Attention + Mamba SSM (val_bpb 1. |
| #914 | 1.1873 | @mkenney2 | [Non-Record] Hymba-LongContext: 32K context training and eva |
| #896 | 1.1900 | @MVPandey | [Non-Record] JEPA Self-Distillation with EMA Target Encoder  |
| #832 | 1.1903 | @jfprincz | Non-record: Byte-level transformer + JEPA auxiliary loss (va |
| #1110 | 1.2249 | @gowtham0992 | Notable Non-Record: Universal Transformer — 1.2249 BPB — Dep |
| #904 | 1.2734 | @anthony-maio | Non-record: Diffusion-Noised Teacher AR Hybrid (val_bpb=1.27 |
| #970 | 1.2907 | @dnldsz | Non-record: GatedDeltaNet SSM via fla library — 1.2907 bpb,  |
| #969 | 1.2907 | @dnldsz | Non-record: GatedDeltaNet SSM via fla library — 1.2907 bpb,  |
| #1193 | 1.4390 | @dentity007 | Non-record: Universal Transformer + Adaptive Density (val_bp |
| #1116 | 1.4447 | @gowtham0992 | Notable Non-Record: JEPA — 1.4447 BPB — Joint Embedding Pred |

## 8. Gap Analysis

### Techniques from our research NOT seen in PRs:

- Signed hashing for n-gram tables
- Distributional Categories (DC500/DC1000)
- English Knowledge Engine (POS, capitalization, context)
- Skip-bigram table (prev2→next)
- Tabulation hashing (XOR lookup)
- Perfect hashing (bbhash)
- NorMuon + MuonEq-R combined
- Trimmed mean loss (trim tails)
- Rho-1 excess loss with n-gram reference
- Dual-codebook n-gram compression
- Dendritic MLP (block-diagonal)
- Codon-style eval tokenization search
- Online eval n-gram mixing (backward-looking)
- Predictive coding gate (suppress predictable)
- Folding shard ordering
- FIM-diverse shard selection

### Underexplored in competition (low PR count):

- FP8: only 1 PRs
- BROTLI: only 2 PRs
- BPE8192: only 2 PRs

## 9. Strategic Recommendations

### Safest path to beat SOTA:
- Stack proven winners: 11L/3xMLP + XSA + NorMuon + BigramHash + GPTQ int6 + sliding window
- Add our unique edge: n-gram logit bias + DC500 + signed hashing + skip-bigram
- Legal eval: temperature scaling + online n-gram cache + legal TTT

### Highest-ceiling path:
- Everything above + Meta-TTT (FOMAML) + parallel TTT search on 8 GPUs
- Factorized embedding → extra layers
- Rho-1 excess loss training

### Most novel/differentiated path:
- Our n-gram bias stack (nobody else has DC categories + skip-bigram + signed hashing)
- Dendritic MLP + predictive coding gate
- Perfect hashing for zero-collision n-grams
- Codon-style eval tokenization search