# Parameter Golf - Results Tracker

## Mac MLX Runs (Apple Silicon)


| Run             | Steps | val_bpb | val_loss | Step Avg | Tok/s | Notes                                                                    |
| --------------- | ----- | ------- | -------- | -------- | ----- | ------------------------------------------------------------------------ |
| mlx_baseline    | 200   | 2.3214  | 3.9196   | 862ms    | 9,364 | Vanilla baseline, 9L 512d, 1 train shard                                 |
| baseline_50step | 50    | 3.2273  | 5.4492   | 1.57s    | 5,193 | Vanilla baseline 9L 512d, 50 steps comparison                            |
| v2_50step       | 50    | 3.3192  | 5.6044   | 1.57s    | 5,226 | 11L 3xMLP all tricks - WORSE at low steps (bigger model needs more data) |
| v3_leakyrelu_50 | 50    | 3.2270  | 5.4486   | 855ms    | 9,506 | Baseline + LeakyReLU(0.5)^2 only - negligible diff at 50 steps           |
| baseline_500    | 500   | 2.0239  | 3.4172   | 1.51s    | 5,424 | Baseline 500 steps, 10 shards                                            |
| v3_leakyrelu_500| 500   | 2.0102 | 3.3941| 1.51s    | 5,515 | LeakyReLU(0.5)^2 validated at scale               |
| v3_smear2_500   | 500   | **2.0054** | 3.3860| 1.79s    | 5,498 | **NEW BEST: -0.019 bpb** LeakyReLU + SmearGate stacked |
| v4_byteweight   | 500   | 2.0065 | 3.3878| 1.46s    | 5,768 | LeakyReLU + byte-weighted loss (-0.017 bpb) |
| v5_combined     | 500   | 2.0093 | 3.3927| 876ms    | 8,873 | All 3 stacked — WORSE than SmearGate alone! ByteWeight hurts SmearGate |
| v6_wd04         | 500   | 2.0036 | 3.3830| 870ms    | 9,411 | LeakyReLU + SmearGate + WD=0.04 |
| v7_depthrecur   | 500   | 2.0024 | 3.3809| 1.01s    | 8,074 | v6 + depth recurrence (repeat layers 3-4) |
| v8_bigram       | 500   | 1.9932 | 3.3655| 1.02s    | 8,128 | Bigram logprob bias w=0.3. BROKE 2.0! |
| v8_bigram_w02   | 500   | 1.9922 | 3.3637| 1.01s    | 8,051 | w=0.2 slightly better than w=0.3 |
| v9_trigram      | 500   | 1.9712 | 3.3282| 1.01s    | 8,040 | Bigram(0.2) + Trigram(0.15) bias |
| v10_4gram       | 500   | 1.9519 | 3.2957| 1.02s    | 8,043 | Bi+Tri+4gram |
| v11_5gram       | 500   | **1.9428** | 3.2803| 1.01s    | 8,086 | **-0.081 bpb total!!** 2-5gram bias (65K buckets). Best absolute score |
| v12_8kbuckets   | 500   | 1.9663 | 3.3201| 1.02s    | 8,012 | 8K buckets: -0.024 penalty vs 65K |
| v13_16kbuckets  | 500   | 1.9567 | 3.3037| 1.02s    | 8,032 | 16K buckets: only -0.014 penalty vs 65K. Best size/quality trade-off |
| baseline_1000   | 1000  | 1.9257 | 3.2515| 1.46s    | 5,715 | Baseline 1000 steps scaling test |
| v12_1000step    | 1000  | 1.8841 | 3.1812| 1.74s    | 4,935 | N-gram 8K 1000 steps |
| bpe8192_base    | 500   | 1.8953 | 4.8956| 989ms    | 8,434 | BPE-8192 vanilla. -0.129 vs 1024v baseline |
| bpe8192_leaky   | 500   | 1.8910 | 4.8845| 1.59s    | 5,214 | BPE-8192 + LeakyReLU |
| bpe8192_ngram   | 500   | 1.8364 | 4.7433| 1.68s    | 4,922 | 8192v+LeakyReLU+ngram(16K). -0.188 total |
| bpe8192_1000    | 1000  | 1.7422 | 4.5001| 969ms    | 8,402 | ngram only 1000 steps |
| **best_1000**   | 1000  | **1.7397** | 4.4937| 2.73s    | 3,433 | **ABSOLUTE BEST! ngram+R. -0.284 vs 1024v baseline!** |
| bpe8192_wave    | 500   | 1.8678 | 4.8245| 2.33s    | 3,516 | Wave equation fwd pass! -0.023 vs LeakyReLU baseline |
| bpe8192_antiEnt | 500   | 1.7943* | 4.6347| 2.39s    | 3,397 | *CAVEAT: -0.05*entropy in loss (unfair bpb comparison) |
| bpe8192_wave_ng | 500   | 1.8368 | 4.7444| 3.15s    | 2,644 | Wave+ngram = SAME as ngram alone. Gains redundant. |
| bpe8192_ng_R    | 500   | 1.8342 | 4.7377| 2.42s    | 3,415 | ngram+period_bias |
| full_stack      | 500   | 1.8328 | 4.7341| 966ms    | 8,462 | ngram+R+knowledge_engine |
| ngram_dc500    | 500   | 1.8318 | 4.7314| 1.01s    | 8,017 | ngram+R+DC500 |
| best_26d      | 500   | 1.8269 | 4.7188| 2.75s    | 2,619 | Dual MLP stack |
| ultimate_v2    | 500   | 1.8220 | 4.7061| 1.70s    | 4,956 | +NorMuon+Turbo+sc20. -0.202 total |
| ultimate_v2_1k   | 1000  | **1.7279** | 4.4632| 3.29s    | 4,679 | Previous best |
| A6_sota_1k     | 1000  | 1.7131 | 4.4249| 1.79s    | 4,655 | Full SOTA arch |
| A1_11L_1k      | 1000  | 1.7110 | 4.4194| 2.77s    | 2,985 | 11L alone > full SOTA. Depth is king.                             |
| wavelet_1k     | 1000  | 1.6929 | 4.3728| 2.50s    | 3,578 | WaveletGPT training best |
| wavelet + eval α=0.08 | eval | 1.6864 | —  | —    | —     | -0.0065 from eval-time prob-space n-gram blend |
| wavelet + α=0.06 + T=0.93 | eval | 1.6805 | —  | —    | —     | -0.0124 from joint alpha+temperature |
| **stage2_lowlr (continual)** | **+300** | **1.6670** | 4.3058 | 3.39s | 2,392 | **NEW ABSOLUTE BEST! -0.0259 from continual training with 4x lower LR. Two-stage training WORKS at Mac scale.** |
| **stage2 + eval α+T (projected)** | eval | **~1.6546** | —  | —    | —     | **EFFECTIVE BEST if eval tricks stack on top: -0.038 vs wavelet_1k. -0.382 total from baseline.** |

## QUICK-REFERENCE: What to Implement Next (FINAL after 12 research cycles, Apr 4)

### TIER 1: Highest Impact, Mac-Testable (do these first)
1. **⭐ Signed hashing for n-gram tables** — 2 lines, **TESTED: +0.003 BPB FREE**
   Apply sign = hash2(prev2,prev1) % 2 * 2 - 1 during both table BUILD and LOOKUP.
2. **Complementary training (PR #803)** — ~10 lines, **PROVEN on H100**, est -0.005 to -0.015 BPB
   weight = clamp(1 - 0.5 * bigram_prob, min=0.1). Online stats accumulate during training.
3. **Dynamic token selection (skip bottom 20% loss)** — 5 lines, est -0.005 to -0.01 BPB
   mask = per_tok_loss > quantile(per_tok_loss, 0.2). Static n-gram selection DEAD on BPE-8192.
4. **Hyper-connections (DeepSeek ICLR 2025)** — ~10 lines, replace residual with learned multi-stream
5. **TOP auxiliary loss** — 1 extra head, ranking loss, est -0.005 to -0.015 BPB

### TIER 2: Fill the 3.6MB Gap
6. **Skip-bigram table** — 3-4 MB, tested +0.28 bits/tok signal, est -0.005 BPB
7. **DC1000 upgrade** — 2.9 MB, exp #135 showed DC1000 > DC500 by -0.079 at 50 steps
8. **5-gram table** — 3.5 MB, exp #147 showed -0.066 at 50 steps

### TIER 3: H100 Only (paradigm-level)
9. **Progressive seq length** (512→1024→2048) — 3x more steps in Phase 1, est -0.005 to -0.015 BPB
10. **Token selection + shard selection** — 2x training efficiency, est -0.02 to -0.05 BPB
11. **Meta-TTT (FOMAML)** — train model to be adaptable, est -0.02 to -0.08 BPB
12. **Parallel TTT search** (8 configs on 8 GPUs) — est -0.01 to -0.03 BPB
13. **SSE eval-time correction** (from cmix/PAQ) — zero model bytes, est -0.01 to -0.03 BPB
14. **Branch-merge training** (8 GPU MoT) — iterative weight averaging, est -0.01 to -0.02 BPB

### MAC LOSERS → H100 WINNERS (confirmed, implement all):
- 3x MLP expansion (all H100 submissions use this)
- WD=0.04 (all H100 submissions use this)
- seq_len=2048 (all H100 submissions use this)
- QK_gain=4.0-5.0 (PR #1176, PR #1302)
- Complementary training (PR #803, proven)
- Split-LR early/late (PR #1172, 1.1015 BPB)

### UNCERTAIN — WORTH H100 TEST (failed on Mac from wrong settings):
- UID at beta=0.0001 (beta=0.001 was too aggressive on Mac)
- WaveletGPT at w=0.001/layer (w=0.1 and 0.01 were 10x too high)
- Truncated backprop after step 3000 (at step 50 layers still learning)
- DC1000 at w=0.05 (w=0.15 overshoots at 500 steps; w=0.05 untested)
- Embedding dropout 5% (timed out on MLX, should work on CUDA)
- Entropy-gated skip connections (timed out on MLX, may work on CUDA)

### CONFIRMED DEAD (do NOT implement):
- MoR / depth recursion (quant error compounds 900x through GPTQ)
- Pure Mamba (282ms, breaks torch.compile, 10-15% tensor core utilization)
- MoE (capacity splitting kills it at 16MB)
- SVD/spectral embedding init (tied embed conflict, structural)
- Focused/self-referential/difficulty loss (catastrophically broken formulas)
- Running doc LayerNorm, bits-back loss, reservoir computing, pure PPM
- Sigma-delta / Floyd-Steinberg quant (weights lack spatial correlation)
- Knowledge distillation (no time for teacher), label smoothing (diluted at V=8192)
- Langevin gradient noise (kills convergence with Muon LR)
- Lookahead optimizer + Muon (slow weight interpolation hurts convergence speed)
- MDL weight decay (soft shrinkage too aggressive, +0.221 at 50 steps)
- Reservoir layers (frozen random blocks bypassed via U-Net residuals)
- Predictive Coding stacking (-0.040 at 50 steps but +0.038 at 1000 — convergence trap)
- Stacking WaveletGPT + SmearGate + 11L (1.7230 vs 1.6929 alone — too much overhead for Mac)
- Ternary STE (BitNet-inspired, +0.174 at 50 steps — too aggressive for Mac scale, H100 only)
- QK gain 4.0 (competition uses it but WORSE on Mac at 50 steps, needs 7000+ steps)
- MixUp for text (incoherent), logistic mixing (negligible vs additive)

### REVIVED FROM "DEAD" (was wrong about these):
- **Continual training works** with low LR (matrix_lr=0.01, 4x lower) — was failing because optimizer reset + full LR pushed model away from minimum. 300 extra steps = -0.0259 BPB.
- The previous "stage 2 fails" finding (#307) was caused by full LR. Lower LR fixes it.

### Current BPB progression:
```
SP-1024 baseline:           2.0239
SP-1024 + all tricks:       1.9428  (-0.081)
BPE-8192 baseline:          1.8953  (-0.129)
BPE-8192 + ngram + R:       1.8342  (-0.190)
BPE-8192 + ngram + R + DC:  1.8318  (-0.192)  ← MAC BEST (500 steps)
BPE-8192 + ngram + R 1000:  1.7397  (-0.284)
BPE-8192 + 11L + ult_v2:   1.7110  (-0.313)
BPE-8192 + WaveletGPT:     1.6929  (-0.331)
+ Continual lowLR (300st): 1.6670  (-0.357)  ← MAC BEST (training only)
+ Eval α=0.06 + T=0.93:    ~1.6546 (-0.382, projected)  ← MAC EFFECTIVE BEST
```

### Path to sub-1.00 on H100:
```
PROTEUS-like 11L stack:                    ~1.08 BPB
+ Token selection (RHO-1):                 -0.03  → 1.05
+ Complementary training:                  -0.01  → 1.04
+ Our n-gram + DC + English engine:        -0.04  → 1.00
+ Meta-TTT:                                -0.03  → 0.97
+ Parallel TTT search:                     -0.02  → 0.95
```


## Key Findings

1. **Architectural improvements don't help at low step counts** - v2 (11L, 3xMLP, all tricks) is WORSE than baseline at 50 steps
2. **LeakyReLU(0.5)^2** - negligible difference at 50 steps (~0.0003 bpb)
  1. **Compression: zstd-22 saves ~500KB over zlib-9** - free headroom for larger models
3. **Mac is for validating convergence-per-step, not absolute bpb** - real gains only show on H100 with full training
4. **Need to focus on H100 submission script**, not Mac optimization

## H100 Reference (from leaderboard)


| Run            | val_bpb | Author        | Notes                            |
| -------------- | ------- | ------------- | -------------------------------- |
| Naive Baseline | 1.2244  | Baseline      | 9L 512d 1024vocab, full training |
| SOTA (merged)  | 1.1147  | abaybektursun | 11L, GPTQ, XSA, BigramHash, TTT  |
| Best standard  | 1.0914  | bigbag        | QK-Gain 4.0 + Muon-TTT + SLOT (#1176) |
| Pending        | 0.93    | resouer       | Under compliance review (#1229) |
| Pending        | 0.40    | n/a           | "Swarm n-gram mixer" — under review (#1094) |

**Competition ends: April 30, 2026**

Key techniques from frontier submissions:
- int6 quant + selective layer precision + late QAT
- Complementary training: tokens predictable by bigram get lower loss weight
- Backoff n-gram mixer with entropy-adaptive alpha (orders 2-10)
- Full GPTQ with Fisher-weighted bit allocation


## Experiment Ideas

### High Priority (try first)

- v2 full stack (running now) - all table stakes ported to MLX
- Curriculum learning: train on simple/common patterns first, complex later
- Frequency-weighted loss: weight common words higher (they contribute more to bpb)
- Adaptive tokenizer: fewer bits for common tokens, more for rare

### Creative / Human-Inspired

- **DATA QUALITY FILTERING (TRY THIS):** Filter FineWeb training shards to only clean English prose. No code, no tables, no URLs, no abbreviations, no slang, no fake/made-up words. Only real English with common vocabulary and well-structured sentences. Hypothesis: with only 10 min of training, the model can't see all of FineWeb anyway — a curated high-quality subset may teach more per step, building stronger fundamental English patterns that generalize to the messy val set. NOTE TO OTHER CLAUDE: try filtering the training data to clean-only English. Strip out any documents containing code snippets, URLs, tables, non-English text, heavy abbreviations, or jargon. Keep only clear prose with real common English words. The val set is fixed FineWeb (messy web text), but cleaner training data might help the model learn fundamental patterns faster in limited steps.
- "Baby brain" progressive growth: start 4L, grow to 11L during training
- English structure prior: embed syntactic patterns (subject-verb-object)
- Semantic clustering: group related tokens, share representations
- Repetition exploitation: English is highly repetitive, exploit n-gram patterns
- Context window tricks: variable sequence length during training

### "Solve English" — Bake Language Knowledge Into the Artifact

English has a theoretical entropy of ~0.8-1.0 bits per byte (Shannon). SOTA is 1.1147 BPB. The remaining gap is mostly about better modeling of English structure, not architecture tricks. The top entries are already partially doing this (BigramHash = byte-pair frequency lookup, TTT = adapting to specific text). We should push this further:

- **N-gram backbone**: A classical trigram/4-gram byte model is tiny and captures most of English predictability. Use the neural model as a RESIDUAL on top — only predict what the n-gram model gets wrong. This means the neural net's limited capacity goes toward genuinely hard predictions instead of re-learning that "th" follows "e".
- **Hardcode grammar/structure priors**: English word order (subject-verb-object), common suffixes (-ing, -tion, -ly, -ed), articles before nouns, capitalization after periods — these are ~100% predictable patterns the model currently wastes parameters re-learning from scratch every training run. Encode these as rules or lookup tables in the artifact.
- **Frequency-weighted parameter allocation**: The top 1000 English words cover ~85% of all text. Dedicate disproportionately more model capacity (wider embeddings, more attention) to high-frequency tokens. Don't spend equal parameters on "the" and "xylophone".
- **Spend FEWER bits on context-specific words**: Words with very specific meanings (like "surgery", "photosynthesis", "parliament") are EASY to predict because they only appear in narrow contexts. The model barely needs capacity for them — context does the work. Spend the model's limited parameters on the HARD predictions: ambiguous words ("get", "thing", "run"), function words that appear everywhere, proper nouns, numbers, URLs. In practice this means: smaller embeddings or lower precision for rare/specific tokens, larger embeddings or higher precision for common/ambiguous tokens. The tokenizer could also help — merge specific-meaning words into single large tokens (cheap predictions covering many bytes) while splitting ambiguous contexts into smaller tokens (giving the model more granular control where it matters).
- **Word/phrase dictionary in the artifact**: Embed a compressed dictionary of common English words and phrases with their frequencies directly in the 16MB. The neural model handles surprises; the dictionary handles the predictable bulk.
- **Expand on BigramHash**: The SOTA already uses BigramHash (3072 features). Push this further — trigram hash, 4-gram hash, or context-dependent hashing. Look at what the top entry is actually doing in `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/` and see if we can extend it.

NOTE TO OTHER CLAUDE: The human's key insight is — instead of only optimizing architecture, try to "solve English" by encoding known language patterns directly. The neural model should be a residual on top of classical English knowledge, not learning everything from scratch. Look at BigramHash in the SOTA entry and see how to extend it (trigram, 4-gram, dictionary-backed). English is ~0.8 BPB theoretical minimum, we're at 1.11 — the gap is linguistic knowledge, not architecture.

### "Invent a New Language" — Tokenizer as a Compression Lever

The tokenizer IS "inventing a language" — it maps English bytes into a token sequence the model predicts. The BPB formula is `(cross_entropy / log(2)) * (tokens / bytes)`. You win by: (1) fewer tokens per byte = fewer predictions, (2) more predictable tokens = lower cross-entropy. The leaderboard already proves this — 8192 BPE beats 1024 BPE. This lever is massively under-explored compared to architecture tricks. The tokenizer is tiny (few hundred KB) so it barely dents the 16MB budget.

- **Deterministic text normalization layer**: Before tokenizing, normalize the text deterministically — lowercase, expand contractions ("don't" → "do not"), spell out numbers ("42" → "forty two"), normalize Unicode/whitespace. This makes the input more regular and predictable. As long as the mapping is deterministic and byte counts are tracked correctly for BPB, this is free entropy reduction. The model sees cleaner patterns.
- **Custom tokenizer trained on FineWeb val distribution**: Instead of generic BPE, build a tokenizer specifically optimized for the byte-level patterns in FineWeb. Maximize bytes-per-token for the most common patterns in the actual eval data. This is legal — you can use any tokenizer.
- **Hierarchical / multi-level encoding**: First pass groups bytes into word-level units, second pass groups those into phrase-level units. Model predicts at phrase level — much more predictable. Like compressing twice.
- **Entropy-rebalancing tokenizer**: Some tokens are trivial to predict ("the", "is", "a"), some are near-impossible (proper nouns, numbers, URLs). Design the tokenizer so every token is roughly equally hard to predict — that's the optimal allocation for a fixed-capacity model. Easy tokens should be merged into larger units; hard tokens should be split smaller.
- **Grammar-aware tokenization**: Instead of statistical BPE, tokenize at syntactic boundaries (noun phrases, verb phrases). The model predicts grammatical units instead of arbitrary subwords — aligns with how English actually works.

### Tokenizer Experiments — ACTUALLY TRY THESE

Almost nobody on the leaderboard has touched the tokenizer (24/25 entries use default 1024 BPE). Only the Ternary Quantization entry uses 8192 BPE. This is a massively under-explored lever. The competition explicitly encourages "novel tokenizers". Tooling already exists — add a spec to `data/tokenizer_specs.json` and run `data/download_hf_docs_and_tokenize.py`.

1. **Bump to 8192 BPE (low-hanging fruit):** Already proven to work. Bigger vocab = fewer tokens = fewer predictions = lower BPB. Only costs a few hundred KB more in the 16MB budget. Try this first.
2. **Go bigger — 16384 or 32768 BPE:** More bytes per token, but the embedding table grows. Find the sweet spot where the vocab size gain outweighs the parameter cost.
3. **Custom BPE trained on FineWeb distribution:** Train the tokenizer specifically on FineWeb to optimize vocabulary for the actual eval text patterns. Probably marginal gains over standard BPE on same-distribution data, but worth trying — it's cheap.
4. **Exotic tokenizers — entropy-balanced, grammar-aware:** Design vocab so every token is roughly equally hard to predict (optimal for fixed-capacity model). Or tokenize at syntactic boundaries instead of statistical BPE boundaries. These are the high-risk high-reward ideas.

NOTE TO OTHER CLAUDE: The tokenizer is the #1 under-explored lever in this competition. Almost nobody has changed it. Start by trying 8192 BPE (proven), then 16384/32768. The tooling exists in `data/tokenizer_specs.json` + `data/download_hf_docs_and_tokenize.py`. Tokenizer only costs a few hundred KB of 16MB budget but directly affects BOTH terms of the BPB formula. WARNING: the README says tokenizer-change submissions get extra scrutiny — make sure BPB calculation is correct.

### Unhinged / Moonshot Ideas — Try These, Embrace Failure

These are genuinely novel ideas that might fail miserably but could also break the game open. Nobody on the leaderboard has tried anything like these.

#### Reframing the Problem

- **PAQ/PPM as the model**: Compression IS prediction. PAQ-style context mixing is literally a language model — it just doesn't use neural nets. PAQ8 achieves ~1.2 BPB on English text in a tiny binary. What if the 16MB artifact IS a PAQ-variant with a neural net as one of its context models? Hybrid classical compression + neural. The PAQ family has 30 years of engineering for exactly this problem.
- **Kolmogorov complexity attack**: Treat the 16MB as a program that outputs predictions. Use genetic programming / program synthesis to search for the shortest program that predicts English. The model IS code, not weights. A sufficiently clever program could encode English grammar, common words, and statistical patterns in far less than 16MB.
- **Learned compression prior / VAE decoder**: Instead of next-token prediction, train a VAE where the decoder IS the model. The 16MB artifact is the compressed latent representation of English itself. At eval time, decompress and decode.
- **Anti-tokenizer (arithmetic coding IS the model)**: Use arithmetic coding as the tokenizer. The coding scheme encodes probability distributions. Every byte is one "token" but the arithmetic coder does the heavy lifting. The tokenizer and the model become the same thing.

#### Architecture Heresy

- **Byte-level mixture of experts**: Different tiny expert networks for different byte contexts — after vowels, after punctuation, after digits, after spaces, etc. Route at the byte level, not token level. 16MB buys you hundreds of tiny experts. English has maybe ~20 distinct byte regimes.
- **Fractal / recursive model**: Same tiny network (~100KB) applied recursively at byte, word, sentence, and paragraph level. Like a U-Net for sequences. The model learns scale-invariant patterns. English has self-similar structure at every scale.
- **Byte-pair graph neural network**: Don't model text as a sequence. Model byte relationships as a GRAPH. Edges connect co-occurring bytes/n-grams. Run a GNN. Text is a path through the graph. The graph structure IS English.
- **Cellular automata language model**: Use Rule 110 or similar Turing-complete CA as the compute primitive instead of matrix multiplies. The "weights" are the initial CA state + rule table. Might be insanely parameter-efficient for the patterns it can express.
- **Spiking neural network**: Binary activations, temporal coding instead of floating point. Could be extremely parameter-efficient — each "weight" is 1 bit but the timing carries the information. There's a whole field of neuromorphic computing that never gets applied to LMs.
- **Tensor network / Matrix Product States**: Quantum-inspired tensor decomposition. MPS can represent exponentially large tensors compactly. A single MPS layer with bond dimension 256 can represent correlations that would take a massive dense layer. Nobody has tried this for byte-level LM.

#### Eval-Time Tricks

- **Retrieval from self**: During eval, use the model's own hidden states from earlier in the document as a key-value memory. Like TTT but at the representation level — no gradient updates, just nearest-neighbor lookup in activation space. The document becomes its own retrieval corpus.
- **Ensemble of checkpoints**: Save 5 checkpoints during training (different epochs). At eval, run all 5 and average predictions. Costs 5x eval compute but eval budget is generous (10 min on 8xH100). Model diversity for free.
- **Adaptive compute per byte**: Some bytes are trivial ("e" after "th"), some are hard (first letter of a new word). Run more layers / more iterations on hard bytes, fewer on easy ones. Early-exit transformer. Spend the eval time budget where it matters.

#### Truly Deranged

- **Train on the tokenizer, not the data**: What if you spend most of your 10 minutes training the TOKENIZER to be optimal, and only a few minutes training the model? An optimal tokenizer with a mediocre model might beat a mediocre tokenizer with an optimal model. The tokenizer determines the upper bound.
- **Random projections + memorization**: Use random FIXED projections (no learned features, zero training needed). Spend all 16MB on memorizing correction factors for the random model's mistakes. Johnson-Lindenstrauss says random projections preserve structure. The corrections might be highly compressible.
- **Physics simulation of language**: Model English as a dynamical system. Words are trajectories in phase space, grammar is the attractor. The model is an ODE/SDE solver. "The" → "cat" is a trajectory, not a probability. Fit the dynamics, not the distribution.
- **Adversarial compression**: Train two models — a generator that tries to produce text the predictor can't predict, and a predictor that tries to predict it. The predictor, hardened by adversarial training, should be better at predicting real text. GAN but for compression.
- **Model stacking / gradient boosting for LMs**: Train 50-100 tiny models (150KB each), where each one predicts the RESIDUAL error of all previous models combined. Gradient boosting but for language modeling. The ensemble fits in 16MB. Each model specializes in what the others miss.
- **DNA-inspired error-correcting weights**: Encode weights with biological-style redundancy — each weight is stored 3x with majority voting. Sounds wasteful but it means you can quantize to 1 bit per copy and still recover 1-bit-error-free weights. 3 bits total per weight with built-in error correction.
- **The quine model**: A model whose training objective includes predicting its own weights. Self-referential. The model that best understands itself best understands structure. Wild, possibly meaningless, but if it works the implications are insane.
- **Synesthesia model**: Map bytes to 2D spatial positions (e.g., by frequency and co-occurrence). Treat text as a trajectory through a 2D image. Use convolutions instead of attention. Image models are extremely parameter-efficient for spatial patterns. Text might have spatial structure we're not exploiting.
- **Lossy English**: What if some bytes literally don't matter? "colour" vs "color", double spaces, optional commas, capitalization inconsistencies. Identify bytes with near-zero information content and predict them as a fixed default. Spend zero model capacity on them. A "lossy" language model that's lossless where it counts.

### Max Out the Code — The 16MB is Code + Model

The artifact is code bytes + compressed model bytes. Everyone focuses on the model. But the CODE is also inside the 16MB. What if the code itself does heavy lifting?

#### Deterministic Pre-Training (Jump-Start Weights)

The idea: compute analytically good starting weights BEFORE gradient descent. Every step of training is precious (10 min cap). If you start closer to the optimum, you need fewer steps.

- **Analytical bigram/trigram embeddings**: Compute exact bigram/trigram statistics from training data. Initialize embedding matrix so that dot(embed[A], embed[B]) ≈ log P(B|A). The model starts already knowing English co-occurrence. This is computable deterministically from data.
- **SVD-initialized projections**: Compute the SVD of the token co-occurrence matrix. Use top-k singular vectors as Q/K/V initialization. The attention weights start aligned with the actual statistical structure of English.
- **Spectral initialization from data**: Compute the eigenvectors of the byte transition matrix. Use these as the initial basis for the model. The model starts in a coordinate system aligned with English, not random.
- **Weight prediction from architecture**: Given the architecture (layer sizes, etc), can you PREDICT what the trained weights will look like? Train a tiny meta-model offline that predicts "good weights for a transformer of size X trained on English." The predicted weights become your initialization. This is like distillation but the teacher is a weight-predictor, not a text-predictor.
- **Deterministic feature engineering in code**: Hardcode a massive lookup table of English patterns AS CODE (not weights). "th" → high probability of "e", "qu" → high probability of "u", etc. This is recomputable from the training data deterministically, so it's legal. The code generates the tables at runtime.
- **Pretrain the tokenizer on the training data statistics**: Before training the model, spend 30 seconds computing optimal token merges, byte frequencies, n-gram statistics. Build these into the model's initial state.

#### Bit-Level Math — Squeeze Every Bit

- **Custom number format**: Instead of int8 or int6, design a number format optimized for neural net weight distributions. Weights are roughly Gaussian — a format with more resolution near 0 and less at the tails could store more information per bit.
- **Entropy-optimal weight encoding**: Weights aren't uniformly distributed. Use Huffman or arithmetic coding on the WEIGHT VALUES (not the text). Common weight values (near 0) get fewer bits, rare values (large magnitude) get more. Could save 10-20% over uniform quantization.
- **Shared exponents (block floating point)**: Groups of 16-32 weights share one exponent. Each weight is just a 4-5 bit mantissa. Microsoft's MSFP format does this. Much denser than per-weight quantization.
- **Delta encoding of weights**: Adjacent weights in the same layer tend to be similar. Store the first weight, then deltas. Deltas are smaller → more compressible. Like how video compression stores frame differences.
- **Weight matrix as an image**: Treat each weight matrix as a grayscale image. Apply JPEG-style compression (DCT + quantization). Neural net weights often have smooth spatial structure that image codecs exploit well.
- **Kolmogorov-optimal code**: The code portion should be as SHORT as possible (every byte of code is a byte not spent on model). Minify aggressively. Use the most terse Python possible. Reuse stdlib functions. The ideal is a tiny decompressor + a blob.

#### Ideas From Outside ML Entirely

These are from other fields. Nobody in ML has tried most of these.

- **Lempel-Ziv as a language model**: LZ77/LZ78 are literally predictive models — they predict the next byte based on previously seen patterns. The dictionary IS the model. What if you run LZ78 on the training data and use the resulting dictionary (which fits in a few MB) as your predictor? Augment with a tiny neural net for novel patterns only.
- **Shannon's guessing game, literally**: Shannon measured English entropy by having humans guess the next letter. What if the model encodes a human-like guessing strategy? First guess: most common byte in context. Second guess: second most common. Etc. The model stores a priority queue of guesses per context. This is exactly what n-gram caches do, but framed differently it might suggest different implementations.
- **Information-theoretic partitioning**: Split the prediction problem by information content. Some bytes carry ~0 bits of information (the "u" after "q", the space after a period). Some carry ~4-5 bits (first letter of a new topic). Build separate sub-models for each information tier. Allocate capacity proportional to information content.
- **Markov chain Monte Carlo at eval time**: Don't just predict the most likely next byte. Sample multiple futures, see which is most consistent with the model's world knowledge, then commit. Like how chess engines look ahead. Costs eval compute but the eval budget is generous.
- **Genetic algorithm weight search**: Don't use gradient descent at all. Start with random 16MB blobs. Mutate them. Select the ones that predict English best. Crossover the winners. Repeat for 10 minutes. Evolution is embarrassingly parallel on 8xH100s. No backprop needed.
- **Analog signal processing on byte streams**: Treat the byte stream as a discrete-time signal. Apply IIR/FIR filters, wavelets, Fourier analysis. English text has characteristic frequency spectra (word length cycles, sentence length cycles, paragraph structure). A bank of matched filters could predict periodicity cheaply.
- **Game theory / minimax**: Model the prediction problem as a two-player game: the predictor vs. English. The predictor commits to a strategy (model weights), English reveals the next byte. The minimax optimal strategy is provably the best worst-case predictor. Solve for it analytically for small contexts, use neural approximation for large ones.
- **Protein folding inspired**: AlphaFold predicts 3D structure from 1D sequence. What if we do the reverse — predict 1D sequence from "structure"? Treat English text as having hidden 3D structure (topic space, sentiment space, formality space). The model predicts the next byte by navigating this structure. Basically a structured latent variable model, but inspired by the protein folding framing.
- **Error-correcting codes as architecture**: Reed-Solomon, turbo codes, LDPC — these are mathematically optimal for transmitting information through noisy channels. English text IS a noisy channel (typos, formatting variation, multiple valid phrasings). Use coding theory to build the model's architecture. The parity check matrix IS the weight matrix.
- **Reservoir computing**: A large RANDOM recurrent network (the "reservoir") with fixed weights + a tiny trained linear readout. The reservoir provides a rich nonlinear expansion of the input for free (no training needed). Only the readout layer needs training. Could fit a massive reservoir in 16MB since the weights are generated from a seed, not stored.
- **Finite automata / regex ensemble**: English grammar is mostly regular/context-free. Build a massive bank of regex patterns and finite automata that match English patterns. Each automaton votes on the next byte. The ensemble of automata IS the model. Fits great in 16MB — automata compress well.
- **Holographic reduced representations**: Encode words as high-dimensional random vectors. Combine them with circular convolution (not concatenation). The resulting vector encodes the entire sequence history in a fixed-size vector. Prediction is inverse convolution. This is from cognitive science — Plate (1995). Extremely compact representation.
- **Number theory / prime factorization encoding**: Map each possible n-gram to a unique prime number. The "state" of the model is the product of all primes seen so far. Factor the product to recover the context. Prediction = which primes are likely to appear next given the current product. Sounds insane but Chinese Remainder Theorem makes this tractable for bounded contexts.
- **Thermodynamic computing**: Model prediction as energy minimization. Each byte has an "energy" in context. The model stores an energy function (Boltzmann machine style). Prediction = lowest energy next byte. Training = fitting the energy landscape to English. Restricted Boltzmann Machines are tiny and trainable.

#### Spatial / Geometric — Solve the Model, Don't Train It

The transformer is not a black box. It's a series of linear maps + nonlinearities. What if we treat it as a GEOMETRY problem and solve for the weights directly?

- **Weight space as a manifold**: The set of "good" 16MB models for English prediction forms a manifold in weight space. Don't random-walk toward it with SGD — try to characterize the manifold analytically and jump onto it. The co-occurrence statistics of English define the target manifold. PCA/SVD of the data gives you the tangent space at the origin.
- **Solve attention analytically**: For a single attention head, the optimal Q/K/V matrices for bigram prediction have a CLOSED FORM solution — it's just the matrix that maps token A to a query that matches token B's key proportionally to P(B|A). Compute this exactly. No training needed for the first layer. Stack learned residuals on top.
- **Geometric embedding**: Place tokens in a metric space where distance = negative log probability of co-occurrence. This is just word2vec done analytically. But then go further — the ATTENTION operation is a nearest-neighbor search in this space. If you initialize Q/K matrices to implement this metric, the model starts with a geometrically correct attention pattern.
- **Layer-wise closed-form solutions**: Each transformer layer takes an input distribution and maps it to an output distribution. For a known input distribution (which you CAN compute from training data statistics), the optimal linear map has a closed-form least-squares solution. Solve each layer in sequence: compute input stats → solve for optimal weights → compute output stats → repeat. Training is just fine-tuning the residual error.
- **Solve the loss landscape directly**: The loss function for a small model on known data statistics is a polynomial in the weights. For very small models (or individual layers), you can find the global minimum analytically using algebraic geometry / Gröbner bases. This is computationally insane for full models but might work layer-by-layer or for individual attention heads.
- **Sparse coding / dictionary learning**: English text lives in a low-dimensional subspace of byte-sequence space. Find the basis vectors (dictionary atoms) of this subspace from training data. The model's job becomes: project input onto dictionary, predict coefficients of next dictionary atom. The dictionary is computable deterministically.
- **Voronoi partition of context space**: Partition all possible byte contexts into regions (Voronoi cells). Each cell gets its own simple predictor (just a probability table). At eval time, find which cell the current context falls in, use that cell's predictor. The partition + per-cell tables fit in 16MB. This is a geometric version of mixture of experts.
- **Optimal transport for prediction**: The next-byte prediction problem is: given a distribution over contexts, produce a distribution over next bytes. This is LITERALLY an optimal transport problem. The Wasserstein distance between your prediction and the truth has known optimal solutions for discrete distributions. The model's weight matrix IS the transport plan.

NOTE TO OTHER CLAUDE: The user wants WEIRD ideas that might fail. Try any that seem tractable. The research below is from Wave 3 (linguistics, math/physics, obscure CS) — the most actionable findings from genuinely novel domains.

### NEW RESEARCH — Wave 3: Actionable Cross-Domain Finds (Apr 3)

#### 1. UID Regularizer — 5 lines, PROVEN with limited data (DO THIS FIRST)
Uniform Information Density (Wei, Meister & Cotterell, ACL 2021). Penalize variance of per-token surprisal.
```python
per_tok_loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), reduction='none')
loss = per_tok_loss.mean() + 0.01 * per_tok_loss.var()  # beta=0.01
```
Effect is LARGER with limited training data — our exact situation. 0.005-0.02 BPB. **Zero cost.**

#### 2. PETE Embeddings — Kill the embedding table, save 4MB (BPE-8192 game-changer)
Replace `nn.Embedding(8192, 512)` with Fourier expansion of token IDs + tiny MLP (arxiv 2505.02266).
Saves ~4MB → 2-3 extra transformer layers. Code: `github.com/HMUNACHI/pete`. Estimated: 0.02-0.05 BPB.

#### 3. MorphBPE — Morpheme-aware tokenizer (arxiv 2502.00894)
Constrain BPE merges to respect morpheme boundaries. Drop-in for SentencePiece. Consistent CE improvement. 0.01-0.03 BPB.

#### 4. DCT Weight Compression — JPEG for weights
Apply 2D DCT to weight matrices before quantization. Zero small coefficients. 10-30% better compression → more params. 0.02-0.05 BPB.

#### 5. Substitution-Tolerant N-gram Hashing
Normalize bytes before hashing (lowercase, collapse whitespace/quotes). Free. From genomic compressor GeCo3. 0.002-0.01 BPB.

### NEW RESEARCH — Wave 4: Training, Architecture, Eval, CS Algorithms (Apr 3)

#### TRAINING — Get More Learning Per Step

**6. WaveletGPT Auxiliary Loss — 40-60% faster convergence, ZERO params (arxiv 2409.12924)**
Apply causal Haar wavelet to intermediate hidden states, add aux prediction losses at coarser scales. Multi-scale loss aligns with how language has structure at byte/word/sentence/paragraph levels. Paper shows 0.04 val loss improvement AND 40-60% faster convergence.
```python
# After each layer, add coarse-grained prediction loss:
h_coarse = (h[:, 0::2, :] + h[:, 1::2, :]) * 0.7071  # Haar downsample
aux_loss += cross_entropy(proj(h_coarse), targets[:, 1::2]) * 0.1 / n_layers
```
**Zero extra params. ~10 lines. 40-60% faster convergence = 40-60% more effective steps in 10 min.**

**7. Late-SAM for Quantization-Friendly Training (ICLR 2025)**
Sharpness-Aware Minimization finds flatter minima that survive GPTQ int6 better. Apply ONLY during warmdown phase (last ~1000 steps). Paper proves late-SAM works as well as full-training SAM.
- Current GPTQ gap: ~0.02 BPB (1.1354 pre-quant → 1.1147 post-quant)
- SAM could reduce this gap by 30-50% = **-0.006 to -0.010 BPB**
- LookSAM variant: apply every 5th step = only 20% overhead
```python
# SAM step: perturb, compute loss, unperturb, use perturbed gradient
eps = 0.05 * grad / grad.norm()
params += eps; loss2 = forward(); params -= eps
# Use grad from loss2 for actual update
```

**8. Progressive Growing — Start small, stack layers during training (NeurIPS 2024 Spotlight)**
G_stack: train 4L model for 2000 steps (cheap, fast), duplicate to 8L, duplicate to 11L. Early layers are well-trained before upper layers begin. 54.6% FLOPs speedup proven.
- 4L steps cost ~40% of 11L steps → get 2.5x more steps in early training
- Combined: ~30-50% more total gradient updates in 10 minutes
- Implementation: duplicate layer weights at growth points, reset LN scales

#### ARCHITECTURE — More Capacity Without More Parameters

**9. Michaelis-Menten Activation — genuinely novel, NEVER tried in ML**
From enzyme kinetics: `f(x) = Vmax * |x| / (Km + |x|) * sign(x)`. Learnable saturation with per-neuron Vmax and Km. Unlike ReLU (hard threshold) or GELU (fixed shape), this adapts sensitivity per feature.
- 2 params per hidden dim = 1024 extra params total. Negligible.
- Each neuron learns its own "sensitivity threshold" — like a learnable softcap.
- **Nobody has ever tried this as a neural net activation. Genuinely novel.**

**10. Timestep-Conditioned Loop — FIX the 3+ loop catastrophe, get free virtual layers**
Our depth recurrence 3+ loops was CATASTROPHIC (+4.3 BPB) — but that was WITHOUT conditioning. With sine-cosine timestep encoding + tiny MLP, each loop does different work (ICML 2025: looped transformers are universal approximators WITH conditioning).
```python
# Per-loop conditioning: tell the block which iteration it is
t_emb = sin_cos_embed(loop_idx)  # [dim] — unique per iteration
gamma1, gamma2 = conditioning_mlp(t_emb)  # tiny 2-layer MLP, ~2K params
h = layer_norm(h) * gamma1  # modulate per-loop
```
- Could enable 3-4 loops safely → 12-15 virtual layers from 9 physical. **Multiplicative capacity gain.**
- Sandwich sharing (unique first/last, shared middle) + conditioning = 40% param reduction (Subformer, EMNLP 2021)
- Also: **Gated Attention** (NeurIPS 2025 Best Paper) — add sigmoid gate after SDPA. 8 params/layer. Free training stability win.

**10b. Factorized Embeddings (ALBERT-style) — save 4MB on BPE-8192**
Instead of 8192×512 (4.19M params), use 8192×64 × 64×512 (524K + 32K = 556K params). Same effective dimension, 7.5x fewer embedding params. ALBERT proved this works. Combined with PETE: potentially even smaller.

#### EVAL-TIME — Squeeze Every Drop From 10-Min Budget

**11. Beam Search Over SLOT Configs — 8 configs in parallel, pick best (HIGHEST VALUE)**
Run 8 different SLOT hyperparameter configs on 8 H100s simultaneously. Pick the winner per batch. Configs: vary lr (0.004/0.008/0.012), steps (8/16/24), weight_decay. Same wall-clock as single SLOT.
- **-0.005 to -0.015 BPB on top of single SLOT. Nearly free compute.**
- Implementation: fork SLOT code, parameterize, distributed reduce.

**12. Progressive Eval Refinement — fast scan then focused compute**
Pass 1: stride=256 (~25s) to estimate per-token entropy. Pass 2: stride=32 on hard regions only (remaining ~475s). Focus compute where it matters most. 80% of tokens are easy; the hard 20% contribute disproportionately to BPB.
- **-0.005 to -0.015 BPB from targeted compute allocation**

**13. Temperature Scaling — trivial, free**
Learn optimal T from scored tokens: `logits_scaled = logits / T`. One scalar, 5 lines. -0.001 to -0.003 BPB. Do this because it costs nothing.

#### OBSCURE CS — Novel Data Structures for N-gram Storage

**14. Count-Min Sketch N-gram Tables — 72x compression!**
Replace 9.2MB hash tables with CMS: 4 rows × 8192 cols × 4 orders = **128KB total**. Frees ~9MB for bigger neural model. Overcounts (never undercounts) but CMS estimator (take min across rows) is surprisingly accurate. Count-Min-Log variant handles low-frequency items better.

**15. Minimal Perfect Hash (MPH) N-gram Tables — dense, exact, ~3x smaller**
Map exactly N observed n-grams to [0..N-1] with zero wasted slots. 1.41 bytes per n-gram (Guthrie & Hepple 2010). Trigram: **1.13MB vs 3.9MB** for 8K-bucket hash table. Library: `bbhash` (Python).

**16. Suffix Automaton at Eval — context-adaptive n-gram weighting, ZERO artifact cost**
Build dynamic suffix automaton over scored tokens during eval. Finds longest suffix match in O(1). Match length determines which n-gram order to trust. Replaces fixed weights with data-driven adaptive weights. From SAM Decoding (ACL 2025, arxiv 2411.10666).

**17. MLP Memory — learned n-gram approximation in 520KB**
Train a tiny 2-layer MLP (256 hidden) to map hidden states → next-token logit bias. Replaces ALL n-gram tables (9.2MB) with a 520KB learned function. Jointly fine-tunable. From kNN-LM + MLP Memory literature.

### Compression — What's Proven and What to Try

Current landscape: baseline uses int8+zlib (~~17M params in 16MB). SOTA uses int6+LZMA+GPTQ (~~22M params). Ternary fits 73.7M params. Binary fits 106.2M. Lower bits = more params = more capacity to model English.

**Free wins (do these first):**

- **LZMA over zlib**: ~0.5MB free savings, zero downside. SOTA already uses it. Just swap the compressor.
- **Int6 instead of int8**: 25% more headroom for parameters. Most top entries already use this.
- **zstd-22 as middle ground**: Better than zlib, faster decompression than LZMA.

**Proven techniques from leaderboard:**

- **Mixed precision by layer**: Int5 for MLP weights (less precision-sensitive), int6 for attention (more sensitive). Proven by thwu1 entry — freed ~1.86MB to add a 10th layer.
- **GPTQ (post-training quantization)**: Instead of naive round-to-nearest, uses Hessian info to minimize quantization error. The SOTA's secret weapon — worth ~0.006 BPB alone. Uses AR self-generated calibration data.
- **QAT (quantization-aware training)**: Train with STE (straight-through estimator) quantization noise so the model learns to be robust to low-precision. Multiple top entries use late-stage QAT (enable when LR scale < 0.15 during warmdown).
- **EMA + SWA weight averaging**: Exponential moving average (decay=0.997) + stochastic weight averaging (every 50 steps). Smooths out noise before quantization. Worth ~0.001-0.002 BPB.

**Aggressive / high-risk:**

- **Ternary {-1,0,+1}**: 1.6 bits/param, fits 73.7M params in 16MB (4x more than int6). Uses base-3 packing (5 trits per byte) + LZMA. Scored 1.1570 BPB. The zero state helps LZMA compress further.
- **Binary {-1,+1}**: 1 bit/param, fits 106.2M params in 16MB (6x more). Scored 1.1239 BPB but needed 2hrs training — doesn't meet 10min constraint yet. If training can be optimized, this is the frontier.
- **FP8 for non-quantized params**: Use e4m3 format for embeddings, norms, scales instead of FP16/FP32. Saves bytes on the "overhead" parameters.

**Budget math (approximate):**

```
16MB total - ~0.3MB tokenizer/code = ~15.7MB for weights
Int8:    15.7MB / 1 byte  = ~15.7M params
Int6:    15.7MB / 0.75    = ~20.9M params  
Int5:    15.7MB / 0.625   = ~25.1M params
Ternary: 15.7MB / 0.2     = ~78.5M params
Binary:  15.7MB / 0.125   = ~125.6M params
```

NOTE TO OTHER CLAUDE: Priority order for compression: (1) swap zlib→LZMA (free), (2) int8→int6 (free params), (3) add late-stage QAT with STE, (4) add EMA/SWA weight averaging before quantization, (5) try GPTQ if time permits. Reference implementations exist in the records/ folder — check the SOTA entry at `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/` and ternary at `records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/`.
| val_28a        | 500   | 1.8538 | 4.7884| 1.49s    | 8,576 | 2.8a CTW entropy mix (bi+tri only). Worse than full stack (1.8269) |
