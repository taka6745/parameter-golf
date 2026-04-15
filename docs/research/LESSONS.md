# Parameter Golf - Lessons Learned

## Mac Development Lessons

### 1. Architectural changes don't help at low step counts
- v2 (11L, 3xMLP, XSA, BigramHash, SmearGate) was WORSE than baseline at 50 steps
- Bigger models need more data/steps to converge. With 1 shard, baseline wins
- At 500 steps with 10 shards, improvements start showing

### 2. LeakyReLU(0.5)^2 is a validated win
- -0.014 bpb at 500 steps (2.0102 vs 2.0239)
- Zero-cost change: one line in MLP forward
- Gap grows with more training steps

### 3. SmearGate is a validated win (-0.019 bpb stacked with LeakyReLU)
- val_bpb 2.0054 vs baseline 2.0239
- mx.concatenate in hot loop caused memory leak — fixed with mx.pad
- Stacks well with LeakyReLU (better than either alone)
- ~20% throughput cost (1.79s vs 1.51s/step) — worth it for the bpb gain

### 3b. Byte-weighted loss is a validated win (-0.017 bpb)
- val_bpb 2.0065 vs baseline 2.0239
- Weight each token's loss by bytes it represents — aligns training with bpb metric
- Zero throughput cost (1.46s/step, same as baseline)
- Train loss numbers are higher due to weighting but val_bpb is what matters

### 3c. Techniques don't always stack additively
- ByteWeight + SmearGate combined (2.0093) was WORSE than SmearGate alone (2.0054)
- ByteWeight changes gradient distribution, which may conflict with SmearGate's learning
- Best known combo: LeakyReLU + SmearGate (no ByteWeight)
- Always test combinations — don't assume gains are additive

### 4. Compression is free improvement
- zstd-22 saves ~500KB over zlib-9 (5.11MB -> 4.61MB)
- LZMA-6 saves ~400KB
- This headroom can be spent on more parameters

### 4b. Depth recurrence is a validated win (-0.001 bpb incremental)
- val_bpb 2.0024 vs v6's 2.0036 (incremental -0.0012 from recurrence alone)
- Repeat encoder layers 3-4 once more: 9 physical layers -> 11 virtual layers
- 16% throughput cost (1.01s vs 0.87s/step)
- Zero extra parameters! Just reuses existing layers
- On H100, throughput cost may be lower (Flash Attention is more efficient with depth)

### 4c. Bigram logit bias is a BREAKTHROUGH (-0.009 bpb incremental, -0.031 total)
- val_bpb 1.9932 — BROKE BELOW 2.0!
- Precompute bigram P(next|prev) from training data (100M tokens, 1024x1024 table)
- Add as additive logit bias: logits += 0.3 * bigram_logprobs[prev_token]
- Model learns RESIDUAL on top of bigram knowledge — doesn't waste capacity on obvious patterns
- Step 1 loss drops from 6.93 to 5.38 — massive head start!
- This validates the user's "solve English" hypothesis: baking language knowledge directly works
- Weight 0.3 was first try — tuning may yield more
- 4MB table size (1024x1024 float32) — tiny cost for huge gain
- On H100 with full training, this should translate to large bpb improvement at competition scale
- NEXT: try trigram, 4-gram, try during eval only vs train+eval
- Weight tuning: w=0.2 is optimal (1.9922), w=0.3 close (1.9932), w=0.5/1.0 worse
- Higher weights overwhelm model's own predictions — 0.2-0.3 is the sweet spot
- DON'T run parallel weight tests on Mac — GPU contention invalidates train loss comparisons

### 4d. Trigram logit bias is ANOTHER massive win (-0.021 on top of bigram)
- val_bpb 1.9712 vs bigram-only 1.9922 — total -0.053 vs baseline!
- Uses hash(prev2, prev1) into 65536 buckets for trigram lookup
- 256MB table (65536 x 1024 float32) — too large for 16MB artifact, needs compression
- Each n-gram order gives ~0.02 bpb: bigram -0.032, trigram -0.021
- Diminishing returns expected: 4-gram might give -0.015, 5-gram -0.010
- The "solve English" approach is VALIDATED — classical n-gram knowledge injection works
- Competition artifact limitation: n-gram tables must fit in 16MB. Solutions:
  - Quantize tables to int8 (4x compression)
  - Use fewer hash buckets (e.g. 16K instead of 65K)
  - Apply zstd compression on top
  - Only use during eval (doesn't need to be in training artifact)
  
### 4e. CRITICAL INSIGHT: n-gram tables for training vs eval are DIFFERENT problems
- **Training**: n-gram bias from training data, baked into the training script. Must fit in artifact or be recomputable
- **Eval**: n-gram cache built dynamically from already-scored validation tokens. This is "legal score-first TTT"
- Eval-time cache is FREE (no artifact cost) and uses actual val distribution (better match)
- Competition allows up to 10 min eval time on 8xH100
- Strategy: use moderate-size n-gram tables for training (fit in artifact), plus eval-time dynamic cache
- 8K buckets: -0.024 penalty vs 65K but feasible for artifact (~15MB int8+zstd)
- 16K buckets: untested, likely ~-0.012 penalty, ~30MB int8+zstd (tight)
- The eval-time dynamic cache could recover or exceed the quality gap from smaller training tables
- 16K buckets is the sweet spot: only -0.014 penalty vs 65K but 4x smaller tables
- Bucket trade-off: 65K=best quality, 16K=-0.014, 8K=-0.024

### 4f. N-gram logit bias: full summary of approach
- **Technique**: Precompute n-gram P(next|context) from training data, add as weighted logit bias
- **Weights found**: bigram=0.2, trigram=0.15, 4gram=0.1, 5gram=0.08 (decreasing with order)
- **Gains by order**: bigram -0.010, trigram -0.021, 4gram -0.019, 5gram -0.009 (total -0.081)
- **Diminishing returns**: 5-gram only gave -0.009. 6+ probably ~-0.005 each
- **Throughput cost**: ~1.01s/step vs 0.87s baseline (~16% slower). Acceptable
- **This approach is our core innovation and competitive differentiator**
- **Competition viability**: need eval-time dynamic cache OR compressed tables in artifact

### 4i. N-gram gains scale with more training but diminish relatively
- At 500 steps: n-gram gives -0.058 bpb (v12 1.9663 vs baseline 2.0239)
- At 1000 steps: n-gram gives -0.042 bpb (v12 1.8841 vs baseline 1.9257)
- The absolute gap shrinks (model eventually learns n-gram patterns itself)
- But the gains are STILL significant at 1000 steps and would be ~-0.03 at 7000 steps (H100 scale)
- Extrapolation: at H100 with full training, n-gram bias could give ~-0.03 bpb on top of SOTA
- This means SOTA 1.1147 -> potentially ~1.08 with our n-gram approach alone
- Combined with other SOTA techniques, potentially sub-1.05

### 4j. seq_len=2048 doesn't help at 200 steps, grad_clip hurts
- Seq 2048 vs 1024: identical train loss at step 200 (3.654 vs 3.653). Extra context unused at low steps
- Gradient clipping (1.0): WORSE by 0.25 train loss. Muon optimizer already normalizes gradients
- Both are negative results for Mac iteration but might matter on H100 with more training
- SKIP both for Mac experiments, consider seq=2048 for H100 submission

### 4l. Wild experiments batch results (experiments 45-49)
- **Smaller models (5L, 7L) + n-gram**: Both WORSE. N-gram doesn't compensate for fewer layers. Model capacity is essential.
- **Reservoir computing (random projection)**: CE 6.79, nearly random. Needs recurrence to capture temporal patterns.
- **Pure PPM (classical n-gram)**: CE 4.38 on val. Neural model (CE 3.32) adds 1.0 nats of value beyond n-grams. Neural is essential.
- **Spectral embedding init**: WORSE (4.375 vs 4.05 ref). ALL geometric inits (SVD, spectral) hurt. Random init is best because model needs freedom.
- **Key lesson**: The neural model is the foundation. N-gram bias is a valuable ADDITION but cannot replace model capacity or training. Both are needed.

### 4k. Mac experiments have reached diminishing returns for architecture changes
- 3x MLP + n-gram: WORSE at 200 steps (3.893 vs 3.653). Same story as v2
- Momentum 0.99: WORSE at 200 steps (3.911 vs 3.653). Too slow to settle
- Pattern: EVERY "bigger/stronger" architectural change is worse on Mac (low steps)
- Only n-gram bias improves per-step convergence (it's external knowledge, not more params)
- **Mac experiments are DONE for architecture changes. Only n-gram tuning left.**
- For actual competition: all the arch improvements (11L, 3x, momentum 0.99, XSA) will help on H100 with 7000 steps
- The Mac iteration loop should STOP and the user should apply for H100 credits

### 4g. CRITICAL: Eval-time n-gram cache DOESN'T WORK on models trained without bias
- Tested: added dynamic n-gram cache at eval time to v7 model (trained without n-gram bias)
- Result: cache_bpb was ~0.01 WORSE than base_bpb (2.0530 vs 2.0425)
- Reason: model trained without bias learned to do full predictions itself. Adding bias throws off its calibration
- N-gram bias MUST be part of training for the model to learn residuals
- This means: n-gram tables MUST be in the 16MB artifact (or reproducibly recomputable from training data)
- Eval-time-only approaches need actual gradient updates (real TTT), not just logit biases
- The precomputed-from-training-data tables are our core approach — compress them to fit in artifact

### 4h. Artifact size solution: 8K buckets + drop 5-gram + int8+zstd = FITS IN 16MB
- Bigram (1024x1024 float32 -> int8+zstd): 0.4MB
- Trigram 8K buckets (int8+zstd): 3.9MB  
- 4-gram 8K buckets (int8+zstd): 4.9MB
- Total tables: 9.2MB. Neural model (int8+zstd): ~5MB. Grand total: ~14.2MB. FITS!
- Drop 5-gram (saves 5MB, only loses -0.009 bpb)
- val_bpb with 8K buckets + bi+tri+4gram (no 5gram): estimated ~1.97 based on v12 (1.9663 with all 4)
- This is achievable WITHOUT changing the 16MB constraint
- Competition strategy: 8K-bucket n-gram tables compressed in artifact, neural model learns residual

### 5. Validation is painfully slow on Mac — USER FIX: skip val for smoke tests
- Full val eval takes ~15-20 min per run
- Need to either skip val for iteration or reduce val set size for quick tests
- Training at 500 steps only takes ~7 min, val dominates total time

### 6. PYTHONUNBUFFERED=1 is required for MLX scripts
- Without it, output is completely swallowed by background processes
- Always use PYTHONUNBUFFERED=1 when running training

## Competition Strategy Lessons

### 7. The SOTA has moved to ~1.07 BPB
- Original SOTA was 1.1147, new entries claim 1.0713-1.0903
- TTT (test-time training) is the single biggest lever (~0.07-0.16 BPB)
- Tokenizer optimization is massively under-explored
- WARP (Word-Aware Representation Priors) is a novel approach worth investigating

### 8. Throughput matters as much as architecture
- At 83ms/step on H100, each ms overhead costs ~7 steps
- Need >0.007 BPB per ms of overhead for any new technique to be worthwhile
- Zero-overhead or eval-time-only techniques are the safest bets

### 9. Int4 quantization is a dead end
- +0.065 BPB gap — catastrophically bad
- Int6 is the sweet spot for most approaches
- Ternary (1.6 bits) fits 73.7M params but needs longer training

## Research Findings (Apr 3 — compression + architecture research)

### 10. ~~Entropy coding saves 20%~~ WRONG — zstd-22 already beats per-value entropy coding!
- Measured: int8+entropy-optimal = 54.1MB, int8+zstd-22 = 35.7MB. **zstd WINS by 34%!**
- zstd captures inter-value correlations (LZ77 dictionary matching) not just per-value frequencies
- Custom entropy coding would be WORSE than zstd, not better
- **Don't implement custom entropy coding. Just use zstd-22 or LZMA.**

### 10b. CORRECTED artifact budget (measured Apr 3 on actual models)
- Neural model alone: **int8+zstd = 13.4MB, int6+zlib = 10.8MB**
- Previous estimate of "~5MB neural" was WILDLY OFF — real number is 10-13MB
- **Int6 quantization is ESSENTIAL for fitting n-gram tables**
- Best fit: int6 neural (10.8MB) + bigram (0.4MB) + trigram 8K (3.9MB) + code (0.3MB) = 15.4MB ✓
- 4-gram table (4.9MB) does NOT fit. Need int5-MLP or smaller model.
- Bigram-only fits easily at 11.5MB with 4.5MB headroom

### 11. SVD on n-gram tables: DEAD END (confirmed by top-K analysis)
- Tables appear rank 2-19 but that's the BACKGROUND distribution, not the predictions
- Top-1 prediction preservation at rank 100: only 24-42%. SVD scrambles the useful variation.
- "Low effective rank" is misleading — prediction-relevant info is spread across hundreds of SVs
- **Lesson: Frobenius error and effective rank can mislead. Always test TOP-K / KL / actual task metric.**
- Stick with: 8K buckets + int8 + entropy coding for n-gram table compression

### 12. Delta encoding is useless for transformer weights
- std(diff)/std(w) ratio = 1.41 (WORSE than raw). Weights have no spatial correlation.
- Only tiny resid_mix vectors benefit (ratio 0.25) — negligible overall.

### 13. SVD on transformer weights is useless — they're full rank
- All weight matrices at 96-100% effective rank. No low-rank structure to exploit.
- This matches the SVD embedding init failure: the model's weight structure is emergent and nonlinear, not low-rank.

### 14. Nacrith-style context mixing is the next frontier
- Nacrith (Feb 2026) achieves 0.94 BPB on enwik8 using transformer + classical n-gram + LEARNED mixer
- Our approach (fixed bias weights w=0.2 etc) is a primitive version of this
- The key improvement: make mixing weights CONTEXT-DEPENDENT and LEARNED ONLINE during eval
- This IS score-first TTT — the mixer trains on already-scored tokens
- Implementation: small logistic regression that takes [neural_entropy, ngram_confidences] → mix weights

### 15. 33% of neural weights are near-zero (pruning opportunity)
- |w| < 0.01 for 33% of weights. MLP proj layers up to 52%.
- Zeroing + sparse encoding would reduce storage significantly
- But naive pruning hurts quality. Need Hessian-aware pruning (like GPTQ but for sparsity)
- Sparse + quantized + entropy coded = maximum compression

### 16. Tensor Train / MPO decomposition — promising alternative to GPTQ
- PicoGPT (Mar 2026): 13x compression with 97% accuracy retention using MPO
- Unlike SVD (which fails on our full-rank matrices), TT captures higher-order structure
- Could enable fitting 12L 768d model in same space as 9L 512d
- Risk: training-time overhead, untested on byte-level LM

### 17. Hrrformer — 23x faster attention, single layer matches multi-layer
- Replaces self-attention with holographic reduced representations (circular convolution)
- 23x faster, 24x less memory. Single layer matches multi-layer transformer.
- Would give 23x more training steps in 10 minutes
- Risk: never tested on byte-level LM specifically

### 18b. Program-as-artifact is a dead end (researched Apr 3)
- cmix (SOTA classical): 1.17 BPB in ~90KB binary but needs 32GB RAM and C++. Not feasible in Python.
- PAQ8: 1.2-1.3 BPB in ~40-200KB binary. Python port would be 100-1000x too slow for eval budget.
- PPM in Python: ~2-5KB code, achieves ~1.8 BPB. Already tested: 4.38 CE vs neural 3.32 — conclusively worse.
- **Our n-gram logit bias IS the best version of this idea.** Classical knowledge baked in, neural learns residual.
- Nacrith (0.94 BPB SOTA) confirms: the right approach is transformer + n-gram + adaptive mixer. That's us.
- **The only remaining improvement: make mixing adaptive at eval time (Nacrith-style learned mixer).**

### 18c. BPE-8192 is the SINGLE BIGGEST WIN in our entire experiment history (exp #61)
- BPE-8192 vanilla baseline: **1.8953 BPB** (500 steps, no tricks)
- SP-1024 baseline: 2.0239 BPB → **-0.129 BPB improvement from tokenizer alone**
- SP-1024 + ALL n-gram tricks (v11): 1.9428 → **BPE-8192 vanilla is STILL 0.048 better**
- 54 experiments of n-gram optimization = -0.081 BPB. One tokenizer change = -0.129 BPB.
- **The tokenizer was always the biggest lever. We should have started here.**
- N-gram logprob tables are now OBSOLETE for our approach (and don't fit at 8192 vocab anyway)
- Next: stack all validated tricks (LeakyReLU, SmearGate, WD, depth recurrence) ON TOP of BPE-8192
- Use BigramHash/EngramLite (hash embeddings) instead of logprob tables for n-gram knowledge

### 19. Complementary training (alpha=0.5) FAILED on Mac (exp #55)
- 4.411 CE vs ref 4.05 — WORSE by 0.36 nats
- Down-weighting easy tokens reduces total gradient signal, hurting convergence at low step counts
- Same pattern as all "bigger/different" changes: needs more steps to show benefit
- **May still work on H100** with 7000 steps, but can't validate on Mac
- Alternatively: try smaller alpha (0.1-0.2) or enable only in final 20% of training

### 20a. Kronecker factorization is DEAD for compression (tested Apr 3)
- With shared quantization: 94% savings but 800x worse RMSE (catastrophic)
- With proper per-factor quantization: -0.3% savings (ZERO benefit)
- Cross-layer weight sharing: cosine similarity ≈ 0 between layers. Completely different weights.
- Shared+delta encoding: 11-15% WORSE than naive (overhead exceeds savings)
- **At 512d, weight matrices are too small and full-rank for structural decomposition.**

### 20. QK-gain 4.0 FAILED on Mac (exp #56)
- 4.368 vs ref 4.05 — WORSE
- Higher QK gain sharpens attention, which helps AFTER the model has learned good patterns
- At 100 steps on Mac, the model hasn't learned enough for sharper attention to help
- **Likely helps on H100** — it's in the 1.0914 stack (PR #1176)

### 21. UID regularizer FAILED on Mac (exp #70)
- 5.823 vs ref ~5.56 — variance penalty inflates the loss, model learns slower
- UID was proven for "limited data" regimes but our regime is "limited STEPS" which is different
- The model sees plenty of data (10 shards) but too few gradient updates to benefit from regularization
- **Skip for Mac. May help on H100 where 7000 steps provides a stronger training signal.**

### 22. Factorized embedding breaks tied head — but 2-matmul fix exists
- 8192×64 + 64×512 factorization saves 3.64M params (1.87MB)
- BUT: tied output head assumes logits = hidden @ embed.T (single matmul)
- FIX: logits = hidden @ proj.T @ small_embed.T (two matmuls, both cheap)
- The 2nd matmul is (B×S, 64) × (64, 8192) — negligible compute
- **Code in RESEARCH.md. Builder should implement this.**

### 23. WD=0.04 and depth recurrence WORSE at 100 steps on BPE-8192 (exp #66, #67)
- Same pattern as SP-1024: bigger/regularized models need more steps
- These will work on H100 with 7000 steps — confirmed by leaderboard entries using both
- **Mac can't validate arch changes. Only validates n-gram and tokenizer.**

### 24. ALL novel activations/losses FAILED on Mac (exp #72-74)
- Gated attention (sigmoid gate): 5.767 vs ~5.56 — gate at 0 kills attention signal
- Michaelis-Menten activation: 5.832 — saturating doesn't match squared pattern
- WaveletGPT aux loss: 6.457 — MUCH WORSE, aux loss inflates total
- **Pattern confirmed: Mac at 100 steps rejects EVERYTHING except n-gram + tokenizer.**
- These may still work on H100 with proper init/tuning, but can't validate on Mac.
- **The only Mac-validated wins remain: BPE-8192 (-0.129) and n-gram bias (-0.055 on 8192v).**

### 25. The definitive Mac lesson after 74 experiments
- On Mac (500 steps): only TWO things reliably improve BPB:
  1. Better tokenizer (BPE-8192: -0.129)
  2. N-gram logit bias (-0.055 on BPE-8192)
- Everything else (architecture, schedule, regularization, novel activations) needs H100.
- **Stop trying new tricks on Mac. Focus on H100 submission prep.**

## Technical Notes

### MLX-specific quirks
- mx.compile captures state at compile time — monkey-patching after compile doesn't work
- mx.concatenate in hot loops can cause memory leaks — use mx.pad instead
- EMA updates every step can blow memory — do every 10 steps with explicit mx.eval
- Eager eval (MLX_EAGER_EVAL=1) essential for 16GB machines

### 26. Sigma-delta quantization is WORSE for transformer weights (tested Apr 4)
- SD quantization feeds quantization error forward to next weight (like audio DACs)
- Theory: should give 4-bit with 8-bit effective quality for correlated signals
- **Result: 41.4% HIGHER RMSE than standard rounding at 6-bit. 43.7% higher at 4-bit.**
- **Reason: transformer weights have NO spatial correlation (delta ratio = 1.41)**
- SD works for audio (highly correlated samples) but fails for random-ish weight matrices
- This also kills Floyd-Steinberg dithering (2D error diffusion) for the same reason
- **Stick with GPTQ int6 + Brotli-11. No novel quantization wins.**

### 27. DCT compressibility of weights: moderate (72%), not enough for compressed sensing (tested Apr 4)
- Top 25% of DCT coefficients contain 72% of weight energy
- Need >80% for compressed sensing to be viable
- Weights are approximately uniformly distributed in frequency — no exploitable structure
- **Skip compressed sensing for weights**

### 28. Hymba (hybrid Mamba+Attention) nearly matches the record (researched Apr 4)
- PR #852: 1.1189 BPB at ~85ms/step — only 0.004 behind merged SOTA
- Pure Mamba is dead (282ms, breaks torch.compile, 10-15% tensor core utilization)
- Hymba uses parallel attn+mamba branches with learned sigmoid gate
- Our n-gram bias stack on Hymba could push to ~1.08 BPB
- Griffin (Google) is untested but potentially more torch.compile compatible

### 29. MoR / depth recursion — ⚠ ORIGINAL CLAIM "DEAD" IS STALE (Apr 7 update)
- ORIGINAL claim (Apr 4): "Quantization error compounds ~900x through 3 recursion cycles" → marked depth recurrence as dead.
- **CONTRADICTION (Apr 7 PR mining)**: 5+ of the top 10 most-recent legal record submissions use depth recurrence:
  - PR #1437 (1.07800): SP8192 + Parallel Residuals + 3-Layer Recurrence
  - PR #1421 (1.09250): 11L Depth Recurrence + EMA Tuning
  - PR #1422: 11L recurrence
  - PR #1429: 13L Int4-Packed + Depth Recurrence
  - PR #1435 (1.09800): 11L Depth Recurrence + BigramHash + EMA 0.9965
- The 900x compounding claim assumed PURE int6 GPTQ on a 1L×3 cycle. Recent winners avoid this by:
  - Using **mixed-precision quant** (INT5 attention + INT6 MLP, or per-layer tuning)
  - Using **3-layer cycles** (3L×N) instead of 1L×N — error compounds less per cycle
  - Using **post-quant EMA** to smooth out the compounding
- ⚠ Original "MoR-1L 3.7MB + tables doesn't fit" budget claim is also stale — 11L depth recurrence fits in 16MB with the same int6 quant.
- **REVISED LESSON**: depth recurrence is NOT dead. It's alive in 5+ recent records. Test it with mixed-precision quant. Patch TODO: USE_DEPTH_RECURRENCE=1 with configurable cycle length + per-layer precision.
- **DO NOT trust "X is DEAD" claims in this file** — re-validate against recent comp records before skipping a technique.

### 30. Logistic mixing (cmix-style) is negligible vs additive (tested Apr 4)
- Tested logistic vs additive mixing on DC500 category-level predictions
- With informed neural expert (70% overlap): additive wins by 0.0003 (effectively tied)
- The mixing method doesn't matter when the neural model captures most of the distribution
- Our additive logit bias approach is fine. Skip cmix-style mixing.

### 31. Q-R trick → skip-bigram table adds +0.28 bits/tok signal (CORRECTED Apr 4)
- Initial claim of "6.35 bits collision damage" was MISLEADING — compared to overfitting oracle
- Rigorous test: Q-R decomposes trigram into bigram (prev1→next) + skip-bigram (prev2→next)
- We already have the bigram. The NEW signal is the skip-bigram.
- Stacking all three (bigram + trigram + skip-bigram): 10.80 vs 11.08 bits/tok for current
- **Real improvement: +0.28 bits/tok additional signal → ~0.005 BPB with bias weight 0.15**
- Modest but additive. Skip-bigram at 8K buckets ≈ 3-4 MB compressed. Fits in 3.6MB gap.
- **Lesson: always validate claims with end-to-end simulation, not just collision counts**

### 32. Canonical trick sweep: nothing missing (researched Apr 4)
- Z-loss: redundant with softcap. Skip.
- QK-norm: already implemented. QK-gain=5.0 might give -0.001.
- Logit softcap: already at 30.0.
- Untied embeddings: catastrophic at this scale (PR #908).
- Post-norm: no evidence at 11L.
- muP: manually approximated via per-group LR.
- Batch size: already throughput-maximized.
- Split-LR (early=0.025, late=0.030): from PR #1172, worth adopting.

### 36. Artifact budget: int6 model is ESSENTIAL to fit bigram table (measured Apr 4)
- Standard model int8+zlib = 12.69 MB. Bigram 8K int8+brotli = 2.94 MB. Total > 16 MB!
- Int6 model (naive, no GPTQ) + brotli = 9.83 MB. Bigram FITS with 2.88 MB to spare!
- 8K buckets = SAME quality as 16K (only 8192 unique prev tokens). Zero quality loss.
- Bigram is worth 0.658 BPB standalone — the most valuable single component after the model.
- GPTQ int6 on H100 is the GATING STEP for our full artifact plan.

### 37. Dual-codebook compression could shrink bigram from 2.94 MB to ~0.77 MB (researched Apr 4)
- K-means cluster 8192 distributions into 64 prototypes + int4 residual
- Would free 2.17 MB for trigram + 4gram tables
- From audio codec research (SemantiCodec). Mac-testable.

### 38. Tabulation hashing (XOR + lookup) is provably better than our linear hash (researched Apr 4)
- Our hash (36313*a + 27191*b) % N is NOT 2-independent — correlated tokens collide systematically
- Tabulation: (T1[a] ^ T2[b]) % N is provably 3-independent. Same speed (XOR = free).
- 4 lines to change. Should reduce the 6.35 bits/tok collision damage we measured.

### 35. Our optimizer hyperparams are BEHIND the competition frontier (researched Apr 4)
- Momentum 0.95 (competition: 0.99), LR 0.04 (competition: 0.02), warmdown 1200 (competition: 3000)
- Grad clipping DISABLED (competition: 0.3), SWA (competition: EMA)
- **Fixing these config-only changes could give -0.01 to -0.02 BPB for ZERO code effort**
- Also discovered: NorMuon (per-row norm after NS, 11% efficiency), MuonEq-R (pre-NS equilibration),
  Turbo-Muon (4 NS steps), Mousse (Shampoo preconditioning), higher WD trend (0.08-0.10)
- Split-LR already confirmed as winner on Mac (exp t214f: -0.038)

### 34. Signed hashing CONFIRMED: +0.003 BPB for 2 lines (tested Apr 4)
- Tested on real BPE-8192 FineWeb data: 12.067 → 12.040 bits/tok = +0.027 bits/tok
- +0.003 BPB improvement. 2 lines of code. Zero overhead. Zero risk.
- Must apply sign BOTH when building the table AND when looking up
- Sign = ((prev2 * 2654435761 + prev1 * 2246822519) % 2) * 2 - 1
- Collision noise becomes zero-mean instead of systematically biased
- **Action: add to every n-gram table (bigram, trigram, 4gram, 5gram)**

### 33. Static n-gram token selection is DEAD on BPE-8192 (tested Apr 4)
- BPE-8192 has mean bigram entropy 9.4 bits (vs ~6 bits for byte-level)
- Only 2.3% of tokens have bigram H < 5 bits (too few to skip meaningfully)
- **Reason: BPE tokenizer already compresses away the easy character patterns**
- Our n-gram tables are less effective on BPE-8192 (-0.055) than SP-1024 (-0.081) for this reason
- **Use DYNAMIC selection instead: skip bottom 20% of model's own per-token loss**
- Dynamic selection adapts as model trains; bigram reference is static and too weak on BPE-8192
- Complementary training (PR #803) also works: weight by inverse bigram prob, gentler approach

### 18. Brotli-11 saves 1.47MB over zstd-22 (measured Apr 3)
- Full artifact with brotli: 12.33MB (neural + bi + tri_8k + 4g_8k)
- Adding 5-gram 4K buckets: total 14.64MB — fits with 1.36MB spare!
- Brotli especially effective on n-gram tables
- **Use brotli-11 not zstd for final artifact**

