# Shannon entropy floor — what we are actually chasing

Computed 2026-04-08 0715Z by sampling 4.19M tokens from `data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin` (8 MiB of uint16 token IDs).

## The numbers (token domain → BPB at 3.5 bytes/token)

| Order | Bits/token | BPB |
|---|---|---|
| H0 unigram | 8.6426 | 2.4693 |
| H1 bigram | 6.0148 | 1.7185 |
| **H2 trigram** | **3.8741** | **1.1069** |
| zlib-9 (raw uint16) | 8.0164 | 2.2904 |
| brotli-11 (raw uint16) | 5.9055 | 1.6873 |

## What this means

- **Trigram H2 = 1.107 BPB** is the floor a perfect bigram+trigram bias model would hit.
- Anything below 1.107 BPB requires *real LM context* (4-gram+ or transformer attention learning higher-order structure).
- Brotli-11 at 1.687 BPB is the non-parametric upper bound for a generic compressor — confirms n-gram tables are a large win on top of brotli.
- **Shannon irreducible limit on English text is ~0.95–1.00 BPB** (Shannon's 1951 estimate, Cover & Thomas 2006 update).

## Where we sit and the gap

| Position | val_bpb | Gap to next |
|---|---|---|
| Our best (L04 gated_attention seed999, projected) | 1.10–1.12 | — |
| Current leaderboard SOTA | 1.07 | -0.03 to -0.05 |
| Trigram-only LM ceiling (H2) | 1.107 | we're AT it |
| Theoretical Shannon limit (English) | ~0.95–1.00 | -0.07 to -0.10 |

### Insight: we are sitting at the trigram floor

Our 1.10–1.12 projected BPB is essentially THE TRIGRAM FLOOR. This means our current model is mostly just memorizing the 3-gram statistics — the LM context isn't adding much beyond what a fixed-table trigram bias could provide.

**To win**, we need to capture higher-order structure that trigrams can't see:
- 4-gram, 5-gram, 7-gram bias tables (LESSONS §31 says +0.28 bits/tok signal exists)
- Real long-range LM attention learning patterns trigrams miss
- Skip-n-gram structure (Q-R trick)
- Compositional structure (morphology, sentence boundaries)

The shipped world-novels that target higher-order context:
- **L09 CTX_PARTITIONED_TAB** — partitioned tabulation for higher-order n-grams (UNVALIDATED)
- **L09 NGR_adaptive_cuckoo_hash** — backlog, zero-collision high-order
- **L09 skip-bigram table** — backlog, the +0.28 bits/tok LESSONS §31 win
- **L01 TOK_INPUT_SMOOTH** — soft input regularization (not directly higher-order)

The shipped world-novels that DON'T target higher-order context (still useful, just not the binding constraint):
- L05 NORM_PCT_DROPOUT, L06 ASYMMETRIC_SKIP_INIT, L07 ASYM_LABEL_SMOOTHING, L08 PER_PROJ_LR_SPLIT — calibration/optimizer tweaks.

## Action implications

1. **Highest-leverage missing work is L09** (n-gram engine). Skip-bigram + adaptive cuckoo hash are still in backlog. Ship them.
2. The **Triton fused n-gram + GQA kernel** (L11 #5) compounds because it makes n-gram lookups faster, freeing budget for more-or-larger-table experiments.
3. Calibration wins (L05/L06/L07) are pure noise reduction — they get us cleaner signal but don't lower the floor.
4. **L01 tokenizer changes** (BPE-8192, Huffman init) move the FLOOR itself by changing the BPB-equivalence of bytes-per-token; worth the full chore-script + n-gram-table-rebuild cost.

## How to use this document

Every campaign decision should be evaluated against:
- "Does this lower our distance to 1.07?"
- "Does this lower our distance to ~1.00?"
- "Does this just trade noise for noise without moving the floor?"

The trigram floor of 1.1069 BPB is our SHORT-TERM target. The Shannon limit of ~1.00 is our LONG-TERM ceiling. The leaderboard 1.07 is between them.

