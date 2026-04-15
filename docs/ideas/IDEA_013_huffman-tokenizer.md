---
id: IDEA-013
slug: huffman-tokenizer
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L01
novelty_class: WN
expected_bpb: [-0.020, -0.005]
cost_hours: 6.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l01-huffman-weighted-token-ids
prior_art_checked: 2026-04-16
next_step: build-huffman-vocab-on-train
---

# IDEA-013: Huffman-weighted token IDs + variant row-tying

> **Hypothesis**: P12 probe showed 97 near-duplicate vocab rows (cos-sim >0.9) and P7 showed a very skewed token-frequency distribution (top-8 tokens = 15.7% of val). A tokenizer that (a) ties near-duplicate BPE pieces and (b) assigns variable-length codes to token IDs by frequency should reduce val_bpb by 0.005-0.020 while shrinking embedding + reclaiming budget.

## Method

Two independent changes, either can ship alone:

**Part A — Variant row-tying**: post-hoc modify the trained model. For each of the 97 cos-sim >0.9 vocab pairs found by P12, merge the rows: store one canonical row + a small 512-dim "delta" encoding the variant. Reclaim 97 × 512 × 8bits = 50 KB.

**Part B — Huffman-weighted IDs**: variable-length encoding of the token-ID stream. Common tokens (top-5%) get short codes (~log2(1/0.05) ≈ 4 bits); rare tokens get log2(8192) = 13 bits. Net: if token distribution has entropy H_token ≈ 10.4 bits (per P5), fixed-13-bit encoding wastes 2.6 bits/token. Reclaim = 2.6 × 40M val tokens × ... but this is within the tokenized stream, not the artifact — need to re-think where the BPB savings land.

Actually for BPB, the model's BPB is `-log2 p(token | context) / bytes_per_token`. Changing the vocab distribution by row-tying doesn't change per-token logprob but DOES change embedding budget. Part B would re-tokenize, changing bytes-per-token and requiring a retrain.

Revised: focus on Part A in the near term (budget reclaim, no retrain needed). Part B is a full retokenize + retrain, defer.

```python
# Part A implementation:
# 1. Load model; find 97 pairs via P12 cos-sim
# 2. For each pair (i, j), compute shared + delta via rank-1 decomposition
# 3. Redirect lookups for j to (shared_i + delta_ij)
# 4. Re-quantize and pack artifact
```

## Expected BPB

- **Range**: [-0.020, -0.005] for Part A + retrain with reclaimed budget on wider MLP
- **Part A alone**: artifact shrink ~50 KB, val_bpb unchanged (at worst)
- **Part A + widened MLP**: val_bpb 0.005-0.010 improvement

## Testable prediction

- Part A: artifact_bytes reduction ~50 KB, val_bpb within ±0.002 of baseline
- Part A + MLP_MULT=4.1 widening: val_bpb ≤ 1.077

## Falsification criterion

Kill Part A if val_bpb ≥ 1.085 (the row-tying lost information).

## Stacking plan

- Composes with IDEA-011 (embed int6) — both reclaim budget.
- If both ship, reclaim ~1.1 MB combined.

## Prior-art audit

- Vocabulary tying is well-studied (weight tying between embed and lm_head, which we already do).
- Near-duplicate row consolidation informed by COS-SIM P12 probe: novel approach to budget reclamation.
- No comp PR ships anything like this.
- Verdict: WN.

## Risks

- Row-tying specific pairs may not work uniformly; some pairs may not actually be near-duplicates semantically.
- Retrain may be needed to let the model learn to use the shared+delta representation.

## Notes

Part A is probe-informed novel allocation. Relatively risky (post-hoc model surgery) but cheap to test.
