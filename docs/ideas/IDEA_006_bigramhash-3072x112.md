---
id: IDEA-006
slug: bigramhash-3072x112
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L08
novelty_class: CP
expected_bpb: [-0.002, -0.001]
cost_hours: 0.5
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l08-bigramhash-3072-x-112
prior_art_checked: 2026-04-16
next_step: bump-bigram-hash-table
---

# IDEA-006: Port BigramHash 3072×112 (comp SOTA #1019, #1408)

> **Hypothesis**: Resizing our bigram bias table from current 16K×(feature_dim) to 3072×112 dense reduces val_bpb by 0.001-0.002 per comp PRs #1019 and #1408.

## Method

Our 1.082 uses 16K hash buckets per order with tabulation hashing. Comp SOTA #1408 uses a denser 3072×112 learned embedding per bigram hash bucket (not tabulation; learnable per-bucket).

```python
# Plumb new env vars and refactor _nlfi_bigram_mult structure
BIGRAM_HASH_BUCKETS=3072
BIGRAM_HASH_EMBED_DIM=112
BIGRAM_HASH_LEARNABLE=1  # was implicit
```

## Expected BPB

- **Range**: [-0.002, -0.001]

## Testable prediction

- val_bpb ≤ 1.0815 at 2 seeds.

## Falsification criterion

Kill if no improvement over 1.082 at 2 seeds.

## Stacking plan

- Composes with L09 n-gram engine changes and L06 n-gram bias mixer.
- Budget: 3072 × 112 × int6 = ~256 KB net addition (may require re-quant shrinking to fit 16 MB).

## Prior-art audit

- Comp PRs #1019, #1408. CP port.

## Risks

- 3072×112 is larger than 16K×small; artifact bytes may exceed 16 MB cap. Verify artifact_size before enabling.

## Notes

Defer if IDEA-001..005 already fill the BPB budget headroom.
