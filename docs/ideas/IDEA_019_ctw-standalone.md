---
id: IDEA-019
slug: ctw-standalone
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L09
novelty_class: WN
expected_bpb: [-0.030, -0.008]
cost_hours: 4.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l09-ctw-context-tree-weighting-online
prior_art_checked: 2026-04-16
next_step: prior-art-audit-then-prototype
---

# IDEA-019: CTW (Context Tree Weighting) as a standalone eval-time predictor

> **Hypothesis**: Running Context Tree Weighting (Willems et al. 1995) as a byte-level online predictor in parallel with the LM and blending via a simple scalar-alpha hedge reduces val_bpb by 0.008-0.030. CTW is provably near-optimal on stationary binary/byte sources with tree-structured priors; FineWeb is locally stationary enough that CTW should capture real structure the LM misses.

## Method

Decompose the idea from IDEA-012 (the full moonshot) into its simplest standalone form: just CTW + LM, no online n-gram cache, no PPM, no learned hedge. This gives us a clean test of "is CTW worth carrying at all?"

At eval time, for each val byte:
```python
# b is the next byte to be scored
lm_logits = model(val_so_far_tokenized)
lm_p = softmax(lm_logits)  # [256] byte distribution projected via lm-to-byte marginalization

# CTW maintains a binary tree of byte-contexts; each tree-node holds
# counts for each of 256 next-byte classes. New bytes update the tree
# BOTTOM-UP after they're scored.
ctw_p = ctw.predict(val_so_far[-D:])  # byte distribution, D = context depth (8-16)

# Simple scalar-alpha hedge
alpha = 0.85  # tune by sweep
mixed_p = alpha * lm_p + (1 - alpha) * ctw_p
byte_cost = -log(mixed_p[b]) / log(2)  # contribution to BPB

# NOW update CTW with (context, b) — causal per MOONSHOT_RULES §1.4
ctw.update(val_so_far[-D:], b)
val_so_far.append(b)
```

CTW implementation: ~150 LOC in Python (or ~200 LOC in C for speed). Context depth D=12 gives 2^13 = 8192 possible binary-split nodes; storage O(n) where n = bytes processed.

**Integration point**: wraps the eval-scoring loop in `submission/train.py`. `CTW_ENABLED=1` env var. CTW state is RAM-only during eval; does not count against 16 MB.

## Expected BPB

- **Range**: [-0.030, -0.008]
- **Mechanism**: CTW is asymptotically optimal on stationary tree sources (Willems 1995 — "Context Tree Weighting: a sequential universal source coding procedure"). On enwik9 cmix uses CTW as a core component and hits ~0.9 BPB. On FineWeb val (less repetitive than Wikipedia), expect 0.008-0.030 contribution when blended with a competent LM.
- **Lower bound**: -0.008 (if LM already captures most of what CTW would capture at short context depths)
- **Upper bound**: -0.030 (if CTW's long-context byte-level statistics materially complement the LM)

## Testable prediction

- **Metric**: val_bpb
- **Threshold**: ≤ 1.074 at seed 42 (-0.008 from 1.082 baseline)
- **After**: 600 s training (stock 1.082 config), new eval with CTW_ENABLED=1
- **Secondary**: alpha sweep {0.70, 0.80, 0.85, 0.90, 0.95} to find optimum

## Falsification criterion

Kill if:
- At best alpha, val_bpb ≥ 1.080 (less than 0.002 improvement)
- OR eval wallclock > 10 min on 8×H100 (blow the eval budget)

## Stacking plan

- **Composes with**: IDEA-012 (moonshot; CTW could be one component of the hedge mixer)
- **Composes with**: all training-time IDEAs (001, 002, 003, 004, etc.) — CTW is an eval-time addition
- **Conflicts with**: nothing
- **Blocks**: nothing
- **Budget footprint**: 0 bytes (eval-time RAM state, no model-artifact addition)

## Prior-art audit

Audited 2026-04-16 by Loop A fire 12 (Explore subagent).

- **Arxiv (2023-2026)**: no papers combine CTW + modern transformer LM. Closest: CTW for real-valued time series (arxiv 2106.03023) — classical setting, not neural.
- **Comp PRs** (openai/parameter-golf): **none** — no PR ships CTW.
- **cmix ecosystem**: cmix (Knoll) uses CTW internally for byte-level compression + blends ~2000 models via LSTM context mixers (Hutter Prize). Proven effective on enwik9 but **operates purely on bytes, no transformer**. Our idea integrates CTW with a modern LM — a distinct step.
- **Verdict**: **world-novel (with strong precursor in cmix)**. CTW as a standalone eval-time hedge with a transformer LM is unpublished. Mechanism (online CTW on bytes/tokens, scalar-alpha blend, legal causal scoring) is novel. cmix validates CTW's core value but doesn't combine it with a neural LM.
- **Checked by**: claude 2026-04-16

## Lineage

- `STACK_NOVELTY_TRACKER_v2.md §L09` open-novelty row "CTW (Context Tree Weighting, online)"
- MOONSHOT_RULES.md §2.6 — rules-confirmed legal (causal update per §1.4)
- RESEARCH_PROTOCOL.md §1 grid cell (L09 × universal source coding / CTW)

## Risks

- CTW memory footprint grows linearly with eval-time context; needs careful O(1) update implementation to not blow eval budget
- Interaction with sp8192 BPE tokenizer: CTW operates on raw bytes, but our val scoring is at token-level. Need to either (a) run CTW on the detokenized byte stream and contribute one BPB fraction at byte level, or (b) adapt CTW to token IDs (loses the tree structure advantage)
- Option (a) requires careful accounting: sum byte-level BPB contributions = same total val_bpb but the per-token hedge-blend becomes non-trivial
- Mitigation: start with option (b) — CTW on token IDs — as a cleaner prototype. Expected BPB drops to the lower end of the range but implementation is straightforward.

## Notes

Cheaper and cleaner test than the full IDEA-012 moonshot. If this lands, IDEA-012 gets a major boost (CTW becomes a validated component of the hedge mixer). If this fails, IDEA-012 loses one of its expected levers and the [-0.05, -0.15] BPB range narrows.
