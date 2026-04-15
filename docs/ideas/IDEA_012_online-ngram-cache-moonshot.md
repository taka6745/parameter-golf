---
id: IDEA-012
slug: online-ngram-cache-moonshot
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: cross-layer
novelty_class: WN
expected_bpb: [-0.15, -0.05]
cost_hours: 8.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l09-online-growing-n-gram-cache-moonshot
prior_art_checked: 2026-04-16
next_step: build-cache-prototype-nonrecord-track
---

# IDEA-012: Online growing n-gram cache at eval time (MOONSHOT)

> **Hypothesis**: A causal online n-gram cache that grows during eval (built from already-scored val tokens) reduces val_bpb by 0.05-0.15 by exploiting FineWeb val's local redundancy that our static cache misses. Cmix achieves 0.9 BPB on enwik9 via online prediction; at 16 MB + LM we should reach 0.90-1.03 BPB.

## Method

At eval time, maintain a hash-map `cache[k-byte-context] -> Counter[next_byte]` and a hedge mixer over (LM, cache).

Protocol per MOONSHOT_RULES.md §1.4 and §2.1 (causal update only):

```python
# During eval loop
context_prefix = []
cache = defaultdict(Counter)
for i in range(len(val)):
    # 1. Predict this byte
    lm_p = softmax(model.forward_logits(tokenize(context_prefix))[-1])
    cache_p = cache_distribution(cache, context_prefix[-k:])  # from scored bytes only
    p = alpha * lm_p + (1 - alpha) * cache_p
    # 2. Score (this is what counts toward BPB)
    bit_cost += -log(p[val[i]])
    # 3. UPDATE cache with (context, b_i) — now b_i is graded
    cache[tuple(context_prefix[-k:])][val[i]] += 1
    context_prefix.append(val[i])
```

Implementation notes:
- `k` = context length: 3-5 bytes (tune)
- Cache starts empty (or pre-seeded from train; train-only is strictly legal per rules)
- Alpha (hedge weight): learned via exponential weights / or trained offline on train data
- Cache stored as hashmap of int → dict → int; grows to ~tens of MB during eval; lives in RAM only, doesn't count toward 16 MB artifact

**Integration point**: eval-time only change; `submission/train.py` gets a new function `online_cache_predict()` called during the eval-scoring loop. The LM + static n-gram bias remain unchanged.

## Expected BPB

- **Range**: [-0.15, -0.05]
- **Mechanism**: cmix-class online prediction + LM hedge. The cache captures repeated n-grams that appear 2+ times in val (which is a big fraction per Zipf). On enwik9, cmix = 0.9 BPB; we have 30M LM + growing cache; reasonable target is ~0.95-1.00 BPB.
- **Lower bound**: -0.05 (if FineWeb val is less internally-redundant than enwik9 — likely)
- **Upper bound**: -0.15 (if FineWeb has comparable redundancy and our hedge mixer is near-optimal)

## Testable prediction

- val_bpb ≤ 1.032 at 2 seeds (target: −0.05 below our 1.082 baseline)
- If we hit val_bpb < 1.0, that's the moonshot landed.
- Eval wallclock ≤ 10 min on 8×H100 (per MOONSHOT_RULES.md §1.2)

## Falsification criterion

Kill if val_bpb ≥ 1.075 at seed 42 (i.e. less than 0.01 improvement — the cache isn't adding anything over static bias).

## Stacking plan

- Composes with everything. Pure eval-time addition.
- No conflict with current stack. The cache operates alongside the LM and existing static bias.
- If this lands, it's the moonshot; everything else is icing.

## Prior-art audit

- Cmix: public general-purpose compressor using online CTW+PPM+LSTM mixture. Achieves 0.9 BPB on enwik9. Not an LM-competition submission.
- PAQ, zpaq: similar family.
- Comp PRs: no PR shipping online n-gram cache (as of 2026-04-15 audit).
- Rule legality: confirmed legal via MOONSHOT_RULES.md §2.1 (causal-only update).
- Verdict: WN moonshot.

## Risks

- Eval wallclock: growing cache + per-token hedge mixer may blow the 10-min eval budget. Mitigation: bucket the cache (hash table size cap), update in batches.
- Alpha tuning: the hedge weight between LM and cache is critical. Too low → cache wasted; too high → cache dominates in high-entropy regions. Mitigation: train α as a per-position scalar on train-set-simulated-online.
- Cache cold-start: first N bytes of val have empty cache. Mitigation: pre-warm with train-derived n-gram counts.

## Non-record track first

Per MOONSHOT_RULES.md §6.3 — prototype on non-record track (no 10-min limit), validate BPB, then optimize for 10-min. Safe POC path.

## Notes

The biggest theoretical BPB payoff in the whole queue. Needs ~8 hours dev time; Loop B can start building this while Loop A keeps researching.
