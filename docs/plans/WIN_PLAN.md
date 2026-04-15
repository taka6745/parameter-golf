# WIN_PLAN.md — moonshot to <1.0 BPB on FineWeb val

Created 2026-04-15. Owner: taka. Deadline: 2026-04-30 (15 days).

## The bet

**Win the comp by hitting val_bpb < 1.0**, beating SOTA (1.07) by enough that nobody catches us in the closing days. Stretch goal: approach the Shannon limit on English text (~0.95–1.00 BPB).

Honest probability: <1.0 ≈ 10–15%. <1.07 (still wins comp) ≈ 50%. The moonshot path *is* the safe path because every component banks BPB independently.

## Reframe: codec, not model

A 16MB int6 model is ~30M effective params. Chinchilla says you can't reach Shannon limit with that. So the artifact is **not a model** — it's a **codec** (model + n-gram tables + tokenizer + scoring rule). The codec is allowed to embed compressed knowledge that vastly exceeds 30M-params-worth.

| Component | Job | Target BPB contribution |
|---|---|---|
| Tiny LM (~5 MB int6) | Long-range, novel structure | ~0.50 |
| Multi-order n-gram engine (skip-bigram + adaptive cuckoo, n=5,7) | Short-range structure trigrams miss | ~0.20 |
| PPM / CTW component | Local statistics LM ignores | ~0.10 |
| Per-token learned hedge mixer | Route per position to best predictor | ~0.05 |
| Online TTT / n-gram cache (rules-permitting) | Adapt to val drift | ~0.05+ |

Sum: ~0.90 BPB if every component delivers and they don't redundantly overlap.

**The novel headline**: *"16MB Hybrid Codec: LM + skip-n-gram engine + PPM, trained end-to-end on arithmetic-coding rate with a per-token hedge mixer."* No comp leader treats the artifact codec-first.

## Phase A — floor measurements + diagnostic (days 1–2)

**Decision gate.** Three estimators of H(val); if all are ≥1.0 the moonshot dies, revert to "win at <1.07".

- **Floor 1 — classical compressors on val bytes**. `zstd -22`, `xz -9e`, `brotli -q11` on raw FineWeb val. Tight model-free upper bound on H(val).
- **Floor 2 — KN n-gram extrapolation (n=1..7+)**. Tells us where the pure n-gram floor is and the marginal value of each extra n-gram order. Directly informs L09 backlog priority.
- **Floor 3 — GPT-4o logprobs on val sample**. Tightest practical upper bound on H(val) from a frontier LM. Burns OpenAI credits, ~$5.

**Diagnostic.** Per-token loss snapshot of our 1.082 stack on val. Bucket by token rarity / domain (code, URLs, prose). Tells us where our 0.012 BPB to SOTA actually lives.

Output: `docs/research/FLOOR_RESULTS.md` with a clear go/no-go for <1.0.

## Phase B — build the codec scaffold (days 3–5)

- Switch training loss from pure CE to **arithmetic-coding-rate** (the actual quantity val_bpb measures, not a proxy).
- Build the **hedge-mixer infrastructure** in eval — N-way scorer that combines log-probs from any subset of: LM, n-gram engine, PPM, brotli posterior. Per-token learned weights.
- Wire the codec end-to-end: bootstrap with current LM + frozen trigram bias, verify val_bpb pipeline matches submission harness.

## Phase C — stack the components (days 6–9)

In parallel, three tracks:

1. **L09 n-gram engine completion** — ship the backlog: skip-bigram (LESSONS §31 +0.28 bits/tok), adaptive cuckoo hash, 5-gram + 7-gram tables. Mac MLX prototyping; cheap pods for ablation.
2. **PPM/CTW augment** — port a minimal PPM / Context-Tree-Weighting predictor as a hedge-mixer input. CTW is provably near-optimal on stationary sources; LM + CTW should compound.
3. **Compression-aware training** (the contingency novel) — co-train for `loss-after-16MB-compression`, not `loss-then-quantize`. Ready to swap in if the eval-time hybrid hits a rules wall.

## Phase D — H100 confirms (days 10–13)

Burn $1k OpenAI credits + remaining $36 RunPod budget on real H100 runs:
- 3-seed × top-2 candidate stacks = ~6 H100 runs (~$30–60 each).
- Final candidate: 5 seeds, full eval, full compression pass.
- Hold $200 of credits for tightening (extra GPT-4o scoring of intermediate failure modes).

## Phase E — submit (days 14–15)

- Final compression under 16MB cap (the bug that bit retry-6).
- Submit, log, monitor. No further changes after submission unless fatal regression.

## Falsification + pivot rules

- **If Floor 1+2+3 all ≥1.0** → moonshot dead; revert to "win <1.07" with comp-port-completeness audit + L09 ship + retry-6 fix.
- **If KN H_5 ≈ H_3** → kill the n-gram leg; pivot to compression-aware training as the novel.
- **If hedge mixer ≤ best single component on val** → mixer doesn't compose; ship best single component.
- **If H100 confirms diverge from cheap-pod by >0.02** → distrust cheap-pod; route everything through H100 (constrains experiments).

## Reading the rules carefully

The eval-time hybrid components (online n-gram cache, TTT during eval) are at the edge of comp rules. **Before days 6–9 ship**, audit the comp's eval harness (`submission/run.sh`, the official scoring code) for:
- Is online learning during eval allowed? (Score-First TTT precedent suggests yes, but online n-gram cache might be different.)
- Is the artifact 16MB hard cap enforced post-decompression or pre-decompression?
- Can the scoring rule be arbitrary code, or does it have to be log-prob from a single model?

If rules forbid the eval-time leg, the codec target shrinks to ~0.95 (model + n-gram + PPM only), still well below SOTA.

## Daily cadence

Cron fires for monitor + research stay running. Personal audits every 3–5 fires (per `feedback_audit_cadence.md`). Spend logs in `docs/research/RESEARCH_LOG.md`.

## Persistent artifact storage (don't recompute)

Anything expensive to compute (n-gram tables, KN-smoothed counts, tokenizer corpora, PPM contexts, GPT-4o val scoring outputs, intermediate H100 checkpoints worth re-using) should be persisted **off-laptop**, not regenerated each session.

- **Home lab (Coolify)**: `http://100.91.109.112:8000/` — primary store. Tag artifacts with the script + commit hash that produced them.
- **S3** (RunPod can sync directly from S3): use as the warm-cache for anything a pod needs to pull at the start of a run. Avoid re-uploading from Mac → pod for files >100MB; push Mac → S3 once, pod pulls from S3.

Workflow: compute on Mac/pod → push to home lab + mirror to S3 → all future runs (Mac, any pod) pull from S3. This avoids: (a) re-tokenizing 10B FineWeb shards, (b) re-building n-gram tables, (c) re-scoring GPT-4o val samples, (d) losing artifacts when pods get destroyed.

## Memory pointers

- Codebase orientation: `CLAUDE.md`
- Knowledge index: `docs/README.md`
- Live research log: `docs/research/RESEARCH_LOG.md`
- Trigram floor (current measurement): `docs/research/SHANNON_FLOOR.md`
- Latest H100 run: `records/track_10min_16mb/2026-04-10_SP8192_NL11_MLP4_int8_ParMuon_PR7_LegalTTT/submission.json`
