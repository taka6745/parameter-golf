# Research Protocol — how to generate PhD-worthy ideas, not brainstorm mush

**Authors**: taka + claude  ·  **Date**: 2026-04-16  ·  **Status**: canonical process doc

**Purpose**: When the user asks me (the LLM) to "find novel / PhD-worthy / world-first ML techniques to improve our 16 MB submission", do NOT free-brainstorm. Run this protocol. It separates "ideas that sound novel" from "ideas that *are* novel, constrained, falsifiable, and rank-ordered by EV." The output is a set of 1-pagers ready to become experiments.

This protocol is the canonical generator for entries in `docs/ideas/`. For the full research-doc architecture (ideas → experiments → findings → findings), see `DOC_SYSTEM.md`.

## 0. Why free brainstorm fails

LLMs (including me) produce "ideas that sound novel but aren't" when asked open-ended research questions:

- Combinations of already-published techniques surface as "new"
- Ideas that need tacit expertise in adjacent fields get re-invented in half-baked form
- Filter for "is this novel vs. the 2024–2026 arxiv + the comp's 1500+ PRs" is never applied
- Falsification criteria are omitted because they're hard to generate cold

**Fix**: force structure. The six-step protocol below makes each step explicit and audit-able. Every candidate that survives has a paper trail.

## 1. The six-step protocol

### Step 1 — Stack × literature grid (the generator)

For each of the 11 stack layers (`L01` tokenizer → `L11` infra), cross-pollinate with a specific non-NLP literature. The query is not "novel ML for layer X" but **"*this layer* × *this foreign field*"**:

| Layer | Current in our stack | Foreign fields to mine |
|---|---|---|
| L01 tokenizer | sp8192 BPE | coding theory (Huffman, arithmetic, Elias, Golomb), DNA/genome compression (BWT, RLE), speech coding (G.711, CELP), universal codes |
| L02 data loader | coprime stride, MDL-compressible-first | data distillation, active learning, curriculum (CL/SPL), sampling theory (MCMC, importance sampling), anti-curriculum |
| L03 architecture | 11L × 4×MLP + depth recurrence + gated attn + parallel residuals | equilibrium models (DEQ), JEPA, state-space (Mamba, S4), neuromorphic (SNN), optical, mixture of depths, hypernetworks, universal transformers |
| L04 optimizer | Muon + NorMuon + ParMuon | Langevin, Hamiltonian MC, evolutionary (CMA-ES), meta-learning (MAML, Reptile), second-order (K-FAC, Shampoo), manifold (Stiefel, Riemannian) |
| L05 training | EMA, warmdown, QAT, Norm-PCT-Dropout | label smoothing variants, distillation (self, mean-teacher), contrastive pretraining, auxiliary losses, compression-aware regularizers |
| L06 eval / scoring rule | sliding window + gated TTT + n-gram bias + DC500 | arithmetic coding, MDL, Context Tree Weighting (CTW, Willems 1995), PPM, Solomonoff prior, Bayesian model averaging, hedge mixer |
| L07 compression | int6 GPTQ + brotli + BSHF stride 2 | BitNet (1.58-bit), fractional bits, sign-magnitude, signed-digit, neural compression (INRs), JPEG-style transform coding, RVQ/ERVQ, tensor-train |
| L08 n-gram bias | bigram + trigram + fourgram + fivegram, 16 K buckets | tabulation hashing (done), adaptive cuckoo, count-min sketch, Bloomier filter, Elias-Fano codes, random projection ngrams |
| L09 n-gram engine | static tables + entropy-adaptive mixer + DC500 | dynamic radix trie, suffix arrays, Burrows-Wheeler, CTW online construction, PPM-Star, LZMA |
| L10 TTT / eval adapt | Score-First TTT (3 SGD epochs on scored chunk) | MAML, gradient-only inference, online convex optimization, FTRL, universal source coding, online n-gram cache |
| L11 infra / speed | FA3, Triton TMA+megakernel, expandable-segments allocator | custom CUDA kernels (fused QKV, fused MLP+norm, fused LM head), PTX hand-tuning, mixed-precision scheduling, fused bit-packing, wgmma |

Generate 5–10 candidates per (layer, foreign field) cell. Output target: 50–150 raw candidates.

### Step 2 — Constraints pass (the rule filter)

Apply `MOONSHOT_RULES.md §3` table to drop:
- Anything that touches val before scoring (forbidden)
- Anything that needs external compute / distillation from pretrained LMs (forbidden)
- Anything that downloads during eval (forbidden)
- Anything that blows the 16 MB cap unless destined for the non-record track

Also drop anything `STACK_UTILISATION_RESULTS.md` proves is already maxed:
- Low-rank factoring (P6: near-full rank)
- Dead-head / dead-neuron pruning (P3/P4: zero)
- BSHF stride tuning (P10: already optimal)

Cross-check `STACK_NOVELTY_TRACKER_v2.md` ❌ killed column to avoid re-proposing known dead ends.

### Step 3 — Priors pass (the EV calibrator)

For each survivor, assign a quantitative score on four axes:

| Axis | Scale | Source |
|---|---|---|
| Expected BPB improvement | 0.001 – 0.15 | inferred from probes (rare-token target? measured slack?) |
| Novelty vs. comp PRs | 0 (done) – 1 (no PR in this direction) | audit via `gh pr list openai/parameter-golf` |
| Compatibility with other candidates | 0 (exclusive) – 1 (stacks cleanly) | composability reasoning |
| Compute cost to prototype | hours on H100 | rough estimate |

Rank by `expected_BPB × novelty × compatibility / compute_cost`. Top 20 advance.

### Step 4 — Prior-art audit (the sanity check)

For each of the top 20, spawn a WebFetch / WebSearch subagent to check:
- Is this in a 2023–2026 arxiv paper?
- Is this in any open or merged comp PR?
- Is this in a github repo that a reviewer could point to?

Kill survivors with strong prior art. Downgrade "partial prior art" ones (mark as port-with-evidence per hard rule #2 instead of world-novel).

### Step 5 — Falsifiability pass (the PhD filter)

Each surviving idea gets a one-pager in `docs/ideas/IDEA_<id>_<slug>.md` using the template at `docs/ideas/TEMPLATE.md`. Required sections:

- **Hypothesis**: one sentence, testable
- **Method**: 2–3 paragraphs + pseudocode
- **Expected BPB**: number + range + reasoning
- **Testable prediction**: what specific metric should move, by how much, after how many training steps
- **Falsification criterion**: what result would kill the idea (specific number, not "it doesn't work")
- **Stacking plan**: can this be combined with the moonshot + current stack?
- **Prior-art audit** (URLs, last-checked date): output of Step 4

Ideas that can't pass falsifiability aren't PhD-worthy, they're vibes. Drop them.

### Step 6 — Rank and execute

Top 3–5 get an experiment doc in `docs/experiments/` (from TEMPLATE.md) and go to H100 non-record track for POC. Results land in a FINDING in `docs/findings/`. The best becomes the moonshot; the rest become stacking wins.

## 2. How to invoke the protocol (canonical prompt)

Open a new session (clean context ideal) with this message:

> Run the research protocol from `RESEARCH_PROTOCOL.md` against our current stack. Priors: `STACK_UTILISATION_RESULTS.md`, `FLOOR_RESULTS.md`, `STACK_NOVELTY_TRACKER_v2.md`, `MOONSHOT_RULES.md`. For each candidate that survives all 6 steps, create an idea doc at `docs/ideas/IDEA_<next_id>_<slug>.md` using `docs/ideas/TEMPLATE.md`. Update `docs/ideas/README.md` index. Use WebFetch/WebSearch subagents for prior-art checks. Goal: 5–10 idea docs ready for experiment.

Expected runtime: 2–3 hours of wall-clock with subagents. Output: 5–10 complete idea 1-pagers, prior-art audited, ranked by EV, ready to prototype.

## 3. What the protocol does NOT do

- Invent techniques that require tacit expertise only a human researcher has. (I can propose "port CTW to byte-level LMs"; I can't tell you whether CTW's convergence guarantees survive non-stationary FineWeb.)
- Predict which idea actually wins. Ranking uses *priors*, not ground truth. Only H100 runs give ground truth.
- Replace the need for a human to read the literature. Subagent audits are coarse — a human spotting "this is actually Paper X" is still the ultimate filter.
- Produce more than ~10 high-quality candidates per run. Going wider dilutes quality.

## 4. Why this is better than ad-hoc brainstorming

The structured protocol gives you an **audit trail**. If a candidate survives all six steps, you can read back exactly why: *"L09 × universal-codes, stacks with n-gram cache, no prior art 2023–2026, expected 0.02 BPB based on P7 rare-token signal, falsifiable via N-seed val_bpb at step K."* That's a PR-worthy justification, not a vibe.

Contrast: "hey LLM, novel ML idea?" → "mixture of experts" → user has no way to tell whether that's novel, defensible, or the LLM's highest-frequency association.

## 5. Running the protocol — meta-rules for the LLM

When running this protocol as the LLM executor:

- **Always start by reading priors**: `STACK_NOVELTY_TRACKER_v2.md`, `STACK_UTILISATION_RESULTS.md`, `FLOOR_RESULTS.md`, `MOONSHOT_RULES.md`. Without these, the protocol degrades to brainstorm.
- **Force the grid literally**: don't skip cells. Every (layer × foreign field) cell must produce ≥3 candidates or an explicit "no candidates — the foreign field has nothing this layer needs" note.
- **Use subagents for prior-art checks**: don't trust your own memory of arxiv. Spawn an Explore agent per top-20 candidate with a web search.
- **Kill ruthlessly in Step 2**: better to lose a hypothetically-good idea to rule misreading than to create a submission that disqualifies.
- **Never skip Step 5**: falsifiability is what separates novelty from vibe. If you can't write a specific metric × threshold × step-count prediction, the idea isn't ready.
- **Write every surviving idea to disk**: one file per idea. Don't generate a combined list — individual files are how the doc system tracks state. See `DOC_SYSTEM.md`.

## 6. Related

- `MOONSHOT_RULES.md` — what's legal
- `DOC_SYSTEM.md` — where the ideas / experiments / findings live and how they're connected
- `STACK_NOVELTY_TRACKER_v2.md` — canonical catalog of all known directions
- `STACK_UTILISATION_RESULTS.md`, `FLOOR_RESULTS.md` — priors that inform Step 3 EV calibration

## 7. Changelog

- **2026-04-16**: extracted from `MOONSHOT_RULES.md §10` so rules doc stays rules-only. Added §5 meta-rules for the LLM executor.
