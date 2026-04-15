# Moonshot Rules — what the comp actually allows, and what it forbids

**Authors**: taka + claude  ·  **Date**: 2026-04-16  ·  **Sources**: `README.md` (repo root FAQ), `submission/run.sh`, `submission/README.md`

## TL;DR

The comp's rule surface is narrower than it sounds. **Most of the "world-class novel" directions from `STACK_UTILISATION_RESULTS.md §12.5` are fully legal** — the only constraints that actually bite a moonshot are:
1. The 16 MB cap is on **code bytes + compressed model bytes** (decimal MB, 16,000,000), not weights alone.
2. **No val access before the val token is scored** — including "paid-prefix" tricks and memorizing val statistics into weights.
3. **Test-time training / online adaptation** is explicitly allowed **on val tokens you have already scored**. That's the single most important rule for the moonshot directions.
4. **Evaluation wallclock** ≤ 10 min on 8×H100 (separate from the 10-min training budget).

**Everything below is derived from primary sources** — quotes are verbatim from `README.md` where rule language is subtle.

## 1. The canonical rules (primary source quotes)

### 1.1 Artifact budget

> "The submission artifact is computed as **code bytes plus compressed model bytes**. All counted code should live in the `train_gpt.py` script. The cap is decimal 16MB, i.e. 16,000,000 total bytes, not 16 MiB / 16,777,216 bytes."  (`README.md`, FAQ)

**What counts toward the 16 MB cap**:
- `train_gpt.py` (or the submission's equivalent), UTF-8 bytes
- The compressed model artifact (e.g. `final_model.int6.ptz`)

**What does NOT count** (free):
- Python package imports (torch, flash-attn, brotli, etc.)
- Custom kernels shipped as CUDA/Triton source via a pip-installable library
- Dataset-tokenization code, as long as it's not called during eval
- The tokenizer file itself? Unclear — but if you edit the tokenizer, the cost has to be accounted for. See §2.3.

### 1.2 Training / eval time budgets

> "Final leaderboard submissions must run in under 10 minutes on 8xH100s." (record-track limit — there's a separate unlimited-compute non-record track)

> "We won't accept submissions that take more than 10 minutes on 8xH100 to evaluate (Note: This limit is in addition to the 10 minutes of training time allowed!)"

So: 10 min train + 10 min eval = 20 min total budget on 8×H100. A moonshot that does expensive eval-time work (online cache, TTT) is fine as long as it fits in the eval budget.

### 1.3 External access (forbidden)

> "No external downloads, training dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible."

So: no phoning home to a bigger model. All information must live inside the 16 MB artifact.

### 1.4 The val/training separation — the single most important rule

> "You CANNOT access validation data during training, e.g. by compressing it into your 16mb with 'paid prefix'."

> "If it isn't abundantly obvious: You can't cheat on your test loss. You can't cheat by training on the validation set before you evaluate on the validation set. **The validation language around test-time training has been confusing people: you are only allowed to test-time train on validation set tokens you've already evaluated your model on, since those tokens have already been graded!**"

**Translation**: the causal-scoring rule. The exact protocol is
```
for token_i in val:
    score = model.logprob(token_i | context_before)       # this gets counted into BPB
    model.adapt(seen = val[0:i+1])                        # this is fine, now that token_i is graded
```
This is what the submission's existing `Legal Score-First TTT` implements and why it's named "Legal".

### 1.5 Tokenizer freedom

> "If changes are made to the tokenizer or dataset, prove with certainty that the val_bpb is correctly calculated. Submissions that edit the tokenizer will be examined much more carefully, since bugs may unjustly improve your score."

So: **tokenizer novelty is legal** but requires extra BPB-correctness verification. Any byte-level BPB math (`nats / ln(2) / bytes`) must use the original UTF-8 byte count, not the new tokenizer's byte count.

### 1.6 Library imports

> "Yes, you're free to import any package or library you want, so long as it does not unjustly violate the rules on evaluation, compute, training time, code size or otherwise. Just include a requirements.txt in your records folder and mention setup instructions in your README.md."

> "You can't sneak in extra compute, capabilities, or massively increase effective code size with custom libraries, but importing FlashAttention, etc. is completely fine."

So: shipping a novel kernel as a pip-installable library is fine. Shipping a 4 GB pretrained model weight file as a pip dependency is not — that's "extra capabilities".

### 1.7 Stats significance bar

> "New SOTA records … must beat the existing SOTA by at least 0.005 nats. All submissions must provide enough run logs to show at p < 0.01 that they achieved the required 0.005-nat improvement."

So: a single-seed 0.005 BPB win is not enough. Need ~3-seed validation by default.

### 1.8 The non-record track

> "We also accept non-record submissions to an unlimited compute track for runs that are not intended to meet the 10-minute cutoff."

Any moonshot that doesn't fit 10 min but is conceptually interesting can land in the non-record track (see Ciprian-Florin Ifrim's 1-bit quantization / 4-hour training entries). **Our moonshot can START in the non-record track for proof-of-concept, then optimize for the 10-min track second.**

## 2. Applying the rules to each moonshot

From `STACK_UTILISATION_RESULTS.md §12.5`:

### 2.1 Eval-time growing n-gram cache  ✅ **Legal** (causal-only)

**Protocol**: n-gram table starts empty (or pre-seeded from train). When val token `i` arrives:
1. Compute `P(token_i | last k bytes of val context, n-gram table)`, mix with LM logprob.
2. Score `-log₂ P(token_i)` → BPB contribution for token `i`.
3. **Then** add `(last k bytes, token_i)` to the table.
4. Proceed to token `i+1` with the updated table.

**Why legal**: step 3 happens after step 2. Every token is graded against a table that only contains already-scored tokens.

**Rule-text match**: the Score-First TTT rule (§1.4) is identical in structure to this — just more general.

**Caveats**:
- The initial state of the table must come from train data only. Pre-seeding with val summary statistics = forbidden.
- Table size counts against the 16 MB cap. If the table starts at 2 MB, you have 14 MB for the rest of the codec.
- Needs a `log(N)` table data structure (probably a hash map with LRU) to fit in H100 RAM + be fast during eval.

**Precedent**: `cmix` (the general-purpose compressor benchmark) uses an online context-tree-weighted predictor that grows during decoding. It gets ~0.9 BPB on enwik9. We can expect similar headroom.

### 2.2 Compression-aware training  ✅ **Trivially legal**

Training-side only. Loss = `NLL + λ × f(θ)` where `f` is an entropy-regularizer or a brotli-size surrogate. No rule touches training-side loss formulation.

**Only caveat**: the final artifact still has to fit in 16 MB and run its own eval under the time budget. λ too high → bad BPB; λ too low → no compression benefit.

### 2.3 Huffman-weighted / novel tokenizer  ✅ **Legal with extra scrutiny**

**Legal** per §1.5, with the requirement to verify BPB calculation. The BPB must be computed on the **original UTF-8 byte count of the FineWeb val** (not the new tokenizer's token count × some mystery factor).

**How to prove correctness**:
1. Before shipping, produce a side-by-side comparison: our BPB vs. a reference BPB using sp1024 on the same val text.
2. Include the tokenizer file in the artifact. It counts against 16 MB (the tokenizer is either code or model bytes).
3. In the README, write the exact BPB formula used: `BPB = (sum_of_nats_across_val_tokens / ln(2)) / len(val_bytes_utf8)`.

**Caveats**:
- If tokenizer changes, the training data tokenization changes, so the model must be retrained with the new tokens. Train + tokenizer + artifact must all be consistent.
- A Huffman-style variable-length token-ID encoding (smaller IDs for common tokens, larger IDs for rare ones) is fine as long as the artifact-bytes accounting is correct.
- Byte-level / char-level tokenizers are explicitly encouraged in the comp's "requests for PRs" (H-net tokenization is listed as a wanted direction).

### 2.4 Per-token hedge mixer (LM + n-gram + CTW + PPM) ✅ **Legal**

Hedge-mixing predicted probabilities over multiple predictors is a standard scoring-rule technique. Rule §1.4 only forbids looking ahead at val; the hedge mixer operates on the current context and previous-token history.

**Caveats**:
- Each component (LM, n-gram, CTW, PPM) has weights / state that count toward the 16 MB. Cost is additive.
- The mixer itself (per-token gating network) needs to fit in budget — probably a tiny MLP.
- Arithmetic-coding loss (the true comp objective) is the natural loss here.

### 2.5 Rare-token MoE specialist  ✅ **Legal**

Architectural novelty, no rule tension. The specialist's parameters count against the 16 MB.

**Caveats**:
- Routing has to be causal (route based on context, not on the target token).
- Gating on "is this token rare?" requires knowing the token — it's fine if the gating is based on context features predicting rarity, not the label itself.

### 2.6 CTW / PPM hybrid (online)  ✅ **Legal** (same rules as §2.1)

Context Tree Weighting is a provably near-optimal universal predictor on stationary sources (Willems et al., 1995). Ships as a ~200-line C / Rust library; compiles to a few KB. Built online during eval.

**Rule fit**: identical to §2.1. Build the tree causally from previously-scored tokens.

### 2.7 Test-time token-aware temperature  ✅ **Legal**

A scoring-rule tweak. Temperature can be a function of context (token rarity indicators, position, output entropy). No rule tension.

## 3. Summary table — what's allowed

| Moonshot | Legal? | Main constraint |
|---|---|---|
| **Eval-time n-gram cache** | ✅ | Must grow from causal val tokens only |
| **Compression-aware training** | ✅ trivial | Training-side only |
| **Huffman-weighted tokenizer** | ✅ scrutinized | Prove BPB formula correct; tokenizer bytes count |
| **Per-token hedge mixer** | ✅ | Each predictor's bytes count |
| **Rare-token MoE specialist** | ✅ | Routing must be causal |
| **CTW / PPM hybrid (online)** | ✅ | Build tree causally from val tokens already scored |
| **Token-aware temperature** | ✅ | None |
| **Custom CUDA kernel for a novel op** | ✅ | Ship via pip lib; no val access |
| **Train for 4 hours on 8×H100** | ✅ (non-record) | Sits in the unlimited-compute track |
| **Train on 0.1% of val, use for arch search** | ❌ | "You CANNOT access validation data during training" |
| **Precompute val n-gram stats, pack as "prefix"** | ❌ | Explicit "paid prefix" example in FAQ |
| **Distill from a pretrained bigger LM (Qwen/Llama/etc.)** | ❌ | Violates spirit: inherits thousands of external GPU-hours of training compute. The FAQ forbids "sneaking in additional compute unfairly." |
| **Use a pretrained model as a measurement reference (score val, compare BPB)** | ✅ | No artifact contamination — the reference model doesn't influence our weights |
| **Download another model weight during eval** | ❌ | "No external downloads, training dataset access, or network calls are allowed during evaluation" |

## 4. Simple worked example — an "eval-time n-gram cache" that passes the rule

Suppose our codec is: `LM + cache`. The cache starts at size 0.

```
# At eval time, given val bytes [b0, b1, b2, ..., bN-1]:
context_prefix = []  # grows with scored bytes
n_gram_cache = {}    # empty map  

total_nats = 0
for i in range(N):
    # Step 1: compute model's distribution
    lm_p = softmax(model.forward_logits(tokenize(context_prefix))[-1])
    # Step 2: look up n-gram from cache (built from already-scored bytes)
    ngram_p = n_gram_cache.get(context_prefix[-k:], uniform)
    # Step 3: hedge-mix
    final_p = alpha * lm_p + (1 - alpha) * ngram_p
    # Step 4: score this byte
    total_nats += -log(final_p[b_i])
    # Step 5: UPDATE cache with (context, b_i) — this is legal because b_i is now "graded"
    n_gram_cache[tuple(context_prefix[-k:])][b_i] += 1
    context_prefix.append(b_i)

val_bpb = total_nats / log(2) / N
```

Step 5 happens AFTER step 4. That's what makes it legal — no val data is used to predict a val byte before that byte is graded.

Rule §1.4 blesses this explicitly: "you are only allowed to test-time train on validation set tokens you've already evaluated your model on, since those tokens have already been graded!"

## 5. The moonshot decision tree

```
Does the novel idea read val before scoring the token?
├── YES → FORBIDDEN. Think harder.
└── NO
    ├── Does it fit in 16 MB (code + model)?
    │   ├── NO → unlimited-compute track, fine for POC
    │   └── YES
    │       ├── Does eval finish in 10 min on 8×H100?
    │       │   ├── NO → unlimited-compute track
    │       │   └── YES → ✅ LEGAL FOR RECORD TRACK
    │       └── Special case: changes tokenizer?
    │           └── YES → also prove BPB formula is correct
    └── Is any training data a subset of val?
        └── YES → FORBIDDEN
```

## 6. What this changes about our plan

### 6.1 Confirmed-legal directions we should attack

All 7 moonshots from `STACK_UTILISATION_RESULTS.md §12.5` are legal. Specifically, **online n-gram cache and CTW/PPM hybrid** — the two with the biggest theoretical payoff (0.05–0.15 BPB based on `cmix` and universal-compression literature) — are fully within the rules.

### 6.2 Re-ordered moonshot priority

Given that nothing is blocked by rules, priority is now pure expected-BPB × novelty:

1. **Online n-gram cache / CTW hybrid** (expected 0.05–0.15 BPB, rules-confirmed)
2. **Compression-aware training** (expected 0.01–0.03 BPB, trivially legal, novel)
3. **Huffman-weighted tokenizer** (expected 0.01–0.02 BPB, extra review but legal)
4. **Per-token hedge mixer + arithmetic-rate loss** (expected 0.02–0.05 BPB, legal)
5. **Drop gated attention** (expected 0.01 BPB, mundane unambiguous win per P15)
6. **Rare-token MoE** (expected 0.01–0.03 BPB, legal)
7. **Token-aware temperature** (expected 0.01–0.02 BPB, legal and cheap)

### 6.3 Non-record track as a staging area

The comp explicitly allows unlimited-compute entries (4-hour trainings, huge models) in a separate leaderboard. **This is where the moonshot should prove itself first** — validate the technique at any scale, then optimize for the 10-min budget. The risk-free path:

1. Prototype the moonshot (online n-gram cache) **without the 10-min constraint** on H100.
2. If val_bpb improves → submit to unlimited-compute track for credit.
3. Optimize for the 10-min track separately. If the win survives the compression, submit for record.

## 7. Limitations of this doc

- Based on my reading of `README.md` and `submission/*` as of 2026-04-16. The comp Discord (channels `#parameter-golf-discussions`, `#parameter-golf-announcements`) may have clarifications we missed.
- Some rules are deliberately vague ("extra compute / capabilities" via libraries) — when uncertain, ask via a GitHub issue or Discord before committing a 1-week project.
- Non-record submissions have a "high bar" per the FAQ but no specific numerical criterion. Expect human review.

## 8. Quick-reference: what to check before shipping a moonshot

- [ ] Artifact (code + compressed model) ≤ 16,000,000 bytes
- [ ] No val tokens read before their corresponding BPB contribution is measured
- [ ] Eval wallclock ≤ 10 min on 8×H100 (or non-record OK if not record-track)
- [ ] All code lives in `train_gpt.py` or sibling scripts as expected
- [ ] If tokenizer modified: BPB formula verified with a reference comparison
- [ ] 3-seed variance shown for > 0.005 BPB improvement (p < 0.01)
- [ ] No network calls during eval
- [ ] `requirements.txt` lists any new Python deps

## 9. Artifacts

| | |
|---|---|
| Source | `README.md` (repo root), `submission/run.sh`, `submission/README.md` |
| Related | `STACK_UTILISATION_RESULTS.md`, `WIN_PLAN.md`, `FLOOR_RESULTS.md`, `STACK_NOVELTY_TRACKER_v2.md` |
| Next | Pick one moonshot + write its design doc; non-record POC on H100 |

## 10. Research + experiment workflow (see separate doc)

The novelty-mining protocol and the full research / experiment / findings doc system have moved to their own docs so this file stays focused on **rules only**:

- **`docs/research/RESEARCH_PROTOCOL.md`** — how to generate PhD-worthy candidates (the 6-step stack × literature grid)
- **`docs/research/DOC_SYSTEM.md`** — the full research-doc architecture (ideas / experiments / findings / logs) + templates + how they link
- **`docs/ideas/`** — one doc per idea, template-driven
- **`docs/experiments/`** — one doc per run, template-driven
- **`docs/findings/`** — one doc per published finding (paper format)

This file (`MOONSHOT_RULES.md`) is authoritative for *what's legal*. All process / protocol lives in the docs above.
