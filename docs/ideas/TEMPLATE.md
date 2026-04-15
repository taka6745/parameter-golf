---
id: IDEA-NNN
slug: short-kebab-case-slug
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: draft                         # draft | audited | approved | in-experiment | validated | killed | superseded
layer: L0X                            # L01..L11 or "cross-layer"
novelty_class: WN                     # WN (world-novel) | CN (comp-novel) | CP (comp-port)
expected_bpb: [-0.02, -0.005]         # [lo, hi] range; negative = improvement
cost_hours: 4                         # estimate wallclock on H100 for a first POC run
depends_on: []                        # other IDEA-### that must land first
blocks: []                            # IDEA-### blocked by this
supersedes: []                        # older IDEA-### replaced by this
stack_row: STACK_NOVELTY_TRACKER_v2.md#l0X-<row-slug>
prior_art_checked: YYYY-MM-DD
---

# IDEA-NNN: <one-line title>

> **Hypothesis** (one sentence, testable):
>
> Replacing X with Y in L0Z of the stack will reduce val_bpb by at least 0.Z BPB at 3-seed mean on the H100 600s budget.

## Method

<2–3 paragraphs. Concrete. Include pseudocode when helpful.>

```python
# pseudocode — how the method plugs into the existing stack
for token in val:
    ...
```

**Integration point**: which file(s) change, which env vars are introduced, how the 16 MB budget is re-allocated.

## Expected BPB

- **Range**: [expected_bpb from frontmatter]
- **Mechanism**: why this should help (which probe / prior / literature result supports the number)
- **Lower bound reasoning**: what's the floor this method can't go below
- **Upper bound reasoning**: what's the ceiling (and why we can't fully reach it)

## Testable prediction

- **Metric**: `val_bpb`
- **Threshold**: ≤ X.XXXX
- **After**: N training steps, K seeds (≥2 for confirmation, ≥3 for SOTA claim)
- **On**: 1×H100 (cheap POC) or 8×H100 SXM (comp-spec)

## Falsification criterion

**Kill this idea if**: after N training steps at 3 seeds, the mean val_bpb is ≥ X.XXXX (specific number, not "it doesn't work").

If val_bpb improves but not by the expected amount, downgrade to "partial" and re-plan stacking strategy rather than shipping alone.

## Stacking plan

- **Composes with**: (list of other IDEA-###s that stack cleanly)
- **Conflicts with**: (list of IDEA-###s that would interfere)
- **Blocks**: (what this enables downstream)
- **Budget footprint**: estimated bytes added to / reclaimed from the 16 MB cap

## Prior-art audit

- **Arxiv (2023–2026)**: <search terms used, last-checked date, hit-count, notable papers + why they're not this exact idea>
- **Comp PRs (openai/parameter-golf)**: <grep terms, hit-count, nearest PR + why ours is different>
- **Public repos / gists**: <any implementations found>
- **Verdict**: world-novel / partial-overlap-with-X / comp-port-from-PR-###
- **Checked by**: <agent name + date>

## Lineage

- **Where this came from**: <research protocol step, or a user ask, or a doc mine, or a STACK_NOVELTY_TRACKER_v2 row>
- **Supersedes**: <older IDEA-###s this replaces, if any>
- **Related findings**: <FINDING-slug URLs>

## Risks / known unknowns

- <List what could go wrong. Not excuses — actual risks.>
- <Each with a mitigation if one exists.>

## Notes

<free-form scratch — anything that doesn't fit above. Revision history at the bottom.>
