---
slug: short-kebab-case
date: YYYY-MM-DD
title: "One-line title — headline metric + scope"
ideas: [IDEA-NNN]
experiments: [EXP-YYYY-MM-DD-NNN, EXP-YYYY-MM-DD-NNN+1]
headline_metric: "val_bpb = X.XXXX (N-seed mean, σ=Y.YYYY)"
tags: [L0X, novelty-class, any other]          # e.g. [L09, WN, moonshot]
updates:                                        # files this finding requires updating
  - STACK_NOVELTY_TRACKER_v2.md#l0X-<row>
  - WIN_PLAN.md#section-N
  - MEMORY.md (if a cross-session rule is established)
novelty_status_after: validated                 # validated | partial | killed
---

# <Title — same as frontmatter>

**Authors**: taka + claude  ·  **Date**: YYYY-MM-DD

## TL;DR

1–3 sentences. What moved, by how much, at what cost, what does it change about the plan.

> Example: *Dropping gated attention from our stack reduces val_bpb from 1.082 to 1.072 at 3-seed mean (p < 0.005) and reclaims 45 KB of budget. This closes 1/13 of our gap to SOTA 1.07 as a single experiment.*

## 1. Background

- What question was being asked
- Which prior docs this builds on (STACK_UTILISATION_RESULTS, FLOOR_RESULTS, a prior FINDING, etc.)
- Why this finding matters now

## 2. Method

- Parent idea(s): [IDEA-NNN](../ideas/IDEA_NNN_<slug>.md)
- Experiment(s) this is built from: links
- Concise method description (2–3 paragraphs), including any differences from the IDEA's original plan
- Pseudocode if the method has a novel core

## 3. Experimental setup

| | |
|---|---|
| Hardware | 1×H100SXM (or 8×H100SXM for comp-spec) |
| Seeds | 42, 314, 999 |
| Training budget | 600 s wallclock |
| Evaluation | sliding window stride=64, seq_len=2048 |
| Baseline | our 1.082 submission (records/track_10min_16mb/2026-04-10_*/) |
| Deltas applied | (list config changes) |

## 4. Results

### 4.1 Primary metric

| seed | val_bpb | artifact_bytes | wallclock |
|---|---:|---:|---:|
| 42  | | | |
| 314 | | | |
| 999 | | | |
| **mean** | | | |
| **σ** | | | |

### 4.2 ASCII diagram of the finding shape

```
(include a diagram — curve, bar chart, or landscape — that shows the shape of the result
 without requiring a reader to scan the table)
```

### 4.3 Significance

- **vs. baseline**: Δ = X.XXXX, p-value = Y.YY (per 3-seed comparison)
- **Comp 0.005-nat bar**: pass | fail
- **Crosses SOTA 1.07**: yes | no

## 5. Simple worked example

Concrete single-token or single-chunk example showing the mechanism at work. One row of the results table, explained so a reader can sanity-check.

## 6. Discussion

### 6.1 Why this worked (or didn't)

### 6.2 What this changes about the plan

- `STACK_NOVELTY_TRACKER_v2.md`: row `<row>` status → `validated`
- `WIN_PLAN.md`: section `<N>` updated; revised expected-BPB budget
- Any new idea candidates this opens up (link to IDEAs)

### 6.3 Stacking implications

- Stacks cleanly with: <other validated findings>
- Conflicts with: <other findings>
- Further compounding: <expected follow-up>

## 7. Limitations

- Sample size (N seeds), val sample size, confidence bounds
- Configurations not tested
- Potential confounds
- What would invalidate this finding

## 8. Follow-up experiments

- [ ] Experiment ideas this opens up, each linking to a new IDEA file
- [ ] Confirmation at 8×H100 comp-spec (if only 1×H100 tested so far)
- [ ] Longer-seed ensemble (N=5 or more) if the p-value is close to threshold

## 9. Artifacts

| | |
|---|---|
| Experiment records | [EXP-...](../experiments/EXP_...md) |
| Pod logs | `/workspace/paramgolf/logs/<run_id>.log` on the H100 pod |
| Homelab cache | `https://paramgolf.koda-software.com/logs/<run_id>.log` |
| Checkpoint (if notable) | homelab path |

## 10. Changelog

- YYYY-MM-DD: initial writeup
- YYYY-MM-DD: corrections, additional seeds, etc.
