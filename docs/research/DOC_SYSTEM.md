# Doc System — how the research / experiment / findings docs connect

**Authors**: taka + claude  ·  **Date**: 2026-04-16  ·  **Status**: canonical architecture

**Purpose**: define the **complete doc architecture** for LLM-driven research on this project. The goal is a system where I (the LLM) can research, experiment, record, and iterate across sessions **without losing state, duplicating work, or producing untrackable output**. Every action has a typed artifact. Every artifact links to its inputs.

## TL;DR

Five kinds of docs, four directories, one append-only log. Ideas become experiments become findings. Findings update the canonical state. The state is small and read every session; the history is large and append-only.

```
docs/
├── research/               — canonical state (read every session)
│   ├── MOONSHOT_RULES.md
│   ├── RESEARCH_PROTOCOL.md
│   ├── DOC_SYSTEM.md                 (this file)
│   ├── STACK_NOVELTY_TRACKER_v2.md   (live)
│   ├── STACK_UTILISATION_RESULTS.md  (finished probe paper)
│   ├── FLOOR_RESULTS.md              (finished probe paper)
│   └── RESEARCH_LOG.md               (append-only, chronological)
│
├── ideas/                  — one file per idea (design-stage)
│   ├── TEMPLATE.md
│   ├── README.md           — auto-updated index of all ideas
│   └── IDEA_###_<slug>.md  — 1-pager per candidate (from RESEARCH_PROTOCOL step 5)
│
├── experiments/            — one file per experiment run
│   ├── TEMPLATE.md
│   ├── README.md           — auto-updated index
│   └── EXP_YYYY-MM-DD_###_<slug>.md  — per-run record
│
├── findings/               — one file per published finding (paper format)
│   ├── TEMPLATE.md
│   ├── README.md           — auto-updated index
│   └── FINDING_YYYY-MM-DD_<slug>.md  — paper-style writeup
│
├── plans/                  — strategic plans (read every fire)
│   ├── WIN_PLAN.md
│   └── ... (older phased plans)
│
├── setup/                  — operational how-to (rarely changes)
└── minipapers/             — legacy papers pre-doc-system
```

## 1. The five doc types

### 1a. CANONICAL STATE (small, always read)

**Purpose**: tell the LLM where we are right now, in <5 minutes of reading.

- `MOONSHOT_RULES.md` — comp rules (what's legal)
- `RESEARCH_PROTOCOL.md` — how to run the novelty mining
- `DOC_SYSTEM.md` — this file
- `STACK_NOVELTY_TRACKER_v2.md` — every idea's status
- `WIN_PLAN.md` — current strategic plan
- Probe papers (`STACK_UTILISATION_RESULTS.md`, `FLOOR_RESULTS.md`) — what we measured

**Rule**: these must be readable in one sitting (<15 min). When something changes materially, update them inline, don't accrete sections.

### 1b. IDEAS (one file per design-stage candidate)

**Purpose**: a falsifiable 1-pager per candidate technique. Output of `RESEARCH_PROTOCOL.md` step 5.

- File path: `docs/ideas/IDEA_<id>_<slug>.md`
- ID format: 3-digit zero-padded, monotonically increasing (next in `docs/ideas/README.md`)
- Slug: kebab-case, short ("online-ngram-cache", "huffman-vocab")
- Template: `docs/ideas/TEMPLATE.md`
- Frontmatter is **REQUIRED** (YAML). The README index is built from frontmatter.

**Lifecycle**:
```
draft → audited → approved → in-experiment → validated | killed | superseded
```

Status transitions happen via edit + README rebuild. Each transition logs to `RESEARCH_LOG.md`.

### 1c. EXPERIMENTS (one file per training/eval run)

**Purpose**: an auditable record of every run. Hypothesis → command → results → next action.

- File path: `docs/experiments/EXP_YYYY-MM-DD_<nnn>_<slug>.md`
  - Date: ISO calendar date the experiment was *started*
  - nnn: monotonic 3-digit counter within the day
- Frontmatter binds the run to: `idea` (IDEA-###), `pod`, `seed`, `git_sha`, `config`, `status`
- Template: `docs/experiments/TEMPLATE.md`

**Lifecycle**:
```
pending → running → complete | failed | crashed | aborted
```

Every experiment must link to an IDEA (one-to-many: an idea can have multiple experiments at different seeds / configs).

### 1d. FINDINGS (one paper-format writeup per real result)

**Purpose**: the public "here's what we learned" document. Paper format per the user's `feedback_paper_style_docs.md` memory (TL;DR / background / method / results / discussion / simple example / limitations).

- File path: `docs/findings/FINDING_YYYY-MM-DD_<slug>.md`
- Links backward to the IDEAs + EXPERIMENTs that produced it
- Updates `STACK_NOVELTY_TRACKER_v2.md` and `WIN_PLAN.md` when relevant
- Template: `docs/findings/TEMPLATE.md`

**When to create a FINDING (vs just leaving the EXPERIMENT record)**:
- The result changes the plan (update WIN_PLAN)
- The result is publishable (new SOTA, new diagnostic, a mini-paper)
- The result warrants cross-session continuity (future sessions need to *cite* this)

Not every experiment produces a finding. Most experiments are "run, record, move on." Findings are the weight-bearing milestones.

### 1e. RESEARCH_LOG (append-only history)

**Purpose**: chronological, grep-able log of every significant action. Never edit past entries.

- File path: `docs/research/RESEARCH_LOG.md`
- Format: one entry per event, header `## YYYY-MM-DD HH:MMZ — <short summary>`
- Log entries:
  - New idea created
  - Idea status change (killed, superseded, validated)
  - Experiment started / completed / failed
  - Finding published
  - Infrastructure changes (new pod, homelab changes, broken tooling)
  - Spend updates
- This is the "git log of research" — a single chronological stream.

## 2. How the types link (data flow)

```
  RESEARCH_PROTOCOL.md
          │
          ▼ (run the protocol)
  docs/ideas/IDEA_###_<slug>.md
          │
          ▼ (approve + schedule)
  docs/experiments/EXP_YYYY-MM-DD_###_<slug>.md
          │
          ▼ (run on H100, capture metrics)
  docs/experiments/EXP_…_<slug>.md  (status=complete)
          │
          ▼ (if finding-worthy)
  docs/findings/FINDING_YYYY-MM-DD_<slug>.md
          │
          ▼ (update canonical state)
  STACK_NOVELTY_TRACKER_v2.md  (row status update)
  WIN_PLAN.md                  (plan revision if needed)
  MEMORY.md (project-level)    (if cross-session rule)
          │
          ▼ (always)
  RESEARCH_LOG.md  (append chronological entry)
```

**Backlinks are canonical**: each doc's frontmatter cites its parents. `IDEA` cites `STACK_NOVELTY_TRACKER_v2.md` row. `EXPERIMENT` cites `IDEA`. `FINDING` cites `EXPERIMENT`s. This means you can read upward from any finding to see the full provenance.

## 3. YAML frontmatter (the metadata contract)

Every idea / experiment / finding starts with YAML frontmatter. The READMEs are built from it, so it must be machine-readable.

### IDEA frontmatter

```yaml
---
id: IDEA-042
slug: online-ngram-cache
created: 2026-04-16
updated: 2026-04-16
status: draft           # draft | audited | approved | in-experiment | validated | killed | superseded
layer: L09              # L01..L11, or "cross-layer"
novelty_class: WN       # WN | CN | CP (see STACK_NOVELTY_TRACKER_v2.md)
expected_bpb: [-0.15, -0.05]
cost_hours: 8           # wallclock on H100
depends_on: [IDEA-003, IDEA-007]   # optional — ideas that must land first
blocks: []              # optional — ideas blocked by this one
supersedes: []          # optional — older ideas this replaces
stack_row: STACK_NOVELTY_TRACKER_v2.md#l09-online-growing-n-gram-cache-moonshot
prior_art_checked: 2026-04-16
---
```

### EXPERIMENT frontmatter

```yaml
---
id: EXP-2026-04-16-001
slug: online-ngram-cache-poc-seed42
idea: IDEA-042
pod: paramgolf-h100   # from POD_HOSTS.env
gpu: 1xH100SXM
seed: 42
git_sha: d8735a0
config:               # overrides vs the default submission/run.sh
  CACHE_SIZE_MB: 4
  CACHE_ORDER: 7
  TTT_ENABLED: 0
started: 2026-04-16T14:00Z
finished: 2026-04-16T14:27Z
status: complete      # pending | running | complete | failed | crashed | aborted
---
```

### FINDING frontmatter

```yaml
---
slug: online-ngram-cache-poc
date: 2026-04-16
title: "Online n-gram cache: 16 MB LM + causal cache hits 1.065 BPB (POC, non-record track)"
ideas: [IDEA-042]
experiments: [EXP-2026-04-16-001, EXP-2026-04-16-002, EXP-2026-04-16-003]
headline_metric: val_bpb = 1.065 (3-seed mean, σ=0.004)
updates:
  - STACK_NOVELTY_TRACKER_v2.md#l09
  - WIN_PLAN.md
novelty_status_after: validated  # matches updated STACK_NOVELTY_TRACKER_v2 status
---
```

## 4. The README indices (auto-summary layer)

Each of `docs/ideas/`, `docs/experiments/`, `docs/findings/` has a `README.md` that is a **small table indexing all files in the directory**, built from their frontmatter. Constraints:

- **Short**: ≤200 lines. Shrink older entries to "archive" section if the active list grows past 150.
- **Always sorted by** newest first (experiments, findings) or by status + expected-BPB (ideas).
- **Regenerated, not curated**: whenever a file is added or frontmatter changes, rebuild the table. The LLM can do this with a simple shell one-liner (see §7 below).

### `docs/ideas/README.md` columns

| ID | Slug | Layer | Status | Expected BPB | Next step |
|---|---|---|---|---|---|

### `docs/experiments/README.md` columns

| ID | Date | Idea | Config | Status | val_bpb |
|---|---|---|---|---|---|

### `docs/findings/README.md` columns

| Date | Title | Ideas | Headline metric | Updates |
|---|---|---|---|---|

## 5. Workflow — the typical session

**Scenario 1: user says "run the novelty-mining protocol"**

1. LLM reads canonical state (MOONSHOT_RULES, STACK_NOVELTY_TRACKER_v2, STACK_UTILISATION_RESULTS, FLOOR_RESULTS)
2. Executes `RESEARCH_PROTOCOL.md` steps 1–5
3. For each surviving candidate → creates `docs/ideas/IDEA_<next_id>_<slug>.md` from template
4. Updates `docs/ideas/README.md` index
5. Appends summary to `RESEARCH_LOG.md`
6. Returns to user: "N new ideas in `docs/ideas/`. Top 3: IDEA-045, IDEA-047, IDEA-048."

**Scenario 2: user says "run experiment for IDEA-042"**

1. LLM reads `docs/ideas/IDEA_042_*.md`
2. Creates `docs/experiments/EXP_YYYY-MM-DD_nnn_<slug>.md` from template with status=pending
3. Executes command on H100 pod (per `EXPERIMENT_PROTOCOL.md` — separate doc for pod mechanics)
4. Live-updates EXP file with status=running, then status=complete + results table
5. Appends entry to `RESEARCH_LOG.md`
6. If the result warrants a FINDING → creates `docs/findings/FINDING_...md`
7. Updates `STACK_NOVELTY_TRACKER_v2.md` row status; idea's status → validated | killed

**Scenario 3: user says "what's the status of our research?"**

1. LLM reads `docs/ideas/README.md`, `docs/experiments/README.md`, `docs/findings/README.md` (all ≤200 lines each)
2. Reads the tail of `RESEARCH_LOG.md` (last ~20 entries)
3. Returns a summary without needing to open individual idea/experiment files

## 6. Rules for the LLM executor

When I'm producing or maintaining docs in this system:

1. **Templates are the law**. Every new idea/experiment/finding starts by *copying* the TEMPLATE, not by writing from scratch. If a template field doesn't apply, explicitly write "N/A — reason".
2. **Frontmatter is required**. Without YAML frontmatter the README index can't find it. Docs missing frontmatter are broken.
3. **IDs are monotonic**. Check `docs/ideas/README.md` for the last IDEA-###, increment. Same for experiments.
4. **One concern per doc**. Don't write a "ideas dump" — write N separate IDEA files. Don't batch experiments into one file.
5. **Edit in place for status changes**. Updating an IDEA from draft→approved is an edit, not a new file. Only create new docs for new ideas/experiments/findings.
6. **Findings are earned, not default**. Most experiments don't warrant a finding. Only promote to FINDING when it changes the plan or produces a defensible new result.
7. **RESEARCH_LOG.md is append-only**. Never edit past entries. If you need to correct something, write a new entry that references the old one.
8. **Paper-style for FINDINGs** (per `feedback_paper_style_docs.md`). TL;DR + Background + Method + Results + Discussion + Simple example + Limitations.
9. **Terse-style for IDEAs and EXPERIMENTs**. These are working docs, not papers. Short sentences, tables, pseudocode. No narrative prose unless the method demands it.
10. **Cross-references are required**. A finding with no experiment back-link is broken. An experiment with no idea back-link is broken.
11. **Update the index after every doc change**. Regenerate `README.md` from frontmatter. (Or: batch at end of a long session, but never ship a session with stale indices.)
12. **When unsure, read the template first**. The template embeds the rules.

## 7. Index-rebuild one-liner

Each READMEs is auto-generated. Standard command (Python, pyyaml):

```python
#!/usr/bin/env python3
# scripts/rebuild_doc_indices.py
import glob, yaml, re
from pathlib import Path

def parse_frontmatter(path):
    text = Path(path).read_text()
    m = re.match(r"^---\n(.*?)\n---\n", text, flags=re.S)
    if not m: return None
    return yaml.safe_load(m.group(1))

# rebuild docs/ideas/README.md
ideas = []
for p in sorted(glob.glob("docs/ideas/IDEA_*.md")):
    fm = parse_frontmatter(p)
    if fm: ideas.append((fm, p))
ideas.sort(key=lambda x: (x[0].get("status",""), -(sum(x[0].get("expected_bpb", [0,0]))/2)))

with open("docs/ideas/README.md", "w") as f:
    f.write("# docs/ideas/ — design-stage candidates\n\n")
    f.write("| ID | Slug | Layer | Status | Expected BPB | Next step |\n|---|---|---|---|---|---|\n")
    for fm, p in ideas:
        lo, hi = fm.get("expected_bpb", [0,0])
        f.write(f"| {fm['id']} | [{fm['slug']}]({Path(p).name}) | {fm.get('layer','')} | {fm.get('status','')} | [{lo}, {hi}] | {fm.get('next_step','')} |\n")

# ...same for experiments/ and findings/
```

Run `python3 scripts/rebuild_doc_indices.py` after any change to idea/experiment/finding frontmatter. (Create this script once; re-run per session.)

## 8. Anti-patterns (what NOT to do)

- **A single "ideas dump" file with 50 candidates**: defeats the system. Each idea must live in its own file so status tracking works.
- **Editing RESEARCH_LOG.md mid-history**: never. Append-only.
- **Skipping templates to save time**: the frontmatter is what makes the README work. Templates are cheap. Coding time isn't the constraint (per `feedback_coding_time_unbounded.md`).
- **Canonical state docs that grow unbounded**: STACK_NOVELTY_TRACKER_v2 and WIN_PLAN are meant to be updated in place, pruned, not accreted. If they grow past ~500 lines, refactor.
- **Findings without back-links**: every finding must cite the experiments and ideas that produced it. Otherwise it's a floating claim.
- **"I'll make a findings doc later"**: either the result deserves a finding now, or skip it. Backfilling findings a week later is unreliable.

## 9. How this supports LLM-driven research across sessions

The hard problem for multi-session LLM work is **state loss**. Each session starts fresh; memory + CLAUDE.md are ~15 KB of context; everything else is on disk.

This doc system solves it by making every action produce a file, and every file have machine-parseable metadata. A fresh session sees:

- **What's in flight**: `docs/ideas/README.md` (status=in-experiment), `docs/experiments/README.md` (status=running)
- **What we just learned**: `docs/findings/README.md` (latest entries)
- **What's canonical now**: `STACK_NOVELTY_TRACKER_v2.md`, `WIN_PLAN.md`
- **What happened last**: tail of `RESEARCH_LOG.md`

Combined, a new session can re-orient in under 5 minutes and resume in-flight work without re-deriving context. Without this system, every session repeats the "what were we doing?" phase.

## 10. Related

- `RESEARCH_PROTOCOL.md` — how to generate ideas
- `MOONSHOT_RULES.md` — what's legal
- `STACK_NOVELTY_TRACKER_v2.md` — canonical catalog
- `docs/ideas/TEMPLATE.md`, `docs/experiments/TEMPLATE.md`, `docs/findings/TEMPLATE.md` — the doc templates
- `feedback_paper_style_docs.md` (memory) — FINDING paper-format style
- `feedback_coding_time_unbounded.md` (memory) — LOC/time not a constraint; write the full doc

## 11. Changelog

- **2026-04-16**: initial design. Spawned from `MOONSHOT_RULES.md §10` split + user's request for a full doc architecture for LLM-driven research.
