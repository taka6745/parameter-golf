# OVERNIGHT_PLAN.md — Autonomous execution playbook for cron-fired Claude sessions

**Window**: 2026-04-07 21:30 → 2026-04-08 09:00 AEST (12 hours)
**Budget**: $36 RunPod credits (track in `RESEARCH_LOG.md` spend section)
**Pod**: tyf0q5l1kgefgx-64410a6f@ssh.runpod.io (RTX 3080 Ti 12GB)
**Goal**: ship novel, PhD-level techniques across the entire stack to win the openai/parameter-golf 16MB byte-level LM challenge.

## 🚨 PRIORITY OVERRIDE (2026-04-08 20:25 UTC)

User identified that the loop has been running with a MICROSCOPIC batch
(`TRAIN_SEQ_LEN=128`, `TRAIN_BATCH_TOKENS=1024` = 8 sequences × 128 tokens
per batch). GPU at 6% memory, 34% util. Model has trained on **0.75% of
intended data**. The "neutrality plateau" was a measurement artifact —
all techniques tested with 1.5M tokens instead of the intended ~200M.

**MANDATORY for every cron fire from now until end of run**:
1. Verify `TRAIN_SEQ_LEN >= 512` and `TRAIN_BATCH_TOKENS >= 32768` in
   `runpod_tests/loop/experiment_runner.py`. If reverted, fix and push.
2. Verify GPU util > 80% via `nvidia-smi` (check `/tmp/podstatus.sh`).
   If <80%, BUMP batch tokens by 2× and push.
3. **Speed before novelty**: every research fire must pick a technique
   that improves THROUGHPUT or UTILIZATION over training time. Stop
   shipping marginal training-side ports until throughput is fixed.
4. Target: <300ms/step on RTX 3080 Ti at >80% GPU util, >50M tokens
   processed per experiment. H100 target is <80ms/step (proportional).
5. **NO MORE OPTIMIZER PORTS** (Mousse/MuonEq-R/NorMuon style) until
   speed is fixed and validated. They've all been measured at the wrong
   scale.

## NEW CRON BEHAVIOR (effective 2026-04-08 20:25 UTC)

The previous "find ONE novel technique" was producing marginal ports.
The new mandate is **breakthrough on speed and utilization**. Specifically:

- **Monitor fires**: ALSO check `nvidia-smi` for GPU util. Flag if <80%.
- **Research fires**: prioritize hardware-side, data-loading, or
  throughput improvements. Skip the comp PR mining grind.
- **Audit fires**: enforce the >32768 batch tokens rule. Auto-fix if
  reverted.

## CRITICAL FILES TO READ FIRST (every cron fire)

If you're a cron-fired session waking up, read these IN THIS ORDER:

1. **`OVERNIGHT_PLAN.md`** (this file) — playbook + decision tree
2. **`RESEARCH_LOG.md`** — what other fires have found, current spend, what's already been pushed
3. **`SESSION3_AUDIT.md`** — last comprehensive audit of patches + leaderboard
4. **`runpod_tests/loop/results.jsonl`** (via `/tmp/podpull.sh`) — current experiment results
5. **Look at the timestamp of the last entry** in RESEARCH_LOG.md. Don't repeat what's already done.

## CRON ROUTING (which fire are you?)

Look at the current minute of the cron expression in your prompt. Different fires have different jobs:

| minute | job | duration |
|---|---|---|
| 2,17,32,47 | MONITOR | <90s |
| 8,38 | RESEARCH | 5-15 min, may spawn subagents |
| 23 | AUDIT | 3-5 min |
| top of hour 9 (Apr 8) | FINAL WRAP | 15-30 min |

## MONITOR FIRES (every 15 min)

**Job**: keep the loop alive, track diff, no patch pushes.

```bash
# 1. Pull current state
/tmp/podrun.sh < /tmp/run_analyze.sh 2>&1 | LANG=C tr -d '\r' | LANG=C perl -pe 's/\e\[[0-9;?]*[a-zA-Z]//g; s/\e\][^\a]*\a//g'

# 2. Check loop alive
/tmp/podrun.sh < /tmp/check_files.sh 2>&1 | grep -E '^PROC|train_gpt|GPU'
```

**If runner is dead**: restart via `/tmp/podrun.sh < /tmp/restart_with_patches_15_16.sh`

**If train_gpt has been running >7 min on the SAME log file**: it's stuck. Kill it: pkill -9 -f train_gpt. The runner will record a crash + move on.

**If a SAME experiment has crashed >3 times in a row** (check `runpod_tests/loop/results.jsonl`): the experiment is broken. Remove it from `runpod_tests/loop/experiments.json` on git, push, runner auto-pulls within 5 min.

**Output**: 1-paragraph diff. Don't dump full leaderboards.

## RESEARCH FIRES (every 30 min)

**Job**: find ONE truly novel technique, implement if confident, log if not.

**Rotate tracks** by minute:
- **Minute 8** → arxiv search via Explore subagent
- **Minute 38** → openai/parameter-golf PR audit (open + closed)

### Track A — arxiv (minute 8)

Spawn an Explore subagent with:
- WebSearch one of: "byte level language model 2025 sub 1 BPB", "transformer test time training NeurIPS 2025", "differential transformer attention 2025", "small language model 16MB compression 2025", "tabulation hashing language model", "ByteSpan tokenization", "EvaByte byte level"
- Pick a search NOT yet done in RESEARCH_LOG.md (check the file before launching)
- Subagent returns: ONE actionable technique with paper URL, hypothesis, implementation difficulty (easy/medium/hard), expected effect, why it's not in any open PR

### Track B — comp PRs (minute 38)

Spawn an Explore subagent with:
- `gh api 'repos/openai/parameter-golf/pulls?state=open&sort=created&direction=desc&per_page=20'`
- Compare against the patches in `runpod_tests/chore/08_patch_train_gpt.sh` markers
- Subagent returns: any technique in the latest 5 PRs that we DON'T have

### Action rules

- **HIGH-confidence (>70%)** novel + actionable: write the patch directly into `08_patch_train_gpt.sh` (idempotent, with marker), add 2-3 experiment configs to `experiments.json`, commit + push
- **MEDIUM-confidence (40-70%)**: append to `RESEARCH_LOG.md` "next fire" section
- **LOW-confidence**: write a 2-line note in `RESEARCH_LOG.md` and move on

**NOVELTY ENFORCEMENT**: any technique already in 3+ open PRs is NOT novel. Ship it ONLY if it's a confirmed missing leverage win (e.g. parallel residuals before we shipped Patch 13).

## AUDIT FIRES (every 60 min, minute 23)

**Job**: enforce constraints, prevent drift.

```bash
# 1. Pod alive
/tmp/podrun.sh < /tmp/check_files.sh 2>&1 | head -20

# 2. Spend estimate (rough)
# pod uptime since session start ≈ how many hours we're into the night
# 3080 Ti = $0.30/h
# alert if total > $25 worth (~83h equiv but we're only running 12h so we're fine)
```

### Constraint enforcement

- **Hypertuning detection**: scan `experiments.json` for any experiment that's just a weight tweak of an already-validated config. If found AND not labeled as multi-seed validation, REMOVE it from the queue.
- **Novelty drift**: any patch we shipped that now appears in a competitor PR → mark "no longer novel" in RESEARCH_LOG.md but DON'T remove (we still want to validate).
- **Stack coverage**: count patches by category. If >70% are training-side, the next research fire MUST target eval / compression / tokenizer / hardware.
- **Spend**: if total exceeds $25, slow research fire interval (skip every other one).
- **Self-audit vs original wishes** (every 2 hours): re-read the "Original wishes" section of this file, audit what's been done vs what's been promised. Append to RESEARCH_LOG.md.

## DECISION TREE — when something interesting happens

### A new top-1 train_loss appears

1. Check if it's a multi-seed result or single-seed
2. If single-seed: queue 2 more seeds in `experiments.json` to validate
3. If multi-seed and clearly better than previous (>0.02 train_loss): append to RESEARCH_LOG.md as "validated win"
4. If it's a NOVEL patch winning, write a mini-paper MD: `MINIPAPER_<patch_name>.md` with: hypothesis, implementation, results, comparison to baseline, comparison to comp

### A patch crashes the loop

1. The runner now skips experiments after 3 crashes (Patch from earlier).
2. If a NEW experiment crashes immediately, it's a patch bug. Look at the latest log file (`/tmp/podpull.sh` or via SSH heredoc).
3. Fix the patch in `08_patch_train_gpt.sh`, push, kill the runner so the wrapper restarts with the new patch.

### Found a high-EV technique

1. If easy to implement (<30 min): WRITE THE PATCH NOW. Push. Kill+restart loop.
2. If medium (30-90 min): write the patch, push, but don't restart loop (let the next experiment finish first to avoid wasted compute).
3. If hard (>90 min): document in RESEARCH_LOG.md, defer to a future fire or to user.

### H100 escalation conditions

ONLY escalate to a 1xH100 spot if ALL of these:
- We have a CHAMPION config validated across 3+ seeds with mean train_loss > 0.05 below baseline
- The current cheap-GPU best train_loss is < 3.20 (so we're confident the H100 number will be < 1.10 BPB)
- It's after 04:00 UTC (8h into the session — we have time for results before wrap-up)

If yes: spin up `runpodctl create pod --gpu H100 --image runpod/pytorch:2.4.0-cuda12.1.0-cudnn8-devel-ubuntu22.04` (or whatever the right command is — figure it out via `runpodctl --help`). Run ONE eval pass with SKIP_FINAL_EVAL=0 + the champion config. Capture `final_int8_zlib_roundtrip val_bpb`. KILL THE H100 IMMEDIATELY after the result lands.

## ORIGINAL WISHES (the user's spec — audit against this every 2 hours)

1. Win the comp / top 0.001% / novel to comp + world
2. PhD-level / bleeding-edge / **NO HYPERTUNING**
3. Attack the **entire stack**: tokenizer, model, n-gram, optimizer, eval, compression, hardware, caches, English engine, programming language
4. Use ALL CPU + ALL GPU + ALL RAM + Mac CPU + Mac GPU
5. Modify industry code (sentencepiece, flash-attn, triton, etc.) if needed
6. Write mini-papers for new techniques
7. SETUP.md with full stack architecture + optimality proofs
8. Nothing should be the same as when we got it
9. 12 hours, $36 budget
10. Spawn subagents often
11. Cross-domain
12. Audit constantly vs comp (open + closed + merged + unmerged)
13. Build off self
14. Track everything

**Self-audit cadence**: every research fire should append a 1-line audit to RESEARCH_LOG.md noting which wishes are being addressed.

## CURRENT STATE (as of 21:38 local, fire 0)

- Pod: running, ~$0.50 spent so far
- Loop: 28 experiments queued, 50+ runs done
- Best single-run train_loss: CHAMP_L5_seed1337 = 3.2734 (5-seed mean: 3.358 ± 0.066)
- Patches shipped tonight: 13 (parallel residuals), 14 (entropy adaptive), 15 (tabulation hash), 16 (gated attention)
- Subagents launched: 5 returned + 2 in flight (BPE-8192 trainer, closed PR audit)
- Cron count: 10 (4 monitor, 4 research, 2 audit, 1 final wrap)
- Critical untouched: SETUP.md, CLAUDE.md, LESSONS.md update, perf optimization, mini-papers

## PHILOSOPHY (read before pushing anything)

**The user is paying for compute and trusting an autonomous Claude session.**

- Don't push junk patches. Every patch should have a clear hypothesis, implementation cost, expected effect, and a falsification criterion.
- Don't run experiments that just twiddle weights of validated configs.
- DO write mini-papers when you ship a novel patch. The user explicitly authorized this.
- DO modify industry code (sentencepiece, flash-attn, custom CUDA kernels) if it unblocks a real win.
- DO cross domains — every research fire should rotate between training / eval / compression / tokenizer / hardware.
- DO audit yourself.
- DO leave a clean handoff at 9am — the user will wake up and read SESSION_OVERNIGHT_FINAL.md first.

## EMERGENCY STOP

If the loop is in a hot crash loop (>10 crashes in 5 min) AND the runner crash-counting fix is somehow broken, kill EVERYTHING manually:
```bash
/tmp/podrun.sh <<'EOF'
pkill -9 -f experiment_runner
pkill -9 -f train_gpt
pkill -9 -f run_forever
echo killed
exit
EOF
```
Then investigate via `/tmp/podstatus.sh`. Restart only when the bug is identified + fixed.
