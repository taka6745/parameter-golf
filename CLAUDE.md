# CLAUDE.md — Project memory + autonomous-execution playbook

This file is auto-loaded by Claude Code in this repo. Read it FIRST when you wake up to a cron fire.

## Quick orientation

**Project**: openai/parameter-golf — 16MB byte-level LM, 10 min training on 8xH100, BPB on FineWeb val. Currently SOTA ≈ 1.07 BPB; we're a small team chasing it.

**Compute we have right now**:
- RTX 3080 Ti pod at `tyf0q5l1kgefgx-64410a6f@ssh.runpod.io` (12GB)
- Mac (this machine) — CPU + Apple Silicon GPU via MLX
- Up to 8 cheap GPUs OK to spin up via runpodctl
- 1xH100 spot OK for ONE eval pass (sparingly!)
- $36 RunPod budget — **track in `RESEARCH_LOG.md` spend section**

**Current status (session start: 2026-04-07 21:30 AEST → 2026-04-08 09:00 AEST, 12h autonomous run)**:
- Loop alive on pod, 50+ runs done, current best single-run train_loss = 3.2734 (5-seed mean: 3.358 ± 0.066)
- 16 patches in `runpod_tests/chore/08_patch_train_gpt.sh` (markers: PATCHED_FOR_TORCH24, PROG_SEQ_INIT, PHASE_TRANSITION, PHASE_TRANSITION_CLAMP, SKIP_FINAL_EVAL, SKIP_LAST_VAL, SKIP_POST_LOOP, NGRAM_BIAS, NGRAM_GATE, SMEAR_GATE, LEAKY_RELU, BYTE_WEIGHT, WAVELET_GPT, PARALLEL_RESIDUALS, ENTROPY_ADAPTIVE_NGRAM, TABULATION_HASH, GATED_ATTENTION)
- 10 crons covering 12 hours (4 monitor / 4 research / 2 audit / 1 final-wrap)

## Pod SSH — the ONLY invocation that works

RunPod's community-pod proxy needs `-tt` but rejects auth when the macOS SSH agent is consulted. **Every pod SSH call must unset the agent and use these flags** — any deviation gets either `Permission denied (publickey)` or `Your SSH client doesn't support PTY`. Full root-cause write-up: `runpod_tests/loop/SSH_TROUBLESHOOTING.md`.

```bash
source POD_HOSTS.env  # pod endpoints live here (gitignored)
unset SSH_AUTH_SOCK
ssh -tt -F /dev/null -i "$POD_DIAG_KEY" \
    -o IdentitiesOnly=yes -o IdentityAgent=none \
    -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=20 \
    "$POD_DIAG_SSH" <<'REMOTE'
# commands on the pod here
REMOTE
```

All pod commands **must** use `run_in_background: true` in the Bash tool (pod latency is unpredictable; foreground blocks the conversation). Reference: memory `feedback_pod_commands_background.md`.

## Token usage (suggestion)

Long loop sessions, >150k context, and heavy subagent fan-out compound cost fast. To save tokens: `/compact` after a discrete unit, `/clear` when switching tasks, skip subagents for ≤3-call work, prefer Explore over general-purpose, route simple lookups to Haiku, and batch independent subagent calls in one message.

## ESSENTIAL READING ORDER (every cron fire)

1. `docs/plans/OVERNIGHT_PLAN.md` — playbook + decision tree (THE CRITICAL FILE)
2. `docs/research/RESEARCH_LOG.md` — what other fires found, current spend, what's already pushed
3. `docs/results/SESSION3_AUDIT.md` — last comprehensive audit
4. `runpod_tests/loop/results.jsonl` — current results (pull via /tmp/podpull.sh)

If `docs/plans/OVERNIGHT_PLAN.md` doesn't exist or is empty, you're in trouble — the autonomous infrastructure isn't set up yet. Read `docs/results/SESSION3_AUDIT.md` instead and ask the user what's going on.

See `docs/README.md` for the full doc index (plans, research, results, setup, minipapers).

## Tools you have

- **Pod control via SSH heredoc** (RunPod's SSH proxy blocks scp/sftp):
  - `/tmp/podrun.sh` — pipe stdin commands into the pod's interactive shell
  - `/tmp/podpull.sh` — tar+base64 pull a directory from the pod
  - `/tmp/podpush.sh` — base64+tar push a file/dir to the pod
  - `/tmp/podstatus.sh` — quick status snapshot
  - `/tmp/run_analyze.sh` — runs `analyze.py` on the pod
  - `/tmp/check_files.sh` — pod proc + GPU + log tail
  - `/tmp/restart_with_patches_15_16.sh` — kill + repatch + restart loop
- **runpodctl** — installed (use `runpodctl --help`); for spinning up additional GPUs
- **gh CLI** — for openai/parameter-golf PR mining
- **WebFetch + WebSearch** — via Explore subagents (don't pollute main context)
- **Bash** — anything

## Hard rules (from the user)

1. **NO HYPERTUNING** — don't push experiments that just twiddle weights of validated configs. Multi-seed validation is OK; weight sweeps of the same family are NOT.
2. **NOVEL OR PORTING-WITH-EVIDENCE** — every patch must either be: (a) truly novel (not in any open or merged competition PR + grounded in our Mac MLX research or a recent paper), or (b) ported from a comp PR that's in the top 10 records. No speculation.
3. **ATTACK ENTIRE STACK** — every research fire should rotate between: training, eval, compression, tokenizer, hardware, English engine. Don't focus on one slice.
4. **DOCUMENT** — any novel patch deserves a mini-paper MD in `docs/minipapers/MINIPAPER_<patch_name>.md`. Format: hypothesis / implementation / results / comparison to baseline / comparison to comp.
5. **MODIFY INDUSTRY CODE** — explicitly OK to fork sentencepiece, write custom CUDA kernels, etc. if it unblocks a real win. Just check it in.
6. **AUDIT CONSTANTLY** — every 2 hours, re-audit our patches against the latest open + closed + merged PRs. Mark "no longer novel" anything that appeared in a competitor submission since you shipped it.
7. **TRACK SPEND** — append spend estimates to docs/research/RESEARCH_LOG.md. Hard cap: $36. Soft cap: $25 (slow down at this point).
8. **USE THE MAC** — Mac CPU + Apple Silicon GPU via MLX is FREE. Spawn subagents to use it. Specifically: build BPE-8192 tokenizer, run web research, profile code.
9. **PHD-DEFENSIBLE** — every patch must have a clear hypothesis + falsification criterion + at least N=2 seed validation before claiming it as a win.

## Decision tree shortcuts

| situation | action |
|---|---|
| Loop dead | restart via /tmp/restart_with_patches_15_16.sh |
| Same experiment crashed >3 times | remove from experiments.json on git, push, runner auto-pulls |
| New top-1 single-run | queue 2 more seeds in experiments.json, push |
| New top-1 multi-seed | append to docs/research/RESEARCH_LOG.md as "validated win", consider H100 escalation |
| Found novel technique high-confidence | patch + push directly, kill+restart loop |
| Found novel technique low-confidence | append to docs/research/RESEARCH_LOG.md "next fire" section, don't push |
| Loop has been idle (no runs) >15 min | something is stuck — investigate via /tmp/check_files.sh |
| Spend > $25 worth | slow research fire interval, no H100 |
| 9am AEST (= 23 UTC) | trigger 9am wrap one-shot OR do it manually |

## H100 escalation rules (sparingly!)

Only escalate to a 1xH100 spot if ALL of these:
1. CHAMPION config validated across 3+ seeds, mean train_loss > 0.05 below baseline
2. Cheap-GPU best train_loss < 3.20 (so we're confident H100 val_bpb < 1.10)
3. After 04:00 UTC (so there's time for results before 9am wrap)

If yes:
```bash
runpodctl create pod --gpu "NVIDIA H100 80GB HBM3" \
  --image "runpod/pytorch:2.4.0-cuda12.1.0-cudnn8-devel-ubuntu22.04" \
  --containerDiskInGb 50 --volumeInGb 50
```
Get the SSH info, push the champion config, run ONE eval pass with `SKIP_FINAL_EVAL=0`, capture `final_int8_zlib_roundtrip val_bpb`, **KILL THE H100 IMMEDIATELY** via `runpodctl remove pod <id>`. Cost: ~$3-5 for one run. Don't waste time once the number lands.

## What's already been tried + what's been validated

(this section gets refreshed by audit fires; check docs/research/RESEARCH_LOG.md for the current truth)

- N-gram bias stack (bigram + trigram + fourgram, 16K hash buckets) — validated, our biggest single win
- LeakyReLU(0.5)² + n-gram — marginal +0.005 win
- Wavelet GPT — NEUTRAL/HURT on cheap GPU
- Byte-weighted loss — metric not comparable, rebuild reporting before judging
- NGRAM_GATE (learned per-position gate) — failed, model couldn't learn it in 1500 steps
- Bigger model (12L 3x MLP) — fewer steps offsets capacity gain; loses

## Things on the user's wish list NOT YET ADDRESSED

(check this every audit fire, work on the topmost item)

- Custom CUDA kernels (modify industry code)
- BPE-8192 tokenizer + n-gram tables (subagent in flight)
- Mini-papers per novel patch
- SETUP.md full stack architecture
- Mac MLX parallel experiment loop
- Re-enabling torch.compile for 25-35% throughput
- Re-testing depth recurrence with mixed-precision quant
- TTT (test-time training)
- Score-First eval n-gram cache
