# PHASE2_AUTOMATION.md — Cron loop plan for Phase 2 shot builder + pod monitor

**Status**: planned, not started. Spin up when explicitly authorized.

**Purpose**: automate the 7 remaining Phase 2 shots via a recurring cron that picks work off a priority list, codes one shot per fire, commits + pushes, and marks state in a tracking file. A second cron monitors the in-flight Pod L (H100 `55fzwdfhbg9n4u`) passively so we catch the Phase 1 dry run result the moment it lands.

---

## Cron A — Shot Builder

**Schedule**: `4,17,30,43,56 * * * *` (every ~13 min, minute offsets that avoid the :00/:30 API hotspots)

**Max fires**: ~15 (3 hours) — stop condition is "all shots built" or user interrupt, whichever first.

**Working directory**: `/Users/takodamundy/Documents/personal_repos/paramgolf`

**Prompt** (self-contained — future-Claude at the cron fire reads this and acts):

```
Phase 2 Shot Builder fire. CWD: /Users/takodamundy/Documents/personal_repos/paramgolf

You are ONE fire of a recurring cron building Phase 2 speed shots. Be fast, be safe,
never break anything. Each fire has a HARD time budget of 10 minutes; over that,
commit WIP and exit.

## Setup (always do first)
1. Read PHASE2_RESEARCH.md and PHASE2_PLAN.md for shot descriptions
2. Read phase2/README.md to see what's already shipped
3. Read PHASE2_AUTOMATION_STATE.md (create if missing) to see shot statuses
4. Read CLAUDE.md for hard rules

## Shot priority list (work in this order)
0. **Shot 0e — NGR_LOG_FREQ_INV serialization fix (URGENT CORRECTNESS BUG)**
   - Discovered from Pod L Phase 1 dry run: post-prequant-ttt val_bpb=1.241 but quantized=3.862 (-2.62 BPB gap!)
   - Root cause: n-gram tables mutated by NGR_LOG_FREQ_INV but buffers are persistent=False, so mutated state is lost on serialize → deserialize → eval runs with unmutated tables → massive mismatch
   - Fix: Option A from PHASE2_PLAN.md "Shot 0e" section — save the 3 multiplier tensors (shape [16384] fp32 each = 200 KB total) as persistent buffers, re-apply on deserialize via new _nlfi_reapply_if_needed() helper
   - ~30 LOC in submission/train.py (GPT.__init__ + new method + forward_logits + deserialize path)
   - BLOCKS all downstream Phase 2 speed shots (can't measure ε drift against broken baseline)
   - Do as a DIAGNOSTIC first: flip USE_NGR_LOG_FREQ_INV=0 + USE_CMP_QUANT_VALUE_DEDUP=0 in run.sh, run once on cheap pod, measure clean quant gap. Then apply the real fix.
1. Shot 17 — Fuzzy LR bandit per microbatch (smallest, ~4h human = 1 fire)
   - Add env var FUZZY_LR_BANDIT_ENABLED=1
   - In step_fn, sample lr from {0.5x, 1x, 2x} * base_lr with online Thompson sampling
   - Track reward (delta loss) per arm
   - ~80 LOC in submission/train.py
2. Shot 0b — Batched + streaming KV sliding eval
   - New function eval_val_sliding_streaming() using persistent KV cache
   - Each window appends 64 new tokens, scores the tail, evicts oldest 64
   - ~250 LOC, probably 2 fires
3. Shot 10 — Parameter Banking + Parallel Muon (PR #399)
   - Restructure Muon matrix params into 4 contiguous 3D banks
   - NS becomes one torch.bmm call
   - ~200 LOC, probably 2 fires
4. Shot 2 — FA3 sourcing
   - phase2/setup_phase2.sh: try pip install flash-attn-3 (fails, we know)
   - Then try pip install flash-attn (FA2)
   - If that works, train.py try/except already handles it via fallback
   - Document in commit what worked
5. Shot 9 — FA3 varlen + window attention + mixed seq_len (PR #1212)
   - Only if Shot 2 landed FA3 cleanly
   - Modify CausalSelfAttention forward to call flash_attn_varlen_func
   - Add window_size=512 on alternating layers
   - ~150 LOC, depends on Shot 2
6. Shot 19 — GPU-resident successive halving
   - Inside the 600s budget, run 4 model replicas × 100 steps with different LR/momentum
   - Pick the winner at step 100, continue training it
   - ~200 LOC, probably 2-3 fires
7. Shot 14 — Training megakernel (biggest, world-first)
   - Persistent SM scheduler for fwd+bwd+optim in one kernel launch
   - ThunderKittens templates or custom Triton
   - 500-1500 LOC, multi-fire sprint

## Per-fire procedure

1. Pick the FIRST shot in the priority list whose status in PHASE2_AUTOMATION_STATE.md
   is "pending" or "in_progress". If all are "done" or "blocked", EXIT.
2. Mark the picked shot as "in_progress" in the state file with a timestamp.
3. Do the coding work for ~8 minutes MAX. Use Edit + Write tools, not Agent subagents
   (too slow for a single fire).
4. Syntax-check EVERYTHING you change:
   - `python3 -c "import ast; ast.parse(open('submission/train.py').read())"`
   - `bash -n submission/run.sh`
   - `bash -n phase2/run.sh` (if changed)
5. If syntax check fails:
   - Revert your changes via `git checkout -- <file>` on each changed file
   - Mark the shot as "needs_human" in the state file
   - Commit just the state file with a note
   - Exit
6. If syntax check passes:
   - Commit with a clear message tagged [phase2-auto]
   - Try `git push origin main`; if HTTP 408, retry once and continue
   - If the shot is DONE (finished the full spec), mark it "done"
   - If still WIP, mark it "in_progress" (next fire resumes)
7. Exit with a brief status report to the user (under 100 words): which shot,
   what got shipped this fire, whether the shot is done or WIP, next fire target.

## Safety rules (non-negotiable)

- NEVER run destructive git commands (push --force, reset --hard) unless the
  user explicitly requested
- NEVER skip hooks (--no-verify, --no-gpg-sign)
- NEVER kill a running GPU process on any pod
- NEVER spawn a nested cron
- NEVER touch submission/train.py without syntax-checking after
- NEVER commit broken code — revert if the check fails
- If you're unsure about a shot's correctness, mark it "needs_human" and skip
- Respect the 10-minute per-fire budget — overtime = WIP commit + exit

## Env var discipline

- All new code is gated behind a new env var (default OFF) unless it's
  strictly additive and zero-risk
- Defaults set in submission/run.sh (for Phase 1 compat) OR phase2/run.sh
  (for Phase 2 runs only)
- Document the env var in the commit message body

## Success criterion

- All 7 shots shipped as clean committable code in the repo
- Each shot gated behind an env var with a safe default
- PHASE2_AUTOMATION_STATE.md shows the final statuses
- phase2/README.md updated with the newly shipped files
- No broken commits, no reverts, no force-pushes
```

---

## Cron B — Pod L Monitor

**Schedule**: `9,30,51 * * * *` (every ~21 min, offset so it doesn't collide with Cron A's minute marks)

**Max fires**: ~9 (3 hours)

**Working directory**: `/Users/takodamundy/Documents/personal_repos/paramgolf`

**Prompt**:

```
Phase 2 Pod L Monitor fire. CWD: /Users/takodamundy/Documents/personal_repos/paramgolf

You are ONE fire of a recurring monitor cron. Read-only on pod state. Never kill
anything. Never touch code. Max 3 minutes per fire.

## Pod info
- ID: 55fzwdfhbg9n4u (paramgolf-dryrun-h100)
- SSH: POD_HOST="55fzwdfhbg9n4u-64411fec@ssh.runpod.io" /tmp/podrun.sh <<EOF ... EOF
- Running the Phase 1 dry run (submission/bootstrap.sh path)
- Expected to finish with a val_bpb landing around ~03:30 UTC ≈ 13:30 AEST

## Per-fire procedure

1. SSH to the pod, snapshot:
   - date -u
   - ps -ef | grep -E 'train_gpt|download_hf|bootstrap' | grep -v grep | head -5
   - nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
   - df -h /workspace /
   - ls /workspace/paramgolf/data/datasets/datasets/fineweb10B_sp8192/ | wc -l
   - tail -15 /tmp/paramgolf_bootstrap.log
   - grep -E 'val_bpb|stopping_early|peak memory' /tmp/paramgolf_bootstrap.log | tail -10

2. Parse the snapshot:
   - If a val_bpb line was found AND PHASE2_RESULTS.md's baseline row is still TBD,
     PATCH PHASE2_RESULTS.md to fill the baseline val_bpb + artifact bytes +
     ms/step from the log. Commit + push. This is the ONLY write operation
     the monitor cron is allowed to do.
   - If the bootstrap log shows an error (Traceback, ERROR, etc), append a
     1-line entry to PHASE2_TROUBLESHOOTING.md and flag the user.
   - Otherwise just append a 1-line progress note to PHASE2_TROUBLESHOOTING.md:
     "<utc> pod L monitor: <state>" where <state> is "tokenize alive / train
     alive / eval phase / done / idle / error".

3. Commit any PHASE2_RESULTS.md or PHASE2_TROUBLESHOOTING.md changes with a
   [phase2-monitor] tag. Push. Retry once on HTTP 408.

4. Report to user (under 80 words): pod state, what phase it's in, any
   interesting numbers, whether Cron A is doing useful work or blocked.

## Safety rules

- NEVER touch submission/ or phase2/ code files
- NEVER kill any process on the pod
- NEVER modify the pod's git state
- NEVER spawn nested crons or subagents
- Read-only on the pod; write-only on PHASE2_RESULTS.md + PHASE2_TROUBLESHOOTING.md
```

---

## State tracking

**File**: `PHASE2_AUTOMATION_STATE.md` (created lazily by the first Cron A fire)

**Format**:

```markdown
# PHASE2_AUTOMATION_STATE.md

| shot | status | last_touch_utc | fires | notes |
|---|---|---|---|---|
| 17_fuzzy_lr_bandit | pending | — | 0 | |
| 0b_streaming_kv_eval | pending | — | 0 | |
| 10_parameter_banking | pending | — | 0 | |
| 2_fa3_sourcing | pending | — | 0 | |
| 9_fa3_varlen_window | pending | — | 0 | blocked_by: 2 |
| 19_gpu_successive_halving | pending | — | 0 | |
| 14_training_megakernel | pending | — | 0 | |
```

Statuses: `pending` / `in_progress` / `done` / `needs_human` / `blocked`

---

## How to start the loop

**Not yet.** When authorized:

```python
# Cron A: shot builder
CronCreate(cron="4,17,30,43,56 * * * *",
           prompt=<text of the Cron A prompt above>,
           durable=True)

# Cron B: pod monitor
CronCreate(cron="9,30,51 * * * *",
           prompt=<text of the Cron B prompt above>,
           durable=True)
```

**How to stop the loop** (emergency or when done):

```python
CronList()           # find the IDs
CronDelete(id=<A>)   # kill shot builder
CronDelete(id=<B>)   # kill monitor
```

## Safety net

- **Session dependency**: cron fires only work if THIS Claude session stays alive. `/exit` kills both crons. The user is the single thread of control.
- **Budget**: each Cron A fire burns ~$0 (local dev) except when it needs to ssh the pod to test compile. Rare.
- **Git flood**: ~15 commits total expected (1 per successful shot + 1 per WIP commit). All on main.
- **Review window**: user can interrupt any fire by typing something — cron waits for REPL idle, so a new user message preempts it.

## What the human sees after 3 hours

- A `phase2/` directory with code for all 7 shots
- A `submission/` directory with any in-place env-var-gated edits needed by shots
- `PHASE2_AUTOMATION_STATE.md` with all shots marked `done` or `needs_human`
- `PHASE2_TROUBLESHOOTING.md` with the per-fire progress log
- `PHASE2_RESULTS.md` with the Phase 1 dry run baseline filled in (if Pod L finished)
- A clean git log of ~15-20 `[phase2-auto]` commits, all pushed to origin
- An explicit list of any shots that were marked `needs_human` with the reason
