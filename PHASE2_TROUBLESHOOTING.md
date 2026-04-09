# PHASE2_TROUBLESHOOTING.md — append-only log

**Comp**: openai/parameter-golf
**Phase**: 2 (speed work on the locked-in Phase 1 model)
**Hardware**: cheap 3090/4070 Ti pods (NO H100 until final submission run)
**Plan reference**: PHASE2_PLAN.md
**Model invariant**: submission/train.py is LOCKED from Phase 1 (10 patches + PR #1477 base). Phase 2 only changes *how* the math runs, not *what* math runs.

This file is append-only. Each entry: timestamp, what broke, what we did, why, and
whether the fix is "permanent" (in the repo) or "ad-hoc" (lives only on a specific pod).

## Operating rules (inherited from Phase 1)

1. **Clean python files only** — all changes land in `submission/train.py`, `submission/run.sh`, or new files under `submission/kernels/`. No patcher hunks.
2. **Every workaround must be repo-checked-in** — if you SSH and `rm`/`mv` files, that's an ad-hoc fix and you must follow up with a permanent fix in the repo so the next clean pod boot can reproduce. Mark each entry below as PERMANENT or AD-HOC.
3. **Document the WHY** — not just what command, but what error/symptom led to it.
4. **Never bypass safety** — never `--no-verify`, never `git push --force`.
5. **val_bpb invariant** — every Phase 2 change must keep val_bpb within ε=0.005 of the Phase 1 baseline. Log any drift in this file.

## Phase 1 baseline to preserve (the floor)

- train.py: 731 lines, git HEAD at 3dfc868
- 10 patches all active in run.sh defaults
- Phase 1 dry run val_bpb: **TBD** (waiting on Pod L `55fzwdfhbg9n4u` dry run to land ~2026-04-09 03:30Z)
- Reference speed: 180 steps in 600s wallclock on 1× H100 SXM (eager mode, SDPA fallback, no compile)

---

<!-- Phase 2 entries get appended below this line. -->
