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

## 2026-04-09 ~03:00Z — BUG DISCOVERED: NGR_LOG_FREQ_INV serialization mismatch

Pod L (`55fzwdfhbg9n4u`) Phase 1 dry run produced these numbers:

```
post-prequant-ttt  val_bpb: 1.24108342   ← unquantized, post-PreQ-TTT — GREAT
quantized          val_bpb: 3.86173549   ← after GPTQ int6 + brotli — BROKEN
sliding_window     val_bpb: 3.86165833   ← same, sliding eval — BROKEN
```

A **-2.62 BPB gap** from quantization. Normal int6 GPTQ gap is 0.005-0.02 BPB.

### Root cause

`GPT.__init__` registers `_bigram_tab`, `_trigram_tab`, `_fourgram_tab` as
non-persistent buffers (`persistent=False`) so they don't bloat the submission
state_dict. But `NGR_LOG_FREQ_INV` (world-novel L09 patch) **mutates them in
place on first forward**:

```python
if self._nlfi_enabled and not self._nlfi_done:
    _bg_mult = 1.0 / torch.log(2.0 + _bg_counts)
    self._bigram_tab.mul_(_bg_mult.to(self._bigram_tab.dtype).unsqueeze(1))
    # same for trigram, fourgram
    self._nlfi_done = True
```

Flow break:
1. Training: tables are mutated on first forward. Model learns to expect muted high-freq buckets.
2. `serialize()`: saves `base_model.state_dict()`, n-gram tables NOT included (persistent=False).
3. `deserialize()`: creates fresh `GPT(h)` → `__init__` reloads tables from disk → UNMUTATED.
4. Quantized eval: runs with unmutated tables, model expects mutated ones → logit bias way off → val_bpb catastrophically bad.

### Fix plan (Shot 0e in PHASE2_PLAN.md)

**Option A** (recommended, keeps world-novel claim):
- Store the NLFI multipliers as `persistent=True` buffers:
  `self._nlfi_bigram_mult`, `_nlfi_trigram_mult`, `_nlfi_fourgram_mult` shape `[16384]` fp32 each = 64 KB × 3 = 192 KB total
- On deserialize, `__init__` reloads fresh tables, state_dict load restores the multipliers, new helper `_nlfi_reapply_if_needed()` re-multiplies the tables
- Submission size impact: +192 KB. Still under 16 MB cap easily.

**Option B** (fallback, loses world-novel claim):
- Set `USE_NGR_LOG_FREQ_INV=0` in `submission/run.sh`
- Quant gap recovers but we lose the world-novel L09 #2 claim

**Option C** (diagnostic FIRST):
- Flip both `USE_NGR_LOG_FREQ_INV=0` AND `USE_CMP_QUANT_VALUE_DEDUP=0` on a cheap pod
- Re-run, measure the quant gap → confirm it drops to ~0.05 BPB
- Then apply Option A for the real fix

### Blocks

All downstream Phase 2 speed work. The val_bpb invariant (ε ≤ 0.005 drift vs baseline) can't be measured against a 3.86 baseline. Need clean baseline first.

### Spend impact

Pod L Phase 1 dry run cost ~$7.50 so far. Still burning ~$2.99/h. Sliding eval is at chunk 51/1238 (~4% done), ~35 more min = ~$1.75 more if we let it finish. **Kill recommendation**: we already have the quantized number; letting sliding eval finish gives us one more number of the same broken model, not useful.
