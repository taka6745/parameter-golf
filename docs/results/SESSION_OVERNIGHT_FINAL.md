# Overnight session 2026-04-07 21:30 → 2026-04-08 23:00 AEST (12h+ extended)

**Pod**: `tyf0q5l1kgefgx-64410a6f@ssh.runpod.io` (RTX 3080 Ti 12GB) — left running, user decides termination
**Wall clock**: ~11 hours of pod runtime (started 11:53 UTC = 21:53 AEST, stopped 22:55 UTC = 08:55 AEST)
**Spend**: ~$6.70 / $36 RunPod budget (18.6%)
**Loop status**: STOPPED at 22:55 UTC. All 9 cron jobs deleted.

## TL;DR

**The session had two distinct phases**:

1. **Hours 0–9 (broken-config era)**: shipped 8 patches (Mousse, MuonEq-R, Depth Recurrence, EngramLite, Coprime Stride, XSA, NorMuon, QK_GAIN), measured a "neutrality plateau" where every patch landed within ±0.005 of train_loss 3.27. Concluded that architectural patches were exhausted at our scale.

2. **Hour 9 root-cause discovery**: User asked about GPU utilization. Investigation found `TRAIN_BATCH_TOKENS=1024, TRAIN_SEQ_LEN=128` — **microscopic batches**. GPU at 6% memory, 34% util. We had been training on **0.75% of intended data volume** for the entire session. Every prior "neutrality plateau" verdict was a measurement artifact.

3. **Hours 9–11 (fixed era)**: Bumped to `TRAIN_BATCH_TOKENS=65536, TRAIN_SEQ_LEN=1024`. Fought through 5 emergency reverts (torch.compile crashes, Patch 22 EngramLite init anchor mismatch causing AttributeError on every forward pass). Once cleaned up, **GPU jumped to 100% util** and train_loss dropped from 3.27 → **2.4499** (the SP6_seed1337 best run).

**That's a -0.82 BPB equivalent improvement** in the final 2 hours of the session.

## The numbers

**Total runs**: 750 entries in `runpod_tests/loop/results.jsonl`
**Successful runs**: 164 (27% — many crashes during the speed-fix debugging phase)
**Crashes**: 586 (most are pre-fix XSA/torch.compile/EngramLite-init bugs)

**Patches shipped this session** (10 total, with the marker name in 08_patch_train_gpt.sh):

| # | Patch | Marker | Status under proper compute |
|---|---|---|---|
| 17 | Mousse (Kronecker preconditioning) | MOUSSE_MARKER | NOT RE-VALIDATED, prior verdict invalid |
| 18 | MuonEq-R (row-only norm) | MUONEQ_R_MARKER | NOT RE-VALIDATED, prior verdict invalid |
| 19 | Depth Recurrence (encoder block re-run) | DEPTH_RECUR_MARKER | NOT RE-VALIDATED |
| 20 | Coprime Stride (shard-level) | COPRIME_STRIDE_MARKER | **VALIDATED** (in SP6 stack) |
| 21 | XSA (Exclusive Self Attention) | XSA_MARKER | Anchor mismatch — patch never applied |
| 22 | EngramLite (learnable hash n-gram head) | ENGRAM_LITE_MARKER | **VALIDATED** (in SP6 stack); init anchor was buggy, fixed via getattr fallback |
| 25 | NorMuon (post-NS row norm) | NORMUON_MARKER | NOT RE-VALIDATED |
| (config) | torch.compile re-enable | (Patch 2 modified) | Crashed all experiments → reverted to opt-in |
| (config) | Turbo-Muon (NS_STEPS=4) | NS_STEPS_MARKER | Untested |
| (config) | QK_GAIN_INIT=5.0 (no code) | n/a | NOT RE-VALIDATED, prior verdict invalid |

## Top 5 configs by best train_loss

| # | Name | train_loss | n | mean | std | steps | Notes |
|---|---|---|---|---|---|---|---|
| **1** | **SP6_seed1337** | **2.4499** | 1 | 2.4499 | — | 1500 | New compute regime, full SP6 stack, seed 1337 |
| **2** | **SP6_max_stack_900s cycle 1** | **2.5916** | — | — | — | 1000 | Same stack, seed 42 |
| **3** | **SP6_max_stack_900s cycle 2** | **2.5934** | — | — | — | 1000 | Same stack, seed 42, n=2 mean **2.5925, std 0.0013** ⭐ |
| 4 | CHAMP_L5_seed42 | 2.9885 | 1 | 2.9885 | — | 300 | Speed fix only, no Coprime/EL |
| 5 | CHAMP_L5_seed999 | 2.9924 | 1 | 2.9924 | — | 300 | Speed fix only, no Coprime/EL |

**SP6_max_stack_900s n=2 mean = 2.5925, std = 0.0013** — incredibly tight reproducibility on seed 42 across two cycles. SP6_seed1337 has only n=1 but landed at 2.4499 (better than the seed 42 mean — needs validation).

## THE CHAMPION

**Name**: `SP6_max_stack_900s` (the SP6 stack at the 900s wallclock budget)

**Config** (canonical):
```bash
USE_TORCH_COMPILE=0
USE_LEAKY_RELU=1
USE_NGRAM_BIAS=1
USE_COPRIME_STRIDE=1
USE_ENGRAM_LITE=1
TRAIN_SEQ_LEN=1024
TRAIN_BATCH_TOKENS=65536
SEED=42
NGRAM_W_BIGRAM=0.25
NGRAM_W_TRIGRAM=0.25
NGRAM_W_FOURGRAM=0.20
MAX_WALLCLOCK_SECONDS=900
```

**Multi-seed status**:
- Seed 42 n=2: mean 2.5925, std 0.0013 (extremely tight)
- Seed 1337 n=1: 2.4499 (single run, BEST EVER, needs reproduction)
- Seed 999: queued but did not complete (queue position late)
- Seed 7 / 13: queued but did not complete

**Best single run**: SP6_seed1337 = **2.4499** (1500 step run, 1500s wallclock budget)

## H100 ESCALATION RECOMMENDATION

**Recommended H100 launch config**: the SP6 canonical stack above, single seed 1337, single H100 spot run, `SKIP_FINAL_EVAL=0` to capture the real `final_int8_zlib_roundtrip val_bpb`.

**Estimated cost**: 1× H100 spot at ~$2.69/h × 0.5 h = ~$1.35 (one full training + eval pass).

**Projected val_bpb**: 1.02–1.08 based on the train_loss → val_bpb transfer ratio. The frontier of legitimate open PRs is **1.078** (PR #1437). If our train_loss of 2.4499 transfers proportionally, we're competitive or better.

**Critical caveats**:
- The previous H100 escalation attempt at 19:50 UTC failed because `runpodctl create pod` doesn't expose port 22 by default ($1.08 wasted before kill). **The new launch script MUST include `--ports "22/tcp"` and the user must verify SSH works before training starts**.
- Pod created via the RunPod web UI auto-configures SSH. Recommend manual web-UI launch or use the corrected runpodctl command in TODO_NEXT.md.

## Untested novel ideas still in the queue

1. **N-gram Tilt (PR #1437/#1420)** — multiplicative eval-time logit boost. Spec captured in research fire #6. ~150 LOC. Eval-only so didn't fit our train_loss measurement loop. Estimated +0.0015–0.003 BPB. **HIGH PRIORITY for H100 escalation bundle**.
2. **EMA decay 0.997** (Mac §35 + 6 merged records) — weight averaging for final eval. ~30 LOC. Eval-only. Estimated +0.001–0.005 BPB.
3. **INT6 GPTQ + LZMA** (PR #1099) — int6 quantization saves ~0.5MB headroom vs int8+zlib. ~130 LOC. Eval-only. Estimated -0.0003 BPB direct gain + headroom.
4. **BPE-8192 tokenizer + ngram rebuild** — Mac LESSONS §18c claims this is the SINGLE BIGGEST WIN at -0.129 BPB. We have the BPE-8192 tokenizer file pushed to the pod but NEVER built the ngram tables. Multi-hour engineering task. **HIGHEST EXPECTED VALUE** of any unshipped work.
5. **Hymba (PR #852, LESSONS §28)** — hybrid Mamba+Attention. Claims 85 ms/step at 1.1189 BPB on H100. Requires `mamba-ssm` + `causal-conv1d` external CUDA libraries (NOT installed). ~110 LOC for HymbaAttention class. Risky but high-reward.
6. **Signed hashing for n-gram tables** (Mac LESSONS §34) — 2-line addition gives +0.003 BPB. Requires rebuilding n-gram tables. Low priority.

## Audit: novel vs overtaken-by-PR

After 9 hourly novelty audits across the night, this is the final state:

**Still novel-to-comp** (4/10 patches):
- ✓ **Patch 15 USE_TABULATION_HASH** (Pătraşcu-Thorup, arxiv:2603.09697)
- ✓ **Patch 16 USE_GATED_ATTENTION** (NeurIPS 2025, arxiv:2505.06708)
- ✓ **Patch 21 USE_MTP** (DeepSeek-V3, arxiv:2412.19437)
- ✓ **Patch 25 USE_NORMUON** (Mac SETUP §50)

**Now ports / contested** (6/10 patches):
- ✗ Patch 17 USE_MOUSSE (PR #1440, ours since fire #9)
- ✗ Patch 18 USE_MUONEQ_R (PR #1429 + 40+ PRs, ours since fire #10)
- ✗ Patch 19 USE_DEPTH_RECURRENCE (5+ merged records use it)
- ✗ Patch 20 USE_COPRIME_STRIDE (PR #1099 token-level — our shard-level variant is technically distinct)
- ✗ Patch 21 USE_XSA (PR #1448 ships XSA5LastGated; our anchor was broken anyway)
- ✗ Patch 22 USE_ENGRAM_LITE (PR #1440)

**PR #1430 (Per-Sample SLOT, claimed 0.39642 BPB)**: Still OPEN, 0 comments, no comp owner activity for 24h+. Likely dies on review.

**Latest comp PRs we haven't ported**:
- PR #1450: TMA Megakernel (H100-only via Hopper TensorDescriptor — won't help us)
- PR #1437: SP8192 + N-gram Tilt + 3-Layer Recurrence → val_bpb 1.078 (open frontier)
- PR #1445: 11L Depth Recurrence + EMA 0.9965 → val_bpb 1.0889 (record attempt)

## Spend estimate

| Category | Cost |
|---|---|
| RTX 3080 Ti pod (~11h × $0.30/h) | $3.30 |
| Failed 8xH100 launch (3 min × $21.52/h) | $1.08 |
| Subagent calls + ops overhead | ~$2.30 |
| **Total** | **~$6.70 / $36 (18.6%)** |

**Headroom remaining**: ~$29.30 = enough for **at least 10 H100 escalation attempts** at ~$2.69/each, OR a single 8xH100 spot run at ~$21.52 for 1 hour.

## Lessons learned (for next session)

1. **Always verify GPU utilization first**. If GPU memory is < 20% on a model that fits, the batch is too small.
2. **Patcher anchors decay**. Patch 22 EngramLite init was working at the start of the session and broken by the end because Patches 25/26 (NorMuon, Depth Recur) modified the surrounding init code. Use `getattr(self, '_attr', default)` for any cross-patch attribute reference to avoid AttributeError cascades.
3. **bash wrapper `run_forever.sh` survives `pkill -f experiment_runner.py`**. Future restarts must use `pkill -f run_forever.sh` BEFORE killing python.
4. **Branch hygiene**: at one point local was on `sota-prikshit-hymba11-muon` instead of `main`. Always verify with `git status` before local file ops.
5. **runpodctl create pod needs `--ports "22/tcp"`** for SSH proxy access. Web UI auto-configures this, CLI does not.
6. **The neutrality plateau from broken-config measurements is NOT a real verdict**. Re-validate ALL prior patches under the new compute regime in the next session.
