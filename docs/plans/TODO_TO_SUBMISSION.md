# TODO_TO_SUBMISSION.md — Paramgolf step-by-step from where we are now to a competition submission

**Status as of 2026-04-08 23:00 UTC**: best train_loss = **2.4499** (SP6_seed1337, single run, never validated on H100 val_bpb metric).
**Comp deadline**: 2026-04-30 (22 days remaining).
**Budget remaining**: ~$29.30 of $36 RunPod credit.

This doc is the path from "we have a defensible cheap-GPU champion" → "we have a comp submission that beats the open frontier (1.078 BPB)". Steps are prioritized by **expected BPB gain × confidence** divided by **engineering effort**. Each step has a clear input, output, and effort estimate.

---

## PHASE 0 — Stop digging the hole (1 hour, do FIRST next session)

The goal of phase 0 is to verify the bug fixes from this session are still in effect and the loop is healthy on the next pod boot.

### Step 0.1 — Branch hygiene

Verify local checkout is on `main`, not the `sota-prikshit-hymba11-muon` orphan branch the session got stuck on at one point.

```bash
cd /Users/takodamundy/Documents/personal_repos/paramgolf
git status -b | head -1   # must say "main"
git pull origin main
ls runpod_tests/loop/experiments.json  # must exist
```

### Step 0.2 — Verify pod is in known state

If pod `tyf0q5l1kgefgx-64410a6f` is still running:

```bash
/tmp/podstatus.sh
# Expect: clean process tree (one wrapper/runner/train_gpt OR no processes)
# Expect: GPU 100% util if running, 0% if stopped
# Expect: results.jsonl > 750 entries
```

If processes are duplicated again (multiple `bash run_forever.sh`), kill them ALL with `pkill -f run_forever.sh` BEFORE restarting anything.

### Step 0.3 — Patcher hygiene fix (Patch 22 EngramLite was broken all session)

The Patch 22 EngramLite init anchor was failing silently. The forward apply referenced `self._engram_lite_enabled` which crashed every experiment with `AttributeError`. Emergency fix was a `getattr(self, '_engram_lite_enabled', False)` wrap.

**Do this BEFORE shipping anything else**: read all the patches in `08_patch_train_gpt.sh` and wrap EVERY cross-patch attribute reference with `getattr(self, '_attr_name', default)`. This prevents future AttributeError cascades when one patch's init anchor breaks.

Estimated effort: 30 min audit + edit.

### Step 0.4 — Patcher should print full marker status at the END

Currently the patcher only prints during application. If a patch silently fails, you don't notice until experiments crash. Add a final status block:

```python
# At the end of the patcher script, after all replacements:
markers = ["NGRAM_BIAS_MARKER", "LEAKY_RELU_MARKER", "SMEAR_GATE_MARKER", "WAVELET_GPT_MARKER",
           "TABULATION_HASH_MARKER", "GATED_ATTENTION_MARKER", "ENGRAM_LITE_MARKER", "MTP_MARKER",
           "MOUSSE_MARKER", "MUONEQ_R_MARKER", "DEPTH_RECUR_MARKER", "NORMUON_MARKER",
           "COPRIME_STRIDE_MARKER", "XSA_MARKER", "NS_STEPS_MARKER"]
print("\n=== FINAL PATCH STATUS ===")
for m in markers:
    print(f"  {'✓' if m in content else '✗'} {m}")
```

Output goes to `wrapper.out` so the runner state is auditable on next monitor.

---

## PHASE 1 — Re-validate "neutrality plateau" patches (4-6 hours, mostly compute-bound)

The session's biggest discovery was that we were measuring everything under broken compute (TRAIN_BATCH_TOKENS=1024 instead of 65536). **All prior patch verdicts are invalid**. They need fresh runs at proper compute to determine which actually help.

### Step 1.1 — Re-test 7 patches at the SP6 baseline

Add experiments to `experiments.json` that test each patch on top of the validated SP6 stack. Each experiment uses the same SP6 base + ONE patch toggled on:

| Test name | Toggle on top of SP6 base |
|---|---|
| RV_mousse | `USE_MOUSSE=1` |
| RV_muoneqr | `USE_MUONEQ_R=1` |
| RV_normuon | `USE_NORMUON=1` |
| RV_depthrecur | `USE_DEPTH_RECURRENCE=1, DEPTH_RECUR_START=3, DEPTH_RECUR_END=3, DEPTH_RECUR_CYCLES=2` |
| RV_qkgain5 | `QK_GAIN_INIT=5.0` |
| RV_gated_attn | `USE_GATED_ATTENTION=1` |
| RV_mtp | `USE_MTP=1, MTP_NUM_HEADS=1, MTP_LOSS_WEIGHT=0.10` |

For each: 3 seeds (1337, 42, 999), MAX_WALLCLOCK_SECONDS=900, the SP6 base config (Coprime + EngramLite + leaky + ngram + L4 weights + seq=1024 + batch=65536).

**Total cost**: 7 patches × 3 seeds × 25 min = 8.75 hours of cheap-GPU runtime. ~$2.65 in pod cost.

**Output**: a clean table of "patches that actually help at proper compute". Many will move from "marginal" to "real win" or "real loss".

### Step 1.2 — Decision matrix

After step 1.1, each patch is in one of three buckets:

- **KEEP** (mean drop > 0.005 vs SP6 base): include in the H100 escalation stack
- **NEUTRAL** (mean within ±0.005): leave as opt-in, don't include in H100 escalation
- **REMOVE** (mean increase > 0.005): mark SKIP, don't include

Update `runpod_tests/loop/experiments.json` to remove the falsified ones.

**Estimated effort**: 8.75 hours wallclock + 30 min analysis.

---

## PHASE 2 — Build BPE-8192 (the SINGLE biggest unshipped win — Mac §18c claims -0.129 BPB)

Mac LESSONS §18c flagged this as the biggest unshipped Mac-validated win. We have the BPE-8192 tokenizer file (`data/tokenizers/fineweb_8192_bpe.model`) on the pod but **never built the n-gram tables**. Without tables, switching tokenizers loses Patch 6 NGRAM_BIAS entirely.

### Step 2.1 — Re-tokenize FineWeb shards with BPE-8192

```bash
# On the pod:
cd /workspace/paramgolf
TOKENIZER_PATH=data/tokenizers/fineweb_8192_bpe.model python3 data/download_hf_docs_and_tokenize.py
# Output: data/datasets/fineweb10B_bpe8192/fineweb_train_*.bin
```

**Estimated effort**: 30-60 min download + tokenize.

### Step 2.2 — Build BPE-8192 ngram tables

```bash
TOKENIZER_PATH=data/tokenizers/fineweb_8192_bpe.model \
DATA_PATTERN="data/datasets/fineweb10B_bpe8192/fineweb_train_*.bin" \
HASH_BUCKETS=16384 \
python3 runpod_tests/chore/04_build_ngrams.py
# Output: data/{bigram,trigram,fourgram}_logprobs_8192v.npy (already partially exist!)
```

Note: looking at git status, files like `bigram_logprobs_8192v.npy`, `trigram_logprobs_8192v.npy`, `fourgram_logprobs_8192v.npy` ALREADY EXIST in the repo. They're untracked. **Verify these are correct BPE-8192 tables, not stale.**

**Estimated effort**: 30 min if existing tables are valid, 2-3 hours if rebuild needed.

### Step 2.3 — Add `TOKENIZER_VARIANT` env var to train_gpt.py

Patch the train_gpt.py to switch between sp1024 and bpe8192 tokenizer paths via env var. The vocab size differs (1024 vs 8192) which affects:
- Embedding matrix size
- N-gram table shape (`(16384, 1024)` vs `(16384, 8192)`)
- Output projection (tied embedding)

This is a non-trivial code change. Add it as **Patch 26 USE_BPE8192** in `08_patch_train_gpt.sh`.

**Estimated effort**: 2-4 hours engineering + testing.

### Step 2.4 — Validate BPE-8192 on cheap GPU

Add experiments:
- `BPE0_bpe8192_baseline` — BPE-8192 tokenizer + same SP6 stack
- `BPE1_bpe8192_seed1337` — multi-seed
- `BPE2_bpe8192_seed999` — multi-seed

Run for 900s wallclock each. **Expected gain at proper compute: -0.05 to -0.13 BPB** based on Mac claim.

**If BPE-8192 wins by > 0.03 BPB**: it becomes the new H100 escalation tokenizer.
**If it doesn't win**: SP-1024 stays. Mark BPE-8192 as "Mac result didn't transfer" and move on.

**Estimated effort**: 75 min + analysis.

**Total Phase 2 effort**: 4-7 hours engineering + 75 min validation.

---

## PHASE 3 — Ship the eval-time bundle (3-spec H100 escalation prep)

Three specs are documented in RESEARCH_LOG.md but never shipped because they're EVAL-only — they don't affect mid-training train_loss so they couldn't be validated on the cheap-GPU loop. They MUST be shipped before the H100 final submission run.

### Step 3.1 — Patch 23 USE_NGRAM_TILT (eval-time multiplicative boost)

**Source**: PR #1437 (1.078 BPB) + PR #1420. Canonical formula in RESEARCH_LOG.md research fire #6.

**Math**: at each eval position, use prefix-only n-gram cache to predict the next token. If the model's prediction matches the cache hint, multiplicatively boost it:

```
hint = ngram_cache.lookup(prefix[:t])  # may be None
Z    = 1 + p_model(hint) * (exp(β) - 1)
p_tilt(x_t) = p_model(x_t) * exp(β * 1[x_t == hint]) / Z
```

**LEGAL** under issue #1017 four conditions (causal, score-before-update, single-pass, full-normalized).

**Implementation**: ~150 LOC in train_gpt.py eval loop. Cache structure is per-shard hash table built from previously-scored tokens. Score-before-update: cache is updated AFTER the score is locked at each position.

**Hyperparameters**: β = 1.5, n-gram order 8-16, hash buckets 4M.

**Expected gain**: +0.0015 to +0.0030 BPB.

**Effort**: 4-6 hours careful engineering + 1 hour test on cheap GPU with `SKIP_FINAL_EVAL=0`.

### Step 3.2 — Patch 24 USE_EMA (decay 0.997)

**Source**: 6 merged records (PR #287, #315, #414, #1019, #1099). Canonical pattern in RESEARCH_LOG.md research fire #7.

**Math**: maintain a shadow copy of model weights in fp32. After each optimizer step, update via:
```python
ema_state[name] = decay * ema_state[name] + (1 - decay) * param.detach().float()
```
Before final eval, swap model weights with EMA shadow.

**Implementation**: ~30 LOC in train_gpt.py:
1. Init shadow at start of training (after optimizer creation)
2. Update shadow after each `opt.step()` (in the training loop)
3. Swap before final eval (in the serialization step)

**Memory cost**: 88 MB (22M params × 4 bytes). Negligible on H100.

**Expected gain**: +0.001 to +0.005 BPB.

**Effort**: 2-3 hours engineering + 1 hour test.

### Step 3.3 — Patch 25 USE_INT6_GPTQ + LZMA (compression-side)

**Source**: PR #1099 (latest merged record), #1019, #1444, #1446. Canonical in RESEARCH_LOG.md research fire #8.

**Math**: per-row 99.95th percentile quantization to int6 (clamped to [-31, 31]), stored in int8 container, then LZMA-22 compression of the resulting state dict.

**Implementation**: ~130 LOC in train_gpt.py serialization step. Replace the existing int8+zlib path with int6+lzma when env var is set.

**Direct gain**: -0.0003 BPB (within noise) BUT saves ~0.5 MB of artifact size headroom which can be spent on more model capacity (more layers / wider).

**Effort**: 3-4 hours engineering + 1 hour test.

**Critical**: this patch modifies the serialization path. ALWAYS test with `SKIP_FINAL_EVAL=0` on a cheap GPU first to verify the int8_zlib_roundtrip → int6_lzma_roundtrip transition doesn't corrupt the model.

### Step 3.4 — Combined H100 escalation bundle

Once all three eval-bundle patches are shipped + tested on cheap GPU with `SKIP_FINAL_EVAL=0`:

```bash
USE_NGRAM_TILT=1 USE_EMA=1 USE_INT6_GPTQ=1 \
USE_LEAKY_RELU=1 USE_NGRAM_BIAS=1 \
USE_COPRIME_STRIDE=1 USE_ENGRAM_LITE=1 \
TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=524288 \
SEED=1337 SKIP_FINAL_EVAL=0 \
python3 train_gpt.py
```

**Combined estimated gain**: +0.003 to +0.008 BPB on top of the SP6 base.

**Total Phase 3 effort**: 9-13 hours engineering + 3 hours testing.

---

## PHASE 4 — Pursue genuine novelty (the part you actually wanted)

Phases 1-3 are all ports. The user mandate was novel-to-world, cross-domain. Phase 4 is the part where we try things nobody has done.

### Step 4.1 — Tokenizer-side novelty: ByteSpan or fused n-gram tokenizer

**Idea**: instead of switching to BPE-8192, use a HYBRID tokenizer that switches between byte-level and BPE based on token frequency. Common tokens use BPE (efficient), rare tokens use bytes (no OOV). This is a **novel-to-comp** idea — no PR uses it.

**Implementation**: write a new tokenizer that wraps both sp1024 + a byte fallback. Tokenize FineWeb with the hybrid scheme. Build n-gram tables. Compare against pure BPE-8192.

**Effort**: 6-8 hours engineering. **Expected gain**: unknown (speculative), -0.02 to -0.05 BPB if the hybrid actually saves more bytes than pure BPE.

**Risk**: high. May not work. But it's genuinely novel.

### Step 4.2 — Eval-side novelty: per-token adaptive n-gram weight

**Idea**: instead of a fixed N-gram Tilt β=1.5, learn β per-token based on the model's confidence. When the model is uncertain (high entropy), use a high β (trust the n-gram more). When confident, use a low β. The β is a scalar function of model entropy.

**Math**:
```
H = entropy(p_model)
β = β_max * sigmoid(H / H_threshold - 1)  # 0 to β_max
p_tilt = p_model * exp(β * 1[x == hint]) / Z
```

**Novel-to-comp?**: We tested entropy-adaptive n-gram bias as Patch 14 (USE_ENTROPY_ADAPTIVE_NGRAM) and it failed. BUT that was at the broken-config scale. The combination "entropy-adaptive multiplier on N-gram TILT (eval-time)" is distinct from "entropy-adaptive on n-gram BIAS (training-time)" and has not been done in any PR.

**Implementation**: ~50 LOC modification of the N-gram Tilt patch (Step 3.1). Effort: 2-3 hours after Tilt is shipped.

**Expected gain**: +0.001 to +0.003 BPB on top of static Tilt. Speculative.

### Step 4.3 — Compression-side novelty: dual-codebook + signed hashing

**Sources**: Mac LESSONS §37 (dual-codebook clustering of bigram distributions) + Mac §34 (signed hashing for n-gram tables).

**Idea**:
- **Signed hashing** (§34): when looking up the n-gram bias for token (prev2, prev1), use `sign = ((prev2 * 2654435761 + prev1 * 2246822519) % 2) * 2 - 1`. Multiply the bias by sign at lookup AND when building the table. Result: collisions become zero-mean noise instead of systematic bias. **+0.003 BPB for 2 lines of code**, never shipped.
- **Dual-codebook** (§37): cluster the 8192 bigram distributions into 64 prototypes (k-means). Store: 64 prototype vectors (full precision) + 8192 prototype indices (int8) + per-token residual (int4). **Saves 2.17 MB** of artifact size, frees space for more layers.

**Combined novelty**: nobody has shipped both together. Together they could give -0.005 BPB + 2 MB headroom.

**Effort**: 8-12 hours engineering for dual-codebook + 1 hour for signed hashing.

### Step 4.4 — Training-side novelty: shared low-rank n-gram embedding

**Idea**: currently we have THREE separate n-gram bias tables (bigram, trigram, fourgram), each of shape `(16384, 1024)`. They're trained on different stats but they all map to the same vocab. Replace with a SHARED low-rank decomposition:

```
U: (16384, R)  [shared across orders]
V_bigram: (R, 1024)
V_trigram: (R, 1024)
V_fourgram: (R, 1024)

bigram_logits[h] = U[h] @ V_bigram
trigram_logits[h] = U[h] @ V_trigram
fourgram_logits[h] = U[h] @ V_fourgram
```

For R=64: total params = 16384*64 + 3*64*1024 = 1.0M + 0.2M = 1.2M (vs current 50M for three separate tables). **Saves ~50 MB**, frees space for a much bigger model.

**Novel-to-comp?**: Yes. No PR uses this structure for n-gram bias.

**Implementation**: requires rebuilding the n-gram tables AND modifying Patch 6 NGRAM_BIAS forward pass. ~6-8 hours engineering.

**Expected gain**: not direct BPB but the freed memory could fund -0.02 BPB worth of extra model capacity.

### Step 4.5 — Hardware-side novelty: torch.compile properly

We never re-enabled torch.compile properly. The first attempt (default mode) crashed every experiment. The dynamic=True attempt crashed with shape errors. The fix is probably:
- Use `mode="reduce-overhead"` (CUDA graphs, handles dynamic shapes better)
- OR: split the model into "compileable" and "non-compileable" submodules. Compile the encoder blocks; leave the n-gram bias lookup in eager.

**Source**: Mac LESSONS wishlist mentions 25-35% throughput speedup if compile works. CLAUDE.md "Things on the user's wish list NOT YET ADDRESSED" lists this.

**Effort**: 4-8 hours debugging.

**Expected gain**: 25-35% more steps in the same wallclock budget. With the SP6 stack already at train_loss 2.45 in 1500 steps, 25-35% more steps could push it to 2.30-2.35.

### Step 4.6 — One truly speculative novel idea (PhD-level swing)

This is the "what if nobody has tried this?" slot. Pick ONE idea you've never seen anywhere:

- **Self-distillation via early-step replay**: at step T, use the model's predictions from step T/2 as soft targets. Provides regularization without a teacher.
- **Curriculum via N-gram predictability**: order training batches by how predictable they are under the static n-gram bias. Start easy, end hard. (Different from existing curriculum learning which orders by length.)
- **Per-position learnable bias on the n-gram lookup**: each of the 1024 token positions in the sequence gets a learnable scalar that scales the n-gram contribution. Makes the early positions trust n-gram more, late positions trust the model more.
- **Inverse tabulation hashing**: instead of mapping token pairs → bucket, learn an INVERSE mapping that tells the model which bucket would best represent a given prefix. Adds learnable parameters but provides signed signal.

**Effort**: 4-8 hours per idea. Risk: high, may not work. Reward: this is the only path to a true novel-to-world breakthrough.

**Total Phase 4 effort**: 30-50 hours engineering, scattered. Only do this AFTER Phases 1-3 give us the safety net.

---

## PHASE 5 — H100 escalation cycles (the actual submission)

After Phase 1 (re-validation) and Phase 3 (eval bundle), we're ready to escalate to H100 for the real val_bpb measurement.

### Step 5.1 — Verify the H100 launch script works

The previous H100 attempt failed because `runpodctl create pod` doesn't expose port 22. The corrected command:

```bash
runpodctl create pod --name "paramgolf-h100-1" \
  --gpuType "NVIDIA H100 80GB HBM3" --gpuCount 1 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --containerDiskSize 50 --volumeSize 50 --volumePath /workspace \
  --secureCloud --cost 3.0 \
  --ports "22/tcp,8888/http"
```

**Critical**: after pod is RUNNING, verify port 22 is in the ports list via `runpodctl get pod <id> -a`. If missing, terminate and re-launch via the RunPod web UI (which auto-configures SSH).

**Test run**: launch with a dummy `python3 -c "print('hello'); import torch; print(torch.cuda.is_available())"` to verify SSH works. Cost: ~$0.10. Then kill the pod.

**Effort**: 30 min (mostly waiting for pod boot).

### Step 5.2 — Bootstrap pod for first real run

After the test run succeeds:

```bash
# On the new H100 pod:
cd /workspace
git clone https://github.com/taka6745/paramgolf.git
cd paramgolf
bash runpod_tests/chore/08_patch_train_gpt.sh

# Download FineWeb shards (10-15 min)
python3 data/download_hf_docs_and_tokenize.py

# Verify n-gram tables exist
ls data/{bigram,trigram,fourgram}_logprobs.npy
```

**Effort**: 30 min.

### Step 5.3 — Run THE ESCALATION

```bash
USE_LEAKY_RELU=1 \
USE_NGRAM_BIAS=1 \
USE_COPRIME_STRIDE=1 \
USE_ENGRAM_LITE=1 \
USE_NGRAM_TILT=1 \
USE_EMA=1 \
USE_INT6_GPTQ=1 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=524288 \
NGRAM_W_BIGRAM=0.25 \
NGRAM_W_TRIGRAM=0.25 \
NGRAM_W_FOURGRAM=0.20 \
SEED=1337 \
SKIP_FINAL_EVAL=0 \
MAX_WALLCLOCK_SECONDS=600 \
python3 train_gpt.py 2>&1 | tee h100_run_seed1337.log
```

**Critical**: this is the FIRST H100 run with the eval bundle. Watch for:
1. No torch.compile crashes
2. Step time < 100 ms
3. GPU util > 80%
4. Final eval line: `final_int8_zlib_roundtrip val_bpb = X.XXXX`
5. KILL THE POD IMMEDIATELY after capturing the val_bpb line

**Cost**: ~$1.35 per run.

**Output**: a real val_bpb number to compare against the comp open frontier (1.078).

### Step 5.4 — Multi-seed for the actual submission

Comp records use 3-seed mean. Run two more H100 escalations with seeds 42 and 999. Total cost: 3 × $1.35 = $4.05.

If the 3-seed mean is below 1.07: **YOU HAVE A SUBMITTABLE RECORD**. Move to Step 5.5.

If the mean is between 1.07-1.10: marginal. Consider: more eval-bundle tweaks, more seeds, OR re-run with longer wallclock budget.

If the mean is above 1.10: something didn't transfer from cheap-GPU to H100. Diagnose by running an ablation (run with SP6 base only, no eval bundle).

### Step 5.5 — Submit the record to openai/parameter-golf

```bash
git checkout -b record/sp6-bundle-1.07
# Commit the canonical train_gpt.py + reproduce script
gh pr create --repo openai/parameter-golf \
  --title "[Submission] SP6 + Eval Bundle: val_bpb X.XXXX (3-seed mean)" \
  --body "$(cat <<'EOF'
## Summary
SP6 stack (Coprime Stride + EngramLite + leaky_relu + n-gram bias) +
Eval bundle (N-gram Tilt + EMA 0.997 + INT6 GPTQ).

## Multi-seed result
- seed 1337: X.XXXX
- seed 42:   X.XXXX
- seed 999:  X.XXXX
- 3-seed mean: X.XXXX (std: X.XXXX)

## Reproduce
\`\`\`bash
[paste the exact command from Step 5.3 with the seed flag templated]
\`\`\`

## Techniques
- Patch 6 NGRAM_BIAS (training-time bigram/trigram/fourgram log-prob tables, 16K hash buckets)
- Patch 9 LEAKY_RELU (leaky_relu(0.5)^2 in MLP)
- Patch 20 USE_COPRIME_STRIDE (shard-level coprime sampling in TokenStream._advance_file)
- Patch 22 USE_ENGRAM_LITE (learnable hash-embedding n-gram head, ported from PR #1440)
- Patch 23 USE_NGRAM_TILT (eval-time multiplicative bias, ported from PR #1437/#1420)
- Patch 24 USE_EMA (decay 0.997, ported from 6 merged records)
- Patch 25 USE_INT6_GPTQ (per-row 99.95th percentile quant + LZMA-22, ported from PR #1099)

EOF
)"
```

**Effort**: 30 min + multi-seed wait.

---

## PHASE 6 — Iterate towards the leaderboard top (the long tail)

After the first submission, iterate by adding the most-validated PR techniques we haven't shipped yet:

1. **N-gram Tilt with higher order** (16-22) — used in PR #1430 for the contested 0.39642 BPB. If PR #1430 gets merged with comp owner approval, port the order-22 variant.
2. **3-layer depth recurrence with mixed-precision quant** — PR #1437 + #1445 use it. Combines Patch 19 (depth recurrence) with int5/int6 mixed quant.
3. **Hymba (Mamba+Attention hybrid)** — PR #852 claims 85 ms/step on H100 at 1.1189 BPB. Requires `mamba-ssm` + `causal-conv1d` libraries. Risky port (1551-line file replacement) but could unlock more steps in the same wallclock budget.
4. **Scylla-998 tokenizer** — RESEARCH.md mentions Scylla + modern stack scoring 0.9485 BPB locally. Requires building a candidate.meta.npz with per-token byte counts. Not in any PR.

Each is a Phase 6 cycle: port → cheap-GPU validate → H100 escalate → submit if it improves.

---

## Schedule

**Realistic timeline assuming 4-6 hours of focused work per day**:

| Day | Phase | Work |
|---|---|---|
| Day 1 | 0 + 1 | Branch hygiene + patcher hygiene + start re-validation runs (compute-bound, runs in background) |
| Day 2 | 1 + 2 | Re-validation analysis + start BPE-8192 build |
| Day 3 | 2 + 3 | Finish BPE-8192 + start N-gram Tilt patch |
| Day 4 | 3 | Ship Tilt + EMA + INT6 GPTQ (the eval bundle) |
| Day 5 | 3 + 5 | Test eval bundle on cheap GPU + first H100 escalation |
| Day 6 | 5 | Multi-seed H100 + first submission |
| Day 7-21 | 4 + 6 | Phase 4 novelty exploration + Phase 6 iteration |
| Day 22 | — | Final submission deadline 2026-04-30 |

**Total compute budget**:
- Phase 1 re-validation: ~$3
- Phase 2 BPE-8192 validation: ~$1
- Phase 5 H100 cycles: ~$15 (10-12 escalation runs)
- Phase 4 novelty exploration: ~$5
- **Total**: ~$24, well within the $36 RunPod budget

---

## What you should NOT do

1. **Don't ship more port-from-comp-PR patches without measuring at proper compute first.** The session shipped 8 ports that all turned out to be falsified at the wrong scale. Re-validate FIRST (Phase 1), then port more.
2. **Don't trust subagent LOC estimates blindly.** The session burned $1.08 on a failed H100 attempt because I trusted a subagent that said "30 LOC, easy". Always read the actual upstream code before estimating effort.
3. **Don't make a new patch the DEFAULT before validating.** The torch.compile re-enable made `USE_TORCH_COMPILE='1'` the default which crashed every experiment for ~30 min. Default = 0, opt-in via env var.
4. **Don't run the H100 escalation without verifying SSH works first.** The last attempt cost $1.08 because port 22 wasn't exposed. Always test with a $0.10 dummy run first.
5. **Don't pursue novelty (Phase 4) before the safety net (Phases 1-3).** If novelty fails, the safety net of validated ports gives you a submittable result. If novelty succeeds, even better. Don't gamble without the safety net.

---

## Final note

The session's biggest gain came from a BUG FIX (the microscopic batch), not from any patch. The lesson is that **measurement quality** is the foundation. Profile first, ship second. Phases 0-2 are all about getting measurement right before pursuing novelty.

The H100 escalation in Phase 5 is what turns "we have a champion config" into "we have a comp record". Don't skip it.

The novel-to-world work in Phase 4 is what turns "we have a comp record" into "we have a paper". Only attempt after Phases 1-3 give you a safety net.
