# TODO_NEXT.md — H100 escalation + post-overnight followups

**Session 2026-04-07/08 ended at 22:55 UTC.** Best train_loss = **2.4499** (SP6_seed1337, 1500 steps under TRAIN_BATCH_TOKENS=65536, TRAIN_SEQ_LEN=1024).

## CRITICAL — H100 Escalation (do this FIRST)

The SP6 stack is the canonical H100 escalation candidate. We have a tight n=2 mean (2.5925, std 0.0013) on seed 42 plus the seed 1337 single-run best (2.4499). **Need to translate this to real `final_int8_zlib_roundtrip val_bpb`** on H100.

### Step 1: launch H100 pod with SSH properly configured

The previous H100 launch attempt at 19:50 UTC failed because `runpodctl create pod` doesn't auto-expose port 22. **The corrected command** (untested but include the missing flag):

```bash
runpodctl create pod --name "paramgolf-h100-final" \
  --gpuType "NVIDIA H100 80GB HBM3" --gpuCount 1 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --containerDiskSize 50 --volumeSize 50 --volumePath /workspace \
  --secureCloud --cost 3.0 \
  --ports "22/tcp,8888/http"
```

**SAFER alternative**: launch via the RunPod WEB UI (which auto-configures SSH). Pick `1× H100 80GB SXM (spot)` and the same image. Get the SSH endpoint from the web UI, NOT from runpodctl.

### Step 2: bootstrap the new pod

```bash
# On the new H100 pod:
cd /workspace
git clone https://github.com/taka6745/paramgolf.git
cd paramgolf

# Apply all patches (assumes the patcher script is idempotent)
bash runpod_tests/chore/08_patch_train_gpt.sh

# Download FineWeb shards (may take 10-15 min)
python3 data/download_hf_docs_and_tokenize.py  # or whatever the data prep script is
```

### Step 3: run the SP6 canonical stack with FULL eval

```bash
USE_LEAKY_RELU=1 \
USE_NGRAM_BIAS=1 \
USE_COPRIME_STRIDE=1 \
USE_ENGRAM_LITE=1 \
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

**Note**: TRAIN_BATCH_TOKENS bumped to 524288 (the upstream default for 8xH100). On 1xH100 80GB, that should fit comfortably and give ~80MS/step.

**Capture**: `final_int8_zlib_roundtrip val_bpb` from the log. That's the comp metric.

### Step 4: KILL THE H100 IMMEDIATELY after capture

```bash
runpodctl remove pod <pod_id>
```

**Estimated cost**: 1× H100 spot at $2.69/h × 0.5h = ~$1.35 per run.

## HIGH-PRIORITY untested patches (port to H100 escalation bundle)

These three are documented in RESEARCH_LOG.md but never shipped because they're eval-only (couldn't validate on the cheap-GPU loop). They should ALL be added to the H100 escalation bundle:

1. **N-gram Tilt** (RESEARCH_LOG fire #6)
   - Multiplicative eval-time logit boost using prefix-only n-gram cache
   - Formula: `p_tilt(x) = p_model(x) * exp(β * 1[x==hint]) / Z` where `Z = 1 + p_model(hint)*(exp(β)-1)`
   - β = 1.5, n-gram order 8-16, cache size 4M buckets
   - Source: PR #1437 (1.078 BPB) + PR #1420 + issue #1017
   - LEGAL under issue #1017 four conditions
   - Implementation: ~150 LOC eval-loop modification
   - **Estimated +0.0015 to +0.0030 BPB**

2. **EMA decay 0.997** (RESEARCH_LOG fire #7)
   - Standard exponential moving average of model weights, swapped in for final eval
   - Source: 6 merged records (PR #287, #315, #414, #1019, #1099)
   - 88 MB shadow params memory cost (negligible on H100)
   - Implementation: ~30 LOC (clone state_dict at init, lerp after each opt.step, swap before final eval)
   - **Estimated +0.001 to +0.005 BPB**

3. **INT6 GPTQ + LZMA serialization** (RESEARCH_LOG fire #8)
   - Per-row 99.95th percentile quantization to int6, then LZMA-22 compression
   - Source: PR #1099, #1019, #1444 + PR #1446
   - Saves ~0.5 MB vs current int8+zlib serialization
   - Implementation: ~130 LOC in train_gpt.py serialization step
   - **Estimated -0.0003 BPB direct + ~0.5 MB headroom for more capacity**

## Re-validate "neutrality plateau" patches under proper compute

The session shipped 7 training-time patches that all showed marginal/null effect under the broken `TRAIN_BATCH_TOKENS=1024` config. **All those verdicts are invalid.** Re-test under the new compute (`TRAIN_BATCH_TOKENS=65536+, TRAIN_SEQ_LEN=1024+`) to see if any actually help:

| Patch | Marker | Estimated re-test cost | Priority |
|---|---|---|---|
| Mousse (Patch 17) | MOUSSE_MARKER | ~25 min × 3 seeds | MEDIUM |
| MuonEq-R (Patch 18) | MUONEQ_R_MARKER | ~25 min × 3 seeds | MEDIUM |
| Depth Recurrence (Patch 19) | DEPTH_RECUR_MARKER | ~30 min × 3 seeds (deeper compute) | HIGH |
| NorMuon (Patch 25) | NORMUON_MARKER | ~25 min × 3 seeds | MEDIUM |
| QK_GAIN_INIT=5.0 (env var) | n/a | ~25 min × 3 seeds | LOW |

**Total re-test cost**: ~5-7 hours of cheap-GPU loop OR done as ablations on H100 cycles.

## Patcher script hygiene (prevent future cascades)

The Patch 22 EngramLite init anchor was broken by Patches 25/26 modifying surrounding code. **Required fixes**:

1. **Wrap ALL cross-patch attribute references with `getattr`** as a defensive default. The forward pass should NEVER crash because an init didn't run — it should fall back to no-op.
2. **Add a sanity print at the END of the patcher** showing which markers ARE present in the final train_gpt.py. Currently the patcher only prints during application — if a patch fails silently the user doesn't notice until experiments crash.
3. **Anchor on more invariant code patterns** (e.g., function signatures, class definitions) rather than line-by-line text that other patches can modify.

## XSA fix (Patch 21)

The XSA patch's anchor never matched the upstream `CausalSelfAttention.forward()` due to my misreading of the SDPA + GATED_ATTENTION block. Result: USE_XSA=1 silently does nothing. The XSA family experiments are all NO-OP runs.

To fix:
1. Read the actual upstream `train_gpt.py` to get the EXACT post-SDPA + post-GATED_ATTENTION code
2. Re-anchor XSA on the verified pattern
3. Test that the anchor matches by looking for "✓ added XSA" in the patcher output

## BPE-8192 build (Mac LESSONS §18c — claimed -0.129 BPB, biggest single Mac win)

We have the BPE-8192 tokenizer file pushed to the pod (`data/tokenizers/fineweb_8192_bpe.model`) but **never built the n-gram tables for it**. Without those tables, switching tokenizers means losing the n-gram bias entirely (which is a -0.05 BPB regression).

**Build steps** (multi-hour, do during a long pod-idle window):
1. Run `data/download_hf_docs_and_tokenize.py` with `TOKENIZER_PATH=data/tokenizers/fineweb_8192_bpe.model` to generate `fineweb_train_*_8192.bin` files
2. Run `runpod_tests/chore/04_build_ngrams.py` against the new bin files to generate `bigram_logprobs_8192v.npy`, `trigram_logprobs_8192v.npy`, `fourgram_logprobs_8192v.npy`
3. Add a `TOKENIZER_VARIANT` env var to train_gpt.py that switches between sp1024 and bpe8192 paths (the n-gram tables have different shapes)
4. Validate on cheap GPU loop with TRAIN_BATCH_TOKENS=65536

**Estimated effort**: 2-3 hours engineering. **Expected gain**: -0.05 to -0.13 BPB depending on whether the Mac result transfers.

## PR #1430 watch (task #57 — currently still pending)

PR #1430 has been OPEN at claimed 0.39642 BPB for 24h+ with zero comp owner activity. **Check status periodically**:

```bash
gh api 'repos/openai/parameter-golf/pulls/1430' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'state={r[\"state\"]}, merged={r[\"merged\"]}, comments={r[\"comments\"]}')"
```

If MERGED with comp owner approval → port the Per-Sample SLOT + N-gram Order-22 + TTT stack immediately. If REVERTED or outlawed → mark dead.

## Long-term: branch cleanup

The session detected at one point that local was on `sota-prikshit-hymba11-muon` instead of `main`. Verify all branches are in expected state:

```bash
git branch -v
git status
```

Make sure the user's intended workflow branch is checked out before next session.

---

**Single-line H100 launch (DO NOT EXECUTE WITHOUT USER APPROVAL)**:

```bash
runpodctl create pod --name "paramgolf-h100-final" --gpuType "NVIDIA H100 80GB HBM3" --gpuCount 1 --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" --containerDiskSize 50 --volumeSize 50 --volumePath /workspace --secureCloud --cost 3.0 --ports "22/tcp,8888/http"
```

After pod is RUNNING, get the SSH endpoint from `runpodctl get pod <id> -a` AND verify port 22 is in the ports list. If port 22 is missing, the SSH proxy will return "container not found" — fall back to web UI launch.
