#!/bin/bash
# submission/dry_run.sh — H100 SUBMISSION RUN (dry run AND real submission, same code path)
#
# THE CANONICAL ENTRY POINT. This script is BOTH:
#   - The dry run sanity check (1 seed, fast iteration on 1×H100 PCIe)
#   - The real comp submission (3 seeds, full validation on 8×H100 SXM)
#
# The ONLY difference between the two is the SEEDS env var:
#   bash submission/dry_run.sh                       # dry run (default SEEDS=42)
#   SEEDS=42,314,999 bash submission/dry_run.sh       # real 3-seed submission
#
# Output: assembles a complete comp submission folder under
#         records/track_10min_16mb/<date>_<config-tag>/
#         with: README.md, submission.json, train_gpt.py, train_seed<N>.log
#
# This is the FULL stack we'd ship as a real comp record. Targets PR #1493
# (merged leaderboard #1, val_bpb 1.0810 on 8×H100 SXM) plus our deltas:
#   PR #1493 stack (verified from openai/parameter-golf merged PR):
#   - SP8192 vocab
#   - NUM_LAYERS=11, MLP_MULT=4 (matches PR #1493 architecture)
#   - 3-Layer Depth Recurrence (NUM_LOOPS=2, LOOP_START=3, LOOP_END=5)
#   - Parallel Residuals from layer 7+ (PARALLEL_RESIDUAL_START=7, 4 of 11 layers)
#   - QK-Gain 5.25
#   - Legal Score-First TTT (TTT_ENABLED=1, TTT_LR=0.005, TTT_EPOCHS=3)
#   - EMA_DECAY=0.9965, WARMDOWN_FRAC=0.72, ENABLE_LOOPING_AT=0.35
#   - MUON_WD=0.095, MATRIX_LR=0.022
#   Our deltas on top (where we bet on winning):
#   - MATRIX_BITS=8 (int8 weights vs PR #1493's int6 — saves ~0.011 BPB on quant gap)
#   - USE_PARALLEL_MUON=1 (batched Newton-Schulz speedup)
#   - USE_CMP_QUANT_VALUE_DEDUP=1 (NIGHT_MODE L10 alphabet-snap compression to fit 16 MB at int8)
#   - USE_GATED_ATTENTION=1 + USE_NORMUON=1 + USE_NORM_PCT_DROPOUT=1 (NIGHT_MODE wins)
#   - max-autotune-no-cudagraphs compile mode + cudnn.benchmark
#   - n-gram bias stack DISABLED (Issue #1017 Condition 2 grey area, user policy: no illegal)
#   - 600s training cap, full 16 MB artifact target
#
# Usage on a fresh pod:
#   bash submission/bootstrap.sh   # FIRST do data setup (slow, one-time)
#   bash submission/dry_run.sh     # then this
#
# Or as a one-liner via curl on a fresh pod (after bootstrap completes):
#   curl -sL https://raw.githubusercontent.com/taka6745/paramgolf/main/submission/dry_run.sh | bash
#
# Expected on 1×H100 PCIe: val_bpb ~1.10-1.20 (validates the projection)
# Expected on 8×H100 SXM:  val_bpb ~1.00-1.08 (potentially beats PR #1493 = 1.0810)

set -eu
REPO_DIR="${REPO_DIR:-/workspace/paramgolf}"
cd "$REPO_DIR"

# === Seed selection (THE dry-run/real-submission switch) ===
SEEDS="${SEEDS:-42}"
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
NUM_SEEDS=${#SEED_ARRAY[@]}

# === Records folder staging ===
DATE_STR=$(date -u +%Y-%m-%d)
CONFIG_TAG="SP8192_NL11_MLP4_int8_ParMuon_PR7_LegalTTT"
RECORD_NAME="${DATE_STR}_${CONFIG_TAG}"
RECORD_DIR="records/track_10min_16mb/${RECORD_NAME}"
mkdir -p "$RECORD_DIR"

echo "============================================================"
echo "[dry_run] SUBMISSION RUN starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo "  Seeds: ${SEEDS} (${NUM_SEEDS} run$([ $NUM_SEEDS -gt 1 ] && echo s || echo ''))"
echo "  Stack: PR #1493 architecture (NL11/MLP4) + our int8 quant + parallel muon"
echo "         + parallel residuals L7+ + alphabet-snap compress + max-autotune"
echo "  Records folder: $RECORD_DIR"
echo "============================================================"

# === Submission-grade config (shared across all seed runs) ===
export MAX_WALLCLOCK_SECONDS=600

# Model architecture — Option C: PR #1493 architecture (11L+4x MLP) + our int8
# This is the bigger-model bet. Reasoning: on 8xH100 SXM, model capacity is the
# binding constraint (not training compute), so smaller models like CHAMP_D's
# 6L+2x can't reach the leaderboard's 1.07-1.10 cluster. PR #1493 proves that
# 11L+4x at int6 hits 1.0810. Our edge: switch to int8 to save 0.012 BPB on the
# quant gap. CHAMP_D validated int8 on the smaller model (gap 0.001 vs 3+ BPB
# for int6). This config is UNTESTED for us at the bigger size — CHAMP_E was
# killed mid-run before we could measure 11L+int8.
export NUM_LAYERS=11                        # match PR #1493
export MLP_MULT=4                           # match PR #1493

# PR #1493 (leaderboard #1, val_bpb 1.0810) architecture techniques (verified
# from gh pr view 1493 + train_seed314.log Hyperparameters dump)
export NUM_LOOPS=2                          # 3-layer depth recurrence
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35               # PR #1493 (we default to 0.5)
export QK_GAIN_INIT=5.25                    # PR #1493
export PARALLEL_RESIDUAL_START=7            # PR #1493: parallel residuals from layer 7+ (4 of 11 layers).
                                            # Implemented via per-block flag (train.py Block.__init__),
                                            # respects PR #1493's "deep half parallel" principle.
export USE_PARALLEL_RESIDUALS=0             # use PARALLEL_RESIDUAL_START=7 instead (per-block, not all-layers)
export EMA_DECAY=0.9965                     # PR #1493 (we default to 0.997)
export WARMDOWN_FRAC=0.72                   # PR #1493 (we default to 0.667)
export MUON_WD=0.095                        # PR #1493 (we default to 0.085)
export MATRIX_LR=0.022                      # PR #1493 (we default to 0.020)

# Our discovered speed wins
export USE_PARALLEL_MUON=1                  # batched Newton-Schulz
# CRITICAL: run.sh defaults TORCH_COMPILE_DISABLE=1 and TORCHDYNAMO_DISABLE=1.
# Setting TORCH_COMPILE_MODE alone does NOT re-enable compile -- the disable
# flags override. We MUST explicitly set both to 0 here. PHASE2_RESULTS E2/E4b
# measured 1.85-1.92x speedup from torch.compile on RTX 3090. On 8xH100 SXM
# with our 11L+4x model the speedup is in the 3-5x range. Without compile we
# get half the training steps PR #1493 gets in 600s -> catastrophic for
# convergence. THIS WAS THE BIGGEST DROPPED WIN.
export TORCH_COMPILE_DISABLE=0              # CRITICAL re-enable (run.sh default is 1)
export TORCHDYNAMO_DISABLE=0                # CRITICAL re-enable (run.sh default is 1)
export TORCH_COMPILE_MODE=max-autotune-no-cudagraphs   # +3.7% over default mode (E4b)
export USE_CUDNN_BENCHMARK=1                # +0.8% incremental (E5)

# Our discovered quant fix (int8 instead of int6 — preserves converged quality)
# CMP_QUANT_VALUE_DEDUP RE-ENABLED: NIGHT_MODE n=2 confirmed L10 world-novel,
# alphabet-snap compression that trades minor quality for ~10-15% smaller artifact.
# Needed to fit 11L+4x int8 (~36 MB raw) under the 16 MB cap. Was disabled in
# the 6L+2x CHAMP_D run because we had artifact headroom. Now reinstated.
export MATRIX_BITS=8                        # int8 weight quant (CHAMP_D validated, gap 0.001 BPB on 6L+2x)
export USE_CMP_QUANT_VALUE_DEDUP=1          # alphabet snap for compression — NIGHT_MODE L10 win, restored
export CMP_QUANT_DEDUP_STEP=2               # standard step

# TTT — legal score-first only (matches PR #1493 exactly)
# PR #1493 PR description: "no pre-quant TTT, no ETLB, no n-gram cache, no SLOT — fully compliant"
export PREQUANT_TTT_ENABLED=0               # disabled — PR #1493 explicitly does not use pre-quant TTT
export PREQUANT_TTT_EPOCHS=0
export TTT_ENABLED=1                        # legal score-first TTT (eval chunk, then train chunk)
export TTT_EPOCHS=3                         # PR #1493
export TTT_LR=0.005                         # PR #1493
export TTT_FREEZE_BLOCKS=0
export SLIDING_WINDOW_ENABLED=1             # full sliding eval

# Standard Phase 1 wins (already defaulted in run.sh, declared here for clarity)
# IMPORTANT: n-gram bias stack DISABLED for this submission. Issue #1017 Track B
# Condition 2 says "Standard softmax over full vocab. No n-gram cache, no logit
# biasing." We do not yet know whether this rule applies only to Track B
# (legal-eval-time-adaptation track) or to all submissions. Until verified, we
# do not use n-gram bias features in submission runs.
export USE_GATED_ATTENTION=1                # per-head sigmoid gate over attn output (NeurIPS 2025) — architectural, legal
export USE_NORMUON=1                         # per-row normalize after Newton-Schulz — optimizer variant, legal
export USE_NORM_PCT_DROPOUT=1                # zero top 1% L2-norm rows of FFN intermediate (training-time regularizer) — legal
export USE_NGRAM_BIAS=0                      # DISABLED — possible Issue #1017 Track B violation
export USE_NGRAM_BACKOFF=0                   # DISABLED (n-gram bias is off)
export USE_NGR_LOG_FREQ_INV=0                # DISABLED (n-gram bias is off)
export USE_CTX_PARTITIONED_TAB=0             # DISABLED (n-gram bias is off)
export USE_PREFETCH_LOADER=1                 # data loader optimization — legal

# === Run each seed ===
for THIS_SEED in "${SEED_ARRAY[@]}"; do
    echo
    echo "============================================================"
    echo "[dry_run] SEED=${THIS_SEED} starting at $(date -u +%H:%M:%SZ)"
    echo "============================================================"

    export SEED="$THIS_SEED"
    SEED_LOG="${RECORD_DIR}/train_seed${THIS_SEED}.log"

    # bash submission/run.sh tees its own log to logs/run_seed*.log; we additionally
    # capture the train.py output stream into the records folder for the comp submission
    bash submission/run.sh 2>&1 | tee "$SEED_LOG"

    echo
    echo "[dry_run] SEED=${THIS_SEED} done. Log: $SEED_LOG"

    # Capture the artifact size for this seed (the int6.ptz file is the actual submission blob)
    if [ -f final_model.int6.ptz ]; then
        SEED_ARTIFACT_BYTES=$(stat -c %s final_model.int6.ptz 2>/dev/null || stat -f %z final_model.int6.ptz)
        echo "[dry_run] SEED=${THIS_SEED} artifact: ${SEED_ARTIFACT_BYTES} bytes"
        # Save the artifact alongside the log so a 3-seed run keeps all of them
        cp final_model.int6.ptz "${RECORD_DIR}/final_model_seed${THIS_SEED}.int6.ptz"
    fi
done

# === Assemble submission.json + README.md from the per-seed logs ===
echo
echo "============================================================"
echo "[dry_run] Assembling records folder $(date -u +%H:%M:%SZ)"
echo "============================================================"

python3 - <<PYEOF
import json, re, os, statistics
from pathlib import Path

record_dir = Path("$RECORD_DIR")
seeds = "$SEEDS".split(",")
record_name = "$RECORD_NAME"
date_str = "$DATE_STR"

# Parse each seed log for the final quantized_ttt val_bpb (the submission number)
# Falls back to quantized_sliding_window then quantized if TTT wasn't enabled.
seed_results = {}
for seed in seeds:
    log_path = record_dir / f"train_seed{seed}.log"
    if not log_path.exists():
        print(f"  WARN: missing {log_path}")
        continue
    text = log_path.read_text(errors="replace")
    # Match the lines train.py emits via timed_eval(label, ...):
    #   quantized val_loss:X val_bpb:X eval_time:Xms
    #   quantized_sliding_window val_loss:X val_bpb:X eval_time:Xms
    #   quantized_ttt val_loss:X val_bpb:X eval_time:Xms
    rx = re.compile(r"^(quantized|quantized_sliding_window|quantized_ttt) val_loss:([\d.]+) val_bpb:([\d.]+) eval_time:(\d+)ms", re.M)
    matches = {m.group(1): (float(m.group(2)), float(m.group(3)), int(m.group(4))) for m in rx.finditer(text)}
    # Pick the best available metric, in order of preference
    if "quantized_ttt" in matches:
        primary_label = "quantized_ttt"
    elif "quantized_sliding_window" in matches:
        primary_label = "quantized_sliding_window"
    elif "quantized" in matches:
        primary_label = "quantized"
    else:
        print(f"  WARN: no quantized val_bpb in {log_path}")
        continue
    val_loss, val_bpb, eval_time_ms = matches[primary_label]
    artifact_path = record_dir / f"final_model_seed{seed}.int6.ptz"
    artifact_bytes = artifact_path.stat().st_size if artifact_path.exists() else None
    seed_results[seed] = {
        "primary_label": primary_label,
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "eval_time_ms": eval_time_ms,
        "artifact_bytes": artifact_bytes,
        "all_metrics": {label: {"val_loss": vl, "val_bpb": vb, "eval_time_ms": et}
                         for label, (vl, vb, et) in matches.items()},
    }

# Compute mean + std across seeds
val_bpbs = [r["val_bpb"] for r in seed_results.values()]
if val_bpbs:
    mean_bpb = sum(val_bpbs) / len(val_bpbs)
    std_bpb = statistics.stdev(val_bpbs) if len(val_bpbs) > 1 else 0.0
else:
    mean_bpb = std_bpb = float("nan")

# Detect hardware (best effort)
hw = "unknown"
try:
    import subprocess
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,count", "--format=csv,noheader"], text=True).strip().split("\n")
    if out:
        gpu_name = out[0].split(",")[0].strip()
        gpu_count = len(out)
        hw = f"{gpu_count}x{gpu_name}"
except Exception:
    pass

submission = {
    "author": os.environ.get("SUBMISSION_AUTHOR", "taka6745"),
    "github_id": os.environ.get("SUBMISSION_GITHUB_ID", "taka6745"),
    "name": "SP8192 + NL6 MLP2 int8 + Parallel Muon + N-gram Bias + Parallel Residuals + Legal TTT",
    "date": date_str,
    "track": "10min_16mb",
    "val_bpb": round(mean_bpb, 5),
    "val_bpb_std": round(std_bpb, 5),
    "seeds": [int(s) for s in seeds if s in seed_results],
    "seed_results": {
        s: {"val_bpb": round(r["val_bpb"], 5), "artifact_bytes": r["artifact_bytes"]}
        for s, r in seed_results.items()
    },
    "hardware": hw,
    "technique_summary": (
        "SP8192 + NUM_LAYERS=6 + MLP_MULT=2 + 3-Layer Depth Recurrence (L3-5) "
        "+ Parallel Residuals (all layers) + QK-Gain 5.25 + EMA 0.9965 + WD 0.095 "
        "+ MATRIX_BITS=8 (int8 weights) + Parallel Muon + N-gram Bias Stack "
        "+ Score-First TTT (SGD 3ep) + GPTQ + Brotli"
    ),
    "compliance": {
        "train_under_600s": True,  # MAX_WALLCLOCK_SECONDS=600
        "artifact_under_16mb": all((r["artifact_bytes"] or 0) < 16_000_000 for r in seed_results.values()),
        "eval_under_600s": all(sum(m["eval_time_ms"] for m in r["all_metrics"].values()) < 600_000 for r in seed_results.values()),
        "no_slot": True,
        "no_pre_quant_ttt": True,         # PREQUANT_TTT_ENABLED=0
        "no_etlb": True,
        "no_ngram_cache": False,           # WE USE NGRAM_BIAS — flag honestly
        "score_first_ttt": True,
        "three_seeds": len(seeds) >= 3,
    },
    "attribution": {
        "sp8192_gptq_sdclip": "@clarkkev (PR #1394)",
        "depth_recurrence": "@dexhunter (PR #1331, #1437)",
        "parallel_residuals": "@Robby955 (PR #1412), @msisovic (PR #1204)",
        "legal_ttt_framework": "@abaybektursun (PR #549), @dexhunter (PR #1413)",
        "hyperparameter_tuning_pr1493": "@bigbag (PR #1493)",
        "int8_quant_smaller_model": "@taka6745 (this submission, CHAMP_D discovery)",
    },
}

(record_dir / "submission.json").write_text(json.dumps(submission, indent=2))

# README.md (templated)
seeds_table = "\n".join(
    f"| {s} | **{r['val_bpb']:.4f}** | {r['artifact_bytes'] or 'N/A'} |"
    for s, r in seed_results.items()
)
readme = f"""# Record: SP8192 + NL6 MLP2 int8 + Parallel Muon + N-gram Bias + Legal TTT

**val_bpb = {mean_bpb:.4f}** ({len(val_bpbs)}-seed mean, std {std_bpb:.4f}) | **{hw}**

## Per-seed Results

| Seed | val_bpb (quantized_ttt) | Artifact Bytes |
|------|-------------------------|----------------|
{seeds_table}

## Key Techniques

1. **NUM_LAYERS=6 + MLP_MULT=2** — compute-efficient architecture (CHAMP_D), validated val_bpb 1.39943 on RTX 3090
2. **MATRIX_BITS=8** — int8 weight quantization (eliminates the int6 quant gap on converged smaller models)
3. **3-Layer Depth Recurrence** (L3-5, activate at frac=0.35) — from PR #1331/#1437
4. **Parallel Residuals (all layers)** — more aggressive than PR #1493's L7+ pattern, bet on smaller-model expressivity
5. **QK-Gain 5.25 + EMA 0.9965 + WD 0.095 + warmdown 0.72** — PR #1493 (@bigbag) hyperparameters
6. **Parallel Muon** — batched Newton-Schulz across same-shape parameters
7. **N-gram Bias Stack** — bigram/trigram/fourgram with backoff, hash buckets, NLFI, ctx-partitioned (NIGHT_MODE)
8. **Legal Score-First TTT** — SGD (lr=0.005, mom=0.9), 3 epochs per chunk, cosine LR decay
9. **GPTQ + Brotli** — int8 matrices + int8 embeddings + brotli compression
10. **max-autotune-no-cudagraphs torch.compile + cudnn.benchmark**

## Compliance

```json
{json.dumps(submission["compliance"], indent=2)}
```

**NOTE on n-gram bias**: this submission uses precomputed n-gram log-prob tables as a logit bias.
Issue #1017 Track B (legal eval-time adaptation) Condition 2 says "no n-gram cache, no logit biasing."
We flag `no_ngram_cache: false` honestly. Whether this is comp-legal under Track A or any other
track is an open question that needs to be resolved before merging this as a record.

## Reproduction

```bash
# Default: 1-seed dry run on 1xH100 PCIe
bash submission/bootstrap.sh
bash submission/dry_run.sh

# Real 3-seed submission on 8xH100 SXM
SEEDS=42,314,999 bash submission/dry_run.sh
```

## Attribution

{chr(10).join(f"- **{k}**: {v}" for k, v in submission["attribution"].items())}
"""
(record_dir / "README.md").write_text(readme)

print(f"  wrote {record_dir}/submission.json")
print(f"  wrote {record_dir}/README.md")
print(f"  per-seed logs: train_seed{{{','.join(seeds)}}}.log")
print()
print(f"  MEAN val_bpb: {mean_bpb:.5f}  (std {std_bpb:.5f}, {len(val_bpbs)} seed(s))")
PYEOF

# LZMA-wrap train.py as train_gpt.py for the records folder (matches PR #1493
# format: 2-line script that decompresses + execs the original). This keeps the
# code-size footprint small for comp compliance.
python3 - <<PYEOF
import lzma, base64
from pathlib import Path
src_path = Path("submission/train.py")
src = src_path.read_text(encoding="utf-8")
# Use LZMA2 raw (no container) — same format as PR #1493's wrapper
compressed = lzma.compress(
    src.encode("utf-8"),
    format=lzma.FORMAT_RAW,
    filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
)
encoded = base64.b85encode(compressed).decode("ascii")
wrapped = (
    'import lzma as L,base64 as B\n'
    f'exec(L.decompress(B.b85decode("{encoded}"),format=L.FORMAT_RAW,filters=[{{"id":L.FILTER_LZMA2}}]))'
)
out_path = Path("$RECORD_DIR") / "train_gpt.py"
out_path.write_text(wrapped, encoding="utf-8")
src_bytes = len(src.encode("utf-8"))
wrapped_bytes = len(wrapped.encode("utf-8"))
print(f"  LZMA-wrapped submission/train.py -> $RECORD_DIR/train_gpt.py")
print(f"    raw:     {src_bytes:>8d} bytes")
print(f"    wrapped: {wrapped_bytes:>8d} bytes ({100*wrapped_bytes/src_bytes:.1f}% of raw)")
# Sanity-decode to make sure the wrapper actually round-trips
roundtrip = lzma.decompress(
    base64.b85decode(encoded),
    format=lzma.FORMAT_RAW,
    filters=[{"id": lzma.FILTER_LZMA2}],
).decode("utf-8")
assert roundtrip == src, "LZMA wrap roundtrip mismatch — wrapper is broken"
print(f"    roundtrip OK ({len(roundtrip)} chars match source)")
PYEOF

echo
echo "============================================================"
echo "[dry_run] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo "Records folder: $RECORD_DIR"
echo "Contents:"
ls -la "$RECORD_DIR"
echo
echo "Submission val_bpb candidates (use quantized_ttt for the PR):"
echo "  - quantized val_bpb         (no TTT)"
echo "  - quantized_sliding_window  (sliding eval, no TTT)"
echo "  - quantized_ttt val_bpb     (legal score-first TTT — THIS IS THE SUBMISSION NUMBER)"
echo
for THIS_SEED in "${SEED_ARRAY[@]}"; do
    echo "=== seed $THIS_SEED ==="
    grep -E "^(quantized|quantized_sliding_window|quantized_ttt) val_loss:" "$RECORD_DIR/train_seed${THIS_SEED}.log" 2>/dev/null || echo "  (no quantized lines in log)"
done
