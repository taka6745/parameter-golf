#!/bin/bash
# results.sh — Aggregate all training runs into a single JSON for analysis
# Usage: ./results.sh                       # creates runpod_tests/logs/results.json
#        ./results.sh > /tmp/results.json   # write to a different file
#
# Scans logs/ for all train_*.log files and extracts:
#   - val_bpb (final + per-step)
#   - val_loss
#   - train_loss curve
#   - step_avg (ms/step)
#   - tokens/sec
#   - GPU
#   - all env vars set at run time (from the head of the log)
#
# Output: JSON, one entry per run, easy to grep / pipe to jq

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

python3 << 'PYEOF'
import os
import re
import json
import glob
from pathlib import Path

LOG_DIR = Path("logs")
runs = []

# Scan all .log files in logs/ and subdirectories
log_files = sorted(LOG_DIR.glob("**/*.log"))
log_files += sorted(LOG_DIR.glob("**/*.txt"))

for log_path in log_files:
    rel = log_path.relative_to(LOG_DIR)
    try:
        with open(log_path, "r", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        continue

    if "train_loss" not in content and "val_bpb" not in content:
        continue  # not a training log

    run = {
        "file": str(rel),
        "size_kb": round(log_path.stat().st_size / 1024, 1),
    }

    # Extract config (lines like "key:value" near the top)
    # Most train_gpt.py runs print: model_params, world_size, num_heads, etc.
    for key in [
        "model_params", "world_size", "grad_accum_steps", "num_heads",
        "num_kv_heads", "tie_embeddings", "embed_lr", "matrix_lr",
        "train_batch_tokens", "train_seq_len", "iterations",
        "max_wallclock_seconds", "seed",
    ]:
        m = re.search(rf"{key}:([^\s]+)", content)
        if m:
            run[key] = m.group(1)

    # Train loss curve
    train_steps = re.findall(r"step:(\d+)/\d+ train_loss:([\d.]+)", content)
    if train_steps:
        run["train_loss_curve"] = [
            {"step": int(s), "loss": float(l)} for s, l in train_steps
        ]
        run["final_train_loss"] = float(train_steps[-1][1])
        run["max_step"] = int(train_steps[-1][0])

    # Step time
    step_times = re.findall(r"step_avg:([\d.]+)ms", content)
    if step_times:
        run["step_avg_ms"] = float(step_times[-1])

    tok_s = re.findall(r"tok_s:(\d+)", content)
    if tok_s:
        run["tokens_per_sec"] = int(tok_s[-1])

    # Validation
    val_match = re.search(r"step:\d+/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)", content)
    if val_match:
        run["val_loss"] = float(val_match.group(1))
        run["val_bpb"] = float(val_match.group(2))

    # Final quantized
    final_match = re.search(
        r"final_int8_zlib_roundtrip val_loss:([\d.]+) val_bpb:([\d.]+)", content
    )
    if final_match:
        run["final_int8_val_loss"] = float(final_match.group(1))
        run["final_int8_val_bpb"] = float(final_match.group(2))

    # Artifact size
    artifact = re.search(r"Total submission size int8\+zlib: (\d+)", content)
    if artifact:
        run["artifact_bytes"] = int(artifact.group(1))
        run["artifact_mb"] = round(int(artifact.group(1)) / 1024 / 1024, 2)

    # GPU detection
    gpu = re.search(r"Device: (NVIDIA[^\n]+)", content)
    if gpu:
        run["gpu"] = gpu.group(1).strip()

    runs.append(run)

# Sort by best val_bpb (final_int8 if available, else val_bpb)
def sort_key(r):
    return r.get("final_int8_val_bpb") or r.get("val_bpb") or float("inf")

runs.sort(key=sort_key)

# Write JSON
out = {
    "n_runs": len(runs),
    "runs": runs,
}

output_file = "logs/results.json"
with open(output_file, "w") as f:
    json.dump(out, f, indent=2)

# Print summary table
print(f"Aggregated {len(runs)} training runs")
print(f"Saved: {output_file}")
print()
print("=" * 100)
print(f"{'File':<40} {'val_bpb':>10} {'final_int8':>12} {'steps':>8} {'ms/step':>10} {'GPU':>20}")
print("=" * 100)
for r in runs[:20]:
    name = r["file"][:38]
    val = r.get("val_bpb", "—")
    if isinstance(val, float):
        val = f"{val:.4f}"
    final = r.get("final_int8_val_bpb", "—")
    if isinstance(final, float):
        final = f"{final:.4f}"
    steps = r.get("max_step", "—")
    ms = r.get("step_avg_ms", "—")
    if isinstance(ms, float):
        ms = f"{ms:.1f}"
    gpu = (r.get("gpu", "—") or "—")[:18]
    print(f"{name:<40} {str(val):>10} {str(final):>12} {str(steps):>8} {str(ms):>10} {gpu:>20}")
print()
PYEOF
