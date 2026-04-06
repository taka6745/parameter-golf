#!/bin/bash
# results.sh — Aggregate ALL training runs from validate/ + unknown/ into JSON
# Usage: ./results.sh
#
# Scans logs/ recursively for per-test training logs and extracts:
#   - val_bpb (final + per-step)
#   - val_loss
#   - train_loss curve (every step we logged)
#   - step_avg (ms/step)
#   - tokens/sec
#   - GPU
#   - architecture config (layers, heads, dims, batch, etc.)
#   - artifact size in bytes
#
# Output:
#   logs/results.json     full per-run data, sorted by val_bpb
#   stdout                top-20 leaderboard table

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

python3 << 'PYEOF'
import os
import re
import json
from pathlib import Path

LOG_DIR = Path("logs")

# Files to EXCLUDE — these are runner summaries that contain multiple
# tests concatenated. Parsing them as a single run gives wrong numbers.
EXCLUDE = {"setup.log", "validate.log", "unknown.log", "results.json"}

# Categorize each log file by phase
def categorize(path: Path) -> str:
    parts = path.parts
    name = path.name
    if name.startswith("v"):
        return "validate"
    if "u01" in parts or name.startswith("u01") or "config_" in name:
        return "u01_arch_sweep"
    if "u02" in parts or name.startswith("u02") or "progressive" in name:
        return "u02_progressive_seq"
    if "u03" in parts or name.startswith("u03"):
        return "u03_eval_cache"
    if "u04" in parts or name.startswith("u04"):
        return "u04_full_stack"
    if "u05" in parts or name.startswith("u05") or "seed_" in name:
        return "u05_3seed_final"
    if "u06" in parts or name.startswith("u06"):
        return "u06_speed_baseline"
    if "u07" in parts or name.startswith("u07"):
        return "u07_gla_shootout"
    if "u08" in parts or name.startswith("u08"):
        return "u08_gla_progressive"
    if "u09" in parts or name.startswith("u09"):
        return "u09_continual_lowlr"
    if "u10" in parts or name.startswith("u10"):
        return "u10_eval_temp_alpha"
    return "other"


# Find all candidate logs
log_files = sorted(LOG_DIR.glob("**/*.log"))
log_files += sorted(LOG_DIR.glob("**/*.txt"))

runs = []
skipped_summary = []
skipped_empty = []

for log_path in log_files:
    rel = log_path.relative_to(LOG_DIR)

    # Skip runner summary files
    if log_path.name in EXCLUDE:
        skipped_summary.append(str(rel))
        continue

    try:
        with open(log_path, "r", errors="ignore") as f:
            content = f.read()
    except Exception:
        continue

    # Must contain at least one training step
    if not re.search(r"step:\d+/\d+ train_loss:", content):
        skipped_empty.append(str(rel))
        continue

    run = {
        "file": str(rel),
        "phase": categorize(log_path),
        "size_kb": round(log_path.stat().st_size / 1024, 1),
    }

    # Architecture / config (lines like "key:value" near the top of train_gpt.py output)
    for key in [
        "model_params", "world_size", "grad_accum_steps", "num_heads",
        "num_kv_heads", "tie_embeddings", "embed_lr", "matrix_lr",
        "train_batch_tokens", "train_seq_len", "iterations",
        "max_wallclock_seconds", "seed",
    ]:
        m = re.search(rf"^{key}:([^\s]+)", content, flags=re.MULTILINE)
        if m:
            run[key] = m.group(1)

    # Train loss curve (every logged step)
    train_steps = re.findall(r"step:(\d+)/\d+ train_loss:([\d.]+)", content)
    if train_steps:
        run["train_loss_curve"] = [
            {"step": int(s), "loss": float(l)} for s, l in train_steps
        ]
        run["initial_train_loss"] = float(train_steps[0][1])
        run["final_train_loss"] = float(train_steps[-1][1])
        run["max_step"] = int(train_steps[-1][0])

    # Step time (last logged step_avg)
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
    artifact = re.search(r"Total submission size int8\+zlib:\s*(\d+)", content)
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

# Group runs by phase for the summary
by_phase = {}
for r in runs:
    by_phase.setdefault(r["phase"], []).append(r)

out = {
    "n_runs": len(runs),
    "phases_present": sorted(by_phase.keys()),
    "skipped_summary_files": skipped_summary,
    "skipped_no_training": skipped_empty[:10],  # cap for brevity
    "runs": runs,
}

with open("logs/results.json", "w") as f:
    json.dump(out, f, indent=2)

# Print summary
print()
print("=" * 100)
print(f"AGGREGATED {len(runs)} TRAINING RUNS")
print("=" * 100)
print()
print(f"Skipped runner summaries: {len(skipped_summary)} ({', '.join(skipped_summary[:3])}...)")
print(f"Skipped non-training logs: {len(skipped_empty)}")
print()
print("Runs by phase:")
for phase in sorted(by_phase.keys()):
    n = len(by_phase[phase])
    best = min((r.get("final_int8_val_bpb") or r.get("val_bpb") or float("inf")) for r in by_phase[phase])
    best_str = f"{best:.4f}" if best != float("inf") else "—"
    print(f"  {phase:25s} {n:3d} runs   best val_bpb: {best_str}")
print()

print("=" * 100)
print("LEADERBOARD (top 20 by val_bpb)")
print("=" * 100)
print(f"{'#':>3}  {'phase':<22} {'file':<35} {'val_bpb':>10} {'final_int8':>12} {'steps':>7} {'ms':>7}")
print("-" * 100)
for i, r in enumerate(runs[:20], 1):
    name = r["file"][:33]
    phase = r["phase"][:20]
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
    print(f"{i:3d}  {phase:<22} {name:<35} {str(val):>10} {str(final):>12} {str(steps):>7} {str(ms):>7}")
print()
print(f"Saved: logs/results.json ({os.path.getsize('logs/results.json')} bytes)")
print()
print("Tip: pipe to jq for analysis:")
print("  jq '.runs[] | {file, phase, val_bpb, max_step, step_avg_ms}' logs/results.json")
print("  jq '.runs | group_by(.phase) | map({phase: .[0].phase, count: length})' logs/results.json")
PYEOF
