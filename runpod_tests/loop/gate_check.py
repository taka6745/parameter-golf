#!/usr/bin/env python3
"""
gate_check.py — performance gate evaluator for the stack-novelty campaign.

Reads results.jsonl + gpu_util.log + train_gpt.py + experiments.json and rewrites
Section F of STACK_NOVELTY_TRACKER.md with the current state of the 5 gates:

  G1: tokens_per_min — all training data seen (per-hardware floor)
  G2: gpu_idle_streak — full 10 minutes used (no idle)
  G3: artifact_bytes — full 16 MB used (compression slack)
  G4: marker_count — patcher integrity (26 markers expected)
  G5: queue_depth — every cheap pod has >=1 pending experiment (PD1 saturation)

Pure stdlib so it runs anywhere. Designed to be called from:
  - bash run_forever.sh (per-cycle G1/G2/G4 update)
  - the C5 monitor RemoteTrigger (cross-pod G5 + sync to tracker)
  - smoke test (`python3 gate_check.py --dry-run`)

Usage:
  python3 runpod_tests/loop/gate_check.py [--dry-run] [--repo-root <path>] [--pod-id <id>]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Markers expected to be present in train_gpt.py after 08_patch_train_gpt.sh runs.
# This list is the source of truth for Gate G4. The 08_patch script appends a
# python integrity check that uses this same list (kept in sync by hand).
EXPECTED_MARKERS = [
    "ASYMMETRIC_SKIP_INIT_MARKER",
    "ASYM_LABEL_SMOOTHING_MARKER",
    "BYTE_WEIGHT_MARKER",
    "CMP_HESSIAN_BIT_BUDGET_MARKER",
    "CMP_QUANT_VALUE_DEDUP_MARKER",
    "COPRIME_PER_HEAD_ROPE_MARKER",
    "COPRIME_STRIDE_MARKER",
    "CTX_PARTITIONED_TAB_MARKER",
    "DAT_BYTE_ENTROPY_CURRICULUM_MARKER",
    "DEPTH_RECUR_MARKER",
    "DYN_LYAPUNOV_CLIP_MARKER",
    "EMB_DCT_COEFFICIENT_ENERGY_TRUNCATE_MARKER",
    "ENGRAM_LITE_MARKER",
    "ENTROPY_ADAPTIVE_NGRAM_MARKER",
    "GATED_ATTENTION_MARKER",
    "KER_TMA_MEGAKERNEL_MARKER",
    "LEAKY_RELU_MARKER",
    "LN_SCALE_MARKER",
    "MOUSSE_MARKER",
    "MTP_MARKER",
    "MUONEQ_R_MARKER",
    "NGRAM_BIAS_MARKER",
    "NGR_LOG_FREQ_INV_MARKER",
    "NGRAM_GATE_MARKER",
    "NORM_PCT_DROPOUT_MARKER",
    "NORMUON_MARKER",
    "NS_STEPS_MARKER",
    "OPT_CHEBYSHEV_NS_MARKER",
    "PARALLEL_RESIDUALS_MARKER",
    "PARTIAL_ROPE_MARKER",
    "PER_PROJ_LR_SPLIT_MARKER",
    "PHASE_TRANSITION_MARKER",
    "PROG_SEQ_INIT_MARKER",
    "SKIP_FINAL_EVAL_MARKER",
    "SKIP_LAST_VAL_MARKER",
    "SKIP_POST_LOOP_MARKER",
    "SMEAR_GATE_MARKER",
    "SPD_NGRAM_TILE_CACHE_MARKER",
    "SPD_PINNED_PREFETCH_MARKER",
    "TABULATION_HASH_MARKER",
    "TOK_INPUT_SMOOTH_MARKER",
    "WAVELET_GPT_MARKER",
    "WEIGHT_EMA_SWA_MARKER",
    "XSA_MARKER",
]

# Per-hardware throughput floors for Gate G1 (tokens per minute).
HARDWARE_FLOORS = {
    "RTX3080Ti": 12_500_000,
    "RTX3090": 15_000_000,
    "RTX4070Ti": 12_500_000,
    "H100": 100_000_000,  # 8xH100 aggregate target / 8
    "DEFAULT": 12_500_000,
}

# Gate G3: 16 MB artifact size in bytes minus 0.5 MB slack tolerance.
ARTIFACT_FLOOR_BYTES = 16 * 1024 * 1024 - 512 * 1024  # 16,252,928
ARTIFACT_CEIL_BYTES = 16 * 1024 * 1024                # 16,777,216


def now_utc() -> str:
    """Format current time as YYYYMMDDTHHMMZ."""
    return time.strftime("%Y%m%dT%H%MZ", time.gmtime())


def read_text(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def gate_g1(repo_root: Path, pod_hardware: str) -> dict:
    """G1: parse results.jsonl, compute mean tokens_per_min from the last 5 successful runs."""
    results_file = repo_root / "runpod_tests" / "loop" / "results.jsonl"
    if not results_file.exists():
        return {"value": "0 tok/min", "state": "UNKNOWN", "_raw_tokens_per_min": 0}

    runs = []
    for line in results_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("exit_code") != 0:
            continue
        ms_step = r.get("ms_step")
        env = r.get("env_overrides", {})
        batch_tok = int(env.get("TRAIN_BATCH_TOKENS", "65536"))
        if not ms_step or ms_step <= 0:
            continue
        tpm = batch_tok * 60_000.0 / ms_step
        runs.append(tpm)

    if not runs:
        return {"value": "0 tok/min", "state": "UNKNOWN", "_raw_tokens_per_min": 0}

    last5 = runs[-5:]
    mean_tpm = sum(last5) / len(last5)
    floor = HARDWARE_FLOORS.get(pod_hardware, HARDWARE_FLOORS["DEFAULT"])
    state = "PASS" if mean_tpm >= floor else "FAIL"
    return {
        "value": f"{mean_tpm/1e6:.1f}M tok/min",
        "state": state,
        "_raw_tokens_per_min": int(mean_tpm),
    }


def gate_g2(repo_root: Path) -> dict:
    """G2: parse gpu_util.log, find util<80% streaks longer than 5 s (1 sample = 5 s)."""
    log = repo_root / "runpod_tests" / "loop" / "gpu_util.log"
    if not log.exists():
        return {"value": "no log yet", "state": "UNKNOWN"}

    lines = log.read_text().splitlines()
    # Take only the last 720 samples (1 hour at 5 s cadence) to avoid scanning
    # the whole history every fire.
    lines = lines[-720:]

    streaks = 0
    current = 0
    max_streak = 0
    for line in lines:
        # Format: "97 %, 21678 MiB" or "  97 %,   21678 MiB"
        m = re.match(r"\s*(\d+)\s*%", line)
        if not m:
            continue
        util = int(m.group(1))
        if util < 80:
            current += 1
            max_streak = max(max_streak, current)
        else:
            if current > 1:  # >1 sample = >5 s
                streaks += 1
            current = 0
    if current > 1:
        streaks += 1

    state = "PASS" if streaks == 0 else "FAIL"
    return {
        "value": f"{streaks} streaks, max={max_streak} samples",
        "state": state,
    }


def gate_g3(repo_root: Path) -> dict:
    """G3: artifact size from latest .int8.ptz under runpod_tests/loop/logs/, fall back to UNKNOWN."""
    logs = repo_root / "runpod_tests" / "loop" / "logs"
    if not logs.exists():
        return {"value": "no logs", "state": "UNKNOWN"}

    candidates = list(logs.glob("*.int8.ptz")) + list(logs.glob("*.ptz"))
    if not candidates:
        # Try parsing the log for `final_int8_zlib_roundtrip ... size_bytes:N`
        for log in sorted(logs.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
            text = log.read_text(errors="replace")
            m = re.search(r"final_int8_zlib_roundtrip.*?size_bytes:(\d+)", text)
            if m:
                size = int(m.group(1))
                state = "PASS" if ARTIFACT_FLOOR_BYTES <= size <= ARTIFACT_CEIL_BYTES else (
                    "FAIL" if size > ARTIFACT_CEIL_BYTES else "SLACK"
                )
                return {"value": f"{size:,} B (from log)", "state": state}
        return {"value": "no artifact yet", "state": "UNKNOWN"}

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    size = latest.stat().st_size
    if size > ARTIFACT_CEIL_BYTES:
        state = "FAIL"
    elif size >= ARTIFACT_FLOOR_BYTES:
        state = "PASS"
    else:
        state = "SLACK"  # under 16 MB → compression slack to spend
    return {"value": f"{size:,} B ({latest.name})", "state": state}


def gate_g4(repo_root: Path) -> dict:
    """G4: count expected markers present in train_gpt.py."""
    src_file = repo_root / "train_gpt.py"
    src = read_text(src_file)
    if not src:
        return {"value": "train_gpt.py missing", "state": "FAIL"}
    present = [m for m in EXPECTED_MARKERS if m in src]
    missing = [m for m in EXPECTED_MARKERS if m not in src]
    state = "PASS" if not missing else "FAIL"
    return {
        "value": f"{len(present)}/{len(EXPECTED_MARKERS)}"
        + (f" missing={missing[:3]}..." if missing else ""),
        "state": state,
    }


def gate_g5(repo_root: Path, pod_id) -> dict:
    """G5: queue depth — count pending experiments matching this pod's filter.

    If pod_id is None, computes the worst-case across all 7 pods (A..G).
    """
    exp_file = repo_root / "runpod_tests" / "loop" / "experiments.json"
    res_file = repo_root / "runpod_tests" / "loop" / "results.jsonl"
    if not exp_file.exists():
        return {"value": "no experiments.json", "state": "FAIL"}

    try:
        experiments = json.loads(exp_file.read_text())
    except json.JSONDecodeError:
        return {"value": "experiments.json malformed", "state": "FAIL"}

    # Build per-experiment attempt count from results.jsonl
    attempts = {}
    if res_file.exists():
        for line in res_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            attempts[r.get("name", "")] = attempts.get(r.get("name", ""), 0) + 1

    pods_to_check = [pod_id] if pod_id else ["A", "B", "C", "D", "E", "F", "G"]
    per_pod_pending = {}
    for pid in pods_to_check:
        eligible = []
        for exp in experiments:
            pf = exp.get("pod_filter")
            if pf and pid not in pf:
                continue
            # "pending" = experiment that this pod hasn't yet completed
            if attempts.get(exp.get("name", ""), 0) < 1:
                eligible.append(exp.get("name", ""))
        per_pod_pending[pid] = len(eligible)

    worst = min(per_pod_pending.values()) if per_pod_pending else 0
    state = "PASS" if worst >= 1 else "FAIL"
    return {
        "value": f"min={worst} pending, per_pod={per_pod_pending}",
        "state": state,
    }


def render_section_f(gates: dict, prior_red_flags: dict) -> str:
    """Render a fresh Section F table from the 5 gate results."""
    ts = now_utc()
    rows = [
        ("G1_tokens_per_min", ">=12.5M (3080Ti) / >=15M (3090)", gates["G1"]),
        ("G2_gpu_idle_streak", "0 streaks >5s util<80%", gates["G2"]),
        ("G3_artifact_bytes", ">=16,252,928 B (16MB-0.5MB)", gates["G3"]),
        ("G4_marker_count", f"{len(EXPECTED_MARKERS)}/{len(EXPECTED_MARKERS)} expected", gates["G4"]),
        ("G5_queue_depth", "every pod >=1 pending", gates["G5"]),
    ]
    out = [
        "## Section F — Performance gate status",
        "",
        "| gate | last_checked_utc | last_value | threshold | state | red_flag_ct |",
        "|---|---|---|---|---|---|",
    ]
    for gate_name, threshold, result in rows:
        ct = prior_red_flags.get(gate_name, 0)
        # Increment red_flag_ct on PASS->FAIL transitions only (caller tracks history).
        if result.get("state") == "FAIL":
            ct += 1
        out.append(
            f"| {gate_name} | {ts} | {result.get('value', '')} | {threshold} | {result.get('state', 'UNKNOWN')} | {ct} |"
        )
    out.append("")
    return "\n".join(out)


def update_tracker(repo_root: Path, section_f: str) -> None:
    """Replace Section F of STACK_NOVELTY_TRACKER.md with the new content."""
    tracker = repo_root / "STACK_NOVELTY_TRACKER.md"
    if not tracker.exists():
        print(f"WARN: {tracker} missing, skipping update", file=sys.stderr)
        return

    text = tracker.read_text()
    # Find the boundaries of Section F. It starts at "## Section F" and ends at
    # the next "---" or EOF.
    pattern = re.compile(
        r"## Section F[^\n]*\n.*?(?=\n---\n|\Z)",
        re.DOTALL,
    )
    if pattern.search(text):
        new_text = pattern.sub(section_f.rstrip() + "\n", text)
    else:
        # Append if not present
        new_text = text.rstrip() + "\n\n---\n\n" + section_f
    tracker.write_text(new_text)


def detect_pod_hardware(repo_root: Path) -> str:
    """Best-effort hardware detection from pod_id.txt + a small lookup."""
    pod_id_file = repo_root / "runpod_tests" / "loop" / "pod_id.txt"
    pod_id = pod_id_file.read_text().strip() if pod_id_file.exists() else "DEFAULT"

    pod_to_hw = {
        "A": "RTX3080Ti",
        "B": "RTX3090",
        "C": "RTX3090",
        "D": "RTX3090",
        "E": "RTX3090",
        "F": "RTX3090",
        "G": "RTX4070Ti",
    }
    return pod_to_hw.get(pod_id, "DEFAULT")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=None)
    p.add_argument("--pod-id", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="print results to stdout, don't write the tracker")
    args = p.parse_args()

    repo_root = Path(args.repo_root or os.environ.get("REPO_ROOT") or Path(__file__).resolve().parents[2])
    pod_id = args.pod_id or os.environ.get("POD_ID")
    pod_hardware = detect_pod_hardware(repo_root)

    gates = {
        "G1": gate_g1(repo_root, pod_hardware),
        "G2": gate_g2(repo_root),
        "G3": gate_g3(repo_root),
        "G4": gate_g4(repo_root),
        "G5": gate_g5(repo_root, pod_id),
    }

    section_f = render_section_f(gates, prior_red_flags={})

    if args.dry_run:
        print(f"# gate_check.py dry-run @ {now_utc()}")
        print(f"# repo_root={repo_root}  pod_id={pod_id}  pod_hardware={pod_hardware}")
        print()
        print(section_f)
        # Exit code reflects worst gate state for shell pipelines:
        # 0 = all PASS or UNKNOWN, 1 = any SLACK, 2 = any FAIL
        states = [g.get("state") for g in gates.values()]
        if "FAIL" in states:
            return 2
        if "SLACK" in states:
            return 1
        return 0

    update_tracker(repo_root, section_f)
    print(f"gate_check.py: wrote Section F to STACK_NOVELTY_TRACKER.md @ {now_utc()}")
    states = [g.get("state") for g in gates.values()]
    if "FAIL" in states:
        return 2
    if "SLACK" in states:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
