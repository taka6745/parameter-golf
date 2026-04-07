#!/usr/bin/env python3
"""
experiment_runner.py — autonomous experiment loop for RunPod cheap-GPU iteration.

Reads runpod_tests/loop/experiments.json (list of {"name": ..., env vars...} dicts).
Picks the experiment with the fewest completed runs (round-robin).
Runs python3 train_gpt.py with the experiment's env vars overlaid on a fast-default base.
Parses train_loss / val_bpb / step_avg from the log and appends one line to
runpod_tests/loop/results.jsonl.

Designed to run forever in `while true`. Stop with: pkill -f experiment_runner.py
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/workspace/paramgolf")
LOOP_DIR = REPO_ROOT / "runpod_tests" / "loop"
EXPERIMENTS_FILE = LOOP_DIR / "experiments.json"
RESULTS_FILE = LOOP_DIR / "results.jsonl"
LOG_DIR = LOOP_DIR / "logs"
LEADERBOARD_FILE = LOOP_DIR / "leaderboard.txt"

# Defaults applied to every experiment unless overridden
BASE_ENV = {
    "TRAIN_SEQ_LEN": "128",
    "TRAIN_BATCH_TOKENS": "1024",
    "VAL_BATCH_SIZE": "131072",
    "VAL_LOSS_EVERY": "0",
    "SKIP_FINAL_EVAL": "1",  # train_loss only — much faster signal
    "WARMUP_STEPS": "10",
    "ITERATIONS": "1000000",
    "MAX_WALLCLOCK_SECONDS": "240",  # 4 min per experiment
    "TRAIN_LOG_EVERY": "100",
    "MODEL_DIM": "512",
    "NUM_LAYERS": "9",
    "MLP_MULT": "2",
    "USE_NGRAM_BIAS": "1",
    "NGRAM_W_BIGRAM": "0.20",
    "NGRAM_W_TRIGRAM": "0.15",
    "NGRAM_W_FOURGRAM": "0.10",
}


def load_experiments() -> list[dict]:
    if not EXPERIMENTS_FILE.exists():
        print(f"FATAL: {EXPERIMENTS_FILE} missing", flush=True)
        sys.exit(1)
    return json.loads(EXPERIMENTS_FILE.read_text())


def load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    out = []
    for line in RESULTS_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def pick_next(experiments: list[dict], results: list[dict]) -> dict:
    """Pick the experiment with the fewest completed (non-crashed) runs."""
    counts: dict[str, int] = {}
    for r in results:
        if r.get("exit_code") == 0 and r.get("train_loss") is not None:
            counts[r["name"]] = counts.get(r["name"], 0) + 1
    indexed = list(enumerate(experiments))
    indexed.sort(key=lambda iv: (counts.get(iv[1]["name"], 0), iv[0]))
    return indexed[0][1]


def parse_log(log_text: str) -> dict:
    out = {
        "val_bpb": None,
        "train_loss": None,
        "max_step": 0,
        "ms_step": None,
        "ngram_loaded": False,
    }
    m = re.search(r"final_int8_zlib_roundtrip val_loss:[\d.]+ val_bpb:([\d.]+)", log_text)
    if m:
        out["val_bpb"] = float(m.group(1))
    matches = re.findall(r"step:(\d+)/\d+ train_loss:([\d.]+)\b.*?step_avg:([\d.]+)", log_text)
    if matches:
        last = matches[-1]
        out["max_step"] = int(last[0])
        out["train_loss"] = float(last[1])
        out["ms_step"] = float(last[2])
    if "NGRAM_BIAS: loaded bigram" in log_text:
        out["ngram_loaded"] = True
    return out


def run_experiment(exp: dict) -> dict:
    name = exp["name"]
    env_overrides = {k: str(v) for k, v in exp.items() if k != "name"}

    env = os.environ.copy()
    for k, v in BASE_ENV.items():
        env[k] = v
    for k, v in env_overrides.items():
        env[k] = v

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    log_path = LOG_DIR / f"{name}_{ts}.log"

    print(f"=== {time.strftime('%H:%M:%S')} START {name} ===", flush=True)
    print(f"  overrides: {env_overrides}", flush=True)

    t0 = time.time()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w") as logf:
        proc = subprocess.run(
            ["python3", "-u", "train_gpt.py"],
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
    duration = int(time.time() - t0)

    parsed = parse_log(log_path.read_text(errors="replace"))

    result = {
        "name": name,
        "timestamp": ts,
        "duration_s": duration,
        "exit_code": proc.returncode,
        "crashed": proc.returncode != 0,
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "env_overrides": env_overrides,
        **parsed,
    }
    return result


def append_result(result: dict) -> None:
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(result) + "\n")


def update_leaderboard(results: list[dict]) -> None:
    by_name: dict[str, list[dict]] = {}
    for r in results:
        if r.get("exit_code") != 0 or r.get("train_loss") is None:
            continue
        by_name.setdefault(r["name"], []).append(r)

    rows = []
    for name, runs in by_name.items():
        losses = [r["train_loss"] for r in runs if r["train_loss"] is not None]
        if not losses:
            continue
        best = min(losses)
        avg = sum(losses) / len(losses)
        steps = [r.get("max_step", 0) for r in runs]
        ms = [r.get("ms_step") for r in runs if r.get("ms_step")]
        rows.append((best, avg, name, len(runs), max(steps), (sum(ms) / len(ms)) if ms else 0))

    rows.sort(key=lambda r: r[0])
    lines = [f"# Leaderboard — updated {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]
    lines.append(f"{'name':<40} {'best_tl':>9} {'avg_tl':>9} {'runs':>5} {'max_step':>9} {'ms/step':>9}\n")
    lines.append("-" * 90 + "\n")
    for best, avg, name, n, mxs, msa in rows[:50]:
        lines.append(f"{name:<40} {best:>9.4f} {avg:>9.4f} {n:>5} {mxs:>9} {msa:>9.1f}\n")
    LEADERBOARD_FILE.write_text("".join(lines))


def main() -> None:
    print(f"=== experiment_runner.py STARTED at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    print(f"REPO_ROOT={REPO_ROOT}", flush=True)
    print(f"EXPERIMENTS_FILE={EXPERIMENTS_FILE}", flush=True)
    print(f"RESULTS_FILE={RESULTS_FILE}", flush=True)
    while True:
        # Auto-pull experiments.json from git so live edits to the queue propagate
        # without requiring a runner restart. --autostash because the patcher
        # modifies train_gpt.py locally.
        try:
            subprocess.run(
                ["git", "pull", "--rebase", "--autostash"],
                cwd=str(REPO_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        except Exception as e:
            print(f"  (git pull failed: {e})", flush=True)
        experiments = load_experiments()
        results = load_results()
        exp = pick_next(experiments, results)
        result = run_experiment(exp)
        append_result(result)
        update_leaderboard(load_results())
        bpb = result.get("val_bpb")
        tl = result.get("train_loss")
        print(
            f"=== {time.strftime('%H:%M:%S')} DONE  {exp['name']} "
            f"train_loss={tl} val_bpb={bpb} steps={result['max_step']} "
            f"ms/step={result.get('ms_step')} crashed={result['crashed']} ===",
            flush=True,
        )
        time.sleep(2)


if __name__ == "__main__":
    main()
