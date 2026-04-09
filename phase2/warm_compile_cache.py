#!/usr/bin/env python3
"""phase2/warm_compile_cache.py — populate the torch.compile / TorchInductor
cache via a SHORT training pass with the real model, so the actual 600s
training run hits the cache and runs with compile enabled for free.

This is the user's "pretime is free" unlock: compile doesn't count toward the
600s budget, only the training loop does. A 5-10 min warmup here buys us the
3-5x compile speedup on every subsequent training run.

Strategy:
- Invoke submission/train.py as a subprocess with:
    TORCH_COMPILE_DISABLE=0
    TORCHDYNAMO_DISABLE=0
    MAX_WALLCLOCK_SECONDS=60         # 60s of actual training (after compile)
    PREQUANT_TTT_ENABLED=0           # skip pre-quant TTT (we don't need it for warmup)
    TTT_ENABLED=0                    # skip eval TTT
    SLIDING_WINDOW_ENABLED=0         # skip slow sliding-window eval
    VAL_LOSS_EVERY=999999            # skip intermediate val
    TRAIN_LOG_EVERY=10               # modest log output
    RUN_ID=phase2_warmup
- Cache lands at ~/.cache/torch/inductor (env TORCHINDUCTOR_CACHE_DIR)
- Subsequent runs with TORCH_COMPILE_DISABLE=0 hit the cache

Safety:
- Max total wallclock (compile + train + eval): 10 min. If it takes longer,
  abort and flag the issue — compile shouldn't be 10+ min on H100.
- Check cache directory exists after the run; if empty, the warmup didn't
  actually populate the cache (likely a code bug) and we should not trust the
  next run to be faster.

Prints structured metrics via phase2.metrics so we can compare warmup vs
real-run step times.

Usage (from phase2/bootstrap.sh or the pod directly):
    python3 phase2/warm_compile_cache.py
or with a different target script:
    WARMUP_TRAIN_SCRIPT=submission/train.py python3 phase2/warm_compile_cache.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Import the metrics helper (relative to phase2/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from metrics import Phase2Metrics
except ImportError:
    Phase2Metrics = None  # type: ignore


REPO_DIR = Path(os.environ.get("REPO_DIR", "/workspace/paramgolf"))
WARMUP_TRAIN_SCRIPT = os.environ.get("WARMUP_TRAIN_SCRIPT", "submission/train.py")
WARMUP_TIMEOUT_SEC = int(os.environ.get("WARMUP_TIMEOUT_SEC", "600"))
WARMUP_MAX_WALLCLOCK = int(os.environ.get("WARMUP_MAX_WALLCLOCK_SECONDS", "60"))
INDUCTOR_CACHE_DIR = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", str(Path.home() / ".cache" / "torch" / "inductor")))


def _count_inductor_artifacts(cache_dir: Path) -> dict:
    """Count files + total bytes in the inductor cache for telemetry."""
    if not cache_dir.exists():
        return {"exists": False, "files": 0, "bytes": 0}
    total_files = 0
    total_bytes = 0
    for root, _dirs, files in os.walk(cache_dir):
        for f in files:
            p = Path(root) / f
            try:
                total_bytes += p.stat().st_size
                total_files += 1
            except OSError:
                pass
    return {"exists": True, "files": total_files, "bytes": total_bytes}


def main() -> int:
    os.chdir(REPO_DIR)

    metrics = None
    if Phase2Metrics is not None:
        metrics = Phase2Metrics(
            run_id="phase2_warmup",
            jsonl_path=str(REPO_DIR / "logs" / "phase2_metrics.jsonl"),
            tag="warmup",
        )
        metrics.mark(
            "warmup_start",
            repo_dir=str(REPO_DIR),
            warmup_script=WARMUP_TRAIN_SCRIPT,
            warmup_wallclock=WARMUP_MAX_WALLCLOCK,
            cache_dir=str(INDUCTOR_CACHE_DIR),
        )

    # Pre-warmup cache state
    pre_state = _count_inductor_artifacts(INDUCTOR_CACHE_DIR)
    print(f"[warmup] pre-warmup inductor cache: {pre_state}", flush=True)
    if metrics is not None:
        metrics.mark("cache_state_pre", **{f"pre_{k}": v for k, v in pre_state.items()})

    # Build the subprocess environment
    env = os.environ.copy()
    env.update(
        {
            "TORCH_COMPILE_DISABLE": "0",
            "TORCHDYNAMO_DISABLE": "0",
            "MAX_WALLCLOCK_SECONDS": str(WARMUP_MAX_WALLCLOCK),
            "PREQUANT_TTT_ENABLED": "0",
            "TTT_ENABLED": "0",
            "SLIDING_WINDOW_ENABLED": "0",
            "VAL_LOSS_EVERY": "999999",
            "TRAIN_LOG_EVERY": "10",
            "RUN_ID": "phase2_warmup",
            # Ensure the cache dir is set even if the user didn't
            "TORCHINDUCTOR_CACHE_DIR": str(INDUCTOR_CACHE_DIR),
            # Inherit all the other Phase 1 / Phase 2 flags from the parent env
        }
    )

    cmd = [sys.executable, "-u", WARMUP_TRAIN_SCRIPT]
    print(f"[warmup] launching: {' '.join(cmd)}", flush=True)
    print(
        f"[warmup] env overrides: TORCH_COMPILE_DISABLE=0 TORCHDYNAMO_DISABLE=0 "
        f"MAX_WALLCLOCK_SECONDS={WARMUP_MAX_WALLCLOCK} "
        f"PREQUANT_TTT_ENABLED=0 TTT_ENABLED=0 SLIDING_WINDOW_ENABLED=0",
        flush=True,
    )
    print(f"[warmup] total timeout: {WARMUP_TIMEOUT_SEC}s", flush=True)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(REPO_DIR),
            timeout=WARMUP_TIMEOUT_SEC,
        )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        print(f"[warmup] TIMEOUT after {WARMUP_TIMEOUT_SEC}s — killing", flush=True)
        rc = 124
    except Exception as e:
        print(f"[warmup] ERROR launching subprocess: {e}", flush=True)
        rc = 1

    elapsed = time.perf_counter() - t0
    print(f"[warmup] done in {elapsed:.1f}s (exit code {rc})", flush=True)

    # Post-warmup cache state
    post_state = _count_inductor_artifacts(INDUCTOR_CACHE_DIR)
    delta_files = post_state["files"] - pre_state["files"]
    delta_bytes = post_state["bytes"] - pre_state["bytes"]
    print(
        f"[warmup] post-warmup inductor cache: {post_state} "
        f"(+{delta_files} files, +{delta_bytes/1024/1024:.1f} MiB)",
        flush=True,
    )

    if metrics is not None:
        metrics.mark(
            "cache_state_post",
            exit_code=rc,
            warmup_elapsed_s=round(elapsed, 1),
            **{f"post_{k}": v for k, v in post_state.items()},
            cache_delta_files=delta_files,
            cache_delta_bytes=delta_bytes,
        )
        metrics.mark("warmup_done", exit_code=rc)
        metrics.print_summary()

    # Success criteria: subprocess exited cleanly AND cache grew
    if rc == 0 and delta_files > 0:
        print(f"[warmup] SUCCESS: cache populated ({delta_files} new files, "
              f"{delta_bytes/1024/1024:.1f} MiB). The real training run should "
              f"hit the cache and skip the compile delay.", flush=True)
        return 0
    elif rc == 0:
        print(f"[warmup] WARNING: subprocess exited 0 but cache didn't grow. "
              f"Inductor may not be writing to {INDUCTOR_CACHE_DIR}. Check "
              f"TORCHINDUCTOR_CACHE_DIR / TORCH_COMPILE_DEBUG env.", flush=True)
        return 0  # non-fatal — continue with the full run anyway
    else:
        print(f"[warmup] FAILED: subprocess exit code {rc}. The real training "
              f"run will hit the same issue.", flush=True)
        return rc


if __name__ == "__main__":
    sys.exit(main())
