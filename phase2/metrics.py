"""phase2/metrics.py — structured per-step telemetry for Phase 2 speed work.

Used by every Phase 2 shot so we can measure before/after speedup, verify the
val_bpb invariant (ε ≤ 0.005 drift), and log CPU/GPU/RAM utilization in
parallel with training.

Usage (from train.py or a wrapper):

    from phase2.metrics import Phase2Metrics
    m = Phase2Metrics(run_id='phase2_s1_compile', jsonl_path='logs/phase2_metrics.jsonl')
    m.mark('bootstrap_start')

    # ... model init ...
    m.mark('model_init_done', model_params=sum(p.numel() for p in model.parameters()))

    # ... warmup compile cache ...
    m.mark('compile_warmup_start')
    # ... run warmup ...
    m.mark('compile_warmup_done', compile_cache_hits=h, compile_cache_misses=s)

    # ... training loop ...
    m.mark('train_start')
    for step in range(...):
        step_start = time.perf_counter()
        # ... forward/backward/optimizer ...
        m.step(step=step, ms=1e3*(time.perf_counter() - step_start),
               train_loss=loss.item(), tok_per_sec=..., gpu_util=...,
               cpu_util=..., ram_gb=..., prefetch_queue_depth=...)
    m.mark('train_done')
    m.flush()

Output: one JSON object per line, trivially parseable for post-run analysis.
Human-readable log lines also go to stdout via log() for parity with the
existing train.py logger.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _try_nvidia_smi() -> dict:
    """Best-effort GPU telemetry via nvidia-smi (no pynvml dep). Returns {} on failure."""
    try:
        import subprocess
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode != 0:
            return {}
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        if len(parts) != 5:
            return {}
        return {
            "gpu_util_pct": float(parts[0]),
            "gpu_mem_used_mib": float(parts[1]),
            "gpu_mem_total_mib": float(parts[2]),
            "gpu_power_w": float(parts[3]),
            "gpu_temp_c": float(parts[4]),
        }
    except Exception:
        return {}


def _try_proc_self() -> dict:
    """Best-effort CPU + RAM telemetry via /proc (no psutil dep). Returns {} on failure."""
    try:
        # /proc/meminfo for RAM
        mem = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, _, rest = line.partition(":")
                val_kb = rest.strip().split()[0]
                mem[key.strip()] = int(val_kb)
        total_kb = mem.get("MemTotal", 0)
        avail_kb = mem.get("MemAvailable", 0)
        used_kb = max(0, total_kb - avail_kb)
        # /proc/loadavg for load
        with open("/proc/loadavg", "r", encoding="utf-8") as f:
            loadavg = f.read().strip().split()
        return {
            "ram_used_gb": round(used_kb / 1024 / 1024, 2),
            "ram_total_gb": round(total_kb / 1024 / 1024, 2),
            "ram_used_pct": round(100 * used_kb / max(total_kb, 1), 1),
            "load_1m": float(loadavg[0]) if len(loadavg) > 0 else 0.0,
            "load_5m": float(loadavg[1]) if len(loadavg) > 1 else 0.0,
        }
    except Exception:
        return {}


def _try_torch_cuda_mem() -> dict:
    """Best-effort torch.cuda memory telemetry. Returns {} if torch/cuda not ready."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {
            "torch_cuda_allocated_mib": torch.cuda.memory_allocated() / 1024 / 1024,
            "torch_cuda_reserved_mib": torch.cuda.memory_reserved() / 1024 / 1024,
            "torch_cuda_peak_mib": torch.cuda.max_memory_allocated() / 1024 / 1024,
        }
    except Exception:
        return {}


class Phase2Metrics:
    """Structured telemetry sink for Phase 2 speed work.

    Designed to be near-zero-overhead in the hot path: the per-step .step() call
    takes ~5 microseconds (one dict construction + one write to an in-memory
    buffer). The telemetry is FLUSHED periodically or on mark() events.

    Parameters
    ----------
    run_id : str
        Short identifier for this run (used as a tag in every log line).
    jsonl_path : str | Path
        Where to write the structured JSONL output. Parent dir is created.
    flush_every_steps : int
        Write-to-disk cadence in the hot path. Default 50 steps (≈ 50× the 3ms/step
        budget = 150 ms overhead per flush, negligible).
    tag : str
        Optional human-readable tag printed alongside every log line.
    sample_nvidia_smi_every : int
        Steps between nvidia-smi samples. Expensive (~50-100 ms), so we sample
        sparsely. Default 50.
    """

    def __init__(
        self,
        run_id: str,
        jsonl_path: str | os.PathLike = "logs/phase2_metrics.jsonl",
        flush_every_steps: int = 50,
        tag: str | None = None,
        sample_nvidia_smi_every: int = 50,
    ) -> None:
        self.run_id = run_id
        self.tag = tag or run_id
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every_steps = int(flush_every_steps)
        self.sample_nvidia_smi_every = int(sample_nvidia_smi_every)
        self._buffer: list[dict] = []
        self._t0 = time.perf_counter()
        self._wall_t0 = time.time()
        self._last_nvidia_sample: dict = {}
        self._last_step_index = 0
        # Warmup nvidia-smi subprocess so the first .step() isn't slow
        _try_nvidia_smi()

    # -----------------------------------------------------------------
    # Phase-level events (setup, bootstrap, train, eval, etc)
    # -----------------------------------------------------------------

    def mark(self, event: str, **extra) -> None:
        """Record a named phase event.

        Automatically stamps wall clock, elapsed since init, and a snapshot of
        RAM + GPU state (the latter via nvidia-smi, ~50 ms cost — OK for
        phase-level events, not hot-path).
        """
        entry = {
            "ts_unix": time.time(),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.run_id,
            "tag": self.tag,
            "kind": "mark",
            "event": event,
            "elapsed_s": round(time.perf_counter() - self._t0, 3),
            **_try_proc_self(),
            **_try_nvidia_smi(),
            **_try_torch_cuda_mem(),
            **extra,
        }
        self._buffer.append(entry)
        # Human-readable echo
        human = f"[phase2:{self.tag}] MARK {event} @ {entry['elapsed_s']}s"
        if "gpu_util_pct" in entry:
            human += f" gpu={entry['gpu_util_pct']:.0f}% mem={entry.get('gpu_mem_used_mib', 0):.0f}MiB"
        if "ram_used_gb" in entry:
            human += f" ram={entry['ram_used_gb']:.1f}GB"
        print(human, flush=True)
        self.flush()

    # -----------------------------------------------------------------
    # Hot-path per-step telemetry
    # -----------------------------------------------------------------

    def step(
        self,
        step: int,
        ms: float,
        train_loss: float | None = None,
        tok_per_sec: float | None = None,
        prefetch_queue_depth: int | None = None,
        ngram_queue_depth: int | None = None,
        **extra,
    ) -> None:
        """Record a training step's timing + optional metrics.

        Light-touch: just appends a dict to an in-memory buffer. nvidia-smi is
        sampled every N steps (default 50) to avoid the per-step cost.
        """
        entry = {
            "kind": "step",
            "run_id": self.run_id,
            "tag": self.tag,
            "step": int(step),
            "ms": round(float(ms), 3),
            "ts_unix": time.time(),
            "elapsed_s": round(time.perf_counter() - self._t0, 3),
        }
        if train_loss is not None:
            entry["train_loss"] = round(float(train_loss), 6)
        if tok_per_sec is not None:
            entry["tok_per_sec"] = round(float(tok_per_sec), 1)
        if prefetch_queue_depth is not None:
            entry["prefetch_queue_depth"] = int(prefetch_queue_depth)
        if ngram_queue_depth is not None:
            entry["ngram_queue_depth"] = int(ngram_queue_depth)
        entry.update(extra)
        # Sparse nvidia-smi sampling — expensive, don't do every step
        if step % max(1, self.sample_nvidia_smi_every) == 0:
            self._last_nvidia_sample = _try_nvidia_smi()
        entry.update(self._last_nvidia_sample)
        self._buffer.append(entry)
        self._last_step_index = step
        if len(self._buffer) >= self.flush_every_steps:
            self.flush()

    # -----------------------------------------------------------------
    # Flush / summary
    # -----------------------------------------------------------------

    def flush(self) -> None:
        """Write buffered entries to the JSONL file and clear the buffer."""
        if not self._buffer:
            return
        try:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                for entry in self._buffer:
                    f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except Exception as e:
            print(f"[phase2:{self.tag}] WARNING: failed to flush metrics: {e}", flush=True)
        self._buffer.clear()

    def summary(self) -> dict:
        """Read back the full JSONL and compute a summary for the current run_id.

        Useful at the end of a run to print a clean table of per-shot speedups.
        """
        rows = []
        try:
            with self.jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("run_id") != self.run_id:
                        continue
                    rows.append(row)
        except FileNotFoundError:
            pass
        step_rows = [r for r in rows if r.get("kind") == "step"]
        mark_rows = [r for r in rows if r.get("kind") == "mark"]
        if step_rows:
            mses = [r["ms"] for r in step_rows if "ms" in r]
            toks = [r["tok_per_sec"] for r in step_rows if "tok_per_sec" in r]
            gpus = [r["gpu_util_pct"] for r in step_rows if "gpu_util_pct" in r]
            cpus = [r["load_1m"] for r in step_rows if "load_1m" in r]
            rams = [r["ram_used_gb"] for r in step_rows if "ram_used_gb" in r]
            return {
                "run_id": self.run_id,
                "total_steps": len(step_rows),
                "ms_per_step_mean": round(sum(mses) / max(len(mses), 1), 2),
                "ms_per_step_min": round(min(mses), 2) if mses else None,
                "ms_per_step_max": round(max(mses), 2) if mses else None,
                "tok_per_sec_mean": round(sum(toks) / max(len(toks), 1), 0) if toks else None,
                "gpu_util_pct_mean": round(sum(gpus) / max(len(gpus), 1), 1) if gpus else None,
                "load_1m_mean": round(sum(cpus) / max(len(cpus), 1), 2) if cpus else None,
                "ram_used_gb_mean": round(sum(rams) / max(len(rams), 1), 1) if rams else None,
                "marks": [r.get("event") for r in mark_rows],
                "total_elapsed_s": round(time.perf_counter() - self._t0, 1),
            }
        return {"run_id": self.run_id, "total_steps": 0, "marks": [r.get("event") for r in mark_rows]}

    def print_summary(self) -> None:
        """Print a compact summary table for the current run to stdout."""
        s = self.summary()
        print(f"[phase2:{self.tag}] === SUMMARY ===", flush=True)
        for k, v in s.items():
            if k == "marks":
                continue
            print(f"  {k}: {v}", flush=True)
        if s.get("marks"):
            print(f"  marks: {', '.join(s['marks'][:10])}{' ...' if len(s['marks']) > 10 else ''}", flush=True)

    # -----------------------------------------------------------------
    # Comparison helpers (for after a run completes)
    # -----------------------------------------------------------------

    @staticmethod
    def compare_runs(
        jsonl_path: str | os.PathLike,
        run_ids: list[str],
    ) -> list[dict]:
        """Load the JSONL and return per-run summaries for the given run_ids.

        Handy for before/after speedup measurements:
            Phase2Metrics.compare_runs('logs/phase2_metrics.jsonl',
                                        ['p1_baseline', 'p2_s0_inductor', 'p2_s1_compile'])
        """
        results = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return results
        for run_id in run_ids:
            step_rows = []
            mark_rows = []
            for line in lines:
                try:
                    row = json.loads(line.strip())
                except Exception:
                    continue
                if row.get("run_id") != run_id:
                    continue
                if row.get("kind") == "step":
                    step_rows.append(row)
                elif row.get("kind") == "mark":
                    mark_rows.append(row)
            if step_rows:
                mses = [r["ms"] for r in step_rows if "ms" in r]
                gpus = [r["gpu_util_pct"] for r in step_rows if "gpu_util_pct" in r]
                results.append({
                    "run_id": run_id,
                    "steps": len(step_rows),
                    "ms_mean": round(sum(mses) / len(mses), 2) if mses else None,
                    "ms_p50": round(sorted(mses)[len(mses) // 2], 2) if mses else None,
                    "gpu_util_mean": round(sum(gpus) / len(gpus), 1) if gpus else None,
                })
            else:
                results.append({"run_id": run_id, "steps": 0})
        return results


if __name__ == "__main__":
    # Quick smoke test — writes a few fake entries and prints the summary
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "metrics.jsonl"
        m = Phase2Metrics(run_id="smoke_test", jsonl_path=path, flush_every_steps=3)
        m.mark("init")
        for i in range(10):
            m.step(step=i, ms=3.2 + i * 0.1, train_loss=4.5 - i * 0.01, tok_per_sec=280000 - i * 100)
        m.mark("done")
        m.flush()
        m.print_summary()
        print(f"\nJSONL written to: {path} ({path.stat().st_size} bytes)")
        print("=== compare_runs demo ===")
        for row in Phase2Metrics.compare_runs(path, ["smoke_test"]):
            print(" ", row)
