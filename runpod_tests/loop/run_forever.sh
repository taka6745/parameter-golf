#!/bin/bash
# run_forever.sh — autonomous experiment loop launcher
#
# Usage:
#   nohup bash runpod_tests/loop/run_forever.sh > runpod_tests/loop/run_forever.out 2>&1 &
#
# Kills any existing train_gpt.py / experiment_runner before taking over the GPU.
# Wraps the runner in an infinite restart loop — if Python dies for any reason,
# we sleep 5s and relaunch. No external cron/watchdog needed.

set -u
cd /workspace/paramgolf

# Take over the GPU (only kill OTHER experiment_runner instances)
ME=$$
for pid in $(pgrep -f experiment_runner.py 2>/dev/null); do
    [ "$pid" != "$ME" ] && kill "$pid" 2>/dev/null || true
done
pkill -f 'python3 train_gpt.py' 2>/dev/null || true
sleep 2

mkdir -p runpod_tests/loop/logs
echo "=== run_forever launched at $(date -u) PID=$ME ==="

while true; do
    # Auto-pull latest experiments / runner code before each restart.
    # --autostash because the patcher modifies train_gpt.py locally.
    git pull --rebase --autostash 2>&1 | tail -3 || true
    # Restore train_gpt.py from backup so patcher applies cleanly even after
    # the patcher itself was edited. The patcher is idempotent within a run
    # but doesn't auto-upgrade old patches when the upstream patcher source changes.
    if [ -f train_gpt.py.bak ]; then
        cp train_gpt.py.bak train_gpt.py
    fi
    bash runpod_tests/chore/08_patch_train_gpt.sh 2>&1 | tail -20 || true
    python3 -u runpod_tests/loop/experiment_runner.py
    echo "=== runner exited with code $? at $(date -u) — restart in 5s ==="
    sleep 5
done
