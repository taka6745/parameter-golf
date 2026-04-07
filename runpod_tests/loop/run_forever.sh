#!/bin/bash
# run_forever.sh — autonomous experiment loop launcher
#
# Usage:
#   nohup bash runpod_tests/loop/run_forever.sh > runpod_tests/loop/run_forever.out 2>&1 &
#
# Kills any existing train_gpt.py / experiment_runner first to take over the GPU.
# Then launches the runner under setsid so it survives shell exit.

set -u
cd /workspace/paramgolf

# Take over the GPU
pkill -f 'experiment_runner.py' 2>/dev/null || true
pkill -f 'python3 train_gpt.py' 2>/dev/null || true
sleep 2

mkdir -p runpod_tests/loop/logs
echo "=== run_forever launched at $(date -u) ==="
exec python3 -u runpod_tests/loop/experiment_runner.py
