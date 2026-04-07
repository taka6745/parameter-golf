#!/bin/bash
# status.sh — quick status check for the autonomous loop
# Pipe a heredoc through this on the pod, or run on the pod directly.
set -u
cd /workspace/paramgolf
echo "=== STATUS at $(date -u) ==="
echo
echo "PROCESS:"
ps -ef | grep -E 'experiment_runner|train_gpt' | grep -v grep || echo "  (none)"
echo
echo "RESULTS COUNT:"
test -f runpod_tests/loop/results.jsonl && wc -l runpod_tests/loop/results.jsonl || echo "  (no results yet)"
echo
echo "LAST 5 RESULTS:"
test -f runpod_tests/loop/results.jsonl && tail -5 runpod_tests/loop/results.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print(f\"  {r.get('name','?'):<30} tl={r.get('train_loss','?')} steps={r.get('max_step',0):>5} ms/step={r.get('ms_step','?')} crashed={r.get('crashed', False)}\")
"
echo
echo "LEADERBOARD:"
test -f runpod_tests/loop/leaderboard.txt && head -20 runpod_tests/loop/leaderboard.txt || echo "  (none yet)"
