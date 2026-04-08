#!/bin/bash
# bootstrap_new_pod.sh — fresh-pod bootstrap for the stack-novelty campaign.
#
# Idempotent. Designed to be ssh'd into a fresh RunPod instance and run once:
#
#   ssh root@<pod-ip> -p <port> 'bash -s' <<'EOF'
#       export POD_ID=B
#       export REPO_URL=https://github.com/taka6745/paramgolf.git
#       export ANCHOR_HOST=root@<anchor-ip>
#       export ANCHOR_PORT=<anchor-port>
#       curl -fsSL https://raw.githubusercontent.com/taka6745/paramgolf/main/runpod_tests/loop/bootstrap_new_pod.sh | bash
#   EOF
#
# Or, after `git clone` is already done:
#   POD_ID=B ANCHOR_HOST=root@<ip> ANCHOR_PORT=<port> bash runpod_tests/loop/bootstrap_new_pod.sh
#
# Required env vars:
#   POD_ID         — single letter (A..G), written to runpod_tests/loop/pod_id.txt
#   ANCHOR_HOST    — anchor pod ssh user@ip (e.g. root@136.61.20.181) for scp pulls
#   ANCHOR_PORT    — anchor pod ssh port    (e.g. 4129)
#
# Optional:
#   REPO_URL       — defaults to https://github.com/taka6745/paramgolf.git
#   WORKSPACE      — defaults to /workspace
#
# Reuses these existing scripts unchanged:
#   runpod_tests/chore/00_setup_pod.sh   — venv + pip install
#   runpod_tests/chore/08_patch_train_gpt.sh — apply 26 patches
#   runpod_tests/loop/install_cron.sh    — install pod-side watchdog cron
#   runpod_tests/loop/run_forever.sh     — main loop launcher

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/taka6745/paramgolf.git}"
WORKSPACE="${WORKSPACE:-/workspace}"
ANCHOR_SSH_FLAGS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

if [ -z "${POD_ID:-}" ]; then
    echo "FATAL: POD_ID env var is required (single letter A..G)" >&2
    exit 1
fi
if [ -z "${ANCHOR_HOST:-}" ] || [ -z "${ANCHOR_PORT:-}" ]; then
    echo "FATAL: ANCHOR_HOST and ANCHOR_PORT env vars are required" >&2
    exit 1
fi

echo "=== bootstrap_new_pod.sh starting on POD_ID=$POD_ID at $(date -u) ==="

# ============================================================
# Block 1: repo + venv (sequential start, then parallel)
# ============================================================
echo "--- Block 1: repo + venv ---"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

if [ ! -d paramgolf ]; then
    git clone "$REPO_URL" paramgolf
fi

cd paramgolf
git fetch origin
git checkout main
git pull --rebase --autostash origin main || true

if [ -f runpod_tests/chore/00_setup_pod.sh ]; then
    bash runpod_tests/chore/00_setup_pod.sh &
    BOOT_PID=$!
else
    echo "WARN: runpod_tests/chore/00_setup_pod.sh missing, skipping venv setup"
    BOOT_PID=""
fi

mkdir -p data/datasets
mkdir -p runpod_tests/loop/logs

[ -n "$BOOT_PID" ] && wait "$BOOT_PID"

# ============================================================
# Block 2: fetch shards + n-gram tables from anchor pod (parallel)
# ============================================================
echo "--- Block 2: fetch data from anchor (parallel scp) ---"

# Helper: scp from anchor with the right flags
scp_anchor() {
    local src="$1"
    local dst="$2"
    scp -P "$ANCHOR_PORT" $ANCHOR_SSH_FLAGS \
        "${ANCHOR_HOST}:/workspace/paramgolf/${src}" "$dst" \
        2>&1 | tail -3 || echo "  (scp $src failed)"
}

# Tokenized shards (~2 GB)
if [ ! -d data/datasets/fineweb10B_sp1024 ] || [ -z "$(ls -A data/datasets/fineweb10B_sp1024 2>/dev/null)" ]; then
    mkdir -p data/datasets/fineweb10B_sp1024
    scp -r -P "$ANCHOR_PORT" $ANCHOR_SSH_FLAGS \
        "${ANCHOR_HOST}:/workspace/paramgolf/data/datasets/fineweb10B_sp1024" \
        data/datasets/ &
    SHARDS_PID=$!
else
    SHARDS_PID=""
fi

# N-gram tables (~few hundred MB total)
NGRAM_FILES=(
    "data/bigram_logprobs.npy"
    "data/trigram_logprobs.npy"
    "data/fourgram_logprobs.npy"
    "data/fivegram_logprobs.npy"
    "data/bigram_logprobs_8192v.npy"
    "data/trigram_logprobs_8192v.npy"
    "data/fourgram_logprobs_8192v.npy"
    "data/fivegram_logprobs_4k_8192v.npy"
    "data/tab_hash_1.npy"
    "data/tab_hash_2.npy"
    "data/tab_hash_3.npy"
)
for src in "${NGRAM_FILES[@]}"; do
    if [ ! -f "$src" ]; then
        scp_anchor "$src" "$src" &
    fi
done
wait

[ -n "$SHARDS_PID" ] && wait "$SHARDS_PID" || true

echo "  Block 2 done. Counting downloads:"
ls -lh data/datasets/fineweb10B_sp1024 2>/dev/null | head -3 || echo "  no shards"
ls -lh data/*.npy 2>/dev/null | head -5 || echo "  no n-gram tables"

# ============================================================
# Block 3: patcher + integrity check (sequential, hard fail on miss)
# ============================================================
echo "--- Block 3: patcher + integrity check ---"

# Always work from a fresh backup so the patcher applies cleanly
if [ ! -f train_gpt.py.bak ]; then
    cp train_gpt.py train_gpt.py.bak
fi
cp train_gpt.py.bak train_gpt.py

bash runpod_tests/chore/08_patch_train_gpt.sh 2>&1 | tail -30
PATCH_EXIT=$?
if [ "$PATCH_EXIT" -ne 0 ]; then
    echo "FATAL: 08_patch_train_gpt.sh exited $PATCH_EXIT — bootstrap aborts (G4 FAIL)" >&2
    exit 4
fi

# Run gate_check.py to verify G4 marker integrity before launching anything
if [ -f runpod_tests/loop/gate_check.py ]; then
    python3 runpod_tests/loop/gate_check.py --dry-run > /tmp/gate_check.out 2>&1 || true
    if grep -q "G4_marker_count.*FAIL" /tmp/gate_check.out; then
        echo "FATAL: G4 marker integrity check failed:" >&2
        grep "G4_marker_count" /tmp/gate_check.out >&2
        exit 5
    fi
fi

echo "  Block 3 done. Patcher applied + G4 PASS."

# ============================================================
# Block 4: install pod_id, cron, launch run_forever
# ============================================================
echo "--- Block 4: install pod_id + cron + launch loop ---"

echo "$POD_ID" > runpod_tests/loop/pod_id.txt
echo "  POD_ID=$POD_ID written to runpod_tests/loop/pod_id.txt"

# Install pod-side watchdog cron (idempotent)
if [ -f runpod_tests/loop/install_cron.sh ]; then
    bash runpod_tests/loop/install_cron.sh || echo "  (install_cron.sh: $?)"
fi

# Kill any existing run_forever.sh BEFORE launching (avoid duplicate)
pkill -f run_forever.sh 2>/dev/null || true
pkill -f experiment_runner.py 2>/dev/null || true
sleep 2

# Launch run_forever in the background
nohup bash runpod_tests/loop/run_forever.sh \
    > runpod_tests/loop/run_forever.out 2>&1 &
disown

sleep 3
if pgrep -f run_forever.sh > /dev/null; then
    echo "  run_forever.sh is alive (pid $(pgrep -f run_forever.sh | head -1))"
else
    echo "  WARN: run_forever.sh did not stay alive"
fi

echo "=== bootstrap_new_pod.sh DONE on POD_ID=$POD_ID at $(date -u) ==="
