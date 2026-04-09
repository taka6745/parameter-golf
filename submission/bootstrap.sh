#!/bin/bash
# bootstrap.sh — one-command setup for a fresh RunPod H100.
# Usage: curl -sL https://raw.githubusercontent.com/taka6745/paramgolf/main/submission/bootstrap.sh | bash
#
# What it does:
#   1. apt-get install git (if missing)
#   2. git clone the repo to /workspace/paramgolf
#   3. cd into submission/
#   4. setup.sh   — torch 2.9.1+cu128, FA3 verify, brotli + sentencepiece
#   5. get_data.sh — docs JSONL stage on container disk + SP8192 tokenize
#   6. run.sh     — train + eval + serialize
#
# Total time on a fresh pod: ~50-80 min (5 min setup + 30-60 min tokenize + 10 min train + 5 min eval).
#
# Idempotent: re-running skips steps that are already done (clone exists, shards exist, etc).

set -eu
exec > >(tee -a /tmp/paramgolf_bootstrap.log) 2>&1

REPO_URL="${REPO_URL:-https://github.com/taka6745/paramgolf.git}"
REPO_DIR="${REPO_DIR:-/workspace/paramgolf}"
BRANCH="${BRANCH:-main}"

echo "============================================================"
echo "= paramgolf bootstrap $(date -u +%Y-%m-%dT%H:%M:%SZ) ="
echo "============================================================"
echo "REPO_URL=$REPO_URL"
echo "REPO_DIR=$REPO_DIR"
echo "BRANCH=$BRANCH"

# Step 1: ensure git exists
if ! command -v git >/dev/null 2>&1; then
    echo "[bootstrap] installing git..."
    apt-get update -qq && apt-get install -y -qq git
fi

# Step 2: clone or update the repo
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[bootstrap] cloning $REPO_URL into $REPO_DIR..."
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
    echo "[bootstrap] repo already cloned, pulling latest $BRANCH..."
    cd "$REPO_DIR"
    git fetch origin "$BRANCH"
    git reset --hard "origin/$BRANCH"
fi

cd "$REPO_DIR/submission"

# Step 3: pod environment
echo
echo "============================================================"
echo "[bootstrap] STEP 1/3: setup.sh"
echo "============================================================"
bash setup.sh

# Step 4: data
echo
echo "============================================================"
echo "[bootstrap] STEP 2/3: get_data.sh"
echo "============================================================"
bash get_data.sh

# Step 5: train + eval + serialize
echo
echo "============================================================"
echo "[bootstrap] STEP 3/3: run.sh"
echo "============================================================"
bash run.sh

echo
echo "============================================================"
echo "[bootstrap] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo "Submission artifact: $REPO_DIR/submission/final_model.int6.ptz"
echo "Log: /tmp/paramgolf_bootstrap.log + $REPO_DIR/submission/logs/"
