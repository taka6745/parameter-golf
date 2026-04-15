#!/bin/bash
# =============================================================
# Automated GPU test runner for Parameter Golf
# Creates pod → uploads code → runs test → downloads results → stops pod
# Usage: ./run_gpu_test.sh [test_name] [gpu_type] [steps]
# Example: ./run_gpu_test.sh baseline "NVIDIA GeForce RTX 3080 Ti" 50
# =============================================================

set -e

TEST_NAME="${1:-baseline}"
GPU_TYPE="${2:-NVIDIA GeForce RTX 3080 Ti}"
STEPS="${3:-50}"
POD_NAME="paramgolf-${TEST_NAME}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
RESULTS_DIR="$REPO_DIR/logs/gpu_${TEST_NAME}"

echo "=== PARAMGOLF GPU TEST: $TEST_NAME ==="
echo "GPU: $GPU_TYPE | Steps: $STEPS"
echo ""

# Step 1: Create pod
echo "[1/7] Creating pod..."
POD_ID=$(runpodctl create pod \
  --name "$POD_NAME" \
  --gpuType "$GPU_TYPE" \
  --gpuCount 1 \
  --containerDiskSize 20 \
  --volumeSize 50 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --communityCloud 2>&1 | grep -o 'pod "[^"]*"' | grep -o '"[^"]*"' | tr -d '"')

if [ -z "$POD_ID" ]; then
  echo "ERROR: Failed to create pod. Try a different GPU type."
  exit 1
fi
echo "Pod created: $POD_ID"

# Step 2: Wait for pod to be ready
echo "[2/7] Waiting for pod to be ready..."
for i in $(seq 1 60); do
  STATUS=$(runpodctl get pod 2>&1 | grep "$POD_ID" | awk '{print $NF}')
  if [ "$STATUS" = "RUNNING" ]; then
    echo "Pod is RUNNING"
    break
  fi
  echo "  Status: $STATUS (waiting...)"
  sleep 10
done

# Step 3: Get SSH connection info and wait for SSH to be ready
echo "[3/7] Connecting via SSH..."
sleep 15  # Give SSH service time to start

SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $SSH_KEY ${POD_ID}@ssh.runpod.io"

# Test SSH connection
for i in $(seq 1 12); do
  if $SSH_CMD "echo 'SSH connected'" 2>/dev/null; then
    break
  fi
  echo "  SSH not ready yet (attempt $i/12)..."
  sleep 10
done

# Step 4: Upload code and data
echo "[4/7] Uploading code..."
$SSH_CMD "mkdir -p /workspace/paramgolf"

# Upload essential files only (not the full repo with .venv)
rsync -avz --progress \
  -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY" \
  --include='train_gpt.py' \
  --include='data/' \
  --include='data/datasets/' \
  --include='data/datasets/fineweb10B_sp1024/***' \
  --include='data/datasets/fineweb10B_sp8192/***' \
  --include='data/tokenizers/***' \
  --include='data/bigram_logprobs_8192v.npy' \
  --include='data/trigram_logprobs_8k.npy' \
  --include='data/dist_cats_500_8192.npz' \
  --include='ngram_logit_bias.py' \
  --exclude='.venv' \
  --exclude='logs/' \
  --exclude='*.npz' \
  --exclude='__pycache__' \
  "$REPO_DIR/" "${POD_ID}@ssh.runpod.io:/workspace/paramgolf/"

# Step 5: Install deps and run test
echo "[5/7] Installing deps and running test ($STEPS steps)..."
$SSH_CMD << REMOTE_SCRIPT
cd /workspace/paramgolf
pip install sentencepiece numpy 2>&1 | tail -3

# Check CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Run baseline test
echo "=== RUNNING $TEST_NAME ($STEPS steps) ==="
PYTHONUNBUFFERED=1 python3 train_gpt.py \
  --iterations $STEPS \
  --seed 42 \
  2>&1 | tee /workspace/paramgolf/gpu_${TEST_NAME}.log

echo "=== TEST COMPLETE ==="
REMOTE_SCRIPT

# Step 6: Download results
echo "[6/7] Downloading results..."
mkdir -p "$RESULTS_DIR"
rsync -avz \
  -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY" \
  "${POD_ID}@ssh.runpod.io:/workspace/paramgolf/gpu_${TEST_NAME}.log" \
  "$RESULTS_DIR/"

# Step 7: Stop pod
echo "[7/7] Stopping pod to save money..."
runpodctl stop pod "$POD_ID"

echo ""
echo "=== DONE ==="
echo "Results saved to: $RESULTS_DIR/gpu_${TEST_NAME}.log"
echo "Pod $POD_ID stopped (not deleted — can restart later)"
echo ""
echo "Key metrics:"
grep "step:${STEPS}/${STEPS}\|train_loss\|val_bpb\|step_avg\|tok_s" "$RESULTS_DIR/gpu_${TEST_NAME}.log" | tail -5
