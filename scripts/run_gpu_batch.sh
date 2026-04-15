#!/bin/bash
# =============================================================
# Batch GPU test runner for Parameter Golf
# Creates pod ONCE → runs N tests → downloads all results → stops pod
#
# Usage: ./run_gpu_batch.sh [gpu_type]
# Tests are defined in the TESTS array below — edit to add/remove
# =============================================================

set -e

GPU_TYPE="${1:-NVIDIA GeForce RTX 3080 Ti}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
RESULTS_DIR="$REPO_DIR/logs/gpu_batch_$(date +%Y%m%d_%H%M)"
POD_NAME="paramgolf-batch"

# =============================================================
# DEFINE YOUR TESTS HERE
# Format: "test_name|steps|env_overrides"
# =============================================================
TESTS=(
  # =============================================================
  # PIPELINE VALIDATION (does it run? 10 steps each, ~1 min)
  # =============================================================

  # 1. Baseline — does competition code run on this GPU?
  "pipeline_baseline|10|"

  # 2. BPE-8192 — does our tokenizer work?
  "pipeline_bpe8192|10|DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192"

  # 3. 11L — does our best Mac config work on CUDA?
  "pipeline_11L|10|NUM_LAYERS=11 DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192"

  # =============================================================
  # SPEED COMPARISON (50 steps, skip first 10 for compile warmup)
  # We measure ms/step at steps 20-50 to compare architectures
  # =============================================================

  # 4. Speed: baseline transformer
  "speed_transformer|50|DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192"

  # 5. Speed: torch.compile enabled
  "speed_compiled|50|TORCH_COMPILE=1 DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192"

  # 6. Speed: seq=256 (should be ~2-4x faster)
  "speed_seq256|50|TRAIN_SEQ_LEN=256 DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192"

  # =============================================================
  # GLA SHOOTOUT (install flash-linear-attention, compare ms/step)
  # Uncomment after verifying fla installs on this GPU
  # =============================================================
  # "gla_install|10|PIP_INSTALL=flash-linear-attention"
  # "speed_gla|50|MODEL_TYPE=gla"
)

echo "=== PARAMGOLF BATCH GPU TEST ==="
echo "GPU: $GPU_TYPE"
echo "Tests: ${#TESTS[@]}"
echo "Results: $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# Step 1: Create pod
echo "[1/6] Creating pod..."
CREATE_OUTPUT=$(runpodctl create pod \
  --name "$POD_NAME" \
  --gpuType "$GPU_TYPE" \
  --gpuCount 1 \
  --containerDiskSize 20 \
  --volumeSize 50 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --communityCloud 2>&1)
echo "$CREATE_OUTPUT"

POD_ID=$(echo "$CREATE_OUTPUT" | grep -o '"[a-z0-9]*"' | tr -d '"' | head -1)
if [ -z "$POD_ID" ]; then
  echo "ERROR: Failed to create pod."
  echo "$CREATE_OUTPUT"
  exit 1
fi
echo "Pod ID: $POD_ID"

# Step 2: Wait for pod
echo "[2/6] Waiting for pod..."
for i in $(seq 1 60); do
  STATUS=$(runpodctl get pod 2>&1 | grep "$POD_ID" | awk '{print $NF}')
  if [ "$STATUS" = "RUNNING" ]; then
    echo "Pod RUNNING after ${i}0 seconds"
    break
  fi
  sleep 10
done
sleep 20  # Extra time for SSH to start

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $SSH_KEY ${POD_ID}@ssh.runpod.io"
SCP="scp -o StrictHostKeyChecking=no -i $SSH_KEY"

# Wait for SSH
echo "[3/6] Connecting SSH..."
for i in $(seq 1 12); do
  if $SSH "echo connected" 2>/dev/null; then
    echo "SSH ready"
    break
  fi
  echo "  Retry $i/12..."
  sleep 10
done

# Step 3: Upload code
echo "[4/6] Uploading code + data..."
$SSH "mkdir -p /workspace/paramgolf/data/tokenizers /workspace/paramgolf/data/datasets"

# Upload training script
$SCP "$REPO_DIR/train_gpt.py" "${POD_ID}@ssh.runpod.io:/workspace/paramgolf/"
# Upload tokenizers
$SCP "$REPO_DIR/data/tokenizers/"*.model "${POD_ID}@ssh.runpod.io:/workspace/paramgolf/data/tokenizers/" 2>/dev/null || true
$SCP "$REPO_DIR/data/tokenizers/"*.vocab "${POD_ID}@ssh.runpod.io:/workspace/paramgolf/data/tokenizers/" 2>/dev/null || true

# Upload data shards (just first 3 for quick tests)
for dir in fineweb10B_sp1024 fineweb10B_sp8192; do
  if [ -d "$REPO_DIR/data/datasets/$dir" ]; then
    $SSH "mkdir -p /workspace/paramgolf/data/datasets/$dir"
    for shard in $(ls "$REPO_DIR/data/datasets/$dir/" | head -3); do
      echo "  Uploading $dir/$shard..."
      $SCP "$REPO_DIR/data/datasets/$dir/$shard" "${POD_ID}@ssh.runpod.io:/workspace/paramgolf/data/datasets/$dir/"
    done
  fi
done

# Install deps
echo "[5/6] Installing deps..."
$SSH "cd /workspace/paramgolf && pip install -q sentencepiece numpy"
$SSH "python3 -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')\""

# Step 4: Run all tests
echo "[6/6] Running ${#TESTS[@]} tests..."
echo ""

START_TIME=$(date +%s)

for i in "${!TESTS[@]}"; do
  IFS='|' read -r NAME STEPS ENVS <<< "${TESTS[$i]}"
  TEST_NUM=$((i + 1))

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "TEST $TEST_NUM/${#TESTS[@]}: $NAME ($STEPS steps)"
  echo "ENV: $ENVS"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  $SSH "cd /workspace/paramgolf && \
    echo '=== GPU METRICS BEFORE ===' && \
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader && \
    echo '=== TRAINING ===' && \
    $ENVS PYTHONUNBUFFERED=1 python3 -c \"
import subprocess, threading, time
# Log GPU utilization every 5 seconds in background
def gpu_monitor():
    while True:
        try:
            out = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).strip()
            print(f'GPU_MONITOR: {out}', flush=True)
        except: pass
        time.sleep(5)
t = threading.Thread(target=gpu_monitor, daemon=True)
t.start()
import os; os.system('$ENVS PYTHONUNBUFFERED=1 python3 train_gpt.py')
\" 2>&1 | head -300" > "$RESULTS_DIR/${NAME}.log" 2>&1 || true

  # Extract key metrics
  LOSS=$(grep "step:${STEPS}/" "$RESULTS_DIR/${NAME}.log" 2>/dev/null | grep -o "train_loss:[0-9.]*" | cut -d: -f2 | tail -1)
  STEP_AVG=$(grep "step:${STEPS}/" "$RESULTS_DIR/${NAME}.log" 2>/dev/null | grep -o "step_avg:[0-9.]*" | cut -d: -f2 | tail -1)
  TOKS=$(grep "step:${STEPS}/" "$RESULTS_DIR/${NAME}.log" 2>/dev/null | grep -o "tok_s:[0-9,]*" | cut -d: -f2 | tail -1)

  GPU_UTIL=$(grep "GPU_MONITOR" "$RESULTS_DIR/${NAME}.log" 2>/dev/null | tail -1 | sed 's/GPU_MONITOR: //')

  if [ -n "$LOSS" ]; then
    echo "  ✅ loss=$LOSS ms/step=$STEP_AVG tok/s=$TOKS gpu=$GPU_UTIL"
  else
    echo "  ❌ No result — check $RESULTS_DIR/${NAME}.log"
    tail -10 "$RESULTS_DIR/${NAME}.log" 2>/dev/null
  fi
  echo ""
done

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

# Stop pod
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All tests complete. Stopping pod..."
runpodctl stop pod "$POD_ID"

COST=$(echo "scale=2; $ELAPSED * ${0.18:-0.18} / 60" | bc 2>/dev/null || echo "~\$0.05-0.50")

echo ""
echo "=== BATCH RESULTS SUMMARY ==="
echo "Pod: $POD_ID (stopped)"
echo "Time: ${ELAPSED} minutes"
echo "Cost: ~\$$COST"
echo "Results: $RESULTS_DIR/"
echo ""
echo "| Test | Loss | ms/step | tok/s |"
echo "|------|------|---------|-------|"

for i in "${!TESTS[@]}"; do
  IFS='|' read -r NAME STEPS ENVS <<< "${TESTS[$i]}"
  LOSS=$(grep "step:" "$RESULTS_DIR/${NAME}.log" 2>/dev/null | grep -o "train_loss:[0-9.]*" | cut -d: -f2 | tail -1)
  STEP_AVG=$(grep "step:" "$RESULTS_DIR/${NAME}.log" 2>/dev/null | grep -o "step_avg:[0-9.]*" | cut -d: -f2 | tail -1)
  TOKS=$(grep "step:" "$RESULTS_DIR/${NAME}.log" 2>/dev/null | grep -o "tok_s:[0-9,]*" | cut -d: -f2 | tail -1)
  echo "| $NAME | ${LOSS:-FAIL} | ${STEP_AVG:-?} | ${TOKS:-?} |"
done
