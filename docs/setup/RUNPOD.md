# RunPod Setup & Run Log

## How to Connect

```bash
# 1. Create pod
runpodctl create pod --name paramgolf --gpuType "NVIDIA GeForce RTX 3080 Ti" \
  --gpuCount 1 --containerDiskSize 20 --volumeSize 50 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --communityCloud --ports "22/tcp"

# 2. Get pod ID and SSH host ID
# Go to runpod.io → Pods → your pod → Connect → SSH
# OR use API:
curl -s -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"query { pod(input: {podId: \"POD_ID\"}) { machine { podHostId } } }"}' \
  https://api.runpod.io/graphql

# 3. SSH in (use the podHostId from above)
ssh PODHOSTID@ssh.runpod.io -i ~/.ssh/id_ed25519

# 4. Inside pod: setup
cd /workspace
git clone https://github.com/taka6745/paramgolf.git
cd paramgolf
pip install -q sentencepiece numpy huggingface_hub datasets tqdm

# 5. Download data
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 3

# 6. Run tests
python3 gpu_quick_test.py     # baseline speed test
python3 gpu_speed_test.py     # speed experiments
```

## Known Issues

- **PyTorch 2.4 + enable_gqa:** `F.scaled_dot_product_attention(enable_gqa=...)` not supported.
  Fix: manually repeat KV heads before attention, or upgrade to PyTorch 2.5+
- **torch.compile:** Fails with `enable_gqa` error. Must patch or disable.
  Fix: `sed -i` to remove enable_gqa and replace torch.compile with passthrough
- **OOM on 12GB GPU:** Default batch (524K tokens) too large.
  Fix: `TRAIN_BATCH_TOKENS=65536` or use the standalone test scripts
- **BPE-8192 data:** Not on HuggingFace. Must upload from Mac or rebuild on pod.
  Fix: use `--variant sp1024` for testing, or scp the data
- **Claude sandbox can't SSH:** RunPod SSH proxy blocks non-PTY connections.
  Fix: user must SSH from their own terminal

## CPU Utilization (CPU is idle during GPU training)

CPU is mostly idle while GPU trains. Use it for:
- Precompute n-gram tables in background
- Build BPE-8192 tokenizer + re-export data
- Compress/GPTQ experiments (post-training)
- Per-shard n-gram table building
- Val data preprocessing

NOT useful for: CPU training (100x slower than GPU)

### During H100 Submission (saves 60-90s = ~1000 extra training steps):
- Build next shard's n-gram tables while GPU trains on current shard
- Generate GPTQ calibration data (autoregressive from checkpoint copy)
- Preprocess eval data (load val shards, build sliding window batches)
- Start GPTQ quantization on CPU while GPU finishes training
- Build eval-time n-gram cache from scored tokens during eval

## Pod Management

```bash
# Stop (saves money, keeps disk)
runpodctl stop pod POD_ID

# Start again
runpodctl start pod POD_ID

# Remove (deletes everything)
runpodctl remove pod POD_ID

# List pods
runpodctl get pod
```

## Run Log

| Date | Pod ID | GPU | Test | Results | Cost |
|------|--------|-----|------|---------|------|
| Apr 5 | tyf0q5l1kgefgx | RTX 3080 Ti | gpu_quick_test.py | 9L: 42.4ms, 11L: 52.2ms, VRAM 1.48GB | ~$0.10 |
| Apr 5 | tyf0q5l1kgefgx | RTX 3080 Ti | gpu_speed_test.py | See below | ~$0.05 |
| Apr 5 | tyf0q5l1kgefgx | RTX 3080 Ti | gpu_progressive_test.py | See below | ~$0.05 |
| Apr 6 | tyf0q5l1kgefgx | RTX 3080 Ti | gpu_timed_test.py | **#7 WINS** | ~$0.10 |

### Timed Quality Test (120s each, eval at seq=1024)
| Strategy | Eval Loss | Steps | vs Ref |
|----------|-----------|-------|--------|
| **7. Mostly short 90%@128 + 10%@1024** | **8.8875** | **13,762** | **-0.44 BEST** |
| 3. Prog seq 70/30 (128→1024) | 9.1687 | 11,185 | -0.16 |
| 6. All seq=128 (max steps) | 9.2063 | 15,066 | -0.12 |
| 1. Standard 11L/1024 (REF) | 9.3250 | 2,114 | baseline |
| 2. Prog seq 50/50 (128→1024) | 9.3656 | 8,573 | +0.04 |
| 4. 3-phase 128→256→1024 | 9.3844 | 7,960 | +0.06 |
| 5. Grow+seq 4L/128→11L/256→11L/1024 | 9.4594 | 14,405 | +0.13 |

### Speed Test Results (30 steps each)
| Config | ms/step | Steps/10min |
|--------|---------|-------------|
| 9L baseline | 41.6 | 14,434 |
| 11L baseline | 51.4 | 11,678 |
| 11L + layer drop 50% | 39.9 | 15,054 |
| 11L + layer drop 80% | 32.6 | 18,407 |
| 11L + seq=256 | 13.2 | 45,289 |
| 11L + seq=128 | 8.4 | 71,768 |
| Tiny 4L/256d | 8.3 | 72,298 |
| Hybrid conv+attn | 46.7 | 12,844 |
| Lossy 50% mask | 52.7 | 11,376 (NO speedup) |

### Progressive Quality Test (1000 steps each, eval at seq=1024)
| Strategy | Eval Loss | Time | vs Reference |
|----------|-----------|------|-------------|
| **4. Prog grow 500@4L + 500@11L** | **9.5625** | **39.8s** | **-0.18 BETTER, 1.4x faster** |
| 6. Layer drop (6L proxy) | 9.6750 | 32.6s | -0.07 better, 1.7x faster |
| 1. Standard 11L/1024 (REF) | 9.7438 | 54.7s | baseline |
| 5. Prog ALL 4L/128→11L/256→11L/1024 | 9.7812 | 24.1s | +0.04 same, 2.3x faster |
| 3. Prog seq 700@128 + 300@1024 | 9.9437 | 22.7s | +0.20 worse, 2.4x faster |
| 2. Prog seq 500@128 + 500@1024 | 10.1000 | 32.2s | +0.36 worse, 1.7x faster |
