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
| Apr 5 | tyf0q5l1kgefgx | RTX 3080 Ti | gpu_speed_test.py | PENDING | — |
