# Chore — One-Time Setup

These run **once per pod creation**. They prepare data, tokenizer, and tables. No experimentation here, just prep.

## Hardware

Runs on a **3060 (12GB)** at $0.18/hr. Total time: ~30 min, total cost: ~$0.10.

---

## What you need (everything 00_setup_pod.sh installs)

### System packages (apt-get)
```
git              # clone repo
curl             # downloads, log uploads
wget             # alternative downloader
htop             # monitoring
bc               # floating-point math in validate/ and unknown/ shell scripts
build-essential  # for pip wheel builds
ca-certificates  # HTTPS certs
```

### Python packages (pip)
```
torch>=2.4.0     # PyTorch (CUDA build)
sentencepiece    # BPE-8192 tokenizer
numpy            # arrays, n-gram tables
huggingface_hub  # FineWeb dataset access
datasets         # FineWeb streaming
tqdm             # progress bars
zstandard        # artifact compression
brotli           # artifact compression (Brotli-11 saves -1.47 MB)
scikit-learn     # KMeans for DC500 categories
```

### Git repos
```
https://github.com/taka6745/paramgolf.git    # this repo
```

### Data sources (downloaded by 01_download_data.sh)
```
HuggingFaceFW/fineweb           # FineWeb dataset (public, no auth)
                                # Used for training data + tokenizer training
                                # Downloads ~10 train shards + 1 val shard
```

### Tokenizer (built by 02_build_tokenizer.sh)
```
data/tokenizers/fineweb_8192_bpe.model
data/tokenizers/fineweb_8192_bpe.vocab
```
Built from a 1M-document FineWeb sample using SentencePiece BPE.

---

## Order

| # | Script | What | Time | Output |
|---|---|---|---|---|
| 00 | `00_setup_pod.sh` | apt + pip + git clone | 2 min | venv, deps installed |
| 01 | `01_download_data.sh` | FineWeb shards | 8 min | `data/datasets/fineweb10B_*` |
| 02 | `02_build_tokenizer.sh` | BPE-8192 tokenizer | 5 min | `data/tokenizers/fineweb_8192_bpe.model` |
| 03 | `03_retokenize.sh` | Re-export with BPE-8192 | 8 min | `data/datasets/fineweb10B_bpe8192/` |
| 04 | `04_build_ngrams.py` | Bigram, trigram, 4-gram | 4 min | `data/*gram*.npy` |
| 05 | `05_build_dc500.py` | DC500 categories | 2 min | `data/dist_cats_500_8192.npz` |
| 06 | `06_lloyd_max.py` | Lloyd-Max codebook | 30 sec | `data/lloyd_max_codebook_64.npy` |
| 07 | `07_verify_data.sh` | Sanity checks | 1 min | (verifies all outputs) |

Run them all in order with: `cd .. && ./setup.sh`

---

## Outputs

After running everything, you should have:

```
data/
├── tokenizers/
│   ├── fineweb_8192_bpe.model              ~250 KB
│   └── fineweb_8192_bpe.vocab              ~120 KB
├── datasets/
│   ├── fineweb10B_sp1024/                  default tokenizer (baseline reference)
│   │   ├── fineweb_train_*.bin
│   │   └── fineweb_val_*.bin
│   └── fineweb10B_bpe8192/                 our tokenizer (the actual stack)
│       ├── fineweb_train_*.bin             (10 shards × ~50 MB each)
│       └── fineweb_val_*.bin               (1 shard × ~50 MB)
├── bigram_tab_8192v.npy                    2.94 MB  (16384 buckets × 8192 vocab, log-probs)
├── trigram_logprobs_8192v.npy              0.64 MB
├── fourgram_logprobs_8192v.npy             0.32 MB
├── dist_cats_500_8192.npz                  0.18 MB  (token→category + category log-probs)
└── lloyd_max_codebook_64.npy               256 B    (non-uniform int6 quantization codebook)
```

Total: ~5 MB of precomputed tables + ~500 MB of tokenized data.

---

## Manual install (if 00_setup_pod.sh fails)

```bash
# 1. System packages
apt-get update
apt-get install -y git curl wget htop bc build-essential ca-certificates

# 2. Clone repo
cd /workspace
git clone https://github.com/taka6745/paramgolf.git
cd paramgolf

# 3. Python deps (use system torch from RunPod base image, install everything else)
pip install sentencepiece huggingface_hub datasets tqdm zstandard brotli scikit-learn

# Only create a venv if system torch doesn't work:
# python3 -m venv .venv --system-site-packages
# source .venv/bin/activate
# pip install torch --index-url https://download.pytorch.org/whl/cu124

# 5. Verify CUDA
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## Skip conditions (idempotent re-runs)

If you already ran chore on a previous pod and **mounted the same volume**, you can skip individual steps:

| Skip if | When |
|---|---|
| 01 (data) | `data/datasets/fineweb10B_sp1024/` exists with ≥10 train shards |
| 02 (tokenizer) | `data/tokenizers/fineweb_8192_bpe.model` exists |
| 03 (retokenize) | `data/datasets/fineweb10B_bpe8192/` exists with ≥10 train shards |
| 04 (ngrams) | `data/bigram_tab_8192v.npy` exists |
| 05 (DC500) | `data/dist_cats_500_8192.npz` exists |
| 06 (lloyd-max) | `data/lloyd_max_codebook_64.npy` exists |

The scripts check these conditions internally, so re-running is safe — they'll just print "already exists, skipping".

**Always run** 00 (deps may have changed) and 07 (verify the volume actually has what you expect) on a new pod.

---

## RunPod pod creation

If you haven't created a pod yet:

```bash
# From your local machine (requires runpodctl installed):
runpodctl create pod \
  --name paramgolf-3060 \
  --gpuType "NVIDIA GeForce RTX 3060" \
  --gpuCount 1 \
  --containerDiskSize 50 \
  --volumeSize 100 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --communityCloud \
  --ports "22/tcp"
```

Or use the web UI at runpod.io → Deploy → choose 3060 + the pytorch:2.4.0 image.

Then SSH in (use the SSH host from runpod.io's "Connect" panel):
```bash
ssh PODHOSTID@ssh.runpod.io -i ~/.ssh/id_ed25519
cd /workspace
git clone https://github.com/taka6745/paramgolf.git
cd paramgolf/runpod_tests
./setup.sh
```

That's it.

---

## Cost notes

- 3060 community cloud: ~$0.18/hr
- Chore takes ~30 min total: **$0.10**
- If you need to re-run because something failed: maybe $0.20 total
- The `volumeSize 100` means you can stop the pod and resume without re-downloading data (volume persists, container resets). Worth it if you'll be iterating over multiple sessions.
