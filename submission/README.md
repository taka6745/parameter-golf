# paramgolf submission

Self-contained submission for the openai/parameter-golf 16 MB byte-LM challenge.

**Track**: 10 min / 16 MB on 1× or 8× H100
**Base**: decoded reproduction of comp PR #1477 (val_bpb 1.0822 on 8×H100 SXM)
**Our changes**: layered on top — see `CHANGELOG.md` (TODO)

## One-command bootstrap

On a freshly-rented RunPod H100 (PCIe or SXM, image = `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`):

```bash
curl -sL https://raw.githubusercontent.com/taka6745/paramgolf/main/submission/bootstrap.sh | bash
```

That single line does:
1. `git clone` the repo to `/workspace/paramgolf`
2. Run `setup.sh` — upgrade torch to 2.9.1+cu128 (matches the FA3 wheel ABI), verify FA3 imports, install brotli + sentencepiece
3. Run `get_data.sh` — fetch `docs_selected.jsonl` from HF, stash on container disk (/root) so it doesn't fill the 50 GB volume quota, symlink into the repo, tokenize SP8192 shards into `/workspace/paramgolf/data/datasets/datasets/fineweb10B_sp8192/`
4. Run `run.sh` — launch `train.py` with the right env vars (compile disabled for fast first run, TTT enabled, all our patches)

Total time on a fresh pod: **~50-80 min** (~5 min setup + ~30-60 min tokenize + ~10 min train).

## Manual / step-by-step

If you want to control each phase (e.g., to reuse a tokenized dataset across multiple runs):

```bash
# 0. Clone the repo
git clone https://github.com/taka6745/paramgolf.git /workspace/paramgolf
cd /workspace/paramgolf/submission

# 1. Pod environment (torch upgrade, FA3 verify, deps)
bash setup.sh

# 2. Get + tokenize data (~30-60 min CPU bound)
bash get_data.sh

# 3. Verify shards exist
ls /workspace/paramgolf/data/datasets/datasets/fineweb10B_sp8192/fineweb_train_*.bin | wc -l
# Should print ~120

# 4. Train + eval + serialize (10 min wallclock for the comp budget)
bash run.sh
```

## Outputs

After `run.sh` completes:
- `logs/train_<run_id>.log` — training log
- `final_model.pt` — fp32 EMA-applied model state dict
- `final_model.int6.ptz` — int6 GPTQ quantized + brotli-11 compressed (the submission artifact)
- Final val_bpb printed to log (look for `quantized_sliding_window val_loss: ... val_bpb: ...`)

## Pod requirements

- 1× NVIDIA H100 PCIe 80GB or 8× H100 SXM
- 100 GB container disk + 50 GB persistent volume
- Image: `runpod/pytorch:2.4.0+` (we'll upgrade torch in setup.sh)

## Cost notes

- 1× H100 PCIe ~$2.39/h on RunPod secure cloud
- 8× H100 SXM ~$22/h on RunPod secure cloud
- A single full submission run takes ~10 min training + ~5 min eval + ~2 min compression = ~17 min wallclock = ~$0.70 on 1× / ~$6.20 on 8×

## Disk topology gotcha

`/workspace` is the **50 GB persistent volume** (network filesystem). The 45 GB `docs_selected.jsonl` would blow this if naively placed there. `get_data.sh` puts it on the **100 GB container disk** at `/root/paramgolf_bigdata/` and symlinks into the repo. Without this, the tokenize fails with disk-full halfway through writing shards. (Lesson learned the painful way on 2026-04-09 — see `../PHASE1_TROUBLESHOOTING.md`.)

## Files

| file | purpose |
|---|---|
| `bootstrap.sh` | one-command setup from a fresh pod |
| `setup.sh` | pod environment (torch upgrade, FA3, deps) |
| `get_data.sh` | docs JSONL stage + SP8192 tokenize |
| `run.sh` | train + eval + serialize |
| `train.py` | the model (decoded PR #1477 base + our patches) |
| `requirements.txt` | python deps |
| `README.md` | this file |
