# PHASE1_TROUBLESHOOTING.md — append-only log

**Comp**: openai/parameter-golf
**Pod**: `9lfji49c6ngy9a` (paramgolf-phase1-h100, NVIDIA H100 PCIe 80GB, RunPod-rented)
**SSH**: `9lfji49c6ngy9a-64410a72@ssh.runpod.io`
**Cost**: $2.39/h, hard cap $15
**Mission**: this is the SAME machine the submission will run on. Every fix we apply
must produce a state that's reproducible on a fresh pod from the repo.

This file is append-only. Each entry: timestamp, what broke, what we did, why, and
whether the fix is "permanent" (in the repo) or "ad-hoc" (lives only on this pod).

## Operating rules

1. **Clean python files only** — no patcher hunks (`08_patch_train_gpt.sh`). The
   training entry point is `train_gpt_phase1.py` (decoded PR #1477 + FA3/SDPA
   fallback). The tokenize entry point is `data/download_hf_docs_and_tokenize.py`.
2. **Every workaround must be repo-checked-in** — if you SSH and `rm`/`mv` files,
   that's an ad-hoc fix and you must follow up with a permanent fix in the repo so
   the next clean pod boot can reproduce. Mark each entry below as PERMANENT or
   AD-HOC.
3. **Document the WHY** — not just what command, but what error/symptom led to it.
4. **No bypassing safety** — never `--no-verify`, never `git push --force`. If a
   commit hook fails, fix it.

---

## 2026-04-08 23:34Z — Pod cold-start state at session pickup

- runpodctl confirms `9lfji49c6ngy9a` is RUNNING.
- SSH works via proxy `9lfji49c6ngy9a-64410a72@ssh.runpod.io`.
- `/workspace/paramgolf` exists, repo at commit `b8f21c6` (NIGHT_MODE TERMINATED
  2106Z) — **5 commits behind `origin/main`**. Need `git pull` to get
  PHASE1_PLAN.md, train_gpt_phase1.py update, phase1_launch.sh.
- `train_gpt_phase1.py` IS on disk (50669 bytes, dated 23:09 UTC). This was pushed
  manually before the launch script commit; the file is current.
- `runpod_tests/loop/phase1_launch.sh` is NOT on the pod yet (post-pull).
- `logs/` is empty — no Phase 1 runs have been done.
- A tokenize job is running in the background: `python3
  data/download_hf_docs_and_tokenize.py --output-root data/datasets
  --tokenizer-config data/tokenizer_specs_8192.json` (PID 991, started 23:08 UTC,
  149% CPU = ~1.5 cores, no shards written after 26 minutes).
- GPU idle (0%, 1 MiB used, 27°C) — H100 burning $$ doing nothing while CPU
  tokenizes.

## 2026-04-08 23:35Z — DISK BLOCKER: 50 GB volume at 96%

- RunPod web telemetry showed Volume usage 48 GB / 50 GB (96%).
- `du -sh data/datasets/*` revealed `docs_selected.jsonl` = **48,166,275,520 bytes
  (45 GB)** sitting on the volume. The download script materialized the raw
  FineWeb-Edu text dump there before tokenize.
- Tokenize would have needed another ~24 GB to write SP8192 shards. 50 GB volume
  cannot fit 45 GB JSONL + 24 GB shards. Tokenize was guaranteed to disk-full.
- Container disk (100 GB overlayfs at `/`) was at 48% — plenty of room.

## 2026-04-08 23:39Z — FIX 1: Move JSONL to container disk (AD-HOC)

```bash
kill 991                              # stop tokenize
mkdir -p /root/paramgolf_bigdata
mv data/datasets/docs_selected.jsonl /root/paramgolf_bigdata/docs_selected.jsonl
ln -sfn /root/paramgolf_bigdata/docs_selected.jsonl data/datasets/docs_selected.jsonl
```

- Frees 45 GB on the volume → 24 GB shards now fit.
- Symlink lets the script find the JSONL at its expected path.
- **AD-HOC** — only lives on this pod. A fresh pod boot would re-download the JSONL
  to the volume and re-trigger the same disk-full. **Permanent fix needed**: change
  download script (or its launcher) to write the JSONL to a path the operator
  controls, not hard-coded to `output_root/`.

## 2026-04-08 23:43Z — Discovery: docs script always re-copies destination

- `copy_from_hf_cache()` in `data/download_hf_docs_and_tokenize.py:87-109` always
  unlinks and re-copies the destination, even when it already exists.
- This means re-running the script after our symlink fix would: unlink the symlink,
  attempt `os.link()` cross-FS (fails with EXDEV), fall back to `shutil.copy2()`,
  re-copy the 45 GB JSONL onto the volume — putting us right back at 96% full.

## 2026-04-08 23:45Z — FIX 2: env-gated skip-if-exists in copy_from_hf_cache (PERMANENT)

- Edited `data/download_hf_docs_and_tokenize.py` `copy_from_hf_cache()` to short-
  circuit when the destination exists AND `MATCHED_FINEWEB_SKIP_HF_COPY=1` is set.
- The env var is opt-in so other users / fresh runs are unaffected.
- Behavior with the var: if the destination file or symlink exists, return True
  without touching the HF cache or filesystem.
- **PERMANENT** — committed to repo. Future submission runs that want to reuse a
  pre-staged JSONL set the env var.

## 2026-04-08 23:45Z — FIX 3: free 45 GB on container disk by deleting HF cache (AD-HOC)

- After Fix 1, container disk `/` was at 93% (45 GB JSONL + 45 GB HF cache copy).
- HF cache (`/root/.cache/huggingface`) is only needed during the download phase
  and we have the JSONL already. Deleted it: `rm -rf /root/.cache/huggingface`.
- After delete, `/` back to 48%.
- **AD-HOC** — fresh pod boots wouldn't have the cache anyway, so this is
  effectively a no-op for the submission run.

## 2026-04-08 23:45Z — FIX 4: push pre-trained tokenizer model to skip 5-10 min train (AD-HOC → permanent later)

- The script's `build_tokenizers()` either trains a fresh SentencePiece BPE 8192
  on the JSONL (slow) or reuses a model file passed via `--reuse-sp-model
  8192=PATH`.
- We have `data/tokenizers/fineweb_8192_bpe.model` (370908 bytes) on the Mac,
  trained earlier. Pushed it via `/tmp/podpush.sh` to
  `/workspace/paramgolf/data/datasets/tokenizers/fineweb_8192_bpe.model`.
- **Permanent fix**: commit the tokenizer model to the repo (or to the HF dataset
  source) so a fresh pod can pull it. TODO before the submission run.

## 2026-04-08 23:45Z — FIX 5: delete corrupt sidecar on pod (AD-HOC)

- `data/datasets/docs_selected.source_manifest.json` (481 bytes) was a partial
  metadata dump from the killed tokenize. With Fix 2 in place, the script will
  short-circuit the sidecar copy if the destination exists, but the existing 481
  bytes is from the previous failed run and may not match the JSONL. Deleted it
  so the script re-fetches it from HF (small file, fast).
- **AD-HOC** — this is just cleanup of stale state.

---

## 2026-04-08 23:53Z — FIX 6: SP_MODEL must live OUTSIDE tokenizers_dir (PERMANENT)

- First tokenize attempt (PID 2124) crashed instantly with `FileNotFoundError:
  /workspace/paramgolf/data/datasets/tokenizers/fineweb_8192_bpe.model`.
- Root cause: `build_sentencepiece_tokenizer()` in
  `data/download_hf_docs_and_tokenize.py:264-272` unlinks `model_path` (= the
  destination model file) BEFORE checking the existence of `reuse_model_path`. If
  the operator passes `--reuse-sp-model 8192=<same path as destination>`, the
  source gets deleted out from under the reuse path.
- Fix: stash the canonical SP model in `/root/sp_models/fineweb_8192_bpe.model`
  (container disk, OUTSIDE the destination tokenizers_dir). Updated
  `runpod_tests/loop/phase1_tokenize.sh` to use that path.
- **PERMANENT** — checked into repo. The script bug is also fixable upstream
  (skip unlink when source == destination), TODO add a one-line guard.

## 2026-04-09 00:08Z — Cron fire 1: tokenize alive, 14 train + 1 val shard, 14 min elapsed

- PID 2280 alive, 2h27m cumulative CPU (~10 cores worth across the elapsed 14 min)
- Shards: 14 train + 1 val written, train_000013 latest
- Sustained rate: ~1 train shard / min (200 MB / 100 M tokens each)
- GPU: 0% (idle, CPU tokenize as expected)
- Disk: / at 48%, /workspace volume not under pressure
- ETA unchanged: ~120 train shards → done ~01:54Z = 11:54 AEST

## 2026-04-08 23:54Z — Tokenize launched, RUNNING ✓

- `bash runpod_tests/loop/phase1_tokenize.sh` from /workspace/paramgolf launched
  PID 2280.
- 326% CPU (3+ cores active), 469 MB resident.
- Log shows `Exporting dataset: fineweb10B_sp8192` — went straight to shard export
  (skipped tokenizer training because `--reuse-sp-model` worked).
- /workspace volume free, container disk / at 47%.
- Expected ETA 30-60 min from launch. First shards should appear in
  `data/datasets/datasets/fineweb10B_sp8192/fineweb_*_*.bin` within ~5-10 min.

---

## What still has to happen for the submission run to be reproducible from a fresh pod

## What still has to happen for the submission run to be reproducible from a fresh pod

(track these as repo TODOs — DON'T let them rot)

- [ ] Make `data/download_hf_docs_and_tokenize.py` accept a `--docs-jsonl` flag so
      the operator can point at a pre-staged JSONL on container disk instead of
      hard-coding `output_root/`. Removes the need for the symlink dance.
- [ ] Patch `build_sentencepiece_tokenizer()` to skip the `model_path.unlink()`
      when `reuse_model_path == model_path` (or use a temp dir for the unlink).
- [ ] Commit `data/tokenizers/fineweb_8192_bpe.model` to the repo (or document
      where to fetch it). Removes the need to push from Mac.
- [ ] Add a Phase 1 README or top-of-PHASE1_PLAN.md note about the 50 GB volume
      vs 100 GB container disk distinction so we don't get bitten again.

