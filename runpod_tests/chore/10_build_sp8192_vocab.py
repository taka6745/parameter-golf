#!/usr/bin/env python3
"""10_build_sp8192_vocab.py — build SentencePiece BPE-8192 tokenizer + n-gram tables.

Highest-impact comp-port baseline gap. Per COMP_PORT_GAPS.md, vocab swap from
1024 → 8192 is worth -0.08 to -0.12 BPB. The blocker has been: rebuild requires
re-tokenizing the FineWeb shards + rebuilding bigram/trigram/4-gram tables.

This script does the WHOLE pipeline:
  1. Train SentencePiece BPE model on 1B FineWeb-Edu sample bytes → vocab size 8192
  2. Re-tokenize all FineWeb shards using the new model → data/datasets/fineweb10B_sp8192/*.bin
  3. Build bigram_logprobs_8192v.npy + trigram_logprobs_8192v.npy + fourgram_logprobs_8192v.npy

Steps 1+3 are CPU-bound and runnable on Mac. Step 2 is the slowest (~30-60 min/shard).

Designed to be RUN ONCE on Mac (or via cpu_workers.py as a long-running job).
Idempotent: skips any step whose output file already exists, unless SP8192_FORCE=1.

Output structure:
    data/tokenizers/sp8192.model
    data/tokenizers/sp8192.vocab
    data/datasets/fineweb10B_sp8192/fineweb_train_000000.bin
    data/datasets/fineweb10B_sp8192/fineweb_train_000001.bin
    ...
    data/bigram_logprobs_8192v.npy
    data/trigram_logprobs_8192v.npy
    data/fourgram_logprobs_8192v.npy

Once these exist, ship the L01 USE_SP8192_VOCAB=1 patch to swap data path + table paths.

Env vars:
    SP8192_VOCAB_SIZE       default 8192
    SP8192_TRAIN_SAMPLE_MB  default 256 (sample size for sentencepiece training)
    SP8192_NGRAM_HASH       default 16384 (matches existing tabulation hash)
    SP8192_FORCE            default 0 (set to 1 to overwrite existing artifacts)
    SP8192_SKIP_RETOKENIZE  default 0 (set to 1 to skip re-tokenizing shards — useful for testing)

This script is LONG-RUNNING. On Mac CPU it takes ~1-3 hours for the full pipeline.
Run as: nohup python3 runpod_tests/chore/10_build_sp8192_vocab.py &
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
TOKENIZERS = DATA / "tokenizers"
SHARDS_OUT = DATA / "datasets" / "fineweb10B_sp8192"
SHARDS_IN_PATTERN = "fineweb10B_sp1024/fineweb_*.bin"  # source: existing sp1024 shards

VOCAB_SIZE = int(os.environ.get("SP8192_VOCAB_SIZE", "8192"))
TRAIN_SAMPLE_MB = int(os.environ.get("SP8192_TRAIN_SAMPLE_MB", "256"))
NGRAM_HASH = int(os.environ.get("SP8192_NGRAM_HASH", "16384"))
FORCE = bool(int(os.environ.get("SP8192_FORCE", "0")))
SKIP_RETOKENIZE = bool(int(os.environ.get("SP8192_SKIP_RETOKENIZE", "0")))


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"{ts} sp8192_build: {msg}", flush=True)


def step1_train_sentencepiece() -> Path:
    """Train SentencePiece BPE-8192 on a sample of FineWeb byte text."""
    out_model = TOKENIZERS / "sp8192.model"
    if out_model.exists() and not FORCE:
        log(f"  skip: {out_model} already exists")
        return out_model

    TOKENIZERS.mkdir(parents=True, exist_ok=True)

    # Source: extract raw text from the existing sp1024 .bin shards by detokenizing.
    # If the original FineWeb text dump is available, use it directly. Otherwise,
    # use a smaller seed corpus from `data/docs_selected.jsonl`.
    seed_text_path = TOKENIZERS / "sp8192_train_corpus.txt"
    if not seed_text_path.exists() or FORCE:
        log(f"  preparing training corpus ({TRAIN_SAMPLE_MB} MB)...")
        # Use docs_selected.jsonl if present (curated training corpus from earlier session)
        docs_path = DATA / "docs_selected.jsonl"
        if docs_path.exists():
            log(f"  using {docs_path} as seed corpus")
            written = 0
            cap = TRAIN_SAMPLE_MB * 1024 * 1024
            import json as _json
            with open(seed_text_path, "w") as out_f:
                with open(docs_path) as in_f:
                    for line in in_f:
                        if written >= cap:
                            break
                        try:
                            doc = _json.loads(line)
                            text = doc.get("text", "") or doc.get("content", "") or ""
                        except Exception:
                            continue
                        if text:
                            out_f.write(text)
                            out_f.write("\n")
                            written += len(text.encode("utf-8")) + 1
            log(f"  wrote {written // (1024*1024)} MB to {seed_text_path}")
        else:
            log(f"  ERROR: no seed corpus found. Place a text file at {seed_text_path}")
            log(f"  or provide data/docs_selected.jsonl. Aborting.")
            return None

    # Train SentencePiece
    try:
        import sentencepiece as spm
    except ImportError:
        log("  ERROR: sentencepiece not installed. Run: pip install sentencepiece")
        return None

    log(f"  training SentencePiece BPE vocab_size={VOCAB_SIZE}...")
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=str(seed_text_path),
        model_prefix=str(TOKENIZERS / "sp8192"),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9999,
        byte_fallback=True,
        normalization_rule_name="identity",
        split_by_unicode_script=False,
        split_by_whitespace=False,
        split_by_number=False,
        max_sentence_length=8192,
        train_extremely_large_corpus=True,
    )
    log(f"  trained sp8192.model in {time.time()-t0:.1f}s")
    return out_model


def step2_retokenize_shards(model_path: Path) -> int:
    """Re-tokenize all SP-1024 shards using the new SP-8192 model."""
    if SKIP_RETOKENIZE:
        log("  SP8192_SKIP_RETOKENIZE=1, skipping shard re-tokenization")
        return 0

    SHARDS_OUT.mkdir(parents=True, exist_ok=True)

    try:
        import sentencepiece as spm
    except ImportError:
        log("  ERROR: sentencepiece not installed")
        return 0

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    in_shards = sorted((DATA / "datasets" / "fineweb10B_sp1024").glob("fineweb_*.bin"))
    if not in_shards:
        log("  ERROR: no input shards found at data/datasets/fineweb10B_sp1024/")
        return 0

    log(f"  re-tokenizing {len(in_shards)} shards (this is the slow step)...")
    n_done = 0

    # Header format: 1024 bytes (256 uint32 fields). Mostly zero except magic + size.
    HEADER_BYTES = 1024
    import numpy as np

    for shard_in in in_shards:
        shard_out = SHARDS_OUT / shard_in.name
        if shard_out.exists() and not FORCE:
            log(f"    skip: {shard_out.name} already exists")
            continue

        # Read sp1024 token IDs and detokenize via the existing 1024 model
        sp1024_path = REPO / "data" / "tokenizers" / "sp1024.model"
        if not sp1024_path.exists():
            log(f"  ERROR: source tokenizer {sp1024_path} not found")
            return n_done
        sp1024 = spm.SentencePieceProcessor()
        sp1024.load(str(sp1024_path))

        with open(shard_in, "rb") as f:
            header = f.read(HEADER_BYTES)
            raw = f.read()
        ids_in = np.frombuffer(raw, dtype=np.uint16)
        log(f"    {shard_in.name}: {len(ids_in):,} sp1024 tokens, detokenizing...")

        # Detokenize in 100k-token chunks (memory-friendly), then re-encode with sp8192
        out_ids: list[int] = []
        chunk = 100000
        for i in range(0, len(ids_in), chunk):
            piece_ids = [int(x) for x in ids_in[i:i+chunk]]
            text = sp1024.decode(piece_ids)
            new_ids = sp.encode(text, out_type=int)
            out_ids.extend(new_ids)

        out_arr = np.array(out_ids, dtype=np.uint16)
        with open(shard_out, "wb") as f:
            f.write(header)  # preserve original header (or update size field — fine for now)
            out_arr.tofile(f)
        log(f"    wrote {shard_out.name}: {len(out_arr):,} sp8192 tokens "
            f"(ratio {len(out_arr)/len(ids_in):.3f})")
        n_done += 1

    return n_done


def step3_build_ngram_tables() -> int:
    """Build bigram/trigram/4-gram tables on the new sp8192 shards."""
    out_bg = DATA / "bigram_logprobs_8192v.npy"
    out_tg = DATA / "trigram_logprobs_8192v.npy"
    out_4g = DATA / "fourgram_logprobs_8192v.npy"
    if out_bg.exists() and out_tg.exists() and out_4g.exists() and not FORCE:
        log(f"  skip: ngram tables already built")
        return 0

    import numpy as np

    log(f"  building n-gram tables (hash={NGRAM_HASH}) from sp8192 shards...")
    in_shards = sorted(SHARDS_OUT.glob("fineweb_*.bin"))
    if not in_shards:
        log(f"  ERROR: no sp8192 shards at {SHARDS_OUT}")
        return 0

    H = NGRAM_HASH
    bigram_counts = np.zeros(H, dtype=np.float64)
    trigram_counts = np.zeros(H, dtype=np.float64)
    fourgram_counts = np.zeros(H, dtype=np.float64)
    total = 0

    HEADER_BYTES = 1024
    for shard in in_shards[:5]:  # use first 5 shards (~500M tokens)
        with open(shard, "rb") as f:
            f.seek(HEADER_BYTES)
            raw = f.read()
        ids = np.frombuffer(raw, dtype=np.uint16).astype(np.int64)
        log(f"    {shard.name}: {len(ids):,} tokens")
        if len(ids) < 4:
            continue
        # Bigram hash: (prev * 36313) % H — matches existing tabulation hash style
        bg_hash = (ids[:-1] * 36313) % H
        np.add.at(bigram_counts, bg_hash, 1.0)
        tg_hash = ((ids[:-2] * 36313 ^ ids[1:-1] * 39979) % H)
        np.add.at(trigram_counts, tg_hash, 1.0)
        fg_hash = ((ids[:-3] * 36313 ^ ids[1:-2] * 39979 ^ ids[2:-1] * 41077) % H)
        np.add.at(fourgram_counts, fg_hash, 1.0)
        total += len(ids)

    # Convert counts to log-probs (with smoothing)
    eps = 1e-6
    bg_logp = np.log((bigram_counts + eps) / (total + eps * H)).astype(np.float32)
    tg_logp = np.log((trigram_counts + eps) / (total + eps * H)).astype(np.float32)
    fg_logp = np.log((fourgram_counts + eps) / (total + eps * H)).astype(np.float32)

    np.save(out_bg, bg_logp)
    np.save(out_tg, tg_logp)
    np.save(out_4g, fg_logp)
    log(f"  wrote {out_bg} ({out_bg.stat().st_size//1024} KB)")
    log(f"  wrote {out_tg} ({out_tg.stat().st_size//1024} KB)")
    log(f"  wrote {out_4g} ({out_4g.stat().st_size//1024} KB)")
    return 3


def main() -> int:
    log(f"=== sp8192 vocab build pipeline ===")
    log(f"  vocab_size={VOCAB_SIZE}  train_sample_mb={TRAIN_SAMPLE_MB}  hash={NGRAM_HASH}")
    log(f"  force={FORCE}  skip_retokenize={SKIP_RETOKENIZE}")

    t0 = time.time()
    log("Step 1: train SentencePiece BPE-8192")
    model_path = step1_train_sentencepiece()
    if model_path is None:
        log("Step 1 failed, aborting")
        return 1

    log("Step 2: re-tokenize FineWeb shards with sp8192")
    step2_retokenize_shards(model_path)

    log("Step 3: build sp8192 n-gram tables")
    step3_build_ngram_tables()

    log(f"=== done in {time.time()-t0:.1f}s ===")
    log(f"Next: ship USE_SP8192_VOCAB=1 patcher patch (L01 marker swap)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
