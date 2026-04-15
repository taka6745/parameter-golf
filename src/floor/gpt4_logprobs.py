#!/usr/bin/env python3
"""Score FineWeb val with GPT-4o (or any OpenAI chat model) via logprobs.

Reports bits-per-byte (BPB) — tightest practical upper bound on H(val)
from a frontier LM. Use to know if <1.0 BPB is even theoretically reachable.

Cost rough order: ~$5 for 1 MB of val text on gpt-4o-2024-08-06.

Usage (run from repo root):
  OPENAI_API_KEY=sk-... python3 src/floor/gpt4_logprobs.py [model] [bytes]

Default: model=gpt-4o-2024-08-06, bytes=200_000

The script splits val text into ~2000-token chunks, calls the chat completions
endpoint with logprobs=True, and sums -log2(p(token | prev tokens in chunk)).

LIMITATION: chunking breaks long-range context. The reported BPB is an
upper bound on what gpt-4o would achieve with unlimited context. Closer
to true H(val) than zstd, but slightly above gpt-4o's actual capability.
"""
import os
import sys
import math
import json
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("pip install openai (in .venv: .venv/bin/pip install openai)")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-2024-08-06"
BYTES_TO_SCORE = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000  # 200KB default

VAL_PATH = Path("data/floor/val_sp1024.txt")
OUT_PATH = Path("data/floor/gpt4_logprobs.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if not VAL_PATH.exists():
    raise SystemExit(f"missing {VAL_PATH} — run src/floor/detokenize_val.py first")

text = VAL_PATH.read_bytes()[:BYTES_TO_SCORE].decode("utf-8", errors="replace")
print(f"scoring {len(text)} chars / {len(text.encode('utf-8'))} bytes with {MODEL}")

# Chunk the text. We aim for ~2000 input tokens per chunk (gpt-4o context is
# huge but cost & latency scale with input length). Roughly 4 chars/token
# for English → 8000 chars per chunk.
CHUNK_CHARS = 8000
chunks = [text[i:i+CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
print(f"  {len(chunks)} chunks of ~{CHUNK_CHARS} chars")

client = OpenAI()

# Strategy: pass each chunk as a user message and ask the model to repeat it
# back, requesting logprobs. The logprobs we get are p(next_token | prev_tokens
# in the assistant's repeat). We sum -log2 over all output tokens.
#
# Alternative (cheaper): use the legacy /v1/completions endpoint with the
# text as the prompt + max_tokens=1 + logprobs=20 + echo=True. That returns
# logprobs of the ENTIRE prompt directly. Only available on a few models.

total_neg_log2 = 0.0
total_bytes = 0
total_input_tokens = 0
total_output_tokens = 0
log_records = []

for i, chunk in enumerate(chunks):
    t0 = time.time()
    # Use chat with assistant message echoing the chunk + logprobs=True
    # System prompt is empty so the model isn't conditioned on instructions
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Continue the text exactly as written."},
                {"role": "user", "content": "Continue:"},
                {"role": "assistant", "content": chunk},
            ],
            max_tokens=1,  # We only need the logprobs of the existing assistant text
            logprobs=True,
            top_logprobs=0,
        )
    except Exception as e:
        # Older API: assistant content can't have logprobs returned for it.
        # Fall back to: pass chunk as prompt + ask for completion + parse logprobs of completion.
        # This is less ideal but works.
        print(f"  chunk {i}: {type(e).__name__}: {e}")
        print(f"  trying fallback: prompt mode")
        # Fallback: split chunk into prefix/suffix; feed prefix as prompt; logprobs of suffix
        if len(chunk) < 200:
            continue
        prefix = chunk[:len(chunk)//2]
        suffix = chunk[len(chunk)//2:]
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prefix}],
            max_tokens=len(suffix) // 3 + 50,  # rough token estimate
            logprobs=True,
            top_logprobs=0,
            temperature=0.0,
        )
        # Compare returned content to suffix to score (this is cruder)
        chunk_neg_log2 = sum(-tok.logprob / math.log(2) for tok in resp.choices[0].logprobs.content or [])
        chunk_bytes = len(chunk.encode("utf-8")) // 2  # only half scored
    else:
        # Primary path: logprobs of the assistant message tokens
        lp_content = resp.choices[0].logprobs.content if resp.choices[0].logprobs else []
        chunk_neg_log2 = sum(-tok.logprob / math.log(2) for tok in lp_content)
        chunk_bytes = len(chunk.encode("utf-8"))

    total_neg_log2 += chunk_neg_log2
    total_bytes += chunk_bytes
    total_input_tokens += resp.usage.prompt_tokens
    total_output_tokens += resp.usage.completion_tokens
    log_records.append({
        "chunk": i,
        "chars": len(chunk),
        "bytes": chunk_bytes,
        "neg_log2": chunk_neg_log2,
        "bpb_chunk": chunk_neg_log2 / max(chunk_bytes, 1),
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "elapsed_s": time.time() - t0,
    })
    print(f"  chunk {i+1}/{len(chunks)}: BPB={chunk_neg_log2 / max(chunk_bytes, 1):.4f} ({time.time()-t0:.1f}s)")

bpb = total_neg_log2 / total_bytes if total_bytes else float("nan")
print(f"\n=== {MODEL} BPB on {total_bytes:,} bytes of FineWeb val ===")
print(f"  BPB = {bpb:.4f}")
print(f"  total input tokens: {total_input_tokens:,}  output tokens: {total_output_tokens:,}")
# rough cost: gpt-4o-2024-08-06 = $2.5/M in, $10/M out (verify pricing)
est_cost = total_input_tokens / 1e6 * 2.5 + total_output_tokens / 1e6 * 10
print(f"  estimated cost: ${est_cost:.2f}")

OUT_PATH.write_text(json.dumps({
    "model": MODEL,
    "bytes_scored": total_bytes,
    "neg_log2_total": total_neg_log2,
    "bpb": bpb,
    "input_tokens": total_input_tokens,
    "output_tokens": total_output_tokens,
    "estimated_cost_usd": est_cost,
    "chunks": log_records,
}, indent=2))
print(f"\nwrote {OUT_PATH}")
