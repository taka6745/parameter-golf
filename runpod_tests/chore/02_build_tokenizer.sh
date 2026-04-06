#!/bin/bash
# 02_build_tokenizer.sh — Build BPE-8192 tokenizer
# Runs on: any pod (CPU work)
# Time: ~5 min
#
# WHY: BPE-8192 is our biggest single lever (-0.129 BPP).
# The default sp1024 tokenizer leaves money on the table.

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=== BUILD BPE-8192 TOKENIZER ==="
echo

if [ -f "data/tokenizers/fineweb_8192_bpe.model" ]; then
    echo "✓ Tokenizer already exists, skipping"
    exit 0
fi

mkdir -p data/tokenizers

# Use sentencepiece BPE training
# Sample 1M docs from FineWeb for vocab building
python3 << 'PYEOF'
import sentencepiece as spm
import os
from datasets import load_dataset

print("Loading FineWeb sample...")
ds = load_dataset("HuggingFaceFW/fineweb", "default", split="train", streaming=True)

# Write samples to a file for SP training
sample_file = "/tmp/fineweb_sample.txt"
n_docs = 0
target_docs = 1_000_000
with open(sample_file, "w") as f:
    for doc in ds:
        if n_docs >= target_docs:
            break
        text = doc.get("text", "")
        if text:
            f.write(text + "\n")
            n_docs += 1
        if n_docs % 50000 == 0:
            print(f"  {n_docs}/{target_docs}")

print(f"Wrote {n_docs} docs to {sample_file}")
print("Training BPE-8192 tokenizer...")

spm.SentencePieceTrainer.train(
    input=sample_file,
    model_prefix="data/tokenizers/fineweb_8192_bpe",
    vocab_size=8192,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    input_sentence_size=1_000_000,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=False,
)

print("✓ Tokenizer trained")
os.remove(sample_file)
PYEOF

ls -la data/tokenizers/
echo
echo "✓ BPE-8192 tokenizer ready"
