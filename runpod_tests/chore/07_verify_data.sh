#!/bin/bash
# 07_verify_data.sh — Sanity check all chore outputs
# Time: ~1 min

set -e
cd /workspace/paramgolf
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=== VERIFY DATA ==="
echo

PASS=0
FAIL=0

check() {
    local name=$1
    local cmd=$2
    if eval "$cmd" > /dev/null 2>&1; then
        echo "  ✓ $name"
        PASS=$((PASS+1))
    else
        echo "  ✗ $name"
        FAIL=$((FAIL+1))
    fi
}

echo "Tokenizer:"
check "BPE-8192 model exists" "[ -f data/tokenizers/fineweb_8192_bpe.model ]"
check "BPE-8192 vocab is 8192" "python3 -c 'import sentencepiece; sp=sentencepiece.SentencePieceProcessor(); sp.load(\"data/tokenizers/fineweb_8192_bpe.model\"); assert sp.vocab_size()==8192'"

echo
echo "Data:"
check "BPE-8192 train shards (>=10)" "[ \$(ls data/datasets/fineweb10B_bpe8192/fineweb_train_*.bin 2>/dev/null | wc -l) -ge 10 ]"
check "BPE-8192 val shard exists" "ls data/datasets/fineweb10B_bpe8192/fineweb_val_*.bin 2>/dev/null"

echo
echo "N-gram tables:"
check "Bigram table" "[ -f data/bigram_tab_8192v.npy ]"
check "Bigram shape (16384, 8192)" "python3 -c 'import numpy; t=numpy.load(\"data/bigram_tab_8192v.npy\"); assert t.shape==(16384,8192), t.shape'"
check "Trigram table" "[ -f data/trigram_logprobs_8192v.npy ]"
check "4-gram table" "[ -f data/fourgram_logprobs_8192v.npy ]"

echo
echo "DC categories:"
check "DC500" "[ -f data/dist_cats_500_8192.npz ]"

echo
echo "Quantization:"
check "Lloyd-Max codebook" "[ -f data/lloyd_max_codebook_64.npy ]"
check "Codebook has 64 levels" "python3 -c 'import numpy; t=numpy.load(\"data/lloyd_max_codebook_64.npy\"); assert len(t)==64'"

echo
echo "GPU:"
check "CUDA available" "python3 -c 'import torch; assert torch.cuda.is_available()'"

echo
echo "==============="
echo "PASS: $PASS / $((PASS+FAIL))"
[ $FAIL -eq 0 ] && echo "✓ ALL CHORES COMPLETE" || echo "✗ $FAIL CHECKS FAILED — fix before proceeding"
exit $FAIL
