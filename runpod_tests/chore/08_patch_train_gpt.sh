#!/bin/bash
# 08_patch_train_gpt.sh — Patch train_gpt.py for PyTorch 2.4 (no enable_gqa)
#
# WHY: The competition's train_gpt.py uses
#   F.scaled_dot_product_attention(..., enable_gqa=True)
# which was added in PyTorch 2.5+. RunPod base images often ship 2.4,
# and we already chose to use the system torch in 00_setup_pod.sh
# to avoid CUDA driver mismatches.
#
# This script:
#   1. Detects whether the pod's torch supports enable_gqa
#   2. If not, patches train_gpt.py to manually repeat KV heads instead
#   3. Disables torch.compile (which fails on the same call)
#   4. Creates train_gpt.py.bak before modifying
#
# Idempotent: if already patched, does nothing.

set -e
echo "=== PATCH train_gpt.py FOR PyTorch 2.4 ==="
echo

# Check if torch supports enable_gqa
SUPPORTS_GQA=$(python3 -c "
import torch
import torch.nn.functional as F
import inspect
sig = inspect.signature(F.scaled_dot_product_attention)
print('yes' if 'enable_gqa' in sig.parameters else 'no')
" 2>/dev/null || echo "no")

echo "  torch supports enable_gqa: $SUPPORTS_GQA"

if [ "$SUPPORTS_GQA" = "yes" ]; then
    echo "  ✓ No patch needed"
    exit 0
fi

# Backup (only if no backup exists yet — never overwrite the original)
if [ ! -f train_gpt.py.bak ]; then
    cp train_gpt.py train_gpt.py.bak
    echo "  ✓ backup → train_gpt.py.bak"
else
    echo "  ✓ backup already exists at train_gpt.py.bak"
fi

# The patcher is idempotent — each individual patch checks for its old
# block, and if the block is missing (because that patch was already
# applied) the replace is a no-op. So it's safe to re-run after we add
# new patches without restoring from backup first.

# Patch 1: replace the F.scaled_dot_product_attention call to manually
# repeat KV heads (since enable_gqa doesn't exist).
# Patch 2: disable torch.compile (also fails on enable_gqa).
# Patch 3: honor SKIP_FINAL_EVAL=1 to bail out before the slow int8+zlib
#          val pass — useful for SIGNAL tests where we just want train_loss
#          and don't care about the final quantized number.
python3 << 'PYEOF'
with open("train_gpt.py", "r") as f:
    content = f.read()

# Add a marker so we know we patched it
if "PATCHED_FOR_TORCH24" not in content:
    content = "# PATCHED_FOR_TORCH24: enable_gqa removed, KV heads repeated manually\n" + content

# Find the SDP call with enable_gqa and replace it with a manual GQA version
old_block = """        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )"""

new_block = """        # PATCHED: manually repeat KV heads for GQA (PyTorch 2.4 has no enable_gqa)
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
        )"""

if old_block in content:
    content = content.replace(old_block, new_block)
    print("  ✓ patched scaled_dot_product_attention call")
else:
    print("  ✗ couldn't find SDP call to patch (already different?)")

# Patch 2: disable torch.compile (fails on same enable_gqa issue)
old_compile = "compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)"
new_compile = "compiled_model = base_model  # PATCHED: torch.compile disabled for PyTorch 2.4"
if old_compile in content:
    content = content.replace(old_compile, new_compile)
    print("  ✓ disabled torch.compile on model")

# Also disable the optimizer compile
old_opt = "zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)"
new_opt = "# PATCHED: zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)"
if old_opt in content:
    content = content.replace(old_opt, new_opt)
    print("  ✓ disabled torch.compile on Newton-Schulz")

# Patch 3: honor PROGRESSIVE_SEQ env var (in-loop seq + LR scheduling).
# Reads PROGRESSIVE_SEQ, PHASE1_SEQ_LEN, PHASE2_SEQ_LEN, PHASE1_LR_MULT,
# PHASE1_FRACTION at startup. When elapsed >= phase1_fraction * wallclock,
# mutates args.train_seq_len to PHASE2_SEQ_LEN and drops the LR multiplier.
if "PROG_SEQ_INIT_MARKER" in content:
    print("  ✓ progressive seq init already applied")
    old_loop_top = None
else:
    old_loop_top = """    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:"""
new_loop_top = """    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # PROG_SEQ_INIT_MARKER — progressive seq scheduling
    _prog_seq = bool(int(os.environ.get("PROGRESSIVE_SEQ", "0")))
    _phase1_seq = int(os.environ.get("PHASE1_SEQ_LEN", "128"))
    _phase2_seq = int(os.environ.get("PHASE2_SEQ_LEN", "1024"))
    _phase1_lr_mult = float(os.environ.get("PHASE1_LR_MULT", "1.0"))
    _phase2_lr_mult = float(os.environ.get("PHASE2_LR_MULT", "1.0"))
    _phase1_frac = float(os.environ.get("PHASE1_FRACTION", "0.85"))
    _current_phase = 1 if _prog_seq else 2
    _wallclock_for_phase = max_wallclock_ms or (args.iterations * 100.0)
    _phase1_end_ms = _wallclock_for_phase * _phase1_frac
    _prog_lr_mult = _phase1_lr_mult if _prog_seq else 1.0
    if _prog_seq:
        args.train_seq_len = _phase1_seq
        log0(f"PROGRESSIVE_SEQ enabled: phase1 seq={_phase1_seq} lr_mult={_phase1_lr_mult} "
             f"phase2 seq={_phase2_seq} lr_mult={_phase2_lr_mult} phase1_end_ms={_phase1_end_ms:.0f}")

    step = 0
    while True:"""
if old_loop_top is not None and old_loop_top in content:
    content = content.replace(old_loop_top, new_loop_top)
    print("  ✓ added progressive seq init block")

# Phase transition check + LR scaling — inject before the optimizer step
if "PHASE_TRANSITION_MARKER" in content:
    print("  ✓ phase transition + LR scaling already applied")
    old_lr_apply = None
else:
    old_lr_apply = """        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale"""
new_lr_apply = """        # PHASE_TRANSITION_MARKER: progressive seq phase transition (uses elapsed_ms)
        if _prog_seq and _current_phase == 1 and elapsed_ms >= _phase1_end_ms:
            _current_phase = 2
            args.train_seq_len = _phase2_seq
            _prog_lr_mult = _phase2_lr_mult
            log0(f"PHASE TRANSITION at step {step}: seq {_phase1_seq} -> {_phase2_seq}, "
                 f"lr_mult {_phase1_lr_mult} -> {_phase2_lr_mult}")
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale * _prog_lr_mult"""
if old_lr_apply is not None and old_lr_apply in content:
    content = content.replace(old_lr_apply, new_lr_apply)
    print("  ✓ added phase transition + LR scaling")

# Patch 4: honor SKIP_FINAL_EVAL=1 to skip the int8+zlib roundtrip eval
# (saves 4-5 minutes per run when we only want signal/relative comparisons).
if "SKIP_FINAL_EVAL_MARKER" in content:
    print("  ✓ SKIP_FINAL_EVAL already applied")
    old_int8 = None
else:
    old_int8 = """    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()"""
    new_int8 = """    # SKIP_FINAL_EVAL_MARKER
    if os.environ.get("SKIP_FINAL_EVAL", "0") == "1":
        log0("SKIP_FINAL_EVAL=1 — skipping int8+zlib roundtrip eval (signal mode)")
        if distributed:
            dist.destroy_process_group()
        sys.exit(0)
    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()"""

if old_int8 is not None and old_int8 in content:
    content = content.replace(old_int8, new_int8)
    print("  ✓ added SKIP_FINAL_EVAL env var support")
    # Make sure sys is imported (it usually is, but check)
    if "import sys" not in content:
        content = "import sys\n" + content

# Patch 5: clamp PHASE2_SEQ_LEN to microbatch capacity at transition.
# Without this, setting train_seq_len = 1024 mid-run when microbatch is 128
# (e.g. TRAIN_BATCH_TOKENS=1024 / hardcoded grad_accum=8 = 128 tokens) causes
# `local[:-1].reshape(-1, 1024)` to crash on the very next next_batch() call.
# This patch upgrades the unclamped V1 phase-transition block (inserted by
# Patch 3 above on first run) to a clamped version. Idempotent via PHASE_TRANSITION_CLAMP marker.
if "PHASE_TRANSITION_CLAMP" in content:
    print("  ✓ phase transition microbatch clamp already applied")
else:
    old_unclamped = """        # PHASE_TRANSITION_MARKER: progressive seq phase transition (uses elapsed_ms)
        if _prog_seq and _current_phase == 1 and elapsed_ms >= _phase1_end_ms:
            _current_phase = 2
            args.train_seq_len = _phase2_seq
            _prog_lr_mult = _phase2_lr_mult
            log0(f"PHASE TRANSITION at step {step}: seq {_phase1_seq} -> {_phase2_seq}, "
                 f"lr_mult {_phase1_lr_mult} -> {_phase2_lr_mult}")"""
    new_clamped = """        # PHASE_TRANSITION_MARKER PHASE_TRANSITION_CLAMP: clamp phase2 seq to microbatch
        if _prog_seq and _current_phase == 1 and elapsed_ms >= _phase1_end_ms:
            _current_phase = 2
            _max_micro = args.train_batch_tokens // (world_size * grad_accum_steps)
            _effective_p2 = min(_phase2_seq, _max_micro)
            args.train_seq_len = _effective_p2
            _prog_lr_mult = _phase2_lr_mult
            log0(f"PHASE TRANSITION at step {step}: seq {_phase1_seq} -> {_effective_p2} (microbatch_max={_max_micro}), "
                 f"lr_mult {_phase1_lr_mult} -> {_phase2_lr_mult}")"""
    if old_unclamped in content:
        content = content.replace(old_unclamped, new_clamped)
        print("  ✓ upgraded phase transition to clamped version (avoids reshape crash)")
    else:
        print("  ! couldn't find unclamped phase transition block — clamp not applied")

# Patch 7: SKIP_FINAL_EVAL also disables the last-step val pass + bails before GPTQ quant
# Without this, even with SKIP_FINAL_EVAL=1 the training loop's `should_validate = last_step or ...`
# triggers a full val eval (62M tokens, ~20 min on a 3080 Ti) when wallclock fires.
# We also bail out earlier — right after `peak memory allocated` — so we skip the post-loop
# int8 quant + zlib compress + checkpoint save when in signal mode.
if "SKIP_LAST_VAL_MARKER" in content:
    print("  ✓ skip-last-val already applied")
else:
    old_should_validate = """        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:"""
    new_should_validate = """        # SKIP_LAST_VAL_MARKER: with SKIP_FINAL_EVAL=1 the last-step val pass is skipped
        _skip_last_val = os.environ.get("SKIP_FINAL_EVAL", "0") == "1"
        should_validate = (last_step and not _skip_last_val) or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:"""
    if old_should_validate in content:
        content = content.replace(old_should_validate, new_should_validate)
        print("  ✓ added skip-last-val patch")

if "SKIP_POST_LOOP_MARKER" in content:
    print("  ✓ skip-post-loop already applied")
else:
    old_post_loop = """    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )"""
    new_post_loop = """    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    # SKIP_POST_LOOP_MARKER: in signal mode, bail out before any quant/save/eval work
    if os.environ.get("SKIP_FINAL_EVAL", "0") == "1":
        log0("SKIP_FINAL_EVAL=1 — bailing before GPTQ quant + zlib + final eval")
        if distributed:
            dist.destroy_process_group()
        sys.exit(0)"""
    if old_post_loop in content:
        content = content.replace(old_post_loop, new_post_loop)
        print("  ✓ added skip-post-loop patch")

# Patch 6: USE_NGRAM_BIAS=1 → load bigram/trigram/fourgram tables and add as logit bias
# Tables are built by 04_build_ngrams.py:
#   bigram_tab_1024v.npy   shape (HASH=2048, V=1024) — log P(next | prev)
#   trigram_logprobs_1024v.npy   shape (2048, 1024) — log P(next | prev2, prev1)
#   fourgram_logprobs_1024v.npy  shape (2048, 1024) — log P(next | prev3, prev2, prev1)
# Hash polynomials match 04_build_ngrams.py exactly:
#   bigram:   (prev * 36313) % 2048
#   trigram:  (prev2 * 36313 + prev1 * 27191) % 2048
#   fourgram: (prev3 * 36313 + prev2 * 27191 + prev1 * 51497) % 2048
# Mac-validated: bigram alone -0.01 BPB, +trigram another -0.02, +fourgram another -0.02.
# Idempotent via NGRAM_BIAS_MARKER.
if "NGRAM_BIAS_MARKER" in content:
    print("  ✓ ngram bias already applied")
else:
    # Inject buffer loading + weights right after self._init_weights() at end of __init__
    old_init_end = """        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()"""
    new_init_end = """        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        # NGRAM_BIAS_MARKER: load precomputed n-gram log-prob tables as buffers
        self._ngram_enabled = bool(int(os.environ.get("USE_NGRAM_BIAS", "0")))
        self._ngram_w_bigram = float(os.environ.get("NGRAM_W_BIGRAM", "0.20"))
        self._ngram_w_trigram = float(os.environ.get("NGRAM_W_TRIGRAM", "0.15"))
        self._ngram_w_fourgram = float(os.environ.get("NGRAM_W_FOURGRAM", "0.10"))
        self._ngram_hash = 2048  # must match 04_build_ngrams.py HASH_BUCKETS
        self.register_buffer("_bigram_tab", torch.zeros(1, dtype=torch.float32), persistent=False)
        self.register_buffer("_trigram_tab", torch.zeros(1, dtype=torch.float32), persistent=False)
        self.register_buffer("_fourgram_tab", torch.zeros(1, dtype=torch.float32), persistent=False)
        if self._ngram_enabled:
            import numpy as _np
            try:
                _bg = _np.load("./data/bigram_tab_{}v.npy".format(vocab_size))
                self._bigram_tab = torch.from_numpy(_bg).float()
                print("NGRAM_BIAS: loaded bigram", _bg.shape, "w=", self._ngram_w_bigram)
            except Exception as _e:
                print("NGRAM_BIAS: bigram load failed:", _e)
            try:
                _tg = _np.load("./data/trigram_logprobs_{}v.npy".format(vocab_size))
                self._trigram_tab = torch.from_numpy(_tg).float()
                print("NGRAM_BIAS: loaded trigram", _tg.shape, "w=", self._ngram_w_trigram)
            except Exception as _e:
                print("NGRAM_BIAS: trigram load failed:", _e)
            try:
                _fg = _np.load("./data/fourgram_logprobs_{}v.npy".format(vocab_size))
                self._fourgram_tab = torch.from_numpy(_fg).float()
                print("NGRAM_BIAS: loaded fourgram", _fg.shape, "w=", self._ngram_w_fourgram)
            except Exception as _e:
                print("NGRAM_BIAS: fourgram load failed:", _e)
        self._init_weights()"""
    if old_init_end in content:
        content = content.replace(old_init_end, new_init_end)
        print("  ✓ added NGRAM_BIAS init/loading")

    # Inject bias-add right after softcap, before cross_entropy
    old_softcap = """        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")"""
    new_softcap = """        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        # NGRAM_BIAS_MARKER apply: blend in precomputed n-gram log-probs
        if self._ngram_enabled and self._bigram_tab.numel() > 1:
            _ids_flat = input_ids.reshape(-1).long()  # (B*S,)
            _H = self._ngram_hash
            _h_bi = (_ids_flat * 36313) % _H
            logits = logits + self._ngram_w_bigram * self._bigram_tab[_h_bi]
            if self._trigram_tab.numel() > 1:
                _B, _S = input_ids.shape
                _ids2 = input_ids
                _prev2 = torch.cat([torch.zeros(_B, 1, device=_ids2.device, dtype=_ids2.dtype), _ids2[:, :-1]], dim=1).reshape(-1).long()
                _h_tri = (_prev2 * 36313 + _ids_flat * 27191) % _H
                logits = logits + self._ngram_w_trigram * self._trigram_tab[_h_tri]
            if self._fourgram_tab.numel() > 1:
                _B, _S = input_ids.shape
                _ids2 = input_ids
                _prev3 = torch.cat([torch.zeros(_B, 2, device=_ids2.device, dtype=_ids2.dtype), _ids2[:, :-2]], dim=1).reshape(-1).long()
                _prev2b = torch.cat([torch.zeros(_B, 1, device=_ids2.device, dtype=_ids2.dtype), _ids2[:, :-1]], dim=1).reshape(-1).long()
                _h_four = (_prev3 * 36313 + _prev2b * 27191 + _ids_flat * 51497) % _H
                logits = logits + self._ngram_w_fourgram * self._fourgram_tab[_h_four]
        return F.cross_entropy(logits.float(), targets, reduction="mean")"""
    if old_softcap in content:
        content = content.replace(old_softcap, new_softcap)
        print("  ✓ added NGRAM_BIAS forward pass")

with open("train_gpt.py", "w") as f:
    f.write(content)
PYEOF

echo
echo "✓ train_gpt.py patched for PyTorch 2.4"
echo "  To revert: cp train_gpt.py.bak train_gpt.py"
