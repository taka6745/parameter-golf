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
        self._ngram_hash = int(os.environ.get("NGRAM_HASH_BUCKETS", "16384"))  # must match 04_build_ngrams.py HASH_BUCKETS
        self.register_buffer("_bigram_tab", torch.zeros(1, dtype=torch.float32), persistent=False)
        self.register_buffer("_trigram_tab", torch.zeros(1, dtype=torch.float32), persistent=False)
        self.register_buffer("_fourgram_tab", torch.zeros(1, dtype=torch.float32), persistent=False)
        # Always init these so Patch 15's forward-pass references don't crash
        self._use_tabulation = bool(int(os.environ.get("USE_TABULATION_HASH", "0")))
        self.register_buffer("_tab_t1", torch.zeros(1, dtype=torch.int64), persistent=False)
        self.register_buffer("_tab_t2", torch.zeros(1, dtype=torch.int64), persistent=False)
        self.register_buffer("_tab_t3", torch.zeros(1, dtype=torch.int64), persistent=False)
        if self._use_tabulation:
            import numpy as _np
            try:
                self._tab_t1 = torch.from_numpy(_np.load("./data/tab_hash_t1.npy")).long()
                self._tab_t2 = torch.from_numpy(_np.load("./data/tab_hash_t2.npy")).long()
                self._tab_t3 = torch.from_numpy(_np.load("./data/tab_hash_t3.npy")).long()
                print("TABULATION_HASH: loaded T1/T2/T3 tables shape", self._tab_t1.shape)
            except Exception as _e:
                print("TABULATION_HASH: load failed (will fall back to polynomial):", _e)
                self._use_tabulation = False
        if self._ngram_enabled:
            import numpy as _np
            _ngsuffix = "_tab" if bool(int(os.environ.get("USE_TABULATION_HASH", "0"))) else ""
            try:
                _bg = _np.load("./data/bigram_tab_{}v{}.npy".format(vocab_size, _ngsuffix))
                self._bigram_tab = torch.from_numpy(_bg).float()
                print("NGRAM_BIAS: loaded bigram", _bg.shape, "w=", self._ngram_w_bigram, "suffix=", _ngsuffix)
            except Exception as _e:
                print("NGRAM_BIAS: bigram load failed:", _e)
            try:
                _tg = _np.load("./data/trigram_logprobs_{}v{}.npy".format(vocab_size, _ngsuffix))
                self._trigram_tab = torch.from_numpy(_tg).float()
                print("NGRAM_BIAS: loaded trigram", _tg.shape, "w=", self._ngram_w_trigram)
            except Exception as _e:
                print("NGRAM_BIAS: trigram load failed:", _e)
            try:
                _fg = _np.load("./data/fourgram_logprobs_{}v{}.npy".format(vocab_size, _ngsuffix))
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

# Patch 12: USE_NGRAM_GATE=1 → LEARNED per-position scalar gate over n-gram bias.
# NOVEL: nobody in the competition (or any PR I can find) has used a learned
# per-position gate over the n-gram logit bias. The closest is fixed-weight blends
# (cmix-style logistic mixing) which Mac §32 found "negligible vs additive". But
# that was for FIXED gate weights — a LEARNED PER-TOKEN gate is fundamentally
# different and adds only 513 params (one Linear: model_dim→1).
#
# Idea:
#   gate = sigmoid(Linear(final_residual))  # (B*S, 1) in [0, 1]
#   logits = logits + gate * (w_bi * bigram_bias + w_tri * trigram_bias + w_four * fourgram_bias)
#
# At init the linear is zero so gate = 0.5 — model starts with half-strength bias
# and learns to up-weight when n-gram is reliable, down-weight when residual is more
# confident. This is the "Nacrith-style learned mixer" RESEARCH.md §14 mentioned
# but applied to additive bias instead of full distribution mixing.
#
# Idempotent via NGRAM_GATE_MARKER.
if "NGRAM_GATE_MARKER" in content:
    print("  ✓ ngram gate already applied")
else:
    # Anchor on Patch 6's NGRAM_BIAS init block end (which runs before Patch 12 in this script).
    # The fourgram print line is unique and stable.
    old_anchor = """                print("NGRAM_BIAS: fourgram load failed:", _e)
        self._init_weights()"""
    new_anchor = """                print("NGRAM_BIAS: fourgram load failed:", _e)
        # NGRAM_GATE_MARKER: learned per-position gate over n-gram bias
        self._ngram_gate_enabled = bool(int(os.environ.get("USE_NGRAM_GATE", "0")))
        if self._ngram_gate_enabled:
            self.ngram_gate_proj = CastedLinear(model_dim, 1, bias=True)
            self.ngram_gate_proj._zero_init = True
        self._init_weights()"""
    if old_anchor in content:
        content = content.replace(old_anchor, new_anchor)
        print("  ✓ added NGRAM_GATE init")

    # Wrap the bigram/trigram/fourgram lookups in a gate factor
    old_apply = """        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        # NGRAM_BIAS_MARKER apply: blend in precomputed n-gram log-probs
        if self._ngram_enabled and self._bigram_tab.numel() > 1:"""
    new_apply = """        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        # NGRAM_GATE_MARKER apply: optional learned per-position gate factor
        _gate = None
        if self._ngram_gate_enabled and self._ngram_enabled:
            _gate = torch.sigmoid(self.ngram_gate_proj(x.detach())).float()  # (B*S, 1) in [0,1]
        # NGRAM_BIAS_MARKER apply: blend in precomputed n-gram log-probs
        if self._ngram_enabled and self._bigram_tab.numel() > 1:"""
    if old_apply in content:
        content = content.replace(old_apply, new_apply)
        print("  ✓ added NGRAM_GATE apply")

    # Multiply the bigram bias by gate
    old_bi_apply = """            logits = logits + self._ngram_w_bigram * self._bigram_tab[_h_bi]"""
    new_bi_apply = """            _bi_bias = self._ngram_w_bigram * self._bigram_tab[_h_bi]
            if _gate is not None:
                _bi_bias = _gate * _bi_bias
            logits = logits + _bi_bias"""
    if old_bi_apply in content:
        content = content.replace(old_bi_apply, new_bi_apply)
        print("  ✓ wrapped bigram lookup with gate")

    # Multiply the trigram bias by gate
    old_tri_apply = """                logits = logits + self._ngram_w_trigram * self._trigram_tab[_h_tri]"""
    new_tri_apply = """                _tri_bias = self._ngram_w_trigram * self._trigram_tab[_h_tri]
                if _gate is not None:
                    _tri_bias = _gate * _tri_bias
                logits = logits + _tri_bias"""
    if old_tri_apply in content:
        content = content.replace(old_tri_apply, new_tri_apply)
        print("  ✓ wrapped trigram lookup with gate")

    # Multiply the fourgram bias by gate
    old_four_apply = """                logits = logits + self._ngram_w_fourgram * self._fourgram_tab[_h_four]"""
    new_four_apply = """                _four_bias = self._ngram_w_fourgram * self._fourgram_tab[_h_four]
                if _gate is not None:
                    _four_bias = _gate * _four_bias
                logits = logits + _four_bias"""
    if old_four_apply in content:
        content = content.replace(old_four_apply, new_four_apply)
        print("  ✓ wrapped fourgram lookup with gate")

# Patch 11: USE_SMEAR_GATE=1 → smear current token activations with previous token before MLP.
# Mac validated -0.019 BPB at 500 steps (LESSONS.md §3 — second-best non-tokenizer Mac trick).
# Implementation: x_for_mlp = alpha * x + (1-alpha) * shift_right(x). Causal, no future leak.
# Idempotent via SMEAR_GATE_MARKER.
if "SMEAR_GATE_MARKER" in content:
    print("  ✓ smear gate already applied")
else:
    old_block_forward = """    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x"""
    new_block_forward = """    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        # SMEAR_GATE_MARKER: smear current token with previous token before MLP (Mac -0.019)
        _mlp_in = self.mlp_norm(x)
        if int(os.environ.get("USE_SMEAR_GATE", "0")):
            _smear_alpha = float(os.environ.get("SMEAR_ALPHA", "0.5"))
            _shifted = F.pad(_mlp_in, (0, 0, 1, 0))[:, :-1]
            _mlp_in = _smear_alpha * _mlp_in + (1.0 - _smear_alpha) * _shifted
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(_mlp_in)
        return x"""
    if old_block_forward in content:
        content = content.replace(old_block_forward, new_block_forward)
        print("  ✓ added SMEAR_GATE pre-MLP smear")

# Patch 15: USE_TABULATION_HASH=1 → swap polynomial n-gram hash for tabulation hash.
# MLX prototype validated -0.0024 BPB across 5 seeds (2.4σ over noise) on 100M tokens
# of fineweb_train_000000.bin. Provably 3-independent vs polynomial which is 2-dependent.
# Build tables MUST be re-built with USE_TABULATION_HASH=1; this patch only handles
# the LOOKUP side in train_gpt.py. The build side is in 04_build_ngrams.py.
# Idempotent via TABULATION_HASH_MARKER.
if "TABULATION_HASH_MARKER" in content:
    print("  ✓ tabulation hash already applied")
else:
    # NOTE: TABULATION_HASH init is now done unconditionally inside Patch 6's
    # NGRAM_BIAS init block — see lines just above the "if self._ngram_enabled"
    # check. This avoids the patcher anchor brittleness that bit us earlier.
    # Patch 15 only handles the LOOKUP-side hash swap below.
    print("  ✓ TABULATION_HASH init handled by Patch 6 unconditionally (no anchor needed)")
    # Add a marker comment so re-runs detect this patch was processed:
    if "# TABULATION_HASH_MARKER processed by Patch 6" not in content:
        content = content.replace(
            "self.register_buffer(\"_tab_t1\", torch.zeros(1, dtype=torch.int64), persistent=False)",
            "# TABULATION_HASH_MARKER processed by Patch 6\n        self.register_buffer(\"_tab_t1\", torch.zeros(1, dtype=torch.int64), persistent=False)",
            1,
        )

    # Replace bigram hash lookup
    old_bi_hash = """            _ids_flat = input_ids.reshape(-1).long()  # (B*S,)
            _H = self._ngram_hash
            _h_bi = (_ids_flat * 36313) % _H"""
    new_bi_hash = """            _ids_flat = input_ids.reshape(-1).long()  # (B*S,)
            _H = self._ngram_hash
            if self._use_tabulation and self._tab_t1.numel() > 1:
                _h_bi = self._tab_t1[_ids_flat] % _H
            else:
                _h_bi = (_ids_flat * 36313) % _H"""
    if old_bi_hash in content:
        content = content.replace(old_bi_hash, new_bi_hash)
        print("  ✓ added TABULATION_HASH bigram lookup")

    # Replace trigram hash lookup
    old_tri_hash = """                _h_tri = (_prev2 * 36313 + _ids_flat * 27191) % _H"""
    new_tri_hash = """                if self._use_tabulation and self._tab_t1.numel() > 1:
                    _h_tri = (self._tab_t1[_prev2] ^ self._tab_t2[_ids_flat]) % _H
                else:
                    _h_tri = (_prev2 * 36313 + _ids_flat * 27191) % _H"""
    if old_tri_hash in content:
        content = content.replace(old_tri_hash, new_tri_hash)
        print("  ✓ added TABULATION_HASH trigram lookup")

    # Replace fourgram hash lookup
    old_four_hash = """                _h_four = (_prev3 * 36313 + _prev2b * 27191 + _ids_flat * 51497) % _H"""
    new_four_hash = """                if self._use_tabulation and self._tab_t1.numel() > 1:
                    _h_four = (self._tab_t1[_prev3] ^ self._tab_t2[_prev2b] ^ self._tab_t3[_ids_flat]) % _H
                else:
                    _h_four = (_prev3 * 36313 + _prev2b * 27191 + _ids_flat * 51497) % _H"""
    if old_four_hash in content:
        content = content.replace(old_four_hash, new_four_hash)
        print("  ✓ added TABULATION_HASH fourgram lookup")

# Patch 16: USE_GATED_ATTENTION=1 → per-head sigmoid gate over attention output.
# From "Gated Attention for Large Language Models" (NeurIPS 2025). Adds a tiny
# Linear (model_dim → num_heads) per attention block that produces a per-position,
# per-head sigmoid gate. Multiplies the attention output (after SDPA, before proj).
# Math: y = (attn @ v) * sigmoid(x @ W_gate + b_gate)
# Init: weight=0, bias=2.94 → sigmoid≈0.95 (near identity, room to learn).
# Cost: ~num_heads * model_dim params per layer = 8*512 = 4096 params for our 9L
# baseline = 36k total. Negligible.
# NOT in any open openai/parameter-golf PR (verified by audit subagent Apr 7).
# Idempotent via GATED_ATTENTION_MARKER.
if "GATED_ATTENTION_MARKER" in content:
    print("  ✓ gated attention already applied")
else:
    # Add gate_proj in CausalSelfAttention __init__
    old_attn_init = """        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)"""
    new_attn_init = """        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        # GATED_ATTENTION_MARKER: per-head sigmoid gate (NeurIPS 2025)
        self.gate_proj = CastedLinear(dim, num_heads, bias=True)
        with torch.no_grad():
            self.gate_proj.weight.zero_()
            if self.gate_proj.bias is not None:
                self.gate_proj.bias.fill_(2.94)  # sigmoid(2.94) ≈ 0.95"""
    if old_attn_init in content:
        content = content.replace(old_attn_init, new_attn_init)
        print("  ✓ added GATED_ATTENTION init")

    # Apply the gate after SDPA, before the transpose+reshape
    old_sdpa_call = """        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)"""
    new_sdpa_call = """        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        # GATED_ATTENTION_MARKER apply: per-head sigmoid gate from input x
        if int(os.environ.get("USE_GATED_ATTENTION", "0")):
            _gate = torch.sigmoid(self.gate_proj(x).float()).to(dtype=y.dtype)  # (B, S, num_heads)
            _gate = _gate.transpose(1, 2).unsqueeze(-1)  # (B, num_heads, S, 1)
            y = y * _gate
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)"""
    if old_sdpa_call in content:
        content = content.replace(old_sdpa_call, new_sdpa_call)
        print("  ✓ added GATED_ATTENTION apply")

# Patch 14: USE_ENTROPY_ADAPTIVE_NGRAM=1 → entropy-gated n-gram bias mixing.
# TRULY NOVEL — not in any PR I've found, not in any Mac experiment, not in any
# arxiv paper I'm aware of. The idea: instead of a fixed or learned gate, use the
# model's OWN per-token softmax entropy as the gate. When the model is uncertain
# (high entropy ≈ uniform distribution), we trust the n-gram bias more. When the
# model is confident (low entropy ≈ peaked distribution), we trust its own
# prediction more. The gate is computed deterministically from the logits — no
# extra learned params, no extra forward pass.
#
# Different from:
#   * Mac §32 cmix-style logistic mixing — that used FIXED scalar weights
#   * Our learned NGRAM_GATE (Patch 12) — that uses a Linear on the residual,
#     and empirically failed (NG1=3.42 vs L5=3.29)
#   * Adaptive softmax / temperature-scaled mixing — those scale the WHOLE
#     distribution, not the additive bias
#
# Math (per token i):
#   logits_i ∈ R^V                                  # post-softcap model logits
#   p_i = softmax(logits_i)                         # model probs
#   H_i = -sum(p_i * log(p_i))                      # model entropy
#   gate_i = H_i / log(V)                           # normalize to [0, 1]
#   bias_i = w_bi*bg[h_bi(i)] + w_tri*tg[h_tri(i)] + w_four*fg[h_four(i)]
#   logits_i_final = logits_i + gate_i * bias_i
#
# Cost: ~4 extra ops per token at the output layer. Negligible vs softmax+matmul.
# Idempotent via ENTROPY_ADAPTIVE_NGRAM_MARKER.
if "ENTROPY_ADAPTIVE_NGRAM_MARKER" in content:
    print("  ✓ entropy-adaptive ngram already applied")
else:
    # Wrap the existing ngram bias add (which Patch 6 inserts) with an entropy gate.
    # We anchor on Patch 6's "logits = logits + _bi_bias" line and replace the bigram/
    # trigram/fourgram gates with entropy-scaled versions when the env var is set.
    old_ng_apply_block = """        # NGRAM_GATE_MARKER apply: optional learned per-position gate factor
        _gate = None
        if self._ngram_gate_enabled and self._ngram_enabled:
            _gate = torch.sigmoid(self.ngram_gate_proj(x.detach())).float()  # (B*S, 1) in [0,1]
        # NGRAM_BIAS_MARKER apply: blend in precomputed n-gram log-probs
        if self._ngram_enabled and self._bigram_tab.numel() > 1:"""
    new_ng_apply_block = """        # NGRAM_GATE_MARKER apply: optional learned per-position gate factor
        _gate = None
        if self._ngram_gate_enabled and self._ngram_enabled:
            _gate = torch.sigmoid(self.ngram_gate_proj(x.detach())).float()  # (B*S, 1) in [0,1]
        # ENTROPY_ADAPTIVE_NGRAM_MARKER: model-entropy-gated n-gram mix
        if self._ngram_enabled and bool(int(os.environ.get("USE_ENTROPY_ADAPTIVE_NGRAM", "0"))):
            _p = torch.softmax(logits.float(), dim=-1)
            _logp = torch.log_softmax(logits.float(), dim=-1)
            _H = -(_p * _logp).sum(dim=-1, keepdim=True)  # (B*S, 1)
            _Hmax = float(torch.log(torch.tensor(float(self._bigram_tab.shape[-1] if self._bigram_tab.numel() > 1 else 1024))))
            _ent_gate = (_H / max(_Hmax, 1e-9)).clamp(0.0, 1.0)
            if _gate is None:
                _gate = _ent_gate
            else:
                _gate = _gate * _ent_gate  # combine learned + entropy gates multiplicatively
        # NGRAM_BIAS_MARKER apply: blend in precomputed n-gram log-probs
        if self._ngram_enabled and self._bigram_tab.numel() > 1:"""
    if old_ng_apply_block in content:
        content = content.replace(old_ng_apply_block, new_ng_apply_block)
        print("  ✓ added ENTROPY_ADAPTIVE_NGRAM gating")

# Patch 22: USE_ENGRAM_LITE=1 → learnable hash-embedding n-gram head (PR #1440).
# From "[Submission] EngramLite + Mousse + Progressive Depth Recurrence + TTT" — claimed
# val_bpb 1.1026 single seed in PR #1440. EngramLite alone attributed -0.003 BPB delta.
#
# This is a GENERALIZATION of our Patch 6 NGRAM_BIAS, which uses FROZEN log-prob tables
# built from training data statistics. EngramLite adds a parallel LEARNABLE n-gram head:
#   - shared nn.Embedding(buckets=3072, embed_dim=112) for both bigram and trigram hashes
#   - separate sigmoid gates per hash order, init at -1.0 (sigmoid≈0.27, conservative start)
#   - shared nn.Linear(112, vocab_size) projection, zero-init so it doesn't dominate early
#   - bigram + trigram contributions summed and added to logits as residual
#
# Stacks with Patch 6: static log-prob bias gives the data-grounded prior, EngramLite
# learns a residual on top. Both contribute to logits. Cost: ~460k params at sp1024,
# ~1M params at sp8192. Idempotent via ENGRAM_LITE_MARKER.
if "ENGRAM_LITE_MARKER" in content:
    print("  ✓ engram lite already applied")
else:
    # Add EngramLiteHead class right BEFORE the GPT class definition (always invariant)
    old_gpt_class = """class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,"""
    new_gpt_class = """class EngramLiteHead(nn.Module):
    # ENGRAM_LITE_MARKER: learnable hash-embedding n-gram head from PR #1440
    def __init__(self, vocab_size: int, hash_buckets: int = 3072, embed_dim: int = 112):
        super().__init__()
        self.hash_buckets = hash_buckets
        self.embed = nn.Embedding(hash_buckets, embed_dim)
        self.proj = nn.Linear(embed_dim, vocab_size, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()  # zero init so it doesn't dominate early training
        self.gate_bigram = nn.Parameter(torch.full((1,), -1.0))
        self.gate_trigram = nn.Parameter(torch.full((1,), -1.0))

    def forward(self, input_ids: Tensor) -> Tensor:
        ids_long = input_ids.long()  # (B, S)
        # Bigram hash on current token
        bi_hash = (ids_long * 36313) % self.hash_buckets
        bi_emb = self.embed(bi_hash)  # (B, S, embed_dim)
        bi_logits = self.proj(bi_emb) * torch.sigmoid(self.gate_bigram)
        # Trigram hash on (prev, current)
        prev = F.pad(ids_long, (1, 0))[:, :-1]
        tri_hash = (prev * 36313 + ids_long * 27191) % self.hash_buckets
        tri_emb = self.embed(tri_hash)
        tri_logits = self.proj(tri_emb) * torch.sigmoid(self.gate_trigram)
        return bi_logits + tri_logits  # (B, S, vocab_size)


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,"""
    if old_gpt_class in content:
        content = content.replace(old_gpt_class, new_gpt_class)
        print("  ✓ added EngramLiteHead class definition")

    # Init the EngramLiteHead in GPT.__init__ — anchor on the MTP init block (stable from Patch 21)
    old_mtp_init_close = """        else:
            self.mtp_blocks = None
        self._init_weights()"""
    new_mtp_init_close = """        else:
            self.mtp_blocks = None
        # ENGRAM_LITE_MARKER: optional learnable n-gram head
        self._engram_lite_enabled = bool(int(os.environ.get("USE_ENGRAM_LITE", "0")))
        if self._engram_lite_enabled:
            self.engram_lite = EngramLiteHead(vocab_size)
        else:
            self.engram_lite = None
        self._init_weights()"""
    if old_mtp_init_close in content:
        content = content.replace(old_mtp_init_close, new_mtp_init_close)
        print("  ✓ added EngramLite init in GPT.__init__")

    # Apply: add the EngramLite logits to the main logits BEFORE the softcap.
    # Anchor on the softcap line which is stable across patches.
    old_softcap_line = """        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        # NGRAM_GATE_MARKER apply: optional learned per-position gate factor"""
    new_softcap_line = """        # ENGRAM_LITE_MARKER apply: add learnable hash n-gram head logits to main logits
        if self._engram_lite_enabled and self.engram_lite is not None:
            _el_logits = self.engram_lite(input_ids)  # (B, S, V)
            _el_logits_flat = _el_logits.reshape(-1, _el_logits.size(-1))
            logits_proj = logits_proj + _el_logits_flat
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        # NGRAM_GATE_MARKER apply: optional learned per-position gate factor"""
    if old_softcap_line in content:
        content = content.replace(old_softcap_line, new_softcap_line)
        print("  ✓ added EngramLite forward apply")

# Patch 21: USE_MTP=1 → Multi-Token Prediction (DeepSeek-V3 style).
# Adds K auxiliary heads that predict tokens i+2, i+3, ..., i+K+1 (vs main head's i+1).
# Each head is a single Block that takes the main model's pre-norm residual and produces
# its own logits via the SHARED tok_emb weight (tied embeddings, no extra params except
# the K Block instances). Auxiliary loss = MTP_LOSS_WEIGHT * mean(aux_losses).
#
# Source: DeepSeek-V3 Technical Report (arxiv:2412.19437) — claims ~0.3 BPB equivalent
# improvement at 671B scale via dense supervision. NOT in any open openai/parameter-golf
# PR (audit Apr 7 confirmed zero matches for "MTP" / "multi-token" / "multitoken").
#
# Why this might transfer to byte-level small LMs:
#   - Byte-level has denser supervision (1 token ≈ 3.5 bytes), so each step provides more
#     information; MTP heads exploit this by adding K auxiliary signals per step.
#   - Our regime is compute-bound, not data-bound. MTP gives more gradient per step.
#   - Pure aux loss — sets to 0 if it hurts (degrades gracefully).
#
# Cost: K extra Block instances ≈ K × (4 * model_dim^2 + mlp_mult * model_dim^2) params.
# For our 9L 512d 2x MLP, K=1 adds ~786K params (~5% overhead) but no new attention heads.
# Idempotent via MTP_MARKER.
if "MTP_MARKER" in content:
    print("  ✓ MTP already applied")
else:
    # Init: add the MTP block list at the end of GPT.__init__ (after _init_weights call)
    # We anchor on the last 2 lines of __init__ which are stable: the n-gram fourgram load
    # plus the _init_weights call. To avoid brittleness, we anchor on _init_weights call only.
    old_init_end = """        self._init_weights()

    def _init_weights(self) -> None:"""
    new_init_end = """        # MTP_MARKER: Multi-Token Prediction auxiliary heads
        self._mtp_enabled = bool(int(os.environ.get("USE_MTP", "0")))
        self._mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", "1"))
        self._mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", "0.10"))
        if self._mtp_enabled and self._mtp_num_heads > 0:
            self.mtp_blocks = nn.ModuleList([
                Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, layer_idx=num_layers + _k)
                for _k in range(self._mtp_num_heads)
            ])
        else:
            self.mtp_blocks = None
        self._init_weights()

    def _init_weights(self) -> None:"""
    if old_init_end in content:
        content = content.replace(old_init_end, new_init_end)
        print("  ✓ added MTP init")

    # Apply: capture pre-norm residual + compute aux losses inside forward.
    # Anchor on the line that reshapes to (B*S, D) — stable across all patches.
    old_reshape = """        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)"""
    new_reshape = """        # MTP_MARKER apply: save pre-norm hidden state for MTP heads
        _mtp_hidden = x if (self._mtp_enabled and self.mtp_blocks is not None) else None
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)"""
    if old_reshape in content:
        content = content.replace(old_reshape, new_reshape)
        print("  ✓ added MTP hidden state capture")

    # Insert MTP loss computation right before the final return statement of forward().
    # Anchor on the F.cross_entropy return line which is stable across n-gram patches.
    old_return = """        return F.cross_entropy(logits.float(), targets, reduction="mean")"""
    new_return = """        _main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        # MTP_MARKER apply: compute aux losses from MTP blocks
        if self._mtp_enabled and self.mtp_blocks is not None and _mtp_hidden is not None:
            _B, _S = input_ids.shape
            _D = _mtp_hidden.size(-1)
            _mtp_total = torch.zeros((), device=_main_loss.device, dtype=_main_loss.dtype)
            for _k, _mtp_block in enumerate(self.mtp_blocks):
                _shift = _k + 1  # MTP head k predicts position i+1+k+1 = i+(k+2)
                if _shift >= _S:
                    continue
                _mtp_x = _mtp_block(_mtp_hidden, x0)  # (B, S, D)
                _mtp_x = self.final_norm(_mtp_x)
                _mtp_x_2d = _mtp_x[:, :_S - _shift, :]  # (B, S-shift, D)
                if self.tie_embeddings:
                    _mtp_logits = F.linear(_mtp_x_2d.reshape(-1, _D), self.tok_emb.weight)
                else:
                    _mtp_logits = self.lm_head(_mtp_x_2d.reshape(-1, _D))
                _mtp_logits = self.logit_softcap * torch.tanh(_mtp_logits / self.logit_softcap)
                _mtp_targets = target_ids[:, _shift:].reshape(-1)
                _mtp_total = _mtp_total + F.cross_entropy(_mtp_logits.float(), _mtp_targets, reduction="mean")
            _main_loss = _main_loss + self._mtp_loss_weight * (_mtp_total / max(self._mtp_num_heads, 1))
        return _main_loss"""
    if old_return in content:
        content = content.replace(old_return, new_return)
        print("  ✓ added MTP forward + auxiliary loss")

# Patch 19: USE_PARTIAL_ROPE=1 → rotate only first ROPE_DIMS dims of each head, leave the rest unrotated.
# In merged records #1019 (val_bpb 1.1147 — the literal #1 SOTA) and #315 (1.1248).
# Rationale: rotary positional info is only needed for the first few dims; leaving the rest
# free of positional bias improves length generalization (the unrotated dims attend purely on
# content, the rotated dims attend on relative position). Zero params, free quality.
# Idempotent via PARTIAL_ROPE_MARKER.
if "PARTIAL_ROPE_MARKER" in content:
    print("  ✓ partial rope already applied")
else:
    old_apply_rotary = """def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)"""
    new_apply_rotary = """def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # PARTIAL_ROPE_MARKER: optionally rotate only the first PARTIAL_ROPE_DIMS dims of each head
    _partial = int(os.environ.get("PARTIAL_ROPE_DIMS", "0"))
    if _partial > 0 and _partial < x.size(-1):
        rot, plain = x[..., :_partial], x[..., _partial:]
        half = rot.size(-1) // 2
        r1, r2 = rot[..., :half], rot[..., half:]
        cos_r = cos[..., :half] if cos.size(-1) >= half else cos
        sin_r = sin[..., :half] if sin.size(-1) >= half else sin
        rotated = torch.cat((r1 * cos_r + r2 * sin_r, r1 * (-sin_r) + r2 * cos_r), dim=-1)
        return torch.cat((rotated, plain), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)"""
    if old_apply_rotary in content:
        content = content.replace(old_apply_rotary, new_apply_rotary)
        print("  ✓ added PARTIAL_ROPE branch in apply_rotary_emb")

# Patch 20: USE_LN_SCALE=1 → scale RMSNorm output by 1/sqrt(layer_idx+1) at each block.
# In PRs #1019 + #315. Damps the contribution of deeper layers, stabilizing training.
# Zero params, ~5-line change. Mac never tested. Free quality win in 2+ merged records.
# Requires Block to know its layer_idx, so we add it to Block.__init__ signature.
# Idempotent via LN_SCALE_MARKER.
if "LN_SCALE_MARKER" in content:
    print("  ✓ LN scale already applied")
else:
    # Add layer_idx to Block.__init__ + store it + apply scale in forward
    old_block_init = """class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()"""
    new_block_init = """class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int = 0,
    ):
        super().__init__()
        # LN_SCALE_MARKER: store layer index for 1/sqrt(layer+1) RMSNorm scaling
        self._layer_idx = layer_idx
        self.attn_norm = RMSNorm()"""
    if old_block_init in content:
        content = content.replace(old_block_init, new_block_init)
        print("  ✓ added LN_SCALE layer_idx storage")

    # Wire layer_idx through GPT.__init__ Block creation
    old_block_creation = """        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )"""
    new_block_creation = """        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )"""
    if old_block_creation in content:
        content = content.replace(old_block_creation, new_block_creation)
        print("  ✓ wired LN_SCALE layer_idx into Block creation")

# Patch 13: USE_PARALLEL_RESIDUALS=1 → compute attn and mlp in parallel from the same input.
# Found in research cron #1 (Apr 7 21:23 local) by mining recent openai/parameter-golf PRs:
#   PR #1437: Record SP8192 + Parallel Residuals + 3-Layer Recurrence — val_bpb 1.07800
#   PR #1420: Triple Loop + Parallel Residuals + N-gram Tilt — val_bpb 1.08014
#   PR #1425: PROTEUS Feature Ablation - Parallel Residuals + Mixed INT5/INT6
# All three top records use parallel residuals; we never tried it.
#
# Anchors on the FIRST 3 lines of Block.forward (def + mix + resid blend) which are
# invariant under Patch 11 (smear gate). Inserts the parallel branch right after, so
# the existing serial path (with smear gate) is preserved as the fallback below.
# Idempotent via PARALLEL_RESIDUALS_MARKER.
if "PARALLEL_RESIDUALS_MARKER" in content:
    print("  ✓ parallel residuals already applied")
else:
    old_first_3 = """    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0"""
    new_first_3 = """    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        # PARALLEL_RESIDUALS_MARKER: compute attn + mlp in parallel from the same pre-norm input
        if int(os.environ.get("USE_PARALLEL_RESIDUALS", "0")):
            attn_in = self.attn_norm(x)
            mlp_in = self.mlp_norm(x)
            attn_out = self.attn(attn_in)
            mlp_out = self.mlp(mlp_in)
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
            return x"""
    if old_first_3 in content:
        content = content.replace(old_first_3, new_first_3)
        print("  ✓ added PARALLEL_RESIDUALS branch")

# Patch 9: USE_LEAKY_RELU=1 → MLP activation is leaky_relu(0.5)^2 instead of relu^2.
# Mac validated -0.014 BPB at 500 steps (LESSONS.md §2). One-line MLP change.
# Idempotent via LEAKY_RELU_MARKER.
if "LEAKY_RELU_MARKER" in content:
    print("  ✓ leaky relu already applied")
else:
    old_mlp_forward = """    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())"""
    new_mlp_forward = """    def forward(self, x: Tensor) -> Tensor:
        # LEAKY_RELU_MARKER: optional leaky_relu(0.5)^2 activation (Mac -0.014 BPB)
        if int(os.environ.get("USE_LEAKY_RELU", "0")):
            x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        else:
            x = torch.relu(self.fc(x))
        return self.proj(x.square())"""
    if old_mlp_forward in content:
        content = content.replace(old_mlp_forward, new_mlp_forward)
        print("  ✓ added LEAKY_RELU MLP option")

# Patch 10: USE_BYTE_WEIGHT=1 → weight each token's CE loss by its byte count.
# Mac validated -0.017 BPB at 500 steps (LESSONS.md §3b). Aligns training with bpb metric.
# We approximate byte count via a precomputed lookup over the tokenizer's vocab.
# Idempotent via BYTE_WEIGHT_MARKER.
if "BYTE_WEIGHT_MARKER" in content:
    print("  ✓ byte-weighted loss already applied")
else:
    # Add the weight init in GPT __init__ (right next to ngram init)
    old_ng_init_close = """        if self._ngram_enabled:
            import numpy as _np"""
    new_ng_init_close = """        # BYTE_WEIGHT_MARKER: optional per-token byte-count loss weighting (Mac -0.017 BPB)
        self._byte_weight_enabled = bool(int(os.environ.get("USE_BYTE_WEIGHT", "0")))
        self.register_buffer("_byte_weight_lut", torch.ones(vocab_size, dtype=torch.float32), persistent=False)
        if self._byte_weight_enabled:
            try:
                import sentencepiece as _spm
                _tk = _spm.SentencePieceProcessor(model_file=os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"))
                _bytes_per_token = []
                for _i in range(vocab_size):
                    try:
                        _bytes_per_token.append(float(len(_tk.id_to_piece(_i).encode("utf-8"))))
                    except Exception:
                        _bytes_per_token.append(1.0)
                _bw = torch.tensor(_bytes_per_token, dtype=torch.float32)
                _bw = _bw / _bw.mean()  # normalize so mean weight = 1.0
                self._byte_weight_lut = _bw
                print("BYTE_WEIGHT: built lookup table mean=", float(_bw.mean()), " min=", float(_bw.min()), " max=", float(_bw.max()))
            except Exception as _e:
                print("BYTE_WEIGHT: build failed:", _e)
                self._byte_weight_enabled = False
        if self._ngram_enabled:
            import numpy as _np"""
    if old_ng_init_close in content:
        content = content.replace(old_ng_init_close, new_ng_init_close)
        print("  ✓ added BYTE_WEIGHT init/loading")

    # Replace cross_entropy reduction with byte-weighted version
    old_ce = """        return F.cross_entropy(logits.float(), targets, reduction="mean")"""
    new_ce = """        if self._byte_weight_enabled:
            _ce = F.cross_entropy(logits.float(), targets, reduction="none")
            _w = self._byte_weight_lut[targets]
            return (_ce * _w).sum() / _w.sum()
        return F.cross_entropy(logits.float(), targets, reduction="mean")"""
    if old_ce in content:
        content = content.replace(old_ce, new_ce)
        print("  ✓ added BYTE_WEIGHT loss path")

# Patch 8: USE_WAVELET=1 → causal multi-scale averaging on half dims after each block.
# Mac validated -0.018 BPP at 1000 steps (best Mac result was 1.6929 with this).
# Implementation lifted from runpod_tests/validate/v03_wavelet_mix.py.
# Receptive field per layer: k_i = min(2^(i+1), seq_len), so deeper layers see more context.
# Idempotent via WAVELET_GPT_MARKER.
if "WAVELET_GPT_MARKER" in content:
    print("  ✓ wavelet GPT already applied")
else:
    # Inject the wavelet_mix helper + USE_WAVELET flag right after NGRAM init block
    old_ng_close = """        self._init_weights()"""
    new_ng_close = """        self._wavelet_enabled = bool(int(os.environ.get("USE_WAVELET", "0")))
        self._wavelet_mix_ratio = float(os.environ.get("WAVELET_MIX_RATIO", "0.20"))
        self._init_weights()

    @staticmethod
    def _wavelet_mix(x, layer_idx, mix_ratio):
        # WAVELET_GPT_MARKER: causal multi-scale averaging on the right half of dims
        B, T, D = x.shape
        half = D // 2
        left = x[..., :half]
        right = x[..., half:]
        k = min(2 ** (layer_idx + 1), T)
        cs = torch.cumsum(right, dim=1)
        shifted = F.pad(cs[:, :-k], (0, 0, k, 0))
        counts = torch.arange(1, T + 1, device=x.device, dtype=right.dtype)
        counts = counts.clamp(max=k).unsqueeze(0).unsqueeze(-1)
        right_avg = (cs - shifted) / counts
        right_mixed = (1.0 - mix_ratio) * right + mix_ratio * right_avg
        return torch.cat([left, right_mixed], dim=-1)"""
    if old_ng_close in content:
        content = content.replace(old_ng_close, new_ng_close, 1)
        print("  ✓ added WAVELET helper + flags")

    # Apply wavelet after each block in the encoder + decoder loops
    old_loop = """        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)"""
    new_loop = """        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            if self._wavelet_enabled:
                x = self._wavelet_mix(x, i, self._wavelet_mix_ratio)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
            if self._wavelet_enabled:
                x = self._wavelet_mix(x, self.num_encoder_layers + i, self._wavelet_mix_ratio)"""
    if old_loop in content:
        content = content.replace(old_loop, new_loop)
        print("  ✓ added WAVELET calls in GPT.forward")

with open("train_gpt.py", "w") as f:
    f.write(content)
PYEOF

echo
echo "✓ train_gpt.py patched for PyTorch 2.4"
echo "  To revert: cp train_gpt.py.bak train_gpt.py"
