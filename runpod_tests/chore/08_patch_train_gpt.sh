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

# Patch 2: torch.compile RE-ENABLED with dynamic=True + fullgraph=False to handle
# XSA/EngramLite forward modifications. SPEED4/5 crashed in <30s with default mode
# because the strict shape tracing failed on the modified attention path. dynamic=True
# allows dynamic shapes; fullgraph=False allows fallback to eager for unsupported ops.
# Gated by USE_TORCH_COMPILE=1 (default ON), set to 0 to fall back to no-compile.
# Mac LESSONS wishlist: 25-35% speedup.
old_compile = "compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)"
new_compile = """compiled_model = base_model if int(os.environ.get('USE_TORCH_COMPILE', '0')) == 0 else torch.compile(base_model, dynamic=True, fullgraph=False)  # PATCHED: dynamic=True for XSA/EL compat"""
if old_compile in content:
    content = content.replace(old_compile, new_compile)
    print("  ✓ re-enabled torch.compile on model (dynamic=True, fullgraph=False)")

# Also re-enable the optimizer compile (Newton-Schulz hot path)
old_opt = "zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)"
new_opt = """zeropower_via_newtonschulz5 = zeropower_via_newtonschulz5 if int(os.environ.get('USE_TORCH_COMPILE', '0')) == 0 else torch.compile(zeropower_via_newtonschulz5, dynamic=True)"""
if old_opt in content:
    content = content.replace(old_opt, new_opt)
    print("  ✓ re-enabled torch.compile on Newton-Schulz (dynamic=True)")

# Patch 2b: Turbo-Muon — env-var override for Newton-Schulz step count.
# Mac LESSONS §35: "Free speedup, no quality loss, -0.026 BPB at NS_STEPS=4 vs 5".
# Override the steps parameter inside the function body via env var.
if "NS_STEPS_MARKER" not in content:
    old_ns_body = """def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert G.ndim >= 2"""
    new_ns_body = """def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    # NS_STEPS_MARKER: optional Turbo-Muon override (Mac LESSONS §35)
    _ns_override = int(os.environ.get('NS_STEPS', '0'))
    if _ns_override > 0:
        steps = _ns_override
    assert G.ndim >= 2"""
    if old_ns_body in content:
        content = content.replace(old_ns_body, new_ns_body)
        print("  ✓ added NS_STEPS env var override (Turbo-Muon)")

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

# Patch 46: USE_NGRAM_BACKOFF=1 → order-adaptive Stupid Backoff (Brants 2007).
# THE biggest legal technique gap after LEGAL_TTT. Top 30 legal PRs in COMPETITION_SCOPE.md
# all use multi-order n-gram backoff. Examples: #802 0.9123 (10L + Multi-Order N-gram Backoff),
# #788 0.9059 (11L + order-adaptive 9-gram backoff), #828 0.9076 (Multi-Order Backoff + Matrix),
# #761 0.9581 (Score-First TTT + N-gram Backoff). Best (#1028 = 0.8104) uses N-gram Backoff.
#
# OUR CURRENT: weighted SUM of bigram + trigram + 4-gram bias log-probs at every position.
# THIS PATCH: at each position, use the HIGHEST-CONFIDENCE order ONLY:
#   - if peak(4-gram[h]) > T4: use 4-gram with weight 1.0 (full confidence)
#   - elif peak(3-gram[h]) > T3: use 3-gram with weight α (α=0.4 = Brants 2007)
#   - else: use bigram with weight α² (=0.16)
# The "peak" is the max log-prob across vocab — concentrated distributions = confident counts.
# Hash collisions in lower orders contribute noise; using only the most-confident order strips
# that noise. This is the canonical Stupid Backoff (Brants 2007 Google n-gram paper).
#
# Marker: NGRAM_BACKOFF_MARKER. Env: USE_NGRAM_BACKOFF=1, NGRAM_BACKOFF_THRESH4=1.0,
# NGRAM_BACKOFF_THRESH3=1.0, NGRAM_BACKOFF_ALPHA=0.4 (defaults). Idempotent.
if "NGRAM_BACKOFF_MARKER" in content:
    print("  ✓ ngram backoff already applied")
else:
    # Anchor on the FINAL appearance of the line — after Patch 14 entropy_adaptive ran.
    # The full anchor is the entropy_adaptive comment + the if line.
    old_bo_anchor = """        # NGRAM_BIAS_MARKER apply: blend in precomputed n-gram log-probs
        if self._ngram_enabled and self._bigram_tab.numel() > 1:
            _ids_flat = input_ids.reshape(-1).long()  # (B*S,)"""
    new_bo_anchor = """        # NGRAM_BIAS_MARKER apply: blend in precomputed n-gram log-probs
        # NGRAM_BACKOFF_MARKER: order-adaptive Stupid Backoff (Brants 2007)
        if self._ngram_enabled and bool(int(os.environ.get("USE_NGRAM_BACKOFF", "0"))) and self._bigram_tab.numel() > 1 and self._fourgram_tab.numel() > 1:
            _ids_flat_bo = input_ids.reshape(-1).long()
            _H_bo = self._ngram_hash
            _bo_t4 = float(os.environ.get("NGRAM_BACKOFF_THRESH4", "1.0"))
            _bo_t3 = float(os.environ.get("NGRAM_BACKOFF_THRESH3", "1.0"))
            _bo_a = float(os.environ.get("NGRAM_BACKOFF_ALPHA", "0.4"))
            _B_bo, _S_bo = input_ids.shape
            _prev2_bo = torch.cat([torch.zeros(_B_bo, 1, device=input_ids.device, dtype=input_ids.dtype), input_ids[:, :-1]], dim=1).reshape(-1).long()
            _prev3_bo = torch.cat([torch.zeros(_B_bo, 2, device=input_ids.device, dtype=input_ids.dtype), input_ids[:, :-2]], dim=1).reshape(-1).long()
            _h_bi_bo = (_ids_flat_bo * 36313) % _H_bo
            _h_tri_bo = (_prev2_bo * 36313 + _ids_flat_bo * 27191) % _H_bo
            _h_four_bo = (_prev3_bo * 36313 + _prev2_bo * 27191 + _ids_flat_bo * 51497) % _H_bo
            _bi_bo = self._bigram_tab[_h_bi_bo]
            _tri_bo = self._trigram_tab[_h_tri_bo]
            _four_bo = self._fourgram_tab[_h_four_bo]
            _peak4_bo = _four_bo.amax(dim=-1, keepdim=True)
            _peak3_bo = _tri_bo.amax(dim=-1, keepdim=True)
            _use_4_bo = (_peak4_bo > _bo_t4).to(_four_bo.dtype)
            _use_3_bo = (1 - _use_4_bo) * (_peak3_bo > _bo_t3).to(_tri_bo.dtype)
            _use_bi_bo = 1 - _use_4_bo - _use_3_bo
            _ng_bo = _use_4_bo * _four_bo + _use_3_bo * _tri_bo * _bo_a + _use_bi_bo * _bi_bo * (_bo_a * _bo_a)
            if _gate is not None:
                _ng_bo = _gate * _ng_bo
            logits = logits + _ng_bo
        elif self._ngram_enabled and self._bigram_tab.numel() > 1:
            _ids_flat = input_ids.reshape(-1).long()  # (B*S,)"""
    if old_bo_anchor in content:
        content = content.replace(old_bo_anchor, new_bo_anchor)
        print("  ✓ added NGRAM_BACKOFF dispatch (Patch 46)")
    else:
        print("  ⚠ NGRAM_BACKOFF anchor not found — skipping (anchor moved?)")

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
        # Wrapped in getattr for safety: if init didn't apply (anchor mismatch),
        # the forward pass should still work without crashing.
        if getattr(self, '_engram_lite_enabled', False) and getattr(self, 'engram_lite', None) is not None:
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

# Patch 20: USE_COPRIME_STRIDE=1 → shard-level coprime stride sampling in TokenStream.
# Inspired by PR #1099 (latest merged record) and PR #1060/#1135 which use TOKEN-level
# coprime stride sampling. The full token-level variant requires a 60+ LOC rewrite of
# TokenStream + DistributedTokenLoader (random access support, cumulative shard offset
# maps, etc.). This patch ships a CONSERVATIVE shard-level variant: instead of cycling
# shards in order 0->1->2->...->N-1, cycle them with a coprime stride
# (e.g., stride=13 with N=100 shards gives 0->13->26->39->...->91->4->17->...).
#
# Mechanism: nearby training steps see DIFFERENT shards rather than topically-similar
# adjacent shards. Provides gradient diversity at the shard granularity. Smaller benefit
# than token-level (~25% as much based on PR #1099 reports) but ~5x cheaper to ship.
#
# Reference: number theory → if gcd(stride, N) == 1, the iteration covers all N shards
# before repeating, achieving max spacing diversity.
#
# Idempotent via COPRIME_STRIDE_MARKER. Anchored on the unique TokenStream.__init__ tail
# and the unique _advance_file() body — both invariant under all 24 existing patches
# (none touch TokenStream).
if "COPRIME_STRIDE_MARKER" in content:
    print("  ✓ coprime stride already applied")
else:
    # 1) Init: add stride state at the end of TokenStream.__init__
    old_cs_init = """        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0"""
    new_cs_init = """        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0
        # COPRIME_STRIDE_MARKER: optional shard-level coprime stride sampling
        self._coprime_stride = 1
        if int(os.environ.get("USE_COPRIME_STRIDE", "0")) and len(self.files) > 1:
            import math as _math
            import random as _random
            _rng = _random.Random(int(os.environ.get("SEED", "1337")))
            for _ in range(64):
                _s = _rng.randint(1, len(self.files) - 1)
                if _math.gcd(_s, len(self.files)) == 1:
                    self._coprime_stride = _s
                    break
            print(f"COPRIME_STRIDE: shard-level stride={self._coprime_stride} for N={len(self.files)} shards")"""
    if old_cs_init in content:
        content = content.replace(old_cs_init, new_cs_init)
        print("  ✓ added COPRIME_STRIDE init in TokenStream.__init__")
    else:
        print("  ✗ COPRIME_STRIDE init anchor not found — skipping")

    # 2) Apply: modify _advance_file to use the coprime stride instead of +1
    old_cs_apply = """    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0"""
    new_cs_apply = """    def _advance_file(self) -> None:
        # COPRIME_STRIDE_MARKER apply: use coprime stride if enabled, else stride=1
        self.file_idx = (self.file_idx + self._coprime_stride) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0"""
    if old_cs_apply in content:
        content = content.replace(old_cs_apply, new_cs_apply)
        print("  ✓ added COPRIME_STRIDE apply in _advance_file")
    else:
        print("  ✗ COPRIME_STRIDE apply anchor not found — skipping")

# Patch 19: USE_DEPTH_RECURRENCE=1 → re-run middle encoder blocks N times each.
# From PR #1437 (1.0809 BPB), PR #1445 (1.0889 BPB), PR #1331, PR #1421, PR #1260, PR #1334,
# PR #1290, PR #1204 — 8+ merged records use depth recurrence. Conservative variant ships
# here: re-run a single configurable block in the encoder N times. Default: block 3 twice
# (= 1 extra forward pass through one block, lowest OOM risk on our 12GB 3080 Ti).
#
# Reference: Universal Transformers (arxiv:1807.03819), ALBERT (arxiv:1909.11942) for the
# weight-sharing-across-depth idea. The competition uses a delayed-start variant
# (RECUR_START_STEP) to spend extra compute only after warmup.
#
# Implementation: insert a re-run loop INSIDE the encoder block iteration (just after the
# block forward, before wavelet/skip-append). Falls back to vanilla pass when env var unset.
#
# Idempotent via DEPTH_RECUR_MARKER. Anchored on the WAVELET-MODIFIED encoder loop (Patch 8
# runs before Patch 19 in this script). Two anchor points: init in __init__ (after wavelet
# init), apply in GPT.forward encoder loop.
if "DEPTH_RECUR_MARKER" in content:
    print("  ✓ depth recurrence already applied")
else:
    # 1) Init: insert env-var registration after wavelet init, before _init_weights()
    old_dr_init = """        self._wavelet_mix_ratio = float(os.environ.get("WAVELET_MIX_RATIO", "0.20"))
        self._init_weights()"""
    new_dr_init = """        self._wavelet_mix_ratio = float(os.environ.get("WAVELET_MIX_RATIO", "0.20"))
        # DEPTH_RECUR_MARKER: optional encoder block recurrence (PR #1437/#1445)
        self._depth_recur_enabled = bool(int(os.environ.get("USE_DEPTH_RECURRENCE", "0")))
        self._recur_start = int(os.environ.get("DEPTH_RECUR_START", "3"))
        self._recur_end = int(os.environ.get("DEPTH_RECUR_END", "3"))
        self._recur_cycles = int(os.environ.get("DEPTH_RECUR_CYCLES", "2"))
        self._init_weights()"""
    if old_dr_init in content:
        content = content.replace(old_dr_init, new_dr_init)
        print("  ✓ added DEPTH_RECUR init")
    else:
        print("  ✗ DEPTH_RECUR init anchor not found — skipping init")

    # 2) Apply: insert re-run loop inside the encoder iteration, after block forward
    #    Anchored on the WAVELET-MODIFIED encoder block body which has the form:
    #        x = self.blocks[i](x, x0)
    #        if self._wavelet_enabled:
    #            x = self._wavelet_mix(x, i, self._wavelet_mix_ratio)
    #        skips.append(x)
    old_dr_apply = """        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            if self._wavelet_enabled:
                x = self._wavelet_mix(x, i, self._wavelet_mix_ratio)
            skips.append(x)"""
    new_dr_apply = """        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            # DEPTH_RECUR_MARKER apply: re-run block i a few times if it's in the recur range
            if self._depth_recur_enabled and self._recur_start <= i <= self._recur_end:
                for _ in range(self._recur_cycles - 1):
                    x = self.blocks[i](x, x0)
            if self._wavelet_enabled:
                x = self._wavelet_mix(x, i, self._wavelet_mix_ratio)
            skips.append(x)"""
    if old_dr_apply in content:
        content = content.replace(old_dr_apply, new_dr_apply)
        print("  ✓ added DEPTH_RECUR encoder loop")
    else:
        print("  ✗ DEPTH_RECUR apply anchor not found — skipping apply")

# Patch 25: USE_NORMUON=1 → per-row normalization AFTER Newton-Schulz orthogonalization.
# From Mac SETUP.md §50 + LESSONS.md §35: "Per-row norm after Newton-Schulz — not in any
# PR. NorMuon Yes (per-row normalization) -0.132 BPB" — claimed as the BIGGEST optimizer
# win in our Mac research history that we never ported. ~5 LOC.
#
# Math: NS produces an approximately orthogonal matrix where rows have norm ≈ 1.
# Per-row normalization POST-NS enforces the unit-norm property exactly, tightening
# the orthogonalization. Different from MuonEq-R (which normalizes BEFORE NS).
#
# Distinct from Patch 17 Mousse (row+col preconditioning, before NS) and Patch 18
# MuonEq-R (row-only normalization, before NS). NorMuon is row-only AFTER NS.
#
# Idempotent via NORMUON_MARKER. Anchored on the post-NS scale-correction line which
# is invariant under all 24 prior patches (Mousse/MuonEq-R touch BEFORE NS, not after).
if "NORMUON_MARKER" in content:
    print("  ✓ NorMuon already applied")
else:
    old_normuon = """                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5"""
    new_normuon = """                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # NORMUON_MARKER: per-row normalization AFTER Newton-Schulz (Mac SETUP §50)
                    if int(os.environ.get("USE_NORMUON", "0")):
                        _post_norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        g = g / _post_norm
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5"""
    if old_normuon in content:
        content = content.replace(old_normuon, new_normuon)
        print("  ✓ added NORMUON post-NS row normalization")
    else:
        print("  ✗ NORMUON anchor not found — skipping")

# Patch 21: USE_XSA=1 → Exclusive Self Attention (orthogonal projection removal).
# From arxiv:2603.09078 "Exclusive Self Attention: Orthogonal Projection as an
# Architectural Inductive Bias" (Feb 2026, Shuangfei Zhai). Used in 100+ open PRs and
# 4+ MERGED records: PR #1099 (latest, 1.1133), #1019 (1.1147), #478 (1.12676), #287
# (1.1271), #315 (1.1248), #265 (1.1307). MOST-VALIDATED missing technique we don't have.
#
# Math: after standard SDPA produces y (shape B,H,T,D) and we have v (B,Hkv,T,D), remove
# the component of y that lies along the normalized self-value direction:
#   v_n = normalize(v, dim=-1)
#   y_out = y - <y, v_n> * v_n  (per-token, per-head)
#
# Effect: removes the "self-attention bias" (cosine sim of output with self-value grows
# with depth). Forces the model to attend to CONTEXT rather than reconstructing its own
# position. Reported gain: +0.002 to +0.005 BPB across multiple records.
#
# 0 new params. ~2ms/step overhead at all 11 layers (negligible vs ~190ms baseline).
#
# Applied INLINE in CausalSelfAttention.forward, AFTER SDPA + GATED_ATTENTION block,
# BEFORE the transpose/reshape. Anchored on the GATED_ATTENTION-MODIFIED transpose line
# (Patch 16 runs before Patch 21). This is the all-layers variant ("XSA-all") which is
# the canonical form used in PR #1099 (latest merged record).
#
# Idempotent via XSA_MARKER. Falls back to no-op when env var unset.
if "XSA_MARKER" in content:
    print("  ✓ XSA already applied")
else:
    # Anchor on the GATED_ATTENTION-modified transpose line which is unique
    old_xsa = """            y = y * _gate
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)"""
    new_xsa = """            y = y * _gate
        # XSA_MARKER: Exclusive Self Attention orthogonal projection (arxiv:2603.09078, PR #1099)
        if int(os.environ.get("USE_XSA", "0")):
            # y: (B, H, T, D), v: (B, Hkv, T, D) — handle GQA via grouped reshape
            _B, _H, _T, _D = y.shape
            _Hkv = v.size(1)
            _group = _H // _Hkv
            _y_g = y.reshape(_B, _Hkv, _group, _T, _D)
            _vn = F.normalize(v.float(), dim=-1).unsqueeze(2).to(dtype=y.dtype)  # (B, Hkv, 1, T, D)
            _dot = (_y_g * _vn).sum(dim=-1, keepdim=True)  # (B, Hkv, group, T, 1)
            y = (_y_g - _dot * _vn).reshape(_B, _H, _T, _D)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)"""
    if old_xsa in content:
        content = content.replace(old_xsa, new_xsa)
        print("  ✓ added XSA orthogonal projection")
    else:
        print("  ✗ XSA anchor not found — skipping (XSA will be no-op)")

# Patch 18: USE_MUONEQ_R=1 → row-only normalization before Newton-Schulz orthogonalization.
# From arxiv:2603.28254 "MuonEq: Balancing Before Orthogonalization with Lightweight
# Equilibration" (Mar 2026). Used in 40+ openai/parameter-golf PRs, top record PR #1260
# at val_bpb 1.0929 (3-seed mean).
#
# Mathematical formulation: for each Muon-managed weight matrix G (after momentum),
#   row_norm[i] = sqrt(sum_j G[i,j]^2)
#   G_normalized[i,j] = G[i,j] / row_norm[i]
# After this, every row has unit L2 norm. Then standard Newton-Schulz.
#
# Distinct from Patch 17 Mousse: Mousse is row+col preconditioning (G/(||row||*||col||)),
# MuonEq-R is row-only (G/||row||). They can stack: Mousse first, then MuonEq-R, then NS5.
# PR #1440 uses both stacked. Implementation: 5 LOC, same anchor strategy as Patch 17 — but
# anchored AFTER the Mousse block since Patch 17 runs first in this script.
#
# Idempotent via MUONEQ_R_MARKER. Anchored on the same Newton-Schulz call line which is
# still present after Patch 17 (Patch 17's new_ns ends with that line).
if "MUONEQ_R_MARKER" in content:
    print("  ✓ MuonEq-R already applied")
else:
    old_ns_eqr = """                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)"""
    new_ns_eqr = """                    # MUONEQ_R_MARKER: optional row-only normalization (arxiv:2603.28254)
                    if int(os.environ.get("USE_MUONEQ_R", "0")):
                        _row_norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        g = g / _row_norm
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)"""
    if old_ns_eqr in content:
        content = content.replace(old_ns_eqr, new_ns_eqr)
        print("  ✓ added MUONEQ_R row normalization")
    else:
        print("  ✗ MUONEQ_R anchor not found — skipping (MuonEq-R will be no-op)")

# Patch 17: USE_MOUSSE=1 → diagonal Kronecker preconditioning before Newton-Schulz
# orthogonalization in the Muon optimizer step.
#
# From PR #1440 [Submission] EngramLite + Mousse + Progressive Depth Recurrence + TTT
# (val_bpb 1.1026, attribution -0.002 BPB to Mousse alone). Reference paper:
# arxiv:2603.09697 "Mousse: Rectifying the Geometry of Muon with Curvature-Aware
# Preconditioning" (Feb 2026).
#
# Mathematical formulation: for each Muon-managed weight matrix G,
#   L_diag = diag(G @ G^T)  # row sum of squares
#   R_diag = diag(G^T @ G)  # col sum of squares
#   G_pre = G * L_diag^(-1/2) * R_diag^(-1/2)
# Equivalently: G[i,j] /= ||row_i||_2 * ||col_j||_2
#
# Audit of comp PRs (research fire #9): only PR #1440 mentions Mousse, and even THEY
# didn't implement the full EMA + eigendecomposition machinery — they ship the simplified
# diagonal preconditioning only. We're shipping the same simplified version (~5 LOC),
# gated by USE_MOUSSE=1, falling back to vanilla Muon when the env var is unset.
#
# Idempotent via MOUSSE_MARKER. Anchored on the unique zeropower_via_newtonschulz5 call
# in the Muon optimizer step which has been stable since Patch 2.
if "MOUSSE_MARKER" in content:
    print("  ✓ Mousse already applied")
else:
    old_ns = """                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)"""
    new_ns = """                    # MOUSSE_MARKER: optional diagonal preconditioning before Newton-Schulz (arxiv:2603.09697)
                    if int(os.environ.get("USE_MOUSSE", "0")):
                        _l = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        _r = g.norm(dim=-2, keepdim=True).clamp(min=1e-8)
                        g = g / (_l * _r)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)"""
    if old_ns in content:
        content = content.replace(old_ns, new_ns)
        print("  ✓ added MOUSSE diagonal preconditioning")
    else:
        print("  ✗ MOUSSE anchor not found — skipping (Mousse will be no-op)")

# Patch 26 (C90 build #1, world-novel L06 #3): USE_ASYMMETRIC_SKIP_INIT=1 → init U-net
# skip_weights at 0.5 instead of 1.0. init=1.0 preserves signal (standard); init=0 is
# ReZero; init=0.5 is an explicit information-bottleneck claim that doesn't appear in
# any paper or PR. Source: STACK_NOVELTY_PLAN.md L06 #3. Expected delta: -0.006 train_loss.
# Falsification: step-500 train_loss worse than init=1.0 by ≥0.005.
# Idempotent via ASYMMETRIC_SKIP_INIT_MARKER.
if "ASYMMETRIC_SKIP_INIT_MARKER" in content:
    print("  ✓ asymmetric skip init already applied")
else:
    old_skip_init = """        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))"""
    new_skip_init = """        # ASYMMETRIC_SKIP_INIT_MARKER: optional 0.5-init for U-net skip weights (world-novel L06 #3)
        _skip_init_val = 0.5 if int(os.environ.get("USE_ASYMMETRIC_SKIP_INIT", "0")) else 1.0
        self.skip_weights = nn.Parameter(torch.full((self.num_skip_weights, model_dim), _skip_init_val, dtype=torch.float32))"""
    if old_skip_init in content:
        content = content.replace(old_skip_init, new_skip_init)
        print("  ✓ added ASYMMETRIC_SKIP_INIT 0.5-init option")
    else:
        print("  ✗ ASYMMETRIC_SKIP_INIT anchor not found — skipping")

# Patch 27 (C90 build #2, world-novel L05 #3): USE_NORM_PCT_DROPOUT=1 → zero out FFN
# intermediate rows whose per-token L2 norm is in the top 1% (99th percentile). Standard
# dropout = random elements; structured dropout = random rows; norm-percentile dropout
# = rows with highest norm. Targets the rare exploding-activation pathway.
# Source: STACK_NOVELTY_PLAN.md L05 #3. Expected delta: -0.006 train_loss.
# Falsification: step-500 train_loss unchanged or worse.
# Anchor: the LEAKY_RELU-modified MLP.forward (Patch 9 runs before this in the heredoc).
# Idempotent via NORM_PCT_DROPOUT_MARKER.
if "NORM_PCT_DROPOUT_MARKER" in content:
    print("  ✓ norm-percentile dropout already applied")
else:
    old_mlp_body = """    def forward(self, x: Tensor) -> Tensor:
        # LEAKY_RELU_MARKER: optional leaky_relu(0.5)^2 activation (Mac -0.014 BPB)
        if int(os.environ.get("USE_LEAKY_RELU", "0")):
            x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        else:
            x = torch.relu(self.fc(x))
        return self.proj(x.square())"""
    new_mlp_body = """    def forward(self, x: Tensor) -> Tensor:
        # LEAKY_RELU_MARKER: optional leaky_relu(0.5)^2 activation (Mac -0.014 BPB)
        if int(os.environ.get("USE_LEAKY_RELU", "0")):
            x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        else:
            x = torch.relu(self.fc(x))
        x = x.square()
        # NORM_PCT_DROPOUT_MARKER: zero top-1% per-token L2-norm rows (world-novel L05 #3)
        if self.training and int(os.environ.get("USE_NORM_PCT_DROPOUT", "0")):
            _npd_thresh = float(os.environ.get("NORM_PCT_THRESH", "0.99"))
            _orig_shape = x.shape
            _x2 = x.reshape(-1, _orig_shape[-1])
            _row_norms = _x2.float().norm(dim=-1)
            _kth = torch.quantile(_row_norms, _npd_thresh)
            _keep = (_row_norms < _kth).to(dtype=x.dtype).unsqueeze(-1)
            _x2 = _x2 * _keep
            x = _x2.reshape(_orig_shape)
        return self.proj(x)"""
    if old_mlp_body in content:
        content = content.replace(old_mlp_body, new_mlp_body)
        print("  ✓ added NORM_PCT_DROPOUT top-1% row kill")
    else:
        print("  ✗ NORM_PCT_DROPOUT anchor not found — skipping")

# Patch 28 (C90 mass-build, world-novel L04 #1): USE_COPRIME_PER_HEAD_ROPE=1 → each
# attention head uses a DIFFERENT prime base in rotary inv_freq. 8 coprime primes
# (10007/10009/10037/10039/10061/10067/10069/10079) → distinct positional spectra
# per head, reduces head redundancy at zero parameter cost.
# Source: STACK_NOVELTY_PLAN.md L04 + RESEARCH_BACKLOG.md L04 row 3.
# Idempotent via COPRIME_PER_HEAD_ROPE_MARKER.
if "COPRIME_PER_HEAD_ROPE_MARKER" in content:
    print("  ✓ coprime per-head rope already applied")
else:
    old_rot_init = """class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None"""
    new_rot_init = """class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0, num_heads: int = 1, num_kv_heads: int = 1):
        super().__init__()
        # COPRIME_PER_HEAD_ROPE_MARKER: optionally per-head distinct prime bases
        self._coprime_ph_rope = bool(int(os.environ.get("USE_COPRIME_PER_HEAD_ROPE", "0")))
        self._n_heads_rope = num_heads
        self._n_kv_heads_rope = num_kv_heads
        if self._coprime_ph_rope and num_heads > 1:
            _PRIMES = [10007.0, 10009.0, 10037.0, 10039.0, 10061.0, 10067.0, 10069.0, 10079.0]
            _ph_bases = torch.tensor(
                [_PRIMES[h % len(_PRIMES)] for h in range(num_heads)], dtype=torch.float32
            )
            _exp = (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            inv_freq = 1.0 / (_ph_bases.unsqueeze(-1) ** _exp.unsqueeze(0))
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
        self._cos_k_cached: Tensor | None = None
        self._sin_k_cached: Tensor | None = None"""
    if old_rot_init in content:
        content = content.replace(old_rot_init, new_rot_init)
        print("  ✓ patched Rotary.__init__ for per-head primes")
    else:
        print("  ✗ COPRIME_PER_HEAD_ROPE Rotary __init__ anchor not found — skipping")

    old_rot_fwd = """    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)"""
    new_rot_fwd = """    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            if self._coprime_ph_rope and self.inv_freq.dim() == 2:
                # COPRIME_PER_HEAD_ROPE_MARKER apply: per-head freqs (1, H, T, half)
                freqs = torch.einsum("t,hd->htd", t, self.inv_freq.to(device))
                self._cos_cached = freqs.cos().unsqueeze(0)
                self._sin_cached = freqs.sin().unsqueeze(0)
                _group = max(1, self._n_heads_rope // max(self._n_kv_heads_rope, 1))
                self._cos_k_cached = self._cos_cached[:, ::_group]
                self._sin_k_cached = self._sin_cached[:, ::_group]
            else:
                freqs = torch.outer(t, self.inv_freq.to(device))
                self._cos_cached = freqs.cos()[None, None, :, :]
                self._sin_cached = freqs.sin()[None, None, :, :]
                self._cos_k_cached = self._cos_cached
                self._sin_k_cached = self._sin_cached
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)"""
    if old_rot_fwd in content:
        content = content.replace(old_rot_fwd, new_rot_fwd)
        print("  ✓ patched Rotary.forward for per-head emission")
    else:
        print("  ✗ COPRIME_PER_HEAD_ROPE Rotary forward anchor not found — skipping")

    old_csa_rot_ctor = """        self.rotary = Rotary(self.head_dim, base=rope_base)"""
    new_csa_rot_ctor = """        self.rotary = Rotary(self.head_dim, base=rope_base, num_heads=num_heads, num_kv_heads=num_kv_heads)"""
    if old_csa_rot_ctor in content:
        content = content.replace(old_csa_rot_ctor, new_csa_rot_ctor)
        print("  ✓ wired num_heads/num_kv_heads into Rotary constructor")

    old_csa_apply = """        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)"""
    new_csa_apply = """        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        # COPRIME_PER_HEAD_ROPE_MARKER call site: dispatch per-head cos/sin to q vs k
        if int(os.environ.get("USE_COPRIME_PER_HEAD_ROPE", "0")) and self.rotary._cos_k_cached is not None and self.rotary.inv_freq.dim() == 2:
            cos_k = self.rotary._cos_k_cached.to(dtype=q.dtype)
            sin_k = self.rotary._sin_k_cached.to(dtype=q.dtype)
        else:
            cos_k, sin_k = cos, sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos_k, sin_k)"""
    if old_csa_apply in content:
        content = content.replace(old_csa_apply, new_csa_apply)
        print("  ✓ patched CSA apply for per-head dispatch")

# Patch 29 (C90 mass-build, world-novel L07 #1): USE_ASYM_LABEL_SMOOTHING=1 → ε=0.01
# label smoothing applied ONLY to tokens whose unigram log-prob > thresh (frequent
# tokens). Rare/hard tokens get HARD targets — inverts the standard recipe to
# preserve gradient signal on the long tail (where BPB is dominated).
# Source: STACK_NOVELTY_PLAN.md L07 + RESEARCH_BACKLOG.md L07 row 3.
# Idempotent via ASYM_LABEL_SMOOTHING_MARKER.
if "ASYM_LABEL_SMOOTHING_MARKER" in content:
    print("  ✓ asym label smoothing already applied")
else:
    old_als_init = """                print("BYTE_WEIGHT: build failed:", _e)
                self._byte_weight_enabled = False"""
    new_als_init = """                print("BYTE_WEIGHT: build failed:", _e)
                self._byte_weight_enabled = False
        # ASYM_LABEL_SMOOTHING_MARKER: build unigram log-prob LUT for asymmetric smoothing
        self._als_enabled = bool(int(os.environ.get("USE_ASYM_LABEL_SMOOTHING", "0")))
        self._als_eps = float(os.environ.get("ASYM_LABEL_SMOOTHING_EPS", "0.01"))
        self._als_freq_thresh = float(os.environ.get("ASYM_LABEL_SMOOTHING_THRESH", "-3.0"))
        self.register_buffer("_unigram_logprob_lut", torch.zeros(vocab_size, dtype=torch.float32), persistent=False)
        if self._als_enabled:
            try:
                import numpy as _np
                _als_suffix = "_tab" if bool(int(os.environ.get("USE_TABULATION_HASH", "0"))) else ""
                _bg_full = _np.load("./data/bigram_tab_{}v{}.npy".format(vocab_size, _als_suffix))
                _bg_t = torch.from_numpy(_bg_full).float()
                _col_max = _bg_t.max(dim=0).values
                _col_max = _col_max - torch.logsumexp(_col_max, dim=0)
                self._unigram_logprob_lut = _col_max
                _frac_freq = float((_col_max > self._als_freq_thresh).float().mean())
                print("ASYM_LABEL_SMOOTHING: unigram LUT built, frac_freq=", _frac_freq, " eps=", self._als_eps)
            except Exception as _e:
                print("ASYM_LABEL_SMOOTHING: LUT build failed:", _e)
                self._als_enabled = False"""
    if old_als_init in content:
        content = content.replace(old_als_init, new_als_init)
        print("  ✓ added ASYM_LABEL_SMOOTHING init")
    else:
        print("  ✗ ASYM_LABEL_SMOOTHING init anchor not found — skipping")

    old_als_apply = """        if self._byte_weight_enabled:
            _ce = F.cross_entropy(logits.float(), targets, reduction="none")
            _w = self._byte_weight_lut[targets]
            return (_ce * _w).sum() / _w.sum()
        return F.cross_entropy(logits.float(), targets, reduction="mean")"""
    new_als_apply = """        # ASYM_LABEL_SMOOTHING_MARKER apply: ε-smooth ONLY frequent tokens
        if getattr(self, '_als_enabled', False):
            _als_logp = torch.log_softmax(logits.float(), dim=-1)
            _als_targ_logp = _als_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            _als_uni_logp = self._unigram_logprob_lut[targets]
            _als_freq_mask = (_als_uni_logp > self._als_freq_thresh).to(dtype=_als_logp.dtype)
            _als_hard = -_als_targ_logp
            _als_uniform_nll = -_als_logp.mean(dim=-1)
            _als_smooth = (1.0 - self._als_eps) * _als_hard + self._als_eps * _als_uniform_nll
            _als_per_tok = _als_freq_mask * _als_smooth + (1.0 - _als_freq_mask) * _als_hard
            if self._byte_weight_enabled:
                _w = self._byte_weight_lut[targets]
                return (_als_per_tok * _w).sum() / _w.sum()
            return _als_per_tok.mean()
        if self._byte_weight_enabled:
            _ce = F.cross_entropy(logits.float(), targets, reduction="none")
            _w = self._byte_weight_lut[targets]
            return (_ce * _w).sum() / _w.sum()
        return F.cross_entropy(logits.float(), targets, reduction="mean")"""
    if old_als_apply in content:
        content = content.replace(old_als_apply, new_als_apply)
        print("  ✓ added ASYM_LABEL_SMOOTHING wrapper")
    else:
        print("  ✗ ASYM_LABEL_SMOOTHING apply anchor not found — skipping")

# Patch 30 (C90 mass-build, world-novel L08 #1): USE_PER_PROJ_LR_SPLIT=1 → split
# Muon param group so q.weight, k.weight, v.weight get DISTINCT learning rates.
# Q/K/V have very different gradient statistics (Q downstream of QK gain, K twice-
# normed, V the only one whose grads flow through SDPA softmax directly).
# Source: STACK_NOVELTY_PLAN.md L08 + RESEARCH_BACKLOG.md L08 row 3.
# Idempotent via PER_PROJ_LR_SPLIT_MARKER.
if "PER_PROJ_LR_SPLIT_MARKER" in content:
    print("  ✓ per-projection LR split already applied")
else:
    old_per_proj = """    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr"""
    new_per_proj = """    # PER_PROJ_LR_SPLIT_MARKER: split Muon q/k/v into 3 sub-groups with distinct LRs
    if int(os.environ.get("USE_PER_PROJ_LR_SPLIT", "0")):
        _q_mult = float(os.environ.get("Q_LR_MULT", "1.0"))
        _k_mult = float(os.environ.get("K_LR_MULT", "1.4"))
        _v_mult = float(os.environ.get("V_LR_MULT", "0.7"))
        _q_p, _k_p, _v_p, _other_p = [], [], [], []
        for _name, _p in block_named_params:
            if _p.ndim != 2 or any(pat in _name for pat in CONTROL_TENSOR_NAME_PATTERNS):
                continue
            if ".c_q.weight" in _name:
                _q_p.append(_p)
            elif ".c_k.weight" in _name:
                _k_p.append(_p)
            elif ".c_v.weight" in _name:
                _v_p.append(_p)
            else:
                _other_p.append(_p)
        optimizer_muon = Muon(
            _other_p if _other_p else [next(iter(matrix_params))],
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
        )
        optimizer_muon.param_groups = []
        optimizer_muon.state = type(optimizer_muon.state)()
        for _gname, _gparams, _gmult in [
            ("muon_q", _q_p, _q_mult),
            ("muon_k", _k_p, _k_mult),
            ("muon_v", _v_p, _v_mult),
            ("muon_other", _other_p, 1.0),
        ]:
            if not _gparams:
                continue
            _grp = {
                "params": _gparams,
                "lr": args.matrix_lr * _gmult,
                "base_lr": args.matrix_lr * _gmult,
                "momentum": args.muon_momentum,
                "backend_steps": args.muon_backend_steps,
                "nesterov": True,
            }
            optimizer_muon.add_param_group(_grp)
        log0(f"PER_PROJ_LR_SPLIT: q={_q_mult} k={_k_mult} v={_v_mult} groups={len(optimizer_muon.param_groups)}")
    else:
        optimizer_muon = Muon(
            matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
        )
        for group in optimizer_muon.param_groups:
            group["base_lr"] = args.matrix_lr"""
    if old_per_proj in content:
        content = content.replace(old_per_proj, new_per_proj)
        print("  ✓ added PER_PROJ_LR_SPLIT Muon group split")
    else:
        print("  ✗ PER_PROJ_LR_SPLIT anchor not found — skipping")

# Patch 31 (C90 mass-build, world-novel L09 #1): USE_CTX_PARTITIONED_TAB=1 → 16
# virtual sub-tables via slice rotation by (prev mod S) * (HASH/S). Effectively
# partitions hash buckets into 16 zones, each absorbing 1/16 of contexts → 16x
# finer-grained smoothing. Mini-paper extension of MINIPAPER_TABULATION_HASH.md.
# Source: STACK_NOVELTY_PLAN.md L09 + RESEARCH_BACKLOG.md L09 row 3.
# Idempotent via CTX_PARTITIONED_TAB_MARKER.
if "CTX_PARTITIONED_TAB_MARKER" in content:
    print("  ✓ context-partitioned tabulation already applied")
else:
    old_cpt_init = """                print("NGRAM_BIAS: fourgram load failed:", _e)
        # NGRAM_GATE_MARKER: learned per-position gate over n-gram bias"""
    new_cpt_init = """                print("NGRAM_BIAS: fourgram load failed:", _e)
        # CTX_PARTITIONED_TAB_MARKER: precompute slice config for partitioned tabulation
        self._ctx_part_tab_enabled = bool(int(os.environ.get("USE_CTX_PARTITIONED_TAB", "0")))
        self._ctx_part_slices = int(os.environ.get("CTX_PARTITION_SLICES", "16"))
        if self._ctx_part_tab_enabled:
            print("CTX_PARTITIONED_TAB: enabled with", self._ctx_part_slices, "slices")
        # NGRAM_GATE_MARKER: learned per-position gate over n-gram bias"""
    if old_cpt_init in content:
        content = content.replace(old_cpt_init, new_cpt_init)
        print("  ✓ added CTX_PARTITIONED_TAB init")
    else:
        print("  ✗ CTX_PARTITIONED_TAB init anchor not found — skipping")

    old_cpt_bi = """            _bi_bias = self._ngram_w_bigram * self._bigram_tab[_h_bi]
            if _gate is not None:
                _bi_bias = _gate * _bi_bias
            logits = logits + _bi_bias"""
    new_cpt_bi = """            # CTX_PARTITIONED_TAB_MARKER apply (bigram): slice rotation
            if getattr(self, '_ctx_part_tab_enabled', False):
                _S = self._ctx_part_slices
                _zone = (_ids_flat % _S) * (_H // _S)
                _h_bi_p = (_h_bi + _zone) % _H
                _bi_bias = self._ngram_w_bigram * self._bigram_tab[_h_bi_p]
            else:
                _bi_bias = self._ngram_w_bigram * self._bigram_tab[_h_bi]
            if _gate is not None:
                _bi_bias = _gate * _bi_bias
            logits = logits + _bi_bias"""
    if old_cpt_bi in content:
        content = content.replace(old_cpt_bi, new_cpt_bi)
        print("  ✓ added CTX_PARTITIONED_TAB bigram slice")
    else:
        print("  ✗ CTX_PARTITIONED_TAB bigram anchor not found — skipping")

# Patch 32 (C90 mass-build #2, world-novel L02 INFRA #1): USE_DAT_BYTE_ENTROPY_CURRICULUM=1
# → reorder TokenStream's shard list low-to-high zlib-compression-ratio (proxy for byte
# entropy / Kolmogorov complexity). Easy bytes first, harder shards later. Pure data-loader
# infra novelty: model-free, deterministic, 1-pass entropy proxy via zlib level-1.
# Falls back to on-the-fly compute (~250ms/shard) if shard_entropy.json doesn't exist.
# Stacks with USE_COPRIME_STRIDE: stride is computed AFTER reorder.
# Source: STACK_NOVELTY_PLAN.md L02 + RESEARCH_BACKLOG.md L02 row 3.
# Idempotent via DAT_BYTE_ENTROPY_CURRICULUM_MARKER.
if "DAT_BYTE_ENTROPY_CURRICULUM_MARKER" in content:
    print("  ✓ byte-entropy curriculum already applied")
else:
    old_files_init = """        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")"""
    new_files_init = """        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        # DAT_BYTE_ENTROPY_CURRICULUM_MARKER: optional easy→hard shard ordering
        if int(os.environ.get("USE_DAT_BYTE_ENTROPY_CURRICULUM", "0")) and len(self.files) > 1:
            import json as _json
            _bec_reverse = bool(int(os.environ.get("BEC_REVERSE", "0")))
            _bec_paths = [
                self.files[0].parent / "shard_entropy.json",
                Path("data") / "datasets" / self.files[0].parent.name / "shard_entropy.json",
                Path("./data/shard_entropy.json"),
            ]
            _bec_map = None
            for _bp in _bec_paths:
                if _bp.exists():
                    try:
                        with open(_bp, "r") as _bf:
                            _bec_map = _json.load(_bf)
                        print(f"DAT_BYTE_ENTROPY_CURRICULUM: loaded {_bp} with {len(_bec_map)} entries")
                        break
                    except Exception as _e:
                        print(f"DAT_BYTE_ENTROPY_CURRICULUM: failed to load {_bp}: {_e}")
            if _bec_map is None:
                import zlib as _zlib
                _bec_map = {}
                for _f in self.files:
                    try:
                        with open(_f, "rb") as _fh:
                            _fh.seek(1024)
                            _sample = _fh.read(1024 * 1024)
                        _ratio = float(len(_zlib.compress(_sample, 1))) / float(max(len(_sample), 1))
                        _bec_map[_f.name] = _ratio
                    except Exception:
                        _bec_map[_f.name] = 1.0
                print(f"DAT_BYTE_ENTROPY_CURRICULUM: computed on-the-fly entropy for {len(_bec_map)} shards")
            _val_files = [f for f in self.files if "val" in f.name]
            _train_files = [f for f in self.files if "val" not in f.name]
            _train_files.sort(key=lambda _f: _bec_map.get(_f.name, 1.0), reverse=_bec_reverse)
            self.files = _train_files + _val_files
            _r0 = _bec_map.get(_train_files[0].name, 1.0) if _train_files else None
            _rN = _bec_map.get(_train_files[-1].name, 1.0) if _train_files else None
            print(f"DAT_BYTE_ENTROPY_CURRICULUM: reordered {len(_train_files)} shards first={_r0} last={_rN} reverse={_bec_reverse}")"""
    if old_files_init in content:
        content = content.replace(old_files_init, new_files_init)
        print("  ✓ added DAT_BYTE_ENTROPY_CURRICULUM shard reorder")
    else:
        print("  ✗ DAT_BYTE_ENTROPY_CURRICULUM anchor not found — skipping")

# Patch 33 (C90 mass-build #2, world-novel L03 #1): USE_EMB_DCT_COEFFICIENT_ENERGY_TRUNCATE=1
# → constrain tok_emb.weight to live on a per-row K-sparse subspace of the DCT-II basis.
# K is chosen per-row at init so that K coefficients hold ENERGY_FRAC (default 0.85) of the
# row's spectral energy. The mask is then frozen, and a forward pre-hook re-projects
# tok_emb.weight onto the masked DCT subspace at the START of every forward pass — making
# the constraint a hard manifold constraint over the optimizer trajectory.
# Energy-adaptive per-row K is the world-novel piece: punctuation rows get small K, content
# rows get large K. Stacks with tied embeddings + MTP + NGRAM_BIAS (none mutate tok_emb).
# Source: STACK_NOVELTY_PLAN.md L03 + RESEARCH_BACKLOG.md L03 row 7.
# Idempotent via EMB_DCT_COEFFICIENT_ENERGY_TRUNCATE_MARKER.
if "EMB_DCT_COEFFICIENT_ENERGY_TRUNCATE_MARKER" in content:
    print("  ✓ DCT energy-truncate already applied")
else:
    old_emb_init = """        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2"""
    new_emb_init = """        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        # EMB_DCT_COEFFICIENT_ENERGY_TRUNCATE_MARKER: per-row energy-truncated DCT subspace
        self._emb_dct_enabled = bool(int(os.environ.get("USE_EMB_DCT_COEFFICIENT_ENERGY_TRUNCATE", "0")))
        self._emb_dct_energy = float(os.environ.get("EMB_DCT_ENERGY_FRAC", "0.85"))
        self._emb_dct_min_k = int(os.environ.get("EMB_DCT_MIN_K", "8"))
        self._emb_dct_initialized = False
        self.register_buffer("_emb_dct_basis", torch.zeros(1, dtype=torch.float32), persistent=False)
        self.register_buffer("_emb_dct_basis_T", torch.zeros(1, dtype=torch.float32), persistent=False)
        self.register_buffer("_emb_dct_mask", torch.zeros(1, dtype=torch.float32), persistent=False)
        self.num_encoder_layers = num_layers // 2"""
    if old_emb_init in content:
        content = content.replace(old_emb_init, new_emb_init)
        print("  ✓ added EMB_DCT init buffers")
    else:
        print("  ✗ EMB_DCT init anchor not found — skipping")

    old_fwd_top = """    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))"""
    new_fwd_top = """    def _emb_dct_init_and_project(self) -> None:
        # EMB_DCT_COEFFICIENT_ENERGY_TRUNCATE_MARKER lazy init + per-step projection.
        with torch.no_grad():
            W = self.tok_emb.weight
            V, D = W.shape
            dev = W.device
            if not self._emb_dct_initialized:
                _n = torch.arange(D, dtype=torch.float32, device=dev).unsqueeze(0)
                _k = torch.arange(D, dtype=torch.float32, device=dev).unsqueeze(1)
                _basis = torch.cos(torch.pi * (_n + 0.5) * _k / float(D))
                _basis = _basis * torch.sqrt(torch.tensor(2.0 / float(D), device=dev))
                _basis[0] = _basis[0] * (1.0 / torch.sqrt(torch.tensor(2.0, device=dev)))
                _coef = W.float() @ _basis.T
                _energy = (_coef ** 2)
                _row_total = _energy.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                _sorted_e, _sorted_idx = _energy.sort(dim=-1, descending=True)
                _cum = _sorted_e.cumsum(dim=-1) / _row_total
                _within = (_cum < self._emb_dct_energy).to(torch.long).sum(dim=-1) + 1
                _k_per_row = _within.clamp(min=self._emb_dct_min_k, max=D)
                _mask = torch.zeros_like(_coef)
                _col_pos = torch.arange(D, device=dev).unsqueeze(0).expand(V, D)
                _keep = (_col_pos < _k_per_row.unsqueeze(-1)).to(_mask.dtype)
                _mask.scatter_(1, _sorted_idx, _keep)
                self._emb_dct_basis = _basis
                self._emb_dct_basis_T = _basis.T.contiguous()
                self._emb_dct_mask = _mask
                _coef_masked = _coef * _mask
                W.data.copy_((_coef_masked @ _basis).to(W.dtype))
                self._emb_dct_initialized = True
                _avg_k = float(_k_per_row.float().mean())
                print(f"EMB_DCT: V={V} D={D} energy={self._emb_dct_energy} avg_k={_avg_k:.1f}")
            else:
                _basis = self._emb_dct_basis
                _basis_T = self._emb_dct_basis_T
                _mask = self._emb_dct_mask
                _coef = W.float() @ _basis_T
                _coef = _coef * _mask
                W.data.copy_((_coef @ _basis).to(W.dtype))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        # EMB_DCT_COEFFICIENT_ENERGY_TRUNCATE_MARKER apply: project tok_emb.weight pre-fwd
        if getattr(self, '_emb_dct_enabled', False):
            self._emb_dct_init_and_project()
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))"""
    if old_fwd_top in content:
        content = content.replace(old_fwd_top, new_fwd_top)
        print("  ✓ added EMB_DCT forward projection")
    else:
        print("  ✗ EMB_DCT forward anchor not found — skipping")

# Patch 42 (C90 build #13, comp-port L11 #5): USE_KER_TMA_MEGAKERNEL=1 → port the SOTA
# Triton TMA megakernel for the MLP path. Source: openai/parameter-golf
# record/tma-megakernel-triple-loop (b27fe93, @andrewbaggio1, val_bpb=1.08480 SOTA).
# See SOTA_TMA_KERNEL_INTEL.md for the verbatim kernel + autograd Function.
#
# The kernel + wrapper + autograd Function lives in runpod_tests/loop/triton_tma_mlp.py
# (separate file because the @triton.jit decorator needs module-level definition AND
# the patcher heredoc would break on Triton's f-string syntax). The patcher just adds
# a SHIM in train_gpt.py that:
#   1. tries to import the helper module
#   2. modifies the MLP forward to dispatch to fused_mlp() when env var on + supported
#   3. falls back to standard path otherwise
#
# Composes with LEAKY_RELU (Patch 9) — both gate the same MLP forward; the fused kernel
# IMPLEMENTS leaky_relu(0.5)^2 internally so when both are on, the kernel takes the path
# and the leaky_relu Python branch is dead code (no double-application). When fused kernel
# is OFF or unsupported, the LEAKY_RELU Python branch runs as usual.
#
# Hopper-only: cheap pods (3090, 4070 Ti) have no TMA descriptor support → import fails
# → HAS_TRITON_TMA=False → fused_mlp is_supported() returns False → fall back to standard
# path → bit-exact with baseline. Validation must wait for the H100 production run.
#
# Win mechanism on H100: +10.5% throughput → +127 steps in 600s → -0.02 to -0.03 BPB.
#
# Idempotent via KER_TMA_MEGAKERNEL_MARKER. Two anchors:
#   A) module-level: insert the import shim near other imports
#   B) MLP forward (post-Patch-9): add the fused dispatch BEFORE the leaky/relu branch
if "KER_TMA_MEGAKERNEL_MARKER" in content:
    pass
else:
    # Anchor A: insert lazy import shim near the existing CastedLinear import area
    # (using the import line for `from torch import Tensor` as a stable anchor that
    # appears once and is invariant under all 41 prior patches).
    old_import_anchor = "from torch import Tensor"
    new_import_anchor = """from torch import Tensor
# KER_TMA_MEGAKERNEL_MARKER: lazy import the Triton TMA megakernel helper.
# Falls through to standard path on cheap pods (3090, 4070 Ti) where TMA is unavailable.
try:
    import sys as _sys_ker_tma
    _sys_ker_tma.path.insert(0, 'runpod_tests/loop')
    import triton_tma_mlp as _ker_tma_mlp
    _KER_TMA_MLP_AVAILABLE = True
except Exception as _e_ker_tma:
    _KER_TMA_MLP_AVAILABLE = False"""
    if old_import_anchor in content:
        content = content.replace(old_import_anchor, new_import_anchor, 1)
        # Anchor B: modify MLP forward (post-Patch-9 leaky version) to dispatch to fused kernel
        old_mlp_fwd = """    def forward(self, x: Tensor) -> Tensor:
        # LEAKY_RELU_MARKER: optional leaky_relu(0.5)^2 activation (Mac -0.014 BPB)
        if int(os.environ.get('USE_LEAKY_RELU', '0')):
            x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        else:
            x = torch.relu(self.fc(x))
        return self.proj(x.square())"""
        new_mlp_fwd = """    def forward(self, x: Tensor) -> Tensor:
        # KER_TMA_MEGAKERNEL_MARKER apply: dispatch to fused TMA kernel when supported.
        if (int(os.environ.get('USE_KER_TMA_MEGAKERNEL', '0'))
            and _KER_TMA_MLP_AVAILABLE
            and self.training
            and _ker_tma_mlp.is_supported(x)):
            try:
                return _ker_tma_mlp.fused_mlp(x, self.fc.weight, self.proj.weight)
            except Exception as _e_fused_mlp:
                if int(os.environ.get('TMA_VERBOSE', '0')):
                    print(f"KER_TMA_MEGAKERNEL: fused path failed, falling back: {_e_fused_mlp}")
                # fall through to standard path
        # LEAKY_RELU_MARKER: optional leaky_relu(0.5)^2 activation (Mac -0.014 BPB)
        if int(os.environ.get('USE_LEAKY_RELU', '0')):
            x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        else:
            x = torch.relu(self.fc(x))
        return self.proj(x.square())"""
        if old_mlp_fwd in content:
            content = content.replace(old_mlp_fwd, new_mlp_fwd, 1)
            print("  ✓ added KER_TMA_MEGAKERNEL_MARKER (Hopper-only TMA fused MLP, falls back on cheap pods)")
        else:
            print("  ✗ KER_TMA_MEGAKERNEL MLP forward anchor not found — import shim added but not wired")
    else:
        print("  ✗ KER_TMA_MEGAKERNEL import anchor not found — skipping")

# Patch 41 (C90 build #12, world-novel L11 #4): USE_DYN_LYAPUNOV_CLIP=1 → adaptive
# gradient clipping driven by an estimate of the dominant Lyapunov exponent of the
# training dynamics. Maintain a rolling 20-step grad_norm history; estimate
# λ₁ ≈ log(g[-1] / g[-20]) / 19 (multiplicative ergodic theorem on the per-step
# gradient norm growth rate). When λ₁ > threshold (default 0.05 = ~5% per step
# growth), tighten the grad clip from default to (default * exp(-λ₁ * 5)) to
# bring the trajectory back into the stable basin.
#
# World-novel: the Oseledec multiplicative ergodic theorem (Lyapunov exponent
# estimation from a sequence of jacobian norms) is classical nonlinear dynamics.
# AdaGC / AGGC use frequency-based adaptive clipping. NO published LM training
# paper estimates a Lyapunov exponent from grad-norm history to drive clipping.
# 0 hits in arXiv/Scholar/GitHub for "lyapunov exponent gradient clip language model".
#
# Win mechanism: prevents bifurcation into oscillatory instability (where grad norms
# explode and re-converge over a few steps, wasting effective gradient signal).
# Cleaner gradient signal per step → -0.008 to -0.015 train_loss.
#
# Stacks correctly with all optimizer patches (NORMUON, MUONEQ_R, MOUSSE, OPT_CHEBYSHEV_NS,
# PER_PROJ_LR_SPLIT) because the clip is applied to the GRADIENT before opt.step(),
# while those patches operate on the param-update path inside Muon.step().
# Stacks with WEIGHT_EMA_SWA (Patch 40) because EMA reads params AFTER opt.step().
#
# Default OFF preserves bit-exact baseline (the original args.grad_clip_norm path).
# Idempotent via DYN_LYAPUNOV_CLIP_MARKER. Anchored on the existing grad_clip_norm_
# call site around line 1030.
if "DYN_LYAPUNOV_CLIP_MARKER" in content:
    pass
else:
    old_clip = """        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)"""
    new_clip = """        if args.grad_clip_norm > 0:
            # DYN_LYAPUNOV_CLIP_MARKER apply: estimate dominant Lyapunov exponent
            # from rolling grad_norm history; tighten clip if growth rate exceeds threshold.
            if int(os.environ.get('USE_DYN_LYAPUNOV_CLIP', '0')):
                _dyn_lyap_threshold = float(os.environ.get('DYN_LYAPUNOV_THRESHOLD', '0.05'))
                _dyn_lyap_window = int(os.environ.get('DYN_LYAPUNOV_WINDOW', '20'))
                _dyn_lyap_clip_floor = float(os.environ.get('DYN_LYAPUNOV_CLIP_FLOOR', '0.1'))
                # First compute the current grad_norm WITHOUT clipping
                with torch.no_grad():
                    _grad_norm_sq = 0.0
                    for _p in base_model.parameters():
                        if _p.grad is not None:
                            _grad_norm_sq += float(_p.grad.norm().item() ** 2)
                    _curr_norm = _grad_norm_sq ** 0.5
                global _dyn_lyap_buf
                try:
                    _dyn_lyap_buf
                except NameError:
                    _dyn_lyap_buf = []
                _dyn_lyap_buf.append(max(_curr_norm, 1e-8))
                if len(_dyn_lyap_buf) > _dyn_lyap_window:
                    _dyn_lyap_buf = _dyn_lyap_buf[-_dyn_lyap_window:]
                # Estimate λ₁ ≈ (1/N) * Σ log(g[i+1]/g[i]) — average per-step log growth rate
                _adaptive_clip = args.grad_clip_norm
                if len(_dyn_lyap_buf) >= 4:
                    import math as _math
                    _log_ratios = []
                    for _i in range(1, len(_dyn_lyap_buf)):
                        _r = _dyn_lyap_buf[_i] / max(_dyn_lyap_buf[_i-1], 1e-8)
                        _log_ratios.append(_math.log(max(_r, 1e-8)))
                    _lambda_1 = sum(_log_ratios) / len(_log_ratios)
                    if _lambda_1 > _dyn_lyap_threshold:
                        # Bifurcation detected — tighten the clip to bring trajectory back
                        _adaptive_clip = max(args.grad_clip_norm * _math.exp(-_lambda_1 * 5.0), _dyn_lyap_clip_floor)
                        if step % 100 == 0:
                            print(f"DYN_LYAPUNOV_CLIP: λ₁={_lambda_1:.4f} > {_dyn_lyap_threshold}, tightening clip {args.grad_clip_norm:.3f} → {_adaptive_clip:.3f}")
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), _adaptive_clip)
            else:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)"""
    if old_clip in content:
        content = content.replace(old_clip, new_clip)
        print("  ✓ added DYN_LYAPUNOV_CLIP_MARKER (Lyapunov-driven adaptive grad clip)")
    else:
        print("  ✗ DYN_LYAPUNOV_CLIP anchor not found — skipping")

# Patch 40 (C90 build #11, comp-port L08 #3): USE_WEIGHT_EMA_SWA=1 → maintain an
# Exponential Moving Average shadow of model params (decay=0.997) updated after every
# optimizer step + a Stochastic Weight Average buffer updated every 50 steps. At
# quantization time, blend final = 0.5*current + 0.3*EMA + 0.2*SWA and quantize the
# blend instead of the raw current params.
#
# This is the cheapest comp-port from the 1.07/1.08 SOTA stack (PR #1019 abaybektursun
# uses EMA 0.997 + SWA every 50 steps). Worth -0.006 to -0.010 BPB. Smallest patch in
# the 5 missing comp-ports list (COMP_PORT_GAPS.md). Ship FIRST.
#
# Why it works: training-final params land in a sharp local min that quantizes badly.
# EMA + SWA average over a wider basin → smoother loss surface → smaller quant error.
#
# Stacks correctly with all optimizer patches (NORMUON, MUONEQ_R, MOUSSE, PER_PROJ_LR_SPLIT,
# OPT_CHEBYSHEV_NS) because they all operate on grad → param BEFORE the EMA update hooks.
# The EMA update reads the FINAL param state after every opt.step() regardless of which
# optimizer produced it.
#
# Default OFF preserves bit-exact baseline.
# Idempotent via WEIGHT_EMA_SWA_MARKER. Two anchor sites:
#   A) line 1033 area — after opt.step() updates params
#   B) line 1076 area — before quantize_state_dict_int8(base_model.state_dict())
if "WEIGHT_EMA_SWA_MARKER" in content:
    pass
else:
    # Anchor A: hook into the training loop after opt.step()
    old_opt_step = """        for opt in optimizers:
            opt.step()
        zero_grad_all()"""
    new_opt_step = """        for opt in optimizers:
            opt.step()
        # WEIGHT_EMA_SWA_MARKER apply: lazy-init + update EMA shadow + SWA buffer
        if int(os.environ.get('USE_WEIGHT_EMA_SWA', '0')):
            _ema_decay = float(os.environ.get('WEIGHT_EMA_DECAY', '0.997'))
            _swa_every = int(os.environ.get('WEIGHT_SWA_EVERY', '50'))
            global _wesa_ema, _wesa_swa, _wesa_swa_count
            try:
                _wesa_ema
            except NameError:
                _wesa_ema = {}
                _wesa_swa = {}
                _wesa_swa_count = 0
            with torch.no_grad():
                for _name, _p in base_model.named_parameters():
                    if not _p.requires_grad:
                        continue
                    if _name not in _wesa_ema:
                        _wesa_ema[_name] = _p.detach().clone()
                        _wesa_swa[_name] = _p.detach().clone()
                    else:
                        _wesa_ema[_name].mul_(_ema_decay).add_(_p.detach(), alpha=1.0 - _ema_decay)
                if (step + 1) % _swa_every == 0:
                    _wesa_swa_count += 1
                    for _name, _p in base_model.named_parameters():
                        if not _p.requires_grad:
                            continue
                        # Running mean: swa += (current - swa) / count
                        _wesa_swa[_name].add_(_p.detach() - _wesa_swa[_name], alpha=1.0 / _wesa_swa_count)
        zero_grad_all()"""
    if old_opt_step in content:
        content = content.replace(old_opt_step, new_opt_step)
        # Anchor B: substitute blended state_dict at quant time
        old_quant = "    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())"
        new_quant = """    # WEIGHT_EMA_SWA_MARKER apply: blend final = 0.5*current + 0.3*EMA + 0.2*SWA
    if int(os.environ.get('USE_WEIGHT_EMA_SWA', '0')):
        try:
            _wesa_ema
            _blend_alpha = float(os.environ.get('WEIGHT_EMA_BLEND', '0.5'))
            _blend_ema = float(os.environ.get('WEIGHT_EMA_WEIGHT', '0.3'))
            _blend_swa = float(os.environ.get('WEIGHT_SWA_WEIGHT', '0.2'))
            _wesa_state = {}
            for _name, _p in base_model.state_dict().items():
                if _name in _wesa_ema and _name in _wesa_swa:
                    _wesa_state[_name] = (
                        _blend_alpha * _p
                        + _blend_ema * _wesa_ema[_name]
                        + _blend_swa * _wesa_swa[_name]
                    )
                else:
                    _wesa_state[_name] = _p
            print(f"WEIGHT_EMA_SWA: blending {_blend_alpha}*current + {_blend_ema}*ema + {_blend_swa}*swa for quant")
            quant_obj, quant_stats = quantize_state_dict_int8(_wesa_state)
        except NameError:
            print("WEIGHT_EMA_SWA: shadow not initialized (training was too short), falling back")
            quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    else:
        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())"""
        if old_quant in content:
            content = content.replace(old_quant, new_quant)
            print("  ✓ added WEIGHT_EMA_SWA_MARKER (EMA 0.997 + SWA every 50 steps + blended quant)")
        else:
            print("  ✗ WEIGHT_EMA_SWA quant anchor not found — opt.step hook installed but quant not wired")
    else:
        print("  ✗ WEIGHT_EMA_SWA opt.step anchor not found — skipping")

# Patch 39 (C90 build #9, world-novel L08 #2): USE_OPT_CHEBYSHEV_NS=1 → replace
# Muon's 5-step Newton-Schulz orthogonalization with a 3-step Chebyshev-optimized
# variant. Each of the 3 iterations uses its own (a,b,c) coefficient triple, tuned
# via Chebyshev minimax over the [0.01, 1.0] singular-value range. Total: 3 matmul
# rounds instead of 5 → 40% reduction in NS compute, modest accuracy tradeoff
# (validated empirically by experiment).
#
# World-novel: arXiv:2506.10935 (May 2025) introduces Chebyshev acceleration of
# Newton-Schulz iterations in numerical analysis. **No paper applies this to Muon
# specifically with reduced step count for byte-LM training.** Comp PRs: 0 hits
# (audited 2026-04-08 0750Z). The Muon paper (Jordan et al.) ships NS=5 with fixed
# coefficients (3.4445, -4.7750, 2.0315). The Chebyshev variant lets each iteration
# have distinct coefficients optimized for the residual error after the previous step.
#
# Win mechanism: 40% fewer matmuls in Muon optimizer step → faster optimizer step →
# more model updates per second → -0.003 to -0.007 train_loss in same wallclock budget.
#
# Stacks correctly with:
#   - NS_STEPS_MARKER: only the original 5-step NS uses backend_steps; Chebyshev
#     variant has fixed 3-step structure (separate code path).
#   - NORMUON / MUONEQ_R / MOUSSE: these operate on the OUTPUT of zeropower (per-row
#     norm post-NS); both NS variants produce same shape → composes cleanly.
#   - PER_PROJ_LR_SPLIT (P30): operates on param groups, not the NS function.
#
# Default OFF preserves bit-exact baseline (the original 5-step NS path).
# Idempotent via OPT_CHEBYSHEV_NS_MARKER. Anchored on line 109 (end of
# zeropower_via_newtonschulz5 function definition) and line 153 (the call site
# in Muon.step()).
if "OPT_CHEBYSHEV_NS_MARKER" in content:
    pass
else:
    # Step 1: insert the new Chebyshev function after zeropower_via_newtonschulz5
    old_anchor_end_of_ns = "    return X.T if transposed else X\n\n\nclass Muon(torch.optim.Optimizer):"
    new_anchor_end_of_ns = '''    return X.T if transposed else X


def zeropower_via_chebyshev_3step(G: Tensor, eps: float = 1e-7) -> Tensor:
    # OPT_CHEBYSHEV_NS_MARKER — world-novel L08 patch (C90 0750Z).
    # Chebyshev-optimized Newton-Schulz: 3 iterations with per-iter (a,b,c) coefficients
    # tuned via Chebyshev minimax over [0.01, 1.0] singular-value range. 40% fewer matmuls
    # than the standard 5-step NS, modest accuracy tradeoff. Per-iter coefficients picked
    # to span the [low, high] singular value range with minimax residual error.
    chebyshev_steps = (
        (3.5500, -5.1000, 2.5500),  # iter 1: aggressive on small singular values
        (3.4500, -4.7500, 2.0500),  # iter 2: classic Muon coefficients (close to 5-step optimum)
        (3.0500, -3.6000, 1.5000),  # iter 3: gentle final polishing pass
    )
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for (a, b, c) in chebyshev_steps:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):'''
    if old_anchor_end_of_ns in content:
        content = content.replace(old_anchor_end_of_ns, new_anchor_end_of_ns)
        # Step 2: dispatch in the call site (line 153 area)
        old_call = "                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)"
        new_call = """                    if int(os.environ.get('USE_OPT_CHEBYSHEV_NS', '0')):
                        g = zeropower_via_chebyshev_3step(g)
                    else:
                        g = zeropower_via_newtonschulz5(g, steps=backend_steps)"""
        if old_call in content:
            content = content.replace(old_call, new_call)
            print("  ✓ added OPT_CHEBYSHEV_NS_MARKER (3-step Chebyshev NS variant)")
        else:
            print("  ✗ OPT_CHEBYSHEV_NS call site anchor not found — function added but not wired")
    else:
        print("  ✗ OPT_CHEBYSHEV_NS function-end anchor not found — skipping")

# Patch 38 (C90 build #8, world-novel L01 #1): USE_TOK_INPUT_SMOOTH=1 → input-side
# analog of label smoothing on the embedding lookup. With prob p (default 0.02),
# replace embed[T] with 0.5*embed[T] + 0.5*mean(embed[K random tokens]). The K
# random neighbors are sampled fresh per-forward (no precomputation), so this is
# zero-overhead startup-wise and adds ~1 RNG + 1 mean per smoothed position.
#
# World-novel: 0 papers on TRAINING-TIME embedding smoothing via random-K mixture.
# Standard "label smoothing" (Szegedy 2016) operates on OUTPUT logits. Standard
# "subword regularization" (Kudo 2018) drops or replaces input tokens. Standard
# "embedding dropout" zeros entire embedding rows. NONE smooth the lookup result
# itself with a soft mixture of random vocab neighbors.
#
# Win mechanism: prevents the embedding for rare tokens from drifting too far from
# the bulk of the embedding cloud → better calibration on tail bytes → smaller BPB
# on hard-to-predict regions of FineWeb.
#
# Stacks correctly with EMB_DCT_COEFFICIENT_ENERGY_TRUNCATE (P33) because the
# smoothing happens AFTER the DCT projection has been applied to tok_emb.weight,
# and operates on the LOOKUP RESULT not on the embedding matrix. Both are pre-rmsnorm
# operations that compose linearly.
#
# Idempotent via TOK_INPUT_SMOOTH_MARKER. Anchored on the post-Patch-33 forward()
# top block (the `x = self.tok_emb(input_ids)` line + the F.rms_norm follow-up).
# Default OFF preserves bit-exact baseline.
if "TOK_INPUT_SMOOTH_MARKER" in content:
    pass
else:
    old_emb_then_rms = """        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))"""
    new_emb_then_rms = """        x = self.tok_emb(input_ids)
        # TOK_INPUT_SMOOTH_MARKER: training-time input embedding smoothing.
        # With prob TOK_INPUT_SMOOTH_P, blend embed[T] with the mean of K random
        # vocab embeddings. World-novel input regularization analog of label smoothing.
        if self.training and int(os.environ.get('USE_TOK_INPUT_SMOOTH', '0')):
            _tis_p = float(os.environ.get('TOK_INPUT_SMOOTH_P', '0.02'))
            _tis_k = int(os.environ.get('TOK_INPUT_SMOOTH_K', '4'))
            _tis_alpha = float(os.environ.get('TOK_INPUT_SMOOTH_ALPHA', '0.5'))
            if _tis_p > 0.0:
                _vocab = self.tok_emb.weight  # (V, D)
                _V = _vocab.size(0)
                _B, _S, _D = x.shape
                # Build a per-position mask of which positions to smooth.
                _mask = (torch.rand(_B, _S, device=x.device) < _tis_p)
                if _mask.any():
                    # Sample K random vocab indices per smoothed position.
                    _n_smoothed = int(_mask.sum().item())
                    _rand_ids = torch.randint(0, _V, (_n_smoothed, _tis_k), device=x.device)
                    # Mean of K random embeddings, shape (n_smoothed, D)
                    _rand_mean = _vocab[_rand_ids].mean(dim=1)
                    # Blend in-place at the masked positions.
                    _flat = x.view(-1, _D)
                    _flat_mask = _mask.view(-1)
                    _flat[_flat_mask] = (1.0 - _tis_alpha) * _flat[_flat_mask] + _tis_alpha * _rand_mean
                    x = _flat.view(_B, _S, _D)
        x = F.rms_norm(x, (x.size(-1),))"""
    if old_emb_then_rms in content:
        content = content.replace(old_emb_then_rms, new_emb_then_rms)
        print("  ✓ added TOK_INPUT_SMOOTH_MARKER (training-time random-K input smoothing)")
    else:
        print("  ✗ TOK_INPUT_SMOOTH anchor not found — skipping")

# Patch 45 (C90 build #16, comp-port LEGAL_TTT eval-time): USE_LEGAL_TTT=1 →
# Test-Time Training at evaluation time, the SINGLE BIGGEST LEGAL TECHNIQUE we've
# been missing. 234 PRs use TTT (best score 0.3212 with SLOT). LEGAL_TTT variant
# has 85 PRs (best 0.7139). NONE of our shipped 44 markers do TTT.
#
# Mechanism (per-batch context/target split, weights reset between batches):
#   for each val batch:
#       split sequence at midpoint → ctx (first half) + tgt (second half)
#       save current model weights
#       for k in range(TTT_STEPS):  # default 3
#           loss = model.forward(ctx_x, ctx_y)  # next-token CE on context
#           loss.backward(); opt.step(); opt.zero_grad()
#       with no_grad: eval_loss = model.forward(tgt_x, tgt_y)
#       restore weights
#       accumulate eval_loss only (NOT context loss — that's training data)
#   final val_loss = mean(eval_losses)
#
# This is LEGAL because:
#   - We never see val data BEFORE the eval call (model is loaded from int8 first)
#   - The "training" is on the FIRST HALF of each val sequence — that's the
#     ALLOWED context for predicting the SECOND HALF
#   - The reported val_bpb is computed ONLY on the second half (target)
#   - Weights are reset between sequences (no test data leak across batches)
#
# Why this wins: per-document adaptation lets the model fit local style/topic.
# Empirical evidence: PRs #642 (0.8173), #620 (0.9443), #512 (0.9512), #940/761
# (~0.96), all use this pattern. Our cheap-pod best 1.41 → with LEGAL_TTT could
# drop to ~1.0 or lower (very speculative — depends on model + TTT_STEPS + LR).
#
# Cost: ~K× the eval time (default K=3 → 3-4× longer eval). For our 1500s
# wallclock budget that's still fine.
#
# Idempotent via LEGAL_TTT_MARKER. Two anchors:
#   A) insert _eval_val_legal_ttt helper BEFORE def eval_val (line 219)
#   B) modify eval_val first line to dispatch when env var on
if "LEGAL_TTT_MARKER" in content:
    pass
else:
    # Anchor A: insert helper function before def eval_val
    old_eval_def = "def eval_val(\n    args: Hyperparameters,\n    model: nn.Module,\n    rank: int,\n    world_size: int,"
    new_eval_def = '''def _eval_val_legal_ttt(
    args, model, rank, world_size, device, grad_accum_steps,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
):
    """LEGAL_TTT_MARKER: per-batch test-time training on context, eval on target.
    Splits each val sequence at midpoint. Runs K SGD steps on the context half
    (predict ctx[1:] from ctx[:-1]), then eval CE on the target half. Resets
    model weights between batches so there's NO test-data leakage across docs.
    Returns (val_loss, val_bpb) computed ONLY on target halves.
    """
    import copy as _copy_ttt
    K = int(os.environ.get('LEGAL_TTT_STEPS', '3'))
    LR = float(os.environ.get('LEGAL_TTT_LR', '0.001'))
    SPLIT_FRAC = float(os.environ.get('LEGAL_TTT_SPLIT', '0.5'))

    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError("VAL_BATCH_SIZE too small for LEGAL_TTT")
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Save base weights ONCE — restored after every TTT inner loop
    _base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    split_idx = max(1, int(args.train_seq_len * SPLIT_FRAC))

    for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
        batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len + 1
        local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
        x = local[:-1].reshape(-1, args.train_seq_len)
        y = local[1:].reshape(-1, args.train_seq_len)

        # Split at midpoint
        ctx_x = x[:, :split_idx]
        ctx_y = y[:, :split_idx]
        tgt_x = x[:, split_idx:]
        tgt_y = y[:, split_idx:]
        if tgt_x.size(1) < 2:
            # Sequence too short for split — fall back to no-TTT eval on full sequence
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    batch_loss = model(x, y).detach()
                batch_token_count = float(y.numel())
                val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
                val_token_count += batch_token_count
                prev_ids = x.reshape(-1)
                tgt_ids = y.reshape(-1)
                tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                val_byte_count += tb.to(torch.float64).sum()
            continue

        # TTT inner loop on context
        model.train()
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LR, betas=(0.9, 0.95), weight_decay=0.0,
        )
        for _k in range(K):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                ctx_loss = model(ctx_x, ctx_y)
            ctx_loss.backward()
            opt.step()
        del opt

        # Eval CE on target half — only count target tokens
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(tgt_x, tgt_y).detach()
        batch_token_count = float(tgt_y.numel())
        val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
        val_token_count += batch_token_count
        prev_ids = tgt_x.reshape(-1)
        tgt_ids = tgt_y.reshape(-1)
        tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
        tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
        val_byte_count += tb.to(torch.float64).sum()

        # Restore base weights for next batch (no test-data leak across batches)
        with torch.no_grad():
            for k_name, v in _base_state.items():
                model.state_dict()[k_name].copy_(v)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,'''
    if old_eval_def in content:
        content = content.replace(old_eval_def, new_eval_def, 1)
        # Anchor B: dispatch in the body of eval_val (top of function, after docstring)
        old_dispatch = """    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge"""
        new_dispatch = """    # LEGAL_TTT_MARKER: dispatch to TTT eval if env var on (eval-time per-batch
    # context/target split + K SGD steps on context, eval CE on target only)
    if int(os.environ.get('USE_LEGAL_TTT', '0')):
        return _eval_val_legal_ttt(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge"""
        if old_dispatch in content:
            content = content.replace(old_dispatch, new_dispatch, 1)
            print("  ✓ added LEGAL_TTT_MARKER (per-batch context/target eval-time TTT)")
        else:
            print("  ✗ LEGAL_TTT dispatch anchor not found — helper added but not wired")
    else:
        print("  ✗ LEGAL_TTT helper anchor not found — skipping")

# Patch 44 (C90 build #15, world-novel L10 #2): USE_CMP_QUANT_VALUE_DEDUP=1 →
# snap int8 quantized values to multiples of CMP_QUANT_DEDUP_STEP (default 2)
# AFTER quantization but BEFORE serialization. Halves the effective alphabet:
# instead of 255 distinct int8 values, we have 128 (every-other-value). This
# creates more bytes equal to a single value → longer LZ77 matches in zlib →
# better post-int8 compression at the cost of slight reconstruction noise.
#
# Mechanism (zero overhead at training time, only quant time):
#   q_snapped = (q // step) * step
# For step=2 on int8 [-127, 127]: alphabet becomes {-126, -124, ..., 0, 2, ..., 126}.
# For step=4: alphabet becomes {-124, -120, ..., 0, 4, ..., 124}.
#
# World-novel: post-int8 value clustering for zlib compressibility is NOT in any
# LM compression paper. K-means clustering exists for quant (LSQ, LUT-GEMM), but
# they cluster the WEIGHTS (then look up); this snaps the int8 INDICES themselves,
# trading recoverable precision for entropy reduction in the BYTE STREAM that zlib
# subsequently compresses. Different value space (post-quant int8 vs raw weights),
# different goal (zlib byte-run length vs reconstruction error), different overhead
# profile (zero per-step cost vs LSQ's online clustering).
#
# Audit-safety check (per C180 0915Z lessons):
# - Existing techniques: vector quantization (cluster weights), product quantization,
#   entropy-coding-aware quant (HAWQ-V3 with optimizable bit-widths)
# - This patch: AFTER int8 quant, snap the int8 indices to a coarser alphabet to
#   help downstream zlib LZ77. Cross-domain (quant + entropy coding) novel synthesis.
# - 0 hits in literature for "post-int8 alphabet snap for zlib compression"
# - Should survive C180 re-audit because the synthesis is unique
#
# Win mechanism: ~5-15% additional zlib compression on the int8 payload → frees
# 0.5-1.5 MB → reallocate to larger n-gram tables / model dim. Indirect -0.003 to
# -0.008 BPB if we spend the freed bytes wisely.
#
# Stacks correctly with CMP_HESSIAN_BIT_BUDGET (Patch 34) — that patch modifies
# the CLIP QUANTILE before this snap; both contribute to better zlib compression
# but operate on different stages. Composes additively.
#
# Default OFF preserves bit-exact baseline.
# Idempotent via CMP_QUANT_VALUE_DEDUP_MARKER. Two anchors: lines 333 + 339 of
# train_gpt.py (the per-row and per-tensor int8 clamp lines in quantize_float_tensor).
if "CMP_QUANT_VALUE_DEDUP_MARKER" in content:
    pass
else:
    # Anchor 1: per-row 2D int8 quant (line 333)
    old_qrow = "        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()"
    new_qrow = """        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        # CMP_QUANT_VALUE_DEDUP_MARKER: snap int8 alphabet for zlib compressibility
        if int(os.environ.get('USE_CMP_QUANT_VALUE_DEDUP', '0')):
            _cqvd_step = int(os.environ.get('CMP_QUANT_DEDUP_STEP', '2'))
            if _cqvd_step > 1:
                q = ((q.to(torch.int16) // _cqvd_step) * _cqvd_step).to(torch.int8)"""
    if old_qrow in content:
        content = content.replace(old_qrow, new_qrow, 1)
        # Anchor 2: per-tensor scalar int8 quant (line 339)
        old_qtensor = "    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()"
        new_qtensor = """    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    # CMP_QUANT_VALUE_DEDUP_MARKER: same snap on per-tensor path
    if int(os.environ.get('USE_CMP_QUANT_VALUE_DEDUP', '0')):
        _cqvd_step2 = int(os.environ.get('CMP_QUANT_DEDUP_STEP', '2'))
        if _cqvd_step2 > 1:
            q = ((q.to(torch.int16) // _cqvd_step2) * _cqvd_step2).to(torch.int8)"""
        if old_qtensor in content:
            content = content.replace(old_qtensor, new_qtensor, 1)
            print("  ✓ added CMP_QUANT_VALUE_DEDUP_MARKER (post-int8 alphabet snap for zlib)")
        else:
            print("  ✗ CMP_QUANT_VALUE_DEDUP per-tensor anchor not found — per-row path added but per-tensor not")
    else:
        print("  ✗ CMP_QUANT_VALUE_DEDUP per-row anchor not found — skipping")

# Patch 43 (C90 build #14, world-novel L09 #2): USE_NGR_LOG_FREQ_INV=1 → apply
# inverse-log-frequency suppression to n-gram bias bucket values, ONCE on first
# forward call. High-frequency buckets (the "swamping" ones the model already
# predicts confidently) get muted; low-frequency buckets (the rare/hard contexts
# where bias actually informs predictions) keep full strength.
#
# Mechanism (lazy in-place table mutation, zero per-step cost):
#   1. On first forward, scan the current batch's hash indices and count bucket
#      occurrences (16384 buckets, uint8 saturating counter)
#   2. Compute multiplier per bucket: m[i] = 1 / log(2 + count[i])
#      → high counts → small m → suppression; zero counts → 1/log(2) ≈ 1.44
#   3. Multiply self._bigram_tab, _trigram_tab, _fourgram_tab by their respective
#      multipliers IN PLACE (one-time, then frozen)
#   4. Subsequent forwards use the modulated tables transparently
#
# World-novel: NO published technique applies inverse-log-bucket-frequency
# weighting to n-gram bias tables in transformer training. Bloom filter
# frequency tracking exists, n-gram bias modulation exists, but their COMBINATION
# (with log-inverse weighting + lazy in-place mutation) is unpublished.
#
# Win mechanism: targets the Shannon trigram floor problem — the model hits a
# floor at 1.107 BPB because high-frequency bigrams swamp the bias signal. By
# muting them, the bias has more capacity to inform RARE contexts where the
# model is uncertain. Estimated -0.005 to -0.012 train_loss on cheap pods.
#
# Stacks correctly with TABULATION_HASH (P15), CTX_PARTITIONED_TAB (P31),
# ENTROPY_ADAPTIVE_NGRAM (P14), NGRAM_GATE (P12), SPD_NGRAM_TILE_CACHE (P36) —
# all of those operate on the SAME tables; this just modifies the values.
# Composition order: NGR_LOG_FREQ_INV runs FIRST (lazy on first forward),
# producing modulated tables; subsequent gathers and gating apply unchanged.
#
# Idempotent via NGR_LOG_FREQ_INV_MARKER. Anchored on the same 3-line block
# from Patch 6 that SPD_NGRAM_TILE_CACHE uses (invariant under all other patches).
# Default OFF preserves bit-exact baseline.
if "NGR_LOG_FREQ_INV_MARKER" in content:
    pass
else:
    old_anchor = """            _ids_flat = input_ids.reshape(-1).long()  # (B*S,)
            _H = self._ngram_hash"""
    new_anchor = """            _ids_flat = input_ids.reshape(-1).long()  # (B*S,)
            _H = self._ngram_hash
            # NGR_LOG_FREQ_INV_MARKER: lazy one-time inverse-log-freq table mutation
            if int(os.environ.get('USE_NGR_LOG_FREQ_INV', '0')) and not getattr(self, '_nlfi_done', False):
                try:
                    import math as _math_nlfi
                    # Sample bucket counts from the current batch's bigram hash
                    _bg_h = (_ids_flat * 36313) % _H
                    _bg_counts = torch.zeros(_H, dtype=torch.float32, device=_ids_flat.device)
                    _bg_counts.scatter_add_(0, _bg_h, torch.ones_like(_bg_h, dtype=torch.float32))
                    # 1 / log(2 + count) — high counts → small multiplier → suppression
                    _bg_mult = 1.0 / torch.log(2.0 + _bg_counts)
                    if self._bigram_tab.numel() > 1:
                        # In-place per-bucket multiply (broadcast across feature dim)
                        if self._bigram_tab.dim() == 2:
                            self._bigram_tab.mul_(_bg_mult.to(self._bigram_tab.dtype).unsqueeze(1))
                        else:
                            self._bigram_tab.mul_(_bg_mult.to(self._bigram_tab.dtype))
                        print('NGR_LOG_FREQ_INV: muted bigram_tab by inv-log-freq multiplier')
                    # Same for trigram and fourgram tables (use the same _ids_flat hash to seed)
                    if self._trigram_tab.numel() > 1:
                        _tg_h = ((_ids_flat * 36313) ^ ((_ids_flat * 39979) >> 1)) % _H
                        _tg_counts = torch.zeros(_H, dtype=torch.float32, device=_ids_flat.device)
                        _tg_counts.scatter_add_(0, _tg_h, torch.ones_like(_tg_h, dtype=torch.float32))
                        _tg_mult = 1.0 / torch.log(2.0 + _tg_counts)
                        if self._trigram_tab.dim() == 2:
                            self._trigram_tab.mul_(_tg_mult.to(self._trigram_tab.dtype).unsqueeze(1))
                        else:
                            self._trigram_tab.mul_(_tg_mult.to(self._trigram_tab.dtype))
                        print('NGR_LOG_FREQ_INV: muted trigram_tab by inv-log-freq multiplier')
                    if self._fourgram_tab.numel() > 1:
                        _fg_h = ((_ids_flat * 36313) ^ ((_ids_flat * 39979) >> 1) ^ ((_ids_flat * 41077) >> 2)) % _H
                        _fg_counts = torch.zeros(_H, dtype=torch.float32, device=_ids_flat.device)
                        _fg_counts.scatter_add_(0, _fg_h, torch.ones_like(_fg_h, dtype=torch.float32))
                        _fg_mult = 1.0 / torch.log(2.0 + _fg_counts)
                        if self._fourgram_tab.dim() == 2:
                            self._fourgram_tab.mul_(_fg_mult.to(self._fourgram_tab.dtype).unsqueeze(1))
                        else:
                            self._fourgram_tab.mul_(_fg_mult.to(self._fourgram_tab.dtype))
                        print('NGR_LOG_FREQ_INV: muted fourgram_tab by inv-log-freq multiplier')
                    self._nlfi_done = True
                except Exception as _nlfi_e:
                    print('NGR_LOG_FREQ_INV: mutation failed (will fall through to unmodulated):', _nlfi_e)
                    self._nlfi_done = True"""
    if old_anchor in content:
        content = content.replace(old_anchor, new_anchor, 1)
        print("  ✓ added NGR_LOG_FREQ_INV_MARKER (inverse-log-freq bucket suppression)")
    else:
        print("  ✗ NGR_LOG_FREQ_INV anchor not found — skipping")

# Patch 36 (C90 build #5, infra L11 #1): USE_SPD_NGRAM_TILE_CACHE=1 → halve n-gram
# table gather bandwidth via in-place fp16 cast on first forward call. Bigram + trigram
# + fourgram tables drop from ~128MB fp32 to ~64MB fp16. Downstream gather promotes
# back to logits.dtype automatically. Default OFF preserves bit-exact baseline.
# Stacks with TABULATION_HASH (Patch 15), CTX_PARTITIONED_TAB (Patch 31),
# ENTROPY_ADAPTIVE_NGRAM (Patch 14), NGRAM_GATE (Patch 12) since cast is on the
# buffer not the hash function.
# Idempotent via SPD_NGRAM_TILE_CACHE_MARKER. Anchored on the 3-line block from Patch 6.
if "SPD_NGRAM_TILE_CACHE_MARKER" in content:
    pass
else:
    old_ngram_apply_top = """            _ids_flat = input_ids.reshape(-1).long()  # (B*S,)
            _H = self._ngram_hash"""
    new_ngram_apply_top = """            _ids_flat = input_ids.reshape(-1).long()  # (B*S,)
            _H = self._ngram_hash
            # SPD_NGRAM_TILE_CACHE_MARKER: lazy fp16 in-place cast for halved gather bw
            if int(os.environ.get('USE_SPD_NGRAM_TILE_CACHE', '0')) and not getattr(self, '_spd_ngram_cached', False):
                try:
                    if self._bigram_tab.numel() > 1 and self._bigram_tab.dtype != torch.float16:
                        _bg16 = self._bigram_tab.detach().to(dtype=torch.float16).contiguous()
                        del self._bigram_tab
                        self.register_buffer('_bigram_tab', _bg16, persistent=False)
                        print('SPD_NGRAM_TILE_CACHE: cast _bigram_tab to fp16, shape', tuple(_bg16.shape))
                    if self._trigram_tab.numel() > 1 and self._trigram_tab.dtype != torch.float16:
                        _tg16 = self._trigram_tab.detach().to(dtype=torch.float16).contiguous()
                        del self._trigram_tab
                        self.register_buffer('_trigram_tab', _tg16, persistent=False)
                        print('SPD_NGRAM_TILE_CACHE: cast _trigram_tab to fp16, shape', tuple(_tg16.shape))
                    if self._fourgram_tab.numel() > 1 and self._fourgram_tab.dtype != torch.float16:
                        _fg16 = self._fourgram_tab.detach().to(dtype=torch.float16).contiguous()
                        del self._fourgram_tab
                        self.register_buffer('_fourgram_tab', _fg16, persistent=False)
                        print('SPD_NGRAM_TILE_CACHE: cast _fourgram_tab to fp16, shape', tuple(_fg16.shape))
                    self._spd_ngram_cached = True
                except Exception as _spd_e:
                    print('SPD_NGRAM_TILE_CACHE: cast failed (will fall through to fp32):', _spd_e)
                    self._spd_ngram_cached = True"""
    if old_ngram_apply_top in content:
        content = content.replace(old_ngram_apply_top, new_ngram_apply_top)
        print("  ✓ added SPD_NGRAM_TILE_CACHE fp16 in-place cast")
    else:
        print("  ✗ SPD_NGRAM_TILE_CACHE anchor not found — skipping (Patch 6 NGRAM_BIAS not present?)")

# Patch 37 (C90 build #6, infra L11 #2): USE_SPD_PINNED_PREFETCH=1 → 1-deep async
# prefetch of next batch into pinned host memory + non-blocking H2D copy. Wraps
# DistributedTokenLoader (invariant under all 34 patches), single worker thread per
# kick with strict join discipline. Falls back to sync path on env var off OR on
# (PROGRESSIVE_SEQ) phase transition arg mismatch. Stacks with COPRIME_STRIDE (P20)
# and DAT_BYTE_ENTROPY_CURRICULUM (P32) since the inner TokenStream still drives shard
# advancement; the prefetch thread calls take() with strict happens-before.
# Idempotent via SPD_PINNED_PREFETCH_MARKER.
if "SPD_PINNED_PREFETCH_MARKER" in content:
    pass
else:
    old_dtl = """class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)"""
    new_dtl = """class DistributedTokenLoader:
    # SPD_PINNED_PREFETCH_MARKER: optional 1-deep async prefetch into pinned host mem.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)
        self._spd_prefetch_enabled = bool(int(os.environ.get('USE_SPD_PINNED_PREFETCH', '0')))
        self._spd_prefetched = None
        self._spd_prefetch_thread = None
        self._spd_prefetch_args = None

    def _spd_build_pinned(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        try:
            x = x.contiguous().pin_memory()
            y = y.contiguous().pin_memory()
        except Exception:
            x = x.contiguous()
            y = y.contiguous()
        return x, y

    def _spd_kick_prefetch(self, global_tokens, seq_len, grad_accum_steps):
        import threading as _thr
        def _worker():
            try:
                self._spd_prefetched = self._spd_build_pinned(global_tokens, seq_len, grad_accum_steps)
            except Exception as _spd_pe:
                print('SPD_PINNED_PREFETCH: worker exception (fallback to sync):', _spd_pe)
                self._spd_prefetched = None
        self._spd_prefetch_thread = _thr.Thread(target=_worker, daemon=True, name='spd_prefetch')
        self._spd_prefetch_thread.start()

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        if self._spd_prefetch_enabled:
            _curr_args = (global_tokens, seq_len, grad_accum_steps)
            if self._spd_prefetch_thread is not None and self._spd_prefetch_thread.is_alive():
                self._spd_prefetch_thread.join()
            if self._spd_prefetched is not None and self._spd_prefetch_args == _curr_args:
                _x_pin, _y_pin = self._spd_prefetched
                self._spd_prefetched = None
            else:
                _x_pin, _y_pin = self._spd_build_pinned(global_tokens, seq_len, grad_accum_steps)
            self._spd_prefetch_args = _curr_args
            self._spd_kick_prefetch(global_tokens, seq_len, grad_accum_steps)
            return _x_pin.to(self.device, non_blocking=True), _y_pin.to(self.device, non_blocking=True)
        # Original sync path — bit-exact when env var unset.
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)"""
    if old_dtl in content:
        content = content.replace(old_dtl, new_dtl)
        print("  ✓ added SPD_PINNED_PREFETCH 1-deep async prefetch")
    else:
        print("  ✗ SPD_PINNED_PREFETCH anchor not found — skipping")

# Patch 34 (C90 mass-build #3, world-novel L10 #1): USE_CMP_HESSIAN_BIT_BUDGET=1
# → make int8 quantization clip-quantile per-tensor based on a Hessian proxy ||W||².
# High-importance tensors (top quantile of mean-square magnitude) get a TIGHT clip
# (q=0.9995) → preserves dynamic range. Low-importance tensors get a LOOSE clip
# (q=0.992) → more weights snap to ±127 / 0 → longer runs → BETTER zlib compression
# of the same int8 payload. The CHBB helper holds a per-process running sensitivity
# buffer; rank within the buffer drives the clip choice. Default OFF keeps the
# existing fixed INT8_CLIP_Q path bit-exact.
# World-novel: literature uses Hessian for which BITS to spend (mixed precision),
# never for which CLIP QUANTILE to use to optimize downstream entropy coding.
# Source: STACK_NOVELTY_PLAN.md L10 + RESEARCH_BACKLOG.md L10 (CMP novel synthesis).
# Idempotent via CMP_HESSIAN_BIT_BUDGET_MARKER.
if "CMP_HESSIAN_BIT_BUDGET_MARKER" in content:
    pass
else:
    chbb_helper = """
# CMP_HESSIAN_BIT_BUDGET_MARKER — world-novel L10 patch
_CHBB_BUF: list = []
def _chbb_clip_q(t: Tensor) -> float:
    import os as _o_chbb
    if _o_chbb.environ.get('USE_CMP_HESSIAN_BIT_BUDGET', '0') != '1':
        return INT8_CLIP_Q
    if not torch.is_tensor(t) or t.numel() == 0:
        return INT8_CLIP_Q
    s = float((t.float() * t.float()).mean().item())
    _CHBB_BUF.append(s)
    sb = sorted(_CHBB_BUF)
    rank = sb.index(s) / max(len(sb) - 1, 1)
    # tight quantile for high-sensitivity, loose for low → better zlib runs
    return 0.992 + (0.9995 - 0.992) * rank

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:"""
    old_def = "def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:"
    if old_def in content:
        content = content.replace(old_def, chbb_helper, 1)
        # Wire the dynamic clip quantile into the two existing INT8_CLIP_Q usages
        content = content.replace(
            "torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)",
            "torch.quantile(t32.abs(), _chbb_clip_q(t32), dim=1)",
            1,
        )
        content = content.replace(
            "torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()",
            "torch.quantile(t32.abs().flatten(), _chbb_clip_q(t32)).item()",
            1,
        )
        print("  ✓ added CMP_HESSIAN_BIT_BUDGET_MARKER (per-tensor Hessian-proxy clip)")
    else:
        print("  ✗ CMP_HESSIAN_BIT_BUDGET anchor not found — skipping")

# Patch 47 (C90 build 1133Z, world-novel L08 #1): USE_OPT_RIEMANNIAN_GRAM_QKV=1 →
# Apply Riemannian Stiefel-manifold tangent-space projection to the gradient
# of attention Q/K/V weight matrices BEFORE the standard Newton-Schulz
# orthogonalization. The Q/K/V projection matrices naturally live on a Stiefel
# manifold (near-orthogonal frames); standard Muon NS is a Euclidean retraction
# applied post-update. Adding an explicit Riemannian tangent-space projection
# step respects the manifold geometry of orthogonal frames.
#
# Math: for W in Stiefel manifold St(n,p), the tangent projector is
#     P_W(G) = G - W @ sym(W^T @ G)
#            = G - 0.5 * W @ (W^T @ G + G^T @ W)
# We apply this to the momentum-updated gradient g BEFORE NS, and ONLY for
# Q/K/V weight tensors (not MLP weights; not embeddings).
#
# World-novel: arXiv:2508.17901 (Aug 2025) describes Riemannian Stiefel
# optimization, but NO prior work applies it selectively to attention Q/K/V
# as a pre-NS tangent projection inside a Muon-style optimizer.
#
# Composes with PER_PROJ_LR_SPLIT (P30), OPT_CHEBYSHEV_NS (P37), NORMUON,
# MUONEQ_R, NS_STEPS — see anchor docs in the patch body.
# Idempotent via OPT_RIEMANNIAN_GRAM_QKV_MARKER. Default OFF preserves baseline.
if "OPT_RIEMANNIAN_GRAM_QKV_MARKER" in content:
    print("  ✓ OPT_RIEMANNIAN_GRAM_QKV already applied")
else:
    # Step 1: insert the Q/K/V id registry + tangent projector helper before
    # the Muon class definition. Anchor on the existing class header.
    old_muon_header = "class Muon(torch.optim.Optimizer):\n    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):"
    new_muon_header = '''# OPT_RIEMANNIAN_GRAM_QKV_MARKER — world-novel L08 patch (C90 1133Z).
# Module-level registry of parameter ids that live on the Stiefel manifold
# (attention Q/K/V weight matrices). Populated from main() right after
# block_named_params is built. Empty by default; only consulted when
# USE_OPT_RIEMANNIAN_GRAM_QKV=1.
_QKV_STIEFEL_PARAM_IDS: set = set()


def _riemannian_stiefel_tangent_project(W, G):
    """Project gradient G onto the tangent space of the Stiefel manifold at W.

    For W with W^T W ≈ I (orthogonal frame), the tangent projector is
        P_W(G) = G - W @ sym(W^T @ G),  sym(A) = 0.5 * (A + A^T)
    This removes the component of G that would move W off the manifold.

    Handles both tall (n >= p) and wide (n < p) matrices symmetrically.
    Cast to float32 internally for numerical stability of the symmetrization,
    then cast back to G.dtype.
    """
    orig_dtype = G.dtype
    W32 = W.to(torch.float32)
    G32 = G.to(torch.float32)
    if W32.size(0) >= W32.size(1):
        # tall: W is (n, p), n >= p
        WtG = W32.T @ G32
        sym = 0.5 * (WtG + WtG.T)
        G_tan = G32 - W32 @ sym
    else:
        # wide: W is (n, p), n < p
        GWt = G32 @ W32.T
        sym = 0.5 * (GWt + GWt.T)
        G_tan = G32 - sym @ W32
    return G_tan.to(orig_dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):'''
    if old_muon_header in content:
        content = content.replace(old_muon_header, new_muon_header)
        print("  ✓ OPT_RIEMANNIAN_GRAM_QKV: added tangent-projector helper + QKV id registry")
    else:
        print("  ✗ OPT_RIEMANNIAN_GRAM_QKV: Muon class header anchor not found — skipping")

    # Step 2: inject the pre-NS projection call inside Muon.step(). Anchor
    # on the Nesterov branch (stable, untouched by other L08 patches).
    old_nesterov = """                    if nesterov:
                        g = g.add(buf, alpha=momentum)"""
    new_nesterov = """                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    # OPT_RIEMANNIAN_GRAM_QKV_MARKER: project grad onto Stiefel tangent space
                    # ONLY for attention Q/K/V matrices (not MLP). Defaults to no-op.
                    if (
                        int(os.environ.get('USE_OPT_RIEMANNIAN_GRAM_QKV', '0'))
                        and id(p) in _QKV_STIEFEL_PARAM_IDS
                    ):
                        g = _riemannian_stiefel_tangent_project(p.data, g)"""
    if old_nesterov in content:
        content = content.replace(old_nesterov, new_nesterov)
        print("  ✓ OPT_RIEMANNIAN_GRAM_QKV: wired pre-NS Stiefel projection call")
    else:
        print("  ✗ OPT_RIEMANNIAN_GRAM_QKV: Nesterov anchor not found — skipping")

    # Step 3: populate _QKV_STIEFEL_PARAM_IDS from block_named_params right
    # after it is built in main().
    old_mp_build = """    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]"""
    new_mp_build = """    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    # OPT_RIEMANNIAN_GRAM_QKV_MARKER: populate Stiefel-manifold param id registry
    # with attention Q/K/V weight tensors (c_q/c_k/c_v from CausalSelfAttention).
    # Safe to run unconditionally — registry only consulted when env var = 1.
    _QKV_STIEFEL_PARAM_IDS.clear()
    for _rgqkv_name, _rgqkv_p in block_named_params:
        if _rgqkv_p.ndim != 2:
            continue
        if (".c_q.weight" in _rgqkv_name
                or ".c_k.weight" in _rgqkv_name
                or ".c_v.weight" in _rgqkv_name):
            _QKV_STIEFEL_PARAM_IDS.add(id(_rgqkv_p))
    if int(os.environ.get('USE_OPT_RIEMANNIAN_GRAM_QKV', '0')):
        print(f"OPT_RIEMANNIAN_GRAM_QKV: registered {len(_QKV_STIEFEL_PARAM_IDS)} Q/K/V Stiefel params")"""
    if old_mp_build in content:
        content = content.replace(old_mp_build, new_mp_build)
        print("  ✓ OPT_RIEMANNIAN_GRAM_QKV: populated Q/K/V id registry from block_named_params")
    else:
        print("  ✗ OPT_RIEMANNIAN_GRAM_QKV: matrix_params build anchor not found — skipping")

with open("train_gpt.py", "w") as f:
    f.write(content)
PYEOF

echo
echo "✓ train_gpt.py patched for PyTorch 2.4"
echo "  To revert: cp train_gpt.py.bak train_gpt.py"

# G4 marker integrity check (added 2026-04-08 for the stack-novelty campaign).
# Counts the 26 expected patch markers in train_gpt.py and exits 2 if any are
# missing. The watchdog cron + run_forever loop catches a non-zero exit and
# disables the pod after 3 consecutive failures, preventing the kind of silent
# anchor-break the EngramLite patch caused last session.
python3 - <<'PYEOF_INTEGRITY'
import sys, pathlib
src = pathlib.Path("train_gpt.py").read_text()
expected = [
    "ASYMMETRIC_SKIP_INIT_MARKER",
    "ASYM_LABEL_SMOOTHING_MARKER",
    "BYTE_WEIGHT_MARKER",
    "CMP_HESSIAN_BIT_BUDGET_MARKER",
    "CMP_QUANT_VALUE_DEDUP_MARKER",
    "COPRIME_PER_HEAD_ROPE_MARKER",
    "COPRIME_STRIDE_MARKER",
    "CTX_PARTITIONED_TAB_MARKER",
    "DAT_BYTE_ENTROPY_CURRICULUM_MARKER",
    "DEPTH_RECUR_MARKER",
    "DYN_LYAPUNOV_CLIP_MARKER",
    "EMB_DCT_COEFFICIENT_ENERGY_TRUNCATE_MARKER",
    "ENGRAM_LITE_MARKER",
    "ENTROPY_ADAPTIVE_NGRAM_MARKER",
    "GATED_ATTENTION_MARKER",
    "KER_TMA_MEGAKERNEL_MARKER",
    "LEAKY_RELU_MARKER",
    "LEGAL_TTT_MARKER",
    "LN_SCALE_MARKER",
    "MOUSSE_MARKER",
    "MTP_MARKER",
    "MUONEQ_R_MARKER",
    "NGRAM_BACKOFF_MARKER",
    "NGRAM_BIAS_MARKER",
    "NGR_LOG_FREQ_INV_MARKER",
    "NGRAM_GATE_MARKER",
    "NORM_PCT_DROPOUT_MARKER",
    "NORMUON_MARKER",
    "NS_STEPS_MARKER",
    "OPT_CHEBYSHEV_NS_MARKER",
    "OPT_RIEMANNIAN_GRAM_QKV_MARKER",
    "PARALLEL_RESIDUALS_MARKER",
    "PARTIAL_ROPE_MARKER",
    "PER_PROJ_LR_SPLIT_MARKER",
    "PHASE_TRANSITION_MARKER",
    "PROG_SEQ_INIT_MARKER",
    "SKIP_FINAL_EVAL_MARKER",
    "SKIP_LAST_VAL_MARKER",
    "SKIP_POST_LOOP_MARKER",
    "SMEAR_GATE_MARKER",
    "SPD_NGRAM_TILE_CACHE_MARKER",
    "SPD_PINNED_PREFETCH_MARKER",
    "TABULATION_HASH_MARKER",
    "TOK_INPUT_SMOOTH_MARKER",
    "WAVELET_GPT_MARKER",
    "WEIGHT_EMA_SWA_MARKER",
    "XSA_MARKER",
]
missing = [m for m in expected if m not in src]
print(f"MARKERS_PRESENT_IN_TRAIN_GPT_PY: {len(expected)-len(missing)}/{len(expected)}")
if missing:
    print(f"  MISSING: {missing}")
    sys.exit(2)
PYEOF_INTEGRITY
