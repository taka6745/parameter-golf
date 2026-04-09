#!/bin/bash
# phase2/run.sh — Phase 2 training run with torch.compile enabled.
#
# Difference from submission/run.sh: this wrapper enables torch.compile by
# flipping TORCH_COMPILE_DISABLE=0 and TORCHDYNAMO_DISABLE=0. The compile
# cache should already be warm at this point (phase2/warm_compile_cache.py
# was called by phase2/bootstrap.sh). If the cache is cold, this run pays
# the first-run compile cost (~3-5 min) and then runs with compile enabled
# for the remainder — still a net win because compile cache persists across
# runs on the same pod.
#
# Everything else (data paths, env var defaults, the 10 Phase 1 patches,
# the Phase 2 Tier A wins like prefetch loader + Inductor patch) inherits
# from submission/run.sh. We just flip the compile flag and exec.

set -eu
REPO_DIR="${REPO_DIR:-/workspace/paramgolf}"
cd "$REPO_DIR"

echo "============================================================"
echo "[phase2/run] Phase 2 training with torch.compile enabled"
echo "============================================================"
echo "  TORCH_COMPILE_DISABLE=0"
echo "  TORCHDYNAMO_DISABLE=0"
echo "  (all Phase 1 patches + Phase 2 Tier A overlays inherited from submission/run.sh)"

# Inductor cache check — if it's empty warn but don't block
CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torch/inductor}"
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    echo "[phase2/run] WARNING: inductor cache at $CACHE_DIR is empty."
    echo "[phase2/run] The first forward pass will pay the full compile cost"
    echo "[phase2/run] (~3-5 min on H100). To avoid this, run"
    echo "[phase2/run]   python3 phase2/warm_compile_cache.py"
    echo "[phase2/run] first, or use phase2/bootstrap.sh which chains both."
else
    CACHE_FILES=$(find "$CACHE_DIR" -type f 2>/dev/null | wc -l)
    CACHE_MIB=$(du -sm "$CACHE_DIR" 2>/dev/null | cut -f1)
    echo "[phase2/run] inductor cache: $CACHE_DIR ($CACHE_FILES files, ${CACHE_MIB} MiB)"
fi

# Flip the compile flag and delegate to submission/run.sh for everything else.
# exec replaces this shell so the environment variables set here apply to the
# child process tree cleanly.
export TORCH_COMPILE_DISABLE=0
export TORCHDYNAMO_DISABLE=0
exec bash submission/run.sh
