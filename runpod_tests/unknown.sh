#!/bin/bash
# unknown.sh — Run all unknown tests, output to logs/unknown.log
# Usage: ./unknown.sh [u01 u02 ... uNN]

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -d "$REPO_ROOT/.venv" ] && [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/unknown.log"
mkdir -p "$LOG_DIR"

{
    echo "================================================================================"
    echo "UNKNOWN RUN — $(date)"
    echo "Host:       $(hostname)"
    echo "Repo:       $REPO_ROOT"
    echo "Tests dir:  $SCRIPT_DIR"
    echo "GPU:        $(python3 -c 'import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print("none")' 2>/dev/null || echo 'unknown')"
    echo "VRAM:       $(python3 -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB") if torch.cuda.is_available() else print("none")' 2>/dev/null || echo 'unknown')"
    echo "================================================================================"
    echo
    echo "WARNING: Running unknown/ tests on a non-H100 GPU will give different"
    echo "absolute numbers than the eventual H100 submission. Use these results"
    echo "for RELATIVE comparisons (config A vs B), not absolute BPP estimates."
    echo
} > "$LOG_FILE"

# Build absolute test list — either from args or default order
TESTS=()
if [ $# -gt 0 ]; then
    for arg in "$@"; do
        for script in "$SCRIPT_DIR/unknown/${arg}"*.sh; do
            if [ -f "$script" ]; then
                TESTS+=("$script")
            fi
        done
    done
else
    # Default order: arch, speed, GLA chain, progressive, cache, full, 3-seed
    for pattern in \
        "$SCRIPT_DIR/unknown/u01_"*.sh \
        "$SCRIPT_DIR/unknown/u06_"*.sh \
        "$SCRIPT_DIR/unknown/u07_"*.sh \
        "$SCRIPT_DIR/unknown/u08_"*.sh \
        "$SCRIPT_DIR/unknown/u02_"*.sh \
        "$SCRIPT_DIR/unknown/u03_"*.sh \
        "$SCRIPT_DIR/unknown/u04_"*.sh \
        "$SCRIPT_DIR/unknown/u05_"*.sh; do
        for script in $pattern; do
            if [ -f "$script" ]; then
                TESTS+=("$script")
            fi
        done
    done
fi

PASS=0
FAIL=0
FAILED_TESTS=()
RESULTS=()

run_test() {
    local script=$1
    local name
    name=$(basename "$script" | sed 's/\.[^.]*$//')

    echo
    echo "================================================================================"
    echo ">>> START: $name"
    echo ">>> TIME: $(date '+%H:%M:%S')"
    echo "================================================================================"

    local start
    start=$(date +%s)
    local exit_code=0

    cd "$REPO_ROOT"
    bash "$script" 2>&1
    exit_code=$?

    local end
    end=$(date +%s)
    local duration=$((end - start))

    echo
    if [ $exit_code -eq 0 ]; then
        echo "<<< RESULT: $name PASS (${duration}s)"
        PASS=$((PASS + 1))
        RESULTS+=("$name PASS ${duration}s")
    else
        echo "<<< RESULT: $name FAIL (exit=$exit_code, ${duration}s)"
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$name")
        RESULTS+=("$name FAIL ${duration}s")
    fi
    echo "================================================================================"
}

{
    if [ ${#TESTS[@]} -eq 0 ]; then
        echo "✗ NO TESTS FOUND in $SCRIPT_DIR/unknown/"
        echo "  args: $*"
        echo "  ls $SCRIPT_DIR/unknown/:"
        ls "$SCRIPT_DIR/unknown/" 2>&1 || echo "  (directory does not exist)"
    fi

    for script in "${TESTS[@]}"; do
        run_test "$script"
    done

    # Summary
    echo
    echo "================================================================================"
    echo "UNKNOWN SUMMARY"
    echo "================================================================================"
    echo "TIME:    $(date)"
    echo "TESTS:   ${#TESTS[@]}"
    echo "PASSED:  $PASS"
    echo "FAILED:  $FAIL"
    echo
    if [ ${#RESULTS[@]} -gt 0 ]; then
        echo "RESULTS:"
        for r in "${RESULTS[@]}"; do
            echo "  $r"
        done
    fi

    # Extract any val_bpb scores found in logs
    echo
    echo "VAL_BPB SCORES (if any):"
    if [ -d "$REPO_ROOT/logs" ]; then
        find "$REPO_ROOT/logs" -name "*.log" -exec grep -H 'final_int8_zlib_roundtrip val_loss' {} \; 2>/dev/null | head -20 || echo "  (none yet)"
    else
        echo "  (no logs/ dir yet)"
    fi

    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo
        echo "FAILED TESTS:"
        for t in "${FAILED_TESTS[@]}"; do
            echo "  - $t"
        done
    fi
    echo "================================================================================"
} 2>&1 | tee -a "$LOG_FILE"

echo
echo "Log saved to: $LOG_FILE"
exit $FAIL
