#!/bin/bash
# setup.sh — Run all chore tests, output to logs/setup.log
# Usage: ./setup.sh

set -u

# Resolve paths once
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"      # .../runpod_tests
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"        # .../paramgolf

# Source venv if it exists (so child processes pick up the right Python)
if [ -d "$REPO_ROOT/.venv" ] && [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/setup.log"
mkdir -p "$LOG_DIR"

# Header
{
    echo "================================================================================"
    echo "SETUP RUN — $(date)"
    echo "Host:       $(hostname)"
    echo "Repo:       $REPO_ROOT"
    echo "Tests dir:  $SCRIPT_DIR"
    echo "GPU:        $(python3 -c 'import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print("none")' 2>/dev/null || echo 'unknown')"
    echo "================================================================================"
    echo
} > "$LOG_FILE"

# Build list of tests (absolute paths) — must do this BEFORE cd elsewhere
TESTS=()
for pattern in \
    "$SCRIPT_DIR/chore/00_"*.sh \
    "$SCRIPT_DIR/chore/01_"*.sh \
    "$SCRIPT_DIR/chore/02_"*.sh \
    "$SCRIPT_DIR/chore/03_"*.sh \
    "$SCRIPT_DIR/chore/04_"*.py \
    "$SCRIPT_DIR/chore/05_"*.py \
    "$SCRIPT_DIR/chore/06_"*.py \
    "$SCRIPT_DIR/chore/07_"*.sh; do
    for script in $pattern; do
        if [ -f "$script" ]; then
            TESTS+=("$script")
        fi
    done
done

PASS=0
FAIL=0
FAILED_TESTS=()

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

    # Each test runs from REPO_ROOT so its relative paths (data/, train_gpt.py) work
    cd "$REPO_ROOT"
    if [[ "$script" == *.py ]]; then
        python3 "$script" 2>&1
        exit_code=$?
    else
        bash "$script" 2>&1
        exit_code=$?
    fi

    local end
    end=$(date +%s)
    local duration=$((end - start))

    echo
    if [ $exit_code -eq 0 ]; then
        echo "<<< RESULT: $name PASS (${duration}s)"
        PASS=$((PASS + 1))
    else
        echo "<<< RESULT: $name FAIL (exit=$exit_code, ${duration}s)"
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$name")
    fi
    echo "================================================================================"
}

# Run all tests
{
    if [ ${#TESTS[@]} -eq 0 ]; then
        echo "✗ NO TESTS FOUND in $SCRIPT_DIR/chore/"
        echo "  ls $SCRIPT_DIR/chore/:"
        ls "$SCRIPT_DIR/chore/" 2>&1 || echo "  (directory does not exist)"
    fi

    for script in "${TESTS[@]}"; do
        run_test "$script"
    done

    # Summary
    echo
    echo "================================================================================"
    echo "SETUP SUMMARY"
    echo "================================================================================"
    echo "TIME:    $(date)"
    echo "TESTS:   ${#TESTS[@]}"
    echo "PASSED:  $PASS"
    echo "FAILED:  $FAIL"
    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo "FAILED TESTS:"
        for t in "${FAILED_TESTS[@]}"; do
            echo "  - $t"
        done
    fi
    echo
    if [ ${#TESTS[@]} -eq 0 ]; then
        echo "STATUS: NO TESTS RAN — check $SCRIPT_DIR/chore/ exists"
    elif [ $FAIL -eq 0 ]; then
        echo "STATUS: ALL CHORES COMPLETE — proceed to ./validate.sh"
    else
        echo "STATUS: $FAIL CHORE(S) FAILED — fix before proceeding"
    fi
    echo "================================================================================"
} 2>&1 | tee -a "$LOG_FILE"

echo
echo "Log saved to: $LOG_FILE"
exit $FAIL
