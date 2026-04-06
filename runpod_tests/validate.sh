#!/bin/bash
# validate.sh — Run all validate tests, output to logs/validate.log
# Usage: ./validate.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -d "$REPO_ROOT/.venv" ] && [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/validate.log"
mkdir -p "$LOG_DIR"

{
    echo "================================================================================"
    echo "VALIDATE RUN — $(date)"
    echo "Host:       $(hostname)"
    echo "Repo:       $REPO_ROOT"
    echo "Tests dir:  $SCRIPT_DIR"
    echo "GPU:        $(python3 -c 'import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print("none")' 2>/dev/null || echo 'unknown')"
    echo "================================================================================"
    echo
} > "$LOG_FILE"

# Build absolute test list
TESTS=()
for pattern in \
    "$SCRIPT_DIR/validate/v01_"*.sh \
    "$SCRIPT_DIR/validate/v02_"*.py \
    "$SCRIPT_DIR/validate/v03_"*.py \
    "$SCRIPT_DIR/validate/v04_"*.sh \
    "$SCRIPT_DIR/validate/v05_"*.py \
    "$SCRIPT_DIR/validate/v06_"*.py \
    "$SCRIPT_DIR/validate/v07_"*.py \
    "$SCRIPT_DIR/validate/v08_"*.py \
    "$SCRIPT_DIR/validate/v09_"*.py \
    "$SCRIPT_DIR/validate/v10_"*.sh; do
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

{
    if [ ${#TESTS[@]} -eq 0 ]; then
        echo "✗ NO TESTS FOUND in $SCRIPT_DIR/validate/"
        echo "  ls $SCRIPT_DIR/validate/:"
        ls "$SCRIPT_DIR/validate/" 2>&1 || echo "  (directory does not exist)"
    fi

    for script in "${TESTS[@]}"; do
        run_test "$script"
    done

    echo
    echo "================================================================================"
    echo "VALIDATE SUMMARY"
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
        echo "STATUS: NO TESTS RAN — check $SCRIPT_DIR/validate/ exists"
    elif [ $FAIL -eq 0 ]; then
        echo "STATUS: ALL VALIDATIONS PASSED — proceed to ./unknown.sh"
    else
        echo "STATUS: $FAIL VALIDATION(S) FAILED — debug before unknown/"
    fi
    echo "================================================================================"
} 2>&1 | tee -a "$LOG_FILE"

echo
echo "Log saved to: $LOG_FILE"
exit $FAIL
