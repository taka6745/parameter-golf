#!/bin/bash
# export_logs.sh — Bundle and upload all logs from runpod_tests/logs/
# Usage: ./export_logs.sh
#
# What it uploads:
#   1. The whole logs/ directory as a single tar.gz (everything: setup.log,
#      validate.log, unknown.log, v01_smoke.log, u01/config_*.log, u02/run_*.log,
#      ..., u10/eval_*.log, results.json — recursively)
#   2. results.json individually (if it exists, since that's the most useful
#      single file for quick analysis)
#   3. Each top-level *.log file individually as a fallback
#
# Methods (tries in order, falls back if one fails):
#   1. transfer.sh (free, no auth, 14 days)
#   2. 0x0.st (free, no auth, 30 days)
#   3. file.io (free, no auth, expires after first download)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ] || [ -z "$(ls -A $LOG_DIR 2>/dev/null)" ]; then
    echo "✗ No logs found in $LOG_DIR"
    echo "  Run setup.sh / validate.sh / unknown.sh first"
    exit 1
fi

# Re-run results.sh to refresh results.json (idempotent)
if [ -f "results.sh" ]; then
    echo "Refreshing results.json..."
    ./results.sh > /dev/null 2>&1 || echo "  (results.sh failed, continuing)"
fi

# Bundle everything in logs/ into a tar.gz (recursive — includes subdirs)
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
HOSTNAME=$(hostname | cut -c1-8)
BUNDLE="paramgolf_logs_${HOSTNAME}_${TIMESTAMP}.tar.gz"

echo "================================================================================"
echo "EXPORT LOGS"
echo "================================================================================"
echo

echo "Bundle contents:"
find "$LOG_DIR" -type f | sort | head -30
N_FILES=$(find "$LOG_DIR" -type f | wc -l)
echo "  ... ($N_FILES total files)"
echo

echo "Creating bundle: $BUNDLE"
tar -czf "/tmp/$BUNDLE" -C "$LOG_DIR" .
SIZE=$(du -h "/tmp/$BUNDLE" | cut -f1)
echo "  Size: $SIZE"
echo

# Upload helper functions
upload_transfer_sh() {
    local file=$1
    local url
    url=$(curl --silent --max-time 60 --upload-file "$file" "https://transfer.sh/$(basename $file)" 2>/dev/null)
    if [ -n "$url" ] && [[ "$url" == https://* ]]; then
        echo "$url"
        return 0
    fi
    return 1
}

upload_0x0_st() {
    local file=$1
    local url
    url=$(curl --silent --max-time 60 -F "file=@$file" "https://0x0.st" 2>/dev/null)
    if [ -n "$url" ] && [[ "$url" == http* ]]; then
        echo "$url"
        return 0
    fi
    return 1
}

upload_file_io() {
    local file=$1
    local response
    response=$(curl --silent --max-time 60 -F "file=@$file" "https://file.io/" 2>/dev/null)
    local url=$(echo "$response" | grep -oE '"link":"[^"]+"' | sed 's/"link":"//;s/"//')
    if [ -n "$url" ]; then
        echo "$url"
        return 0
    fi
    return 1
}

upload_any() {
    local file=$1
    local url
    for method in upload_transfer_sh upload_0x0_st upload_file_io; do
        url=$($method "$file")
        if [ -n "$url" ]; then
            echo "$url"
            return 0
        fi
    done
    return 1
}

# === MAIN BUNDLE ===
echo "Uploading bundle..."
BUNDLE_URL=$(upload_any "/tmp/$BUNDLE")

echo
echo "================================================================================"
if [ -n "$BUNDLE_URL" ]; then
    echo "✓ BUNDLE UPLOADED"
    echo
    echo "  $BUNDLE_URL"
    echo
    echo "Download on your laptop:"
    echo "  curl -O '$BUNDLE_URL'"
    echo "  tar -xzf $BUNDLE"
    echo "  ls -la"
else
    echo "✗ BUNDLE UPLOAD FAILED — all 3 services rejected it"
    echo
    echo "Bundle is at: /tmp/$BUNDLE ($SIZE)"
    echo
    echo "Manual download options:"
    echo "  1. From your laptop, scp -i ~/.ssh/id_ed25519 PODHOST@ssh.runpod.io:/tmp/$BUNDLE ./"
    echo "  2. base64 /tmp/$BUNDLE > /tmp/bundle.b64  (then copy-paste into terminal)"
fi
echo "================================================================================"

# === results.json (most useful single file) ===
if [ -f "$LOG_DIR/results.json" ]; then
    echo
    echo "Uploading results.json (the leaderboard) individually..."
    RESULTS_URL=$(upload_any "$LOG_DIR/results.json")
    if [ -n "$RESULTS_URL" ]; then
        echo "  ✓ $RESULTS_URL"
        echo
        echo "  Quick view:"
        echo "    curl -s '$RESULTS_URL' | jq '.runs[] | {phase, file, val_bpb, max_step}' | head -50"
    fi
fi

# === Top-level summary logs (setup, validate, unknown) ===
echo
echo "Uploading runner summaries..."
for name in setup.log validate.log unknown.log; do
    if [ -f "$LOG_DIR/$name" ]; then
        size=$(du -h "$LOG_DIR/$name" | cut -f1)
        echo
        echo "  $name ($size):"
        URL=$(upload_any "$LOG_DIR/$name")
        if [ -n "$URL" ]; then
            echo "    $URL"
        else
            echo "    (upload failed — see bundle)"
        fi
    fi
done

echo
echo "================================================================================"
echo "TIP: Save these URLs! transfer.sh expires in 14 days, 0x0.st in 30 days,"
echo "     file.io after first download."
echo
echo "If you need a per-test file (e.g. logs/u02/run_B_progressive.log) just"
echo "extract it from the bundle:"
echo "  tar -xzf $BUNDLE u02/run_B_progressive.log"
echo "================================================================================"
