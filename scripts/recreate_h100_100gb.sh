#!/bin/bash
# recreate_h100_100gb.sh — destroy the current H100 pod + create a new one with
# correct disk config (100 GB container + 50 GB volume) per submission/README.md.
#
# Why this exists: Loop B fire 2 on 2026-04-16 hit a disk-headroom failure in
# submission/get_data.sh because the current pod was created with
# --containerDiskSize 60 --volumeSize 0. SP8192 data needs docs_selected.jsonl
# (48 GB) + 24 GB of tokenized shards = 72 GB total. 60 GB doesn't fit.
#
# This script is one-shot: destroys old pod, creates new one, updates
# POD_HOSTS.env. Run manually when ready.
#
# USAGE:
#   bash scripts/recreate_h100_100gb.sh
#
# NOTE: after running, you MUST manually:
#   1. Copy the new pod's SSH proxy string from RunPod web UI (the
#      `<podid>-<suffix>` format — runpodctl doesn't print the suffix)
#   2. Paste it into POD_HOSTS.env as POD_H100_SSH
#   3. Also update the kill-switch cron prompt (which has the OLD pod id
#      hardcoded) via `CronList` + CronDelete + CronCreate — or just
#      manually run the teardown at 9am.

set -eu
cd "$(dirname "$0")/.."

source POD_HOSTS.env 2>/dev/null || true

# --- 1. Destroy old pod (if any) ---
if [ -n "${POD_H100_ID:-}" ] && [ "${POD_H100_NOTE:-}" != "DESTROYED" ]; then
    echo "=== destroying old pod $POD_H100_ID ==="
    runpodctl remove pod "$POD_H100_ID" 2>&1 | tail -3
    # mark POD_HOSTS.env to note destruction (manual update of note field)
    sed -i.bak "s/^POD_H100_NOTE=.*$/POD_H100_NOTE=\"DESTROYED 2026-04-16 — replaced with 100GB disk\"/" POD_HOSTS.env
    rm -f POD_HOSTS.env.bak
fi

# --- 2. Create new pod ---
echo "=== creating new H100 pod with 100 GB container + 50 GB volume ==="
# Try HBM3 secure first (what we had last time), fall back through other SKUs
for ATTEMPT in \
    '--gpuType "NVIDIA H100 80GB HBM3" --secureCloud' \
    '--gpuType "NVIDIA H100 PCIe" --communityCloud' \
    '--gpuType "NVIDIA H100 NVL" --communityCloud' \
    '--gpuType "NVIDIA H100 PCIe" --secureCloud' ; do
    echo "---trying: $ATTEMPT---"
    eval runpodctl create pod --name paramgolf-h100-v2 \
        --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
        --containerDiskSize 100 --volumeSize 50 \
        --ports '22/tcp' --mem 40 --vcpu 16 \
        $ATTEMPT 2>&1 | head -3
    if runpodctl get pod 2>&1 | grep -q paramgolf-h100-v2 ; then
        echo "GOT ONE"
        break
    fi
done

echo "=== new pod ==="
runpodctl get pod -a 2>&1 | grep paramgolf-h100-v2 || { echo "FAILED to create pod"; exit 1; }

cat <<'AFTER'

NEXT STEPS (manual):
1. Open https://www.runpod.io/console/pods and find the new pod
2. Click it → SSH connection → copy the "SSH via pod id" string (like "abcd-ef12@ssh.runpod.io")
3. Edit POD_HOSTS.env:
     POD_H100_SSH="<copied-string>"
     POD_H100_ID="<pod id shown in runpodctl output>"
     POD_H100_NOTE="paramgolf-h100-v2 | 1×H100 | 100+50 GB | recreated 2026-04-16"
4. If loops are still running, they'll pick up the new pod on next fire
5. The 08:55 kill-switch cron has OLD pod id hardcoded — update manually or let it harmlessly fail on old id
AFTER
