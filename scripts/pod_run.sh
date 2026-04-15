#!/bin/bash
# pod_run.sh — execute a shell script on the paramgolf-diag pod via the
# SSH-over-base64 pattern documented in runpod_tests/loop/SSH_TROUBLESHOOTING.md.
#
# Usage:
#   ./scripts/pod_run.sh <path/to/remote_script.sh>
#   # or stream from stdin:
#   echo 'nvidia-smi' | ./scripts/pod_run.sh -
#
# Reads POD_HOSTS.env for the pod endpoint.

set -eu
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source POD_HOSTS.env

if [ $# -lt 1 ]; then
    echo "usage: $0 <script.sh | ->" >&2
    exit 1
fi

if [ "$1" = "-" ]; then
    INNER=$(cat)
else
    INNER=$(cat "$1")
fi

B64=$(echo "$INNER" | base64 | tr -d '\n')

unset SSH_AUTH_SOCK
ssh -tt -F /dev/null -i "$POD_DIAG_KEY" \
    -o IdentitiesOnly=yes \
    -o IdentityAgent=none \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=20 \
    -o ServerAliveInterval=15 \
    "$POD_DIAG_SSH" <<SSH_STDIN
echo $B64 | base64 -d > /tmp/pod_inner.sh && bash /tmp/pod_inner.sh
exit
SSH_STDIN
