#!/bin/bash
# pod_run_h100.sh — run a shell script on the H100 pod via the SSH-over-base64 pattern.
# Usage:  echo '<cmds>' | scripts/pod_run_h100.sh -
#         scripts/pod_run_h100.sh path/to/script.sh
set -eu
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source POD_HOSTS.env

if [ $# -lt 1 ]; then echo "usage: $0 <script|->" >&2; exit 1; fi
if [ "$1" = "-" ]; then INNER=$(cat); else INNER=$(cat "$1"); fi

B64=$(echo "$INNER" | base64 | tr -d '\n')

unset SSH_AUTH_SOCK
ssh -tt -F /dev/null -i "$POD_H100_KEY" \
    -o IdentitiesOnly=yes -o IdentityAgent=none \
    -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=20 -o ServerAliveInterval=15 \
    "$POD_H100_SSH" <<SSH_STDIN
echo $B64 | base64 -d > /tmp/pod_inner.sh && bash /tmp/pod_inner.sh
exit
SSH_STDIN
