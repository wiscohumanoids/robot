#!/usr/bin/env bash
# Forward both perception servers from the GPU VM to this machine, until Ctrl-C:
#   127.0.0.1:8100  Grounding DINO + SAM 2.1   (segmenter dino_sam2)
#   127.0.0.1:8101  SAM 3                      (segmenter sam3)
# The VM is VM_HOST from the repo's .env (an ssh host or ~/.ssh/config alias). On the VM, start the servers with
# services/perception_server/run_server.sh and run_sam3_server.sh.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
VM_HOST="${VM_HOST:-$(grep -E '^VM_HOST=' "$REPO/.env" 2>/dev/null | cut -d= -f2- | sed 's/[[:space:]]*#.*//; s/^["'\'']//; s/["'\'']$//')}"
[ -n "$VM_HOST" ] || { echo "set VM_HOST in $REPO/.env" >&2; exit 1; }
echo "forwarding 8100 (dino_sam2) and 8101 (sam3) from $VM_HOST; Ctrl-C to stop"
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 8100:localhost:8100 -L 8101:localhost:8101 "$VM_HOST"
