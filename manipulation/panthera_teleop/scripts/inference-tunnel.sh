#!/usr/bin/env bash
set -euo pipefail
exec ssh -N -T -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8765:127.0.0.1:8765 "${PANTHERA_SSH_HOST:-3.83.15.102}"
