#!/usr/bin/env bash
# Forward local port 8765 to the inference server on the GPU host over SSH (kept alive; used by the tunnel service).
set -euo pipefail
# Machine settings (PANTHERA_SSH_HOST) come from the repo's .env; see .env.example.
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [ -f "$repo_root/.env" ]; then set -a; . "$repo_root/.env"; set +a; fi
: "${PANTHERA_SSH_HOST:?set PANTHERA_SSH_HOST (user@host of the inference machine) in the repo .env}"
exec ssh -N -T -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8765:127.0.0.1:8765 "$PANTHERA_SSH_HOST"
