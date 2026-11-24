#!/usr/bin/env bash
# Start the SAM 3 perception server on the GPU machine beside the Grounding DINO + SAM 2.1 one: tmux session
# hsim-svc-perception-sam3, 127.0.0.1:8101, same /detect API. SAM 3 is gated: the HF token in
# ~/hsim_data/hf-cache/token must have been granted access to facebook/sam3. Extra args go to the server.
# Stop: tmux kill-session -t hsim-svc-perception-sam3
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SESSION=hsim-svc-perception-sam3 LOGFILE=server_sam3.log \
  exec bash "$HERE/run_server.sh" --backend sam3 --port 8101 --verifier none "$@"
