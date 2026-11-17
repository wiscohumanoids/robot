#!/usr/bin/env bash
# Start the perception server on the GPU machine in tmux session hsim-svc-perception (127.0.0.1:8100). Extra args go to the server,
# e.g. --dino IDEA-Research/grounding-dino-tiny. Stop: tmux kill-session -t hsim-svc-perception
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HOME/hsim_runs/perception"; mkdir -p "$LOG"
SESSION=hsim-svc-perception
tmux has-session -t "$SESSION" 2>/dev/null && { echo "already running: tmux attach -t $SESSION"; exit 0; }
# LD_LIBRARY_PATH unset: the VM's default puts system CUDA 13.2 libs first, so torch would load the system cuDNN
# instead of the one it ships with (that breaks cuDNN ops in some environments).
tmux new-session -d -s "$SESSION" "cd '$HERE' && env -u LD_LIBRARY_PATH HF_HOME=\$HOME/hsim_data/hf-cache HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-0} \
  \$HOME/hsim_envs/perception/bin/python -m perception_server.server --backend real $* 2>&1 | tee -a '$LOG/server.log'"
echo "started tmux session $SESSION; log $LOG/server.log; health: curl -s localhost:8100/health"
