#!/usr/bin/env bash
# Start the VLA server on the GPU VM in tmux session hsim-svc-vla (port 8200, localhost only).
#   services/vla_server/run_vla_server.sh [checkpoint dir | --mock]
# Default checkpoint: newest ~/hsim_runs/M1_finetune_v1/checkpoint-*.
# GROOT=n17|n16 picks the env (must match the version the checkpoint was trained with; see its logs/run_info.txt).
# Stop: tmux kill-session -t hsim-svc-vla. Mac side: ssh -N -L 8200:localhost:8200 hsim-gpu
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT="${1:-$(ls -d "$HOME"/hsim_runs/M1_finetune_v1/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)}"
if [ "$CKPT" = "--mock" ]; then ARGS="--mock"; GROOT="${GROOT:-n17}"
else
  [ -d "$CKPT" ] || { echo "no checkpoint (pass a path or --mock)"; exit 1; }
  if [ -z "${GROOT:-}" ]; then GROOT=$(grep -o "GR00T n1[67]" "$(dirname "$CKPT")/logs/run_info.txt" 2>/dev/null | awk '{print $2}' || true); GROOT=${GROOT:-n17}; fi
  ARGS="--model-path $CKPT --modality-config $HERE/../../training/manipulation/groot/g1_dex1_config.py"
fi
REPO="$HOME/hsim_envs/gr00t/Isaac-GR00T"; [ "$GROOT" = n16 ] && REPO="$HOME/hsim_envs/gr00t/Isaac-GR00T-n1d6"
LOG="$HOME/hsim_runs/svc_vla"; mkdir -p "$LOG"
tmux kill-session -t hsim-svc-vla 2>/dev/null || true
tmux new-session -d -s hsim-svc-vla "cd $REPO && env -u LD_LIBRARY_PATH HF_HOME=$HOME/hsim_data/hf-cache HF_TOKEN_PATH=$HOME/.cache/huggingface/token .venv/bin/python $HERE/server.py $ARGS 2>&1 | tee -a $LOG/server.log"
echo "started hsim-svc-vla (GR00T $GROOT, $ARGS); log $LOG/server.log"
