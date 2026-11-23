#!/usr/bin/env bash
# One phase of the daily learning cycle, on the GPU VM (learning/curriculum.py):
#
#   bash manipulation/panthera_agent/learning/daily.sh morning     # practice (demos, DAgger, RL if enabled)
#   bash manipulation/panthera_agent/learning/daily.sh night       # retrain, held-out tests, promote, Claude's step
#   bash manipulation/panthera_agent/learning/daily.sh status
#
# Launched from the laptop by tools/learning_vm.sh (through the job queue: panthera_agent_morning,
# panthera_agent_night). Everything lives in ~/hsim_data/panthera_agent: episodes/, memory/, models/,
# curriculum.json, and the one log, learning.log. The first run starts the curriculum.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PHASE="${1:?usage: daily.sh morning|night|status}"
PY="${PANTHERA_AGENT_PYTHON:-$HOME/hsim_envs/panthera_agent/bin/python}"
export PANTHERA_AGENT_DATA="${PANTHERA_AGENT_DATA:-$HOME/hsim_data/panthera_agent}"
export PANTHERA_AGENT_MEMORY="${PANTHERA_AGENT_MEMORY:-$PANTHERA_AGENT_DATA/memory}"
export PANTHERA_AGENT_MODELS="${PANTHERA_AGENT_MODELS:-$PANTHERA_AGENT_DATA/models}"
export PANTHERA_AGENT_LOG="${PANTHERA_AGENT_LOG:-$PANTHERA_AGENT_DATA/learning.log}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONUNBUFFERED=1
# The VM's system CUDA libraries on LD_LIBRARY_PATH shadow the cuDNN PyTorch ships with (cuDNN "sublibrary version
# mismatch" on the first convolution): this environment uses its own.
unset LD_LIBRARY_PATH
[[ -x "$PY" ]] || { echo "no Python at $PY: run learning/setup_vm.sh first" >&2; exit 1; }
mkdir -p "$PANTHERA_AGENT_DATA" "$PANTHERA_AGENT_MODELS"
cd "$HERE"
[[ -f "$PANTHERA_AGENT_DATA/curriculum.json" ]] || "$PY" -m learning.curriculum init
if ! "$PY" -m learning.curriculum "$PHASE"; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') [$PHASE] FAILED: see the job's stdout.log (tools/vm.sh jobs logs panthera_agent_$PHASE)" \
    >> "$PANTHERA_AGENT_LOG"
  exit 1
fi
