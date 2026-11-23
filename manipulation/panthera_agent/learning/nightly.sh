#!/usr/bin/env bash
# The overnight learning job, on the GPU VM: train on every episode collected so far, evaluate in the sim
# (headless), promote only if better, write a report (learning/nightly.py).
#
#   bash manipulation/panthera_agent/learning/nightly.sh [run folder]     # default ~/hsim_runs/panthera_agent_nightly
#
# Launched by the job queue (infra/vm_jobs/jobs.yaml: panthera_agent_nightly; from the laptop:
# tools/vm.sh jobs launch panthera_agent_nightly) or by cron, e.g. at 02:00:
#   0 2 * * * bash ~/robot/manipulation/panthera_agent/learning/nightly.sh >> ~/hsim_runs/panthera_agent_nightly/cron.log 2>&1
# Data comes from the laptop and the rig (tools/learning_sync.sh push); results go back with tools/learning_sync.sh pull.
#
# Environment: PANTHERA_AGENT_PYTHON (default ~/hsim_envs/panthera_agent/bin/python, made by learning/setup_vm.sh),
# PANTHERA_AGENT_DATA (default ~/hsim_data/panthera_agent: episodes/, memory/), SKILLS (default: insert),
# NIGHTLY_ARGS (extra arguments for learning/nightly.py, e.g. --export-lerobot).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${1:-$HOME/hsim_runs/panthera_agent_nightly}"
PY="${PANTHERA_AGENT_PYTHON:-$HOME/hsim_envs/panthera_agent/bin/python}"
export PANTHERA_AGENT_DATA="${PANTHERA_AGENT_DATA:-$HOME/hsim_data/panthera_agent}"
export PANTHERA_AGENT_MEMORY="${PANTHERA_AGENT_MEMORY:-$PANTHERA_AGENT_DATA/memory}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONUNBUFFERED=1
# Checkpoints live next to the data on the VM, so a repo sync never touches them.
mkdir -p "$PANTHERA_AGENT_DATA/models"
[[ -e "$HERE/models" ]] || ln -s "$PANTHERA_AGENT_DATA/models" "$HERE/models"
[[ -x "$PY" ]] || { echo "no Python at $PY: run learning/setup_vm.sh first" >&2; exit 1; }
cd "$HERE"
stamp=$(date +%Y%m%d_%H%M%S)
for skill in ${SKILLS:-insert}; do
  "$PY" -m learning.nightly --skill "$skill" --out "$OUT_ROOT/$stamp/$skill" ${NIGHTLY_ARGS:-}
done
echo "reports: $OUT_ROOT/$stamp/*/report.md"
