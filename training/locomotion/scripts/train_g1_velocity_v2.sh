#!/usr/bin/env bash
# J1: G1 velocity v2 (legs+waist policy, scripted arms, v2 recipe; docs/runs/2026-11-10_overnight_plan.md).
# Runs from ~/humanoid-sim-stack/training/locomotion with its uv .venv (mjlab 1.6.0, GPU).
# Usage: scripts/train_g1_velocity_v2.sh --log-root ~/hsim_runs/J1_g1_velocity_v2 [extra mjlab train flags...]
#   NUM_ENVS (default 4096) and ITERS (default 20000) can be set in the environment.
#   Resume: add  --agent.resume True --agent.load-run <run dir name>
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_MODE=offline
if [[ "$(uname)" == Linux ]]; then export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi   # EGL is Linux-only
LOG_ROOT=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --log-root=*) LOG_ROOT="${1#*=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
LOG_ROOT="${LOG_ROOT:-$HOME/hsim_runs/J1_g1_velocity_v2}"
mkdir -p "$LOG_ROOT"
exec .venv/bin/python hsim_tasks/train.py Hsim-G1-Velocity-Flat-v2 \
  --env.scene.num-envs "${NUM_ENVS:-4096}" \
  --agent.max-iterations "${ITERS:-20000}" \
  --agent.logger tensorboard \
  --log-root "$LOG_ROOT" \
  ${ARGS[@]+"${ARGS[@]}"}
