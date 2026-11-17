#!/usr/bin/env bash
# G1 velocity arms-free (legs+waist policy, scripted arms; hsim_tasks/g1_velocity_arms_free.py).
# Runs from ~/robot/training/locomotion with its uv .venv (mjlab 1.6.0, GPU).
# Usage: scripts/train_g1_velocity_arms_free.sh --log-root ~/hsim_runs/g1_velocity_arms_free [extra mjlab train flags...]
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
LOG_ROOT="${LOG_ROOT:-$HOME/hsim_runs/g1_velocity_arms_free}"
mkdir -p "$LOG_ROOT"
exec .venv/bin/python hsim_tasks/train.py Hsim-G1-Velocity-Flat-ArmsFree \
  --env.scene.num-envs "${NUM_ENVS:-4096}" \
  --agent.max-iterations "${ITERS:-20000}" \
  --agent.logger tensorboard \
  --log-root "$LOG_ROOT" \
  ${ARGS[@]+"${ARGS[@]}"}
