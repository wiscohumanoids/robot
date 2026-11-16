#!/usr/bin/env bash
# J3: train the Wisco V1 walking policy (arms-free recipe: legs + waist, 15 actions) on the GPU VM.
# Runs from ~/robot/training/locomotion with its .venv (mjlab 1.6.0, GPU torch).
#   training/locomotion/wisco_v1/vm/train_velocity.sh [--log-root DIR] [extra mjlab train flags...]
# Defaults: --log-root ~/hsim_runs/wisco_v1_velocity_arms_free, 4096 envs, 20000 iterations (stop when good; checkpoints every 50),
# TensorBoard logger (set LOGGER=wandb for W&B offline). Resume: add --agent.resume True --agent.load-run <run dir name>.
set -euo pipefail
cd "$(dirname "$0")/../.."          # training/locomotion (mjlab venv)
export MUJOCO_GL=egl WANDB_MODE=offline
LOG_ROOT="$HOME/hsim_runs/wisco_v1_velocity_arms_free"
ARGS=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--log-root" ]]; then LOG_ROOT="$2"; shift 2; else ARGS+=("$1"); shift; fi
done
mkdir -p "$LOG_ROOT"
exec .venv/bin/python wisco_v1/train.py Wisco-V1-Velocity-Flat \
  --env.scene.num-envs "${NUM_ENVS:-4096}" \
  --agent.max-iterations "${ITERS:-20000}" \
  --agent.logger "${LOGGER:-tensorboard}" \
  --log-root "$LOG_ROOT" \
  "${ARGS[@]}"
