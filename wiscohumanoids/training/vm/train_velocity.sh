#!/usr/bin/env bash
# J3: train the Wisco V2 walking policy (v2 recipe: legs + waist, 15 actions) on the GPU VM.
# Runs from ~/humanoid-sim-stack/training/locomotion with its .venv (mjlab 1.6.0, GPU torch).
#   wiscohumanoids/training/vm/train_velocity.sh [--log-root DIR] [extra mjlab train flags...]
# Defaults: --log-root ~/hsim_runs/wisco_velocity_v2, 4096 envs, 20000 iterations (stop when good; checkpoints every 50),
# TensorBoard logger (set LOGGER=wandb for W&B offline). Resume: add --agent.resume True --agent.load-run <run dir name>.
set -euo pipefail
cd "$(dirname "$0")/../../../training/locomotion"
export MUJOCO_GL=egl WANDB_MODE=offline
LOG_ROOT="$HOME/hsim_runs/wisco_velocity_v2"
ARGS=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--log-root" ]]; then LOG_ROOT="$2"; shift 2; else ARGS+=("$1"); shift; fi
done
mkdir -p "$LOG_ROOT"
exec .venv/bin/python ../../wiscohumanoids/training/mjlab_task/train.py Wisco-V2-Velocity-Flat \
  --env.scene.num-envs "${NUM_ENVS:-4096}" \
  --agent.max-iterations "${ITERS:-20000}" \
  --agent.logger "${LOGGER:-tensorboard}" \
  --log-root "$LOG_ROOT" \
  "${ARGS[@]}"
