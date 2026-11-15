#!/usr/bin/env bash
# Policy A: velocity-command walking (drives /cmd_vel later).
# Usage: scripts/train_velocity.sh [extra mjlab flags...]
# Defaults: 4096 envs, 10000 PPO updates (~1B env steps). Override with NUM_ENVS / ITERS.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs "${NUM_ENVS:-4096}" \
  --agent.max-iterations "${ITERS:-10000}" \
  --log-root logs/rsl_rl \
  "$@"
