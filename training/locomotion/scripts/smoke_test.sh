#!/usr/bin/env bash
# A few PPO updates on a small batch of envs. Proves mjlab + MuJoCo-Warp + torch work on this GPU.
# Logs to W&B in offline mode, so no login is needed yet.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl WANDB_MODE=offline
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs 256 \
  --agent.max-iterations 5 \
  --log-root logs/smoke
echo "Smoke test passed. Outputs in training/locomotion/logs/smoke/"
