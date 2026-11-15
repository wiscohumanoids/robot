#!/usr/bin/env bash
# Policy B: BeyondMimic-style motion tracking of one LAFAN1 clip.
# Usage: scripts/train_tracking.sh <registry-name> [extra mjlab flags...]
#   registry-name looks like <entity>-org/wandb-registry-Motions/walk1_subject1 (copy it from the W&B registry page)
set -euo pipefail
cd "$(dirname "$0")/.."
REG="${1:?usage: train_tracking.sh <wandb-registry-name>}"; shift
export MUJOCO_GL=egl
uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --registry-name "$REG" \
  --env.scene.num-envs "${NUM_ENVS:-4096}" \
  --agent.max-iterations "${ITERS:-10000}" \
  --log-root logs/rsl_rl \
  "$@"
