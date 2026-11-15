#!/usr/bin/env bash
# J1c: G1 velocity v2, resumed from an existing checkpoint into a NEW log root.
# mjlab resumes from <log-root>/g1_velocity_v2/<load-run>/<load-checkpoint>, so this copies the source
# checkpoint there first (run dir "resume_<source run>"), then trains with --agent.resume True.
# The new run is written next to it (<log-root>/g1_velocity_v2/<timestamp>/). The curriculum level restarts at 0.
# Usage:
#   scripts/train_g1_velocity_v2_resume.sh --from ~/hsim_runs/J1b_g1_vel_v2/g1_velocity_v2/<run>/model_500.pt \
#       --log-root ~/hsim_runs/J1c_g1_vel_v2 [extra mjlab train flags...]
#   NUM_ENVS (default 4096) and ITERS (default 20000) as in train_g1_velocity_v2.sh. PYTHON overrides the interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_MODE=offline
if [[ "$(uname)" == Linux ]]; then export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi   # EGL is Linux-only
FROM="" LOG_ROOT=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ -f "$FROM" ]] || { echo "--from checkpoint not found: $FROM" >&2; exit 1; }
LOG_ROOT="${LOG_ROOT:-$HOME/hsim_runs/J1c_g1_vel_v2}"
SRC_RUN="$(basename "$(dirname "$FROM")")"
CKPT="$(basename "$FROM")"
RUN="resume_${SRC_RUN}"
DEST="$LOG_ROOT/g1_velocity_v2/$RUN"
mkdir -p "$DEST"
[[ -f "$DEST/$CKPT" ]] || cp -p "$FROM" "$DEST/$CKPT"
echo "resuming from $DEST/$CKPT"
exec "${PYTHON:-.venv/bin/python}" hsim_tasks/train.py Hsim-G1-Velocity-Flat-v2 \
  --env.scene.num-envs "${NUM_ENVS:-4096}" \
  --agent.max-iterations "${ITERS:-20000}" \
  --agent.logger tensorboard \
  --agent.resume True --agent.load-run "$RUN" --agent.load-checkpoint "$CKPT" \
  --log-root "$LOG_ROOT" \
  ${ARGS[@]+"${ARGS[@]}"}
