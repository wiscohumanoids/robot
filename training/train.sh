#!/usr/bin/env bash
# One entry point for every training job on the GPU machine: picks the script for a robot and task and passes the
# remaining arguments through (each script documents its own flags and environment overrides).
#
#   training/train.sh <robot> <task> [task arguments] [extra flags]
#
#   robot   task                  what runs
#   wisco   walk                  Wisco-V1-Velocity-Flat, arms-free recipe       locomotion/wisco_v1/vm/train_velocity.sh
#   wisco   track <clip>          Wisco-V1-Tracking-Flat on one retargeted clip  tracking/wisco_v1/train_wisco_tracking.sh
#   g1      walk                  Hsim-G1-Velocity-Flat-ArmsFree                 locomotion/scripts/train_g1_velocity_arms_free.sh
#   g1      walk-all-joints       mjlab's Mjlab-Velocity-Flat-Unitree-G1          locomotion/scripts/train_velocity.sh
#   g1      track <clip>          Mjlab-Tracking-Flat-Unitree-G1 on one clip      tracking/train_g1_tracking.sh
#   g1      manipulate <version>  GR00T fine-tune, G1 + Dex1 at a counter          manipulation/groot/finetune.sh
#   any     smoke                 5 PPO updates on 256 envs (checks the GPU stack) locomotion/scripts/smoke_test.sh
#
# Arm manipulation policies (ACT, pi0.5, GR00T on an arm profile) train through manipulation/finetune.sh.
# Examples:
#   training/train.sh wisco walk --log-root ~/hsim_runs/wisco_walk
#   training/train.sh g1 track dance1_subject2
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }
[[ $# -ge 1 && ( "$1" == -h || "$1" == --help ) ]] && usage 0
[[ $# -ge 2 ]] || usage
robot="$1" task="$2"; shift 2

case "$robot/$task" in
  wisco/walk)            exec "$HERE/locomotion/wisco_v1/vm/train_velocity.sh" "$@" ;;
  wisco/track)           exec "$HERE/tracking/wisco_v1/train_wisco_tracking.sh" "$@" ;;
  g1/walk)               exec "$HERE/locomotion/scripts/train_g1_velocity_arms_free.sh" "$@" ;;
  g1/walk-all-joints)    exec "$HERE/locomotion/scripts/train_velocity.sh" "$@" ;;
  g1/track)              exec "$HERE/tracking/train_g1_tracking.sh" "$@" ;;
  g1/manipulate)         exec "$HERE/manipulation/groot/finetune.sh" "$@" ;;
  */smoke)               exec "$HERE/locomotion/scripts/smoke_test.sh" "$@" ;;
  *) echo "unknown robot/task: $robot $task" >&2; usage ;;
esac
