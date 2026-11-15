#!/usr/bin/env bash
# Resume a Wisco velocity run from the NEWEST checkpoint of a previous run, into a new log root.
#   resume_latest.sh <src_log_root> <dst_log_root> [extra train_velocity.sh / mjlab flags...]
# Copies <src>/wisco_v2_velocity/<newest run>/model_<highest number>.pt (numeric sort, not mjlab's alphabetical one)
# to <dst>/wisco_v2_velocity/resume_src/ and launches train_velocity.sh with --agent.resume True on it.
# Curriculum start level: set WISCO_START_LEVEL (and WISCO_START_LEVEL2) in the environment.
# WISCO_CKPT_N=<number> picks model_<number>.pt instead of the highest.
set -euo pipefail
SRC="$1"; DST="$2"; shift 2
RUN=$(ls -d "$SRC"/wisco_v2_velocity/20*/ | sort | tail -1)
N=${WISCO_CKPT_N:-$(ls "$RUN" | grep -E '^model_[0-9]+\.pt$' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1)}
mkdir -p "$DST/wisco_v2_velocity/resume_src"
cp -p "$RUN/model_$N.pt" "$DST/wisco_v2_velocity/resume_src/"
echo "resuming from $RUN/model_$N.pt (start level ${WISCO_START_LEVEL:-0})"
exec "$(dirname "$0")/train_velocity.sh" --log-root "$DST" --agent.resume True --agent.load-run resume_src \
  --agent.load-checkpoint "model_$N.pt" "$@"
