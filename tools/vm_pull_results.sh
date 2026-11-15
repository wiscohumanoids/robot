#!/usr/bin/env bash
# Pull small overnight results from the GPU VM to the Mac (never datasets, never all checkpoints).
#   tools/vm_pull_results.sh [DEST] [--runs REMOTE_RUNS_DIR] [--list]
# DEST defaults to runs/2026-11-10 at the repo root; remote defaults to ~/hsim_runs on the VM.
# What comes over (decided on the VM by infra/vm_jobs/pull_manifest.sh): *.onnx *.json *.yaml *.md *.html *.png
# *.mp4 *.csv *.txt plus ONLY the newest model_*.pt of each run dir. Never: data/, datasets/, raw/, wandb/,
# stdout logs, tensorboard events, other checkpoints. Checks local free space first (keeps 5 GB spare).
set -euo pipefail
HOST="${VM_HOST:-hsim-gpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/runs/2026-11-10"; REMOTE='$HOME/hsim_runs'; LIST_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) REMOTE="$2"; shift 2;;
    --list) LIST_ONLY=1; shift;;
    *) DEST="$1"; shift;;
  esac
done
manifest=$(ssh "$HOST" "bash ~/humanoid-sim-stack/infra/vm_jobs/pull_manifest.sh $REMOTE")
total=$(echo "$manifest" | awk '/^TOTAL / {print $2}')
files=$(echo "$manifest" | grep -v '^TOTAL ' || true)
n=$(echo "$files" | grep -c . || true)
mkdir -p "$DEST"
free=$(df -k "$DEST" | awk 'NR==2 {print $4 * 1024}')
printf "to pull: %d files, %.1f MB; free here: %.1f GB\n" "$n" "$(echo "${total:-0}/1048576" | bc -l)" "$(echo "$free/1073741824" | bc -l)"
if [[ $LIST_ONLY == 1 ]]; then echo "$files"; exit 0; fi
if (( ${total:-0} + 5*1024*1024*1024 > free )); then echo "not enough local disk (keeping 5 GB spare); aborting"; exit 1; fi
[[ $n -gt 0 ]] || { echo "nothing to pull"; exit 0; }
echo "$files" | ssh "$HOST" "cd $REMOTE && tar cf - -T -" | tar xf - -C "$DEST"
echo "pulled into $DEST"
