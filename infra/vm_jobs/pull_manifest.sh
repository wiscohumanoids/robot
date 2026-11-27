#!/usr/bin/env bash
# VM side of `tools/dev/vm.sh pull-results`: print the files worth pulling (relative to RUNS_DIR), then a final
# line "TOTAL <bytes>". Small artifacts + only the newest model_*.pt per run dir; never data/datasets/raw/wandb,
# stdout logs, tensorboard events, or (unless pulling it explicitly) the vmjobs test folder and vm_backup.
#   pull_manifest.sh RUNS_DIR
set -euo pipefail
RUNS="${1:-$HOME/hsim_runs}"
cd "$RUNS"
prune=(-name data -o -name datasets -o -name raw -o -name wandb -o -path ./vm_backup)
[[ "$RUNS" == *_vmjobs_test* ]] || prune+=(-o -path ./_vmjobs_test)
list=$(mktemp)
find . \( "${prune[@]}" \) -prune -o -type f \( -name '*.onnx' -o -name '*.json' -o -name '*.yaml' -o -name '*.md' \
  -o -name '*.html' -o -name '*.png' -o -name '*.mp4' -o -name '*.csv' -o -name '*.txt' \) -print > "$list"
find . \( "${prune[@]}" \) -prune -o -type f -name 'model_*.pt' -printf '%h\n' | sort -u | while read -r d; do
  ls -1 "$d"/model_*.pt | sort -V | tail -1
done >> "$list"
total=0
while read -r f; do total=$((total + $(stat -c %s "$f"))); echo "$f"; done < "$list"
rm -f "$list"
echo "TOTAL $total"
