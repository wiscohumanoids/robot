#!/usr/bin/env bash
# Move panthera_agent's learning data to the GPU VM and its results back (the laptop or the rig side).
#
#   tools/learning_sync.sh push       episodes and the seen-objects list -> VM ~/hsim_data/panthera_agent/
#   tools/learning_sync.sh pull       trained checkpoints (models/) and nightly reports <- VM
#   tools/learning_sync.sh status     what is on each side (counts only)
#
# Episodes are only ever added (rsync without --delete on push), so data from the rig and from the laptop's sim
# runs accumulate on the VM. The VM's address comes from the repo's .env (VM_HOST, an ssh alias); nothing secret
# is printed. Run tools/vm.sh sync first if the code changed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
HOST="${VM_HOST:-hsim-gpu}"
DATA="${PANTHERA_AGENT_DATA:-$HERE/data}"
MEMORY="${PANTHERA_AGENT_MEMORY:-$HERE/memory}"
REMOTE='~/hsim_data/panthera_agent'
case "${1:-}" in
  push)
    ssh "$HOST" "mkdir -p $REMOTE/episodes $REMOTE/memory $REMOTE/models"
    [ -d "$DATA/episodes" ] && rsync -az "$DATA/episodes/" "$HOST:$REMOTE/episodes/"
    [ -d "$MEMORY" ] && rsync -az "$MEMORY/" "$HOST:$REMOTE/memory/"
    echo "pushed episodes and memory"
    ;;
  pull)
    mkdir -p "$HERE/models" "$HERE/runs/nightly"
    rsync -az "$HOST:$REMOTE/models/" "$HERE/models/"
    rsync -az --include '*/' --include 'report.*' --exclude '*' "$HOST:~/hsim_runs/panthera_agent_nightly/" \
      "$HERE/runs/nightly/" 2>/dev/null || true
    echo "pulled models/ and runs/nightly/"
    ;;
  status)
    echo "here: $(find "$DATA/episodes" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ') episodes, $(ls "$HERE/models" 2>/dev/null | wc -l | tr -d ' ') model folders"
    ssh "$HOST" "echo \"VM: \$(find $REMOTE/episodes -name '*.npz' 2>/dev/null | wc -l) episodes, \$(ls $REMOTE/models 2>/dev/null | wc -l) model folders\""
    ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
