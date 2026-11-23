#!/usr/bin/env bash
# The daily learning cycle on the GPU VM, driven from here (the laptop). All code lives in this repo; the VM runs it.
#
#   tools/learning_vm.sh setup            once: sync, build the VM's environment (learning/setup_vm.sh), copy the
#                                         Anthropic key into the VM's data folder (mode 600; never printed)
#   tools/learning_vm.sh sync             copy the code the cycle needs to the VM (adds and updates, never deletes)
#   tools/learning_vm.sh push             this machine's episodes and seen-objects list to the VM (adds only)
#   tools/learning_vm.sh morning          sync, then start the morning phase (practice)
#   tools/learning_vm.sh night            sync, then start the night phase (retrain, test, promote, Claude's step)
#   tools/learning_vm.sh wait             until the running phase ends; prints how it ended
#   tools/learning_vm.sh log [N]          the last N lines (default 40) of the one learning log
#   tools/learning_vm.sh status           the curriculum's tasks, the promoted policy, the jobs
#   tools/learning_vm.sh pull             the promoted weights and all of models/local, curriculum.json, the log ->
#                                         here (models/local/, runs/vm/), so the policy can be tried locally
#
# The VM is VM_HOST (an ssh alias) from the repo's .env; its repo copy is ~/$VM_REPO (default robot). The VM keeps
# everything it makes in ~/hsim_data/panthera_agent (episodes/, memory/, models/, curriculum.json, learning.log).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
HOST="${VM_HOST:-hsim-gpu}"
VM_REPO="${VM_REPO:-robot}"
DATA='~/hsim_data/panthera_agent'
REMOTE_AGENT="~/$VM_REPO/manipulation/panthera_agent"
cmd="${1:-}"; shift || true

sync_code() {
  local ex=(--exclude .venv --exclude '.venv-*' --exclude __pycache__ --exclude .pytest_cache --exclude data
            --exclude models --exclude memory --exclude runs --exclude outputs)
  ssh "$HOST" "mkdir -p ~/$VM_REPO/manipulation/panthera_sim ~/$VM_REPO/manipulation/profiles ~/$VM_REPO/infra/vm_jobs $DATA"
  rsync -az "${ex[@]}" "$HERE/" "$HOST:$REMOTE_AGENT/"
  rsync -az "${ex[@]}" "$ROOT/manipulation/panthera_sim/sim/" "$HOST:~/$VM_REPO/manipulation/panthera_sim/sim/"
  rsync -az "$ROOT/manipulation/profiles/panthera_local.yaml" "$HOST:~/$VM_REPO/manipulation/profiles/"
  rsync -az "$ROOT/infra/vm_jobs/jobs.yaml" "$HOST:~/$VM_REPO/infra/vm_jobs/jobs.yaml"
  echo "code synced to $HOST:~/$VM_REPO (panthera_agent, panthera_sim/sim, jobs.yaml)"
}

job_state() {     # the job queue's state of a job: running, done, failed, ...
  ssh "$HOST" "python3 -c \"import json,os;p=os.path.expanduser('~/hsim_runs/$1/state.json');print(json.load(open(p)).get('state','?') if os.path.exists(p) else 'none')\""
}

case "$cmd" in
  sync) sync_code ;;
  setup)
    sync_code
    ssh "$HOST" "bash $REMOTE_AGENT/learning/setup_vm.sh"
    if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
      printf 'ANTHROPIC_API_KEY=%s\n' "$ANTHROPIC_API_KEY" | ssh "$HOST" "umask 077; cat > $DATA/anthropic.env"
      echo "Anthropic key copied to the VM's data folder (mode 600)"
    else
      echo "no ANTHROPIC_API_KEY here: the curriculum step will be skipped on the VM"
    fi
    ;;
  push)
    ssh "$HOST" "mkdir -p $DATA/episodes $DATA/memory"
    [ -d "$HERE/data/episodes" ] && rsync -az "$HERE/data/episodes/" "$HOST:$DATA/episodes/"
    [ -d "$HERE/memory" ] && rsync -az "$HERE/memory/" "$HOST:$DATA/memory/"
    echo "pushed episodes and memory"
    ;;
  morning|night)
    sync_code
    "$ROOT/tools/vm.sh" jobs launch "panthera_agent_$cmd"
    ;;
  wait)
    for j in panthera_agent_morning panthera_agent_night; do
      while [ "$(job_state $j)" = running ]; do sleep 30; done
    done
    echo "morning: $(job_state panthera_agent_morning), night: $(job_state panthera_agent_night)"
    ;;
  log) ssh "$HOST" "tail -n ${1:-40} $DATA/learning.log 2>/dev/null || echo 'no log yet'" ;;
  status)
    ssh "$HOST" "cd $REMOTE_AGENT && PANTHERA_AGENT_DATA=$DATA PANTHERA_AGENT_MODELS=$DATA/models \
      ~/hsim_envs/panthera_agent/bin/python -m learning.curriculum status 2>/dev/null || echo 'no curriculum yet'"
    "$ROOT/tools/vm.sh" jobs status 2>/dev/null | grep -i panthera_agent || true
    ;;
  pull)
    mkdir -p "$HERE/models/local" "$HERE/runs/vm"
    rsync -az "$HOST:$DATA/models/local/" "$HERE/models/local/"
    rsync -az "$HOST:$DATA/curriculum.json" "$HOST:$DATA/learning.log" "$HERE/runs/vm/" 2>/dev/null || true
    rsync -az --include '*/' --include 'reward.pt' --exclude '*' "$HOST:$DATA/models/" "$HERE/models/" 2>/dev/null || true
    echo "pulled: models/local/ (promoted: $(python3 -c "import json;print(json.load(open('$HERE/models/local/latest.json'))['name'])" 2>/dev/null || echo none)), runs/vm/curriculum.json, runs/vm/learning.log"
    ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
