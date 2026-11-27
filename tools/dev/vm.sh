#!/usr/bin/env bash
# GPU VM helper: copy code to the VM, fetch trained policies and results, and drive the job queue.
# Runs on the laptop. The VM keeps a copy of this repo in ~/$VM_REPO (default ~/robot); its training
# environments live inside that copy (training/locomotion/.venv) and are never synced.
#
#   tools/dev/vm.sh sync                          copy the repo (code only: no builds, logs, datasets or weights)
#   tools/dev/vm.sh pull-policy <run-dir> <id>    export a run's ONNX + manifest on the VM, copy to models/locomotion/<id>/
#                                             (run-dir is relative to training/locomotion, e.g. logs/rsl_rl/<exp>/<time>)
#   tools/dev/vm.sh pull-results [dest] [--list]  copy small results (ONNX, reports, plots, videos, newest checkpoint per
#                                             run) from ~/hsim_runs into dest (default results/), keeping 5 GB free
#   tools/dev/vm.sh jobs <args>                   job queue (infra/vm_jobs/): list | status | launch <job>... | launch-all |
#                                             logs <job> [-f] [-n N] | stop <job>... | resume <job>...
#
# Environment (the repo's .env, or the shell): VM_HOST (ssh host alias, default hsim-gpu), VM_REPO (repo folder under ~
# on the VM, default robot).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
HOST="${VM_HOST:-hsim-gpu}"
VM_REPO="${VM_REPO:-robot}"
cmd="${1:-}"; shift || true

case "$cmd" in
  sync)
    # Build outputs, run outputs, datasets, secrets and weights stay where they are made.
    rsync -az --delete \
      --exclude .git --exclude .venv --exclude '.venv-*' --exclude .env --exclude __pycache__ \
      --exclude 'ros2_ws/build' --exclude 'ros2_ws/install' --exclude 'ros2_ws/log' \
      --exclude 'training/*/logs' --exclude 'training/*/data' --exclude wandb --exclude results \
      --exclude 'assets/envs/**/build*' --exclude '*.pt' \
      "$ROOT/" "$HOST:~/$VM_REPO/"
    echo "synced to $HOST:~/$VM_REPO"
    ;;
  pull-policy)
    run="${1:?usage: tools/dev/vm.sh pull-policy <run-dir> <policy-id>}"
    id="${2:?usage: tools/dev/vm.sh pull-policy <run-dir> <policy-id>}"
    ssh "$HOST" "cd ~/$VM_REPO/training/locomotion && ~/.local/bin/uv run python scripts/export_manifest.py '$run' --policy-id '$id'"
    mkdir -p "$ROOT/models/locomotion/$id"
    rsync -az "$HOST:~/$VM_REPO/models/locomotion/$id/" "$ROOT/models/locomotion/$id/"
    echo "pulled to models/locomotion/$id/"
    ;;
  pull-results)
    dest="$ROOT/results"; list_only=0; remote='$HOME/hsim_runs'
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --list) list_only=1; shift ;;
        --runs) remote="$2"; shift 2 ;;
        *) dest="$1"; shift ;;
      esac
    done
    # The VM decides what is small enough to bring over (infra/vm_jobs/pull_manifest.sh) and reports the total.
    manifest=$(ssh "$HOST" "bash ~/$VM_REPO/infra/vm_jobs/pull_manifest.sh $remote")
    total=$(echo "$manifest" | awk '/^TOTAL / {print $2}')
    files=$(echo "$manifest" | grep -v '^TOTAL ' || true)
    n=$(echo "$files" | grep -c . || true)
    mkdir -p "$dest"
    free=$(df -k "$dest" | awk 'NR==2 {print $4 * 1024}')
    printf "to pull: %d files, %.1f MB; free here: %.1f GB\n" "$n" "$(echo "${total:-0}/1048576" | bc -l)" "$(echo "$free/1073741824" | bc -l)"
    if [[ $list_only == 1 ]]; then echo "$files"; exit 0; fi
    if (( ${total:-0} + 5*1024*1024*1024 > free )); then echo "not enough local disk (keeping 5 GB spare)"; exit 1; fi
    [[ $n -gt 0 ]] || { echo "nothing to pull"; exit 0; }
    echo "$files" | ssh "$HOST" "cd $remote && tar cf - -T -" | tar xf - -C "$dest"
    echo "pulled into $dest"
    ;;
  jobs)
    # -f (follow logs) needs a terminal on the remote side.
    args=$(printf " %q" "$@")
    tty=(); [[ " $* " == *" -f "* ]] && tty=(-t)
    exec ssh "${tty[@]}" "$HOST" "/usr/bin/python3 ~/$VM_REPO/infra/vm_jobs/vmjobs.py$args"
    ;;
  *)
    sed -n '2,16p' "$0"; exit 1 ;;
esac
