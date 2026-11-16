#!/usr/bin/env bash
# Overnight job control for the GPU VM. Same commands on the Mac (runs over ssh) and on the VM.
#   tools/vm_jobs.sh list | status | launch <job>... | launch-all | logs <job> [-f] [-n N]
#   tools/vm_jobs.sh stop <job>... | resume <job>... | gpulog-start | gpulog-stop
# Jobs: infra/vm_jobs/jobs.yaml. Outputs: ~/hsim_runs/<job>/ on the VM. Details: infra/vm_jobs/README.md.
# Mac: VM_HOST overrides the ssh host (default hsim-gpu).
set -euo pipefail
if [[ "$(uname)" == "Darwin" ]]; then
  HOST="${VM_HOST:-hsim-gpu}"
  args=$(printf " %q" "$@")
  if [[ " $* " == *" -f "* ]]; then
    exec ssh -t "$HOST" "~/humanoid-sim-stack/tools/vm_jobs.sh$args"
  fi
  exec ssh "$HOST" "~/humanoid-sim-stack/tools/vm_jobs.sh$args"
fi
exec /usr/bin/python3 "$HOME/humanoid-sim-stack/infra/vm_jobs/vmjobs.py" "$@"
