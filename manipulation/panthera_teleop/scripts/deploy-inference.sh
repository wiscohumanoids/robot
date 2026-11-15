#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
inference_host="${PANTHERA_SSH_HOST:-3.83.15.102}"
ssh -o BatchMode=yes "$inference_host" 'mkdir -p /mnt/pi05/inference ~/.config/systemd/user'
scp "$task_root/infra/inference/server.py" "$task_root/infra/inference/action_limits.py" "$task_root/infra/inference/contract.json" "$inference_host:/mnt/pi05/inference/"
scp "$task_root/infra/inference/panthera-inference.service" "$inference_host:.config/systemd/user/"
ssh "$inference_host" 'sudo loginctl enable-linger admin'
ssh "$inference_host" 'systemctl --user daemon-reload && systemctl --user enable --now panthera-inference.service && systemctl --user restart panthera-inference.service'
ssh "$inference_host" 'systemctl --user status panthera-inference.service --no-pager'
