#!/usr/bin/env bash
# Copy the inference server to the GPU host (PANTHERA_SSH_HOST) and (re)start it as a user systemd service.
set -euo pipefail
# Machine settings (PANTHERA_SSH_HOST) come from the repo's .env; see .env.example.
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [ -f "$repo_root/.env" ]; then set -a; . "$repo_root/.env"; set +a; fi
: "${PANTHERA_SSH_HOST:?set PANTHERA_SSH_HOST (user@host of the inference machine) in the repo .env}"
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
inference_host="$PANTHERA_SSH_HOST"
ssh -o BatchMode=yes "$inference_host" 'mkdir -p /mnt/pi05/inference ~/.config/systemd/user'
scp "$task_root/infra/inference/server.py" "$task_root/infra/inference/action_limits.py" "$task_root/infra/inference/contract.json" "$inference_host:/mnt/pi05/inference/"
scp "$task_root/infra/inference/panthera-inference.service" "$inference_host:.config/systemd/user/"
ssh "$inference_host" 'sudo loginctl enable-linger admin'
ssh "$inference_host" 'systemctl --user daemon-reload && systemctl --user enable --now panthera-inference.service && systemctl --user restart panthera-inference.service'
ssh "$inference_host" 'systemctl --user status panthera-inference.service --no-pager'
