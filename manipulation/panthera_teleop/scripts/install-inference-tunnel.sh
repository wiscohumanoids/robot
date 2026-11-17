#!/usr/bin/env bash
# Install and start the user systemd service that keeps the inference SSH tunnel up.
set -euo pipefail
# Machine settings (PANTHERA_SSH_HOST) come from the repo's .env; see .env.example.
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [ -f "$repo_root/.env" ]; then set -a; . "$repo_root/.env"; set +a; fi
: "${PANTHERA_SSH_HOST:?set PANTHERA_SSH_HOST (user@host of the inference machine) in the repo .env}"
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/panthera-inference-tunnel.service" <<EOF
[Unit]
Description=SSH tunnel to Panthera AWS inference
After=network-online.target

[Service]
Type=simple
Environment=PANTHERA_SSH_HOST=${PANTHERA_SSH_HOST}
ExecStart="$task_root/scripts/inference-tunnel.sh"
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now panthera-inference-tunnel.service
systemctl --user status panthera-inference-tunnel.service --no-pager
