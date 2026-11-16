#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/panthera-inference-tunnel.service" <<EOF
[Unit]
Description=SSH tunnel to Panthera AWS inference
After=network-online.target

[Service]
Type=simple
Environment=PANTHERA_SSH_HOST=${PANTHERA_SSH_HOST:-3.83.15.102}
ExecStart="$task_root/scripts/inference-tunnel.sh"
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now panthera-inference-tunnel.service
systemctl --user status panthera-inference-tunnel.service --no-pager
