#!/usr/bin/env bash
# Start the inference dashboard (app/inference_dashboard.py).
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$task_root/work/teleop-venv/bin/python" -u "$task_root/app/inference_dashboard.py" "$@"
