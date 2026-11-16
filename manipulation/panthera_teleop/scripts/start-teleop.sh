#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$task_dir/logs"
exec flock -n "$task_dir/logs/control.lock" "$task_dir/work/teleop-venv/bin/python" "$task_dir/app/launch-teleop.py" "$@"
