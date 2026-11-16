#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$task_dir/logs"
exec "$task_dir/work/teleop-venv/bin/python" -u "$task_dir/app/hand_guide.py" --run "$@" > >(tee -a "$task_dir/logs/gravity-assist.log") 2>&1
