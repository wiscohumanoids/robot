#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$task_dir/work/teleop-venv/bin/python" "$task_dir/app/hand_guide.py" "$@"
