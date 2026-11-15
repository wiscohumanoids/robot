#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$task_dir/work/platform-tools:$PATH"
cd "$task_dir/work/Quest_controller_stream"
exec "$task_dir/work/teleop-venv/bin/python" quest_stream_ui.py
