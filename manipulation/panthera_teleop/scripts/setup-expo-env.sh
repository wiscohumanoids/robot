#!/usr/bin/env bash
# Create an isolated local interpreter while sharing the existing Torch packages.
# This never installs into, activates, or modifies the source environment.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_env="${EXPO_SOURCE_ENV:-$task_root/../RobotArmLearning/.venv-act}"
target_env="$task_root/work/expo-venv"
source_python="$source_env/bin/python"
if [[ ! -x "$source_python" ]]; then
    printf 'Missing source interpreter: %s\n' "$source_python" >&2
    exit 2
fi
if [[ -e "$target_env" ]]; then
    if [[ ! -f "$target_env/expo-source.json" ]]; then
        printf 'Refusing to overwrite an unrecognized environment: %s\n' "$target_env" >&2
        exit 2
    fi
else
    uv venv --python "$source_python" "$target_env"
fi
PYTHONDONTWRITEBYTECODE=1 "$source_python" - "$target_env" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

target = Path(sys.argv[1]).resolve()
source = Path(sys.prefix).resolve()
source_packages = Path(sysconfig.get_paths()['purelib']).resolve()
target_packages = Path(subprocess.check_output(
    [str(target / 'bin/python'), '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
    text=True).strip())
source_info = {'source_env': str(source), 'source_python': sys.executable,
               'shared_site_packages': str(source_packages), 'python': sys.version,
               'packages': {name: metadata.version(name) for name in ('torch', 'torchvision', 'numpy')}}
marker = target / 'expo-source.json'
if marker.exists() and json.loads(marker.read_text()) != source_info:
    raise SystemExit('Source interpreter/packages changed; inspect expo-source.json before rebuilding')
# Plain path shares packages without executing arbitrary .pth hooks from the source.
# Disable bytecode writes so imported shared packages retain their existing files.
(target_packages / 'expo_shared.pth').write_text(
    'import sys; sys.dont_write_bytecode = True\n' + str(source_packages) + '\n')
marker.write_text(json.dumps(source_info, indent=2) + '\n')
(target / 'README.txt').write_text(
    'EXPO local environment\n\n'
    f'Shares installed packages read-only by convention from {source_packages}.\n'
    'Source .pth hooks are not executed. New packages belong in this environment.\n'
    'Do not upgrade or uninstall the source packages while using this environment.\n'
    'Versions and interpreter are recorded in expo-source.json.\n'
    'Use scripts/expo-preflight.py before initializing CUDA.\n'
    'This setup does not contact AWS or initialize CUDA.\n')
print(json.dumps(source_info, indent=2))
PY
PYTHONDONTWRITEBYTECODE=1 "$target_env/bin/python" -c \
    'import torch, torchvision; print("Shared Torch:", torch.__version__, "CUDA build:", torch.version.cuda); print("torchvision:", torchvision.__version__)'
