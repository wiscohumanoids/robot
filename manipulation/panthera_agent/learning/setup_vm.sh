#!/usr/bin/env bash
# One-time set-up of panthera_agent's training environment on the GPU VM (Linux, NVIDIA): a uv virtualenv in
# ~/hsim_envs/panthera_agent with this folder's requirements (PyTorch's Linux wheels from PyPI bundle CUDA), then a
# check that the GPU, MuJoCo's headless (EGL) rendering and the DINOv3 download work.
#
#   bash manipulation/panthera_agent/learning/setup_vm.sh
#
# Leaves the other environments in ~/hsim_envs alone (the perception servers', the pi0.5 server's).
set -euo pipefail
unset LD_LIBRARY_PATH        # the system CUDA libraries would shadow PyTorch's own cuDNN (see learning/daily.sh)
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENV="${PANTHERA_AGENT_ENV:-$HOME/hsim_envs/panthera_agent}"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[[ -x "$UV" ]] || { echo "uv not found (expected in ~/.local/bin)" >&2; exit 1; }
[[ -d "$ENV" ]] || "$UV" venv --python 3.10 "$ENV"
"$UV" pip install --python "$ENV/bin/python" -r "$HERE/requirements.txt"
MUJOCO_GL=egl "$ENV/bin/python" - <<'PY'
import torch, mujoco
print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size="1"/></worldbody></mujoco>')
r = mujoco.Renderer(m, 64, 64); r.update_scene(mujoco.MjData(m)); r.render(); print('MuJoCo', mujoco.__version__, 'EGL rendering ok')
import sys; sys.path.insert(0, '.')
PY
cd "$HERE" && MUJOCO_GL=egl "$ENV/bin/python" -c "
import numpy as np
from perception.dino import shared
d = shared(); f = d(np.zeros((480, 640, 3), np.uint8)); print('DINOv3 on', d.device, 'features', f.shape)"
echo "ready: $ENV"
