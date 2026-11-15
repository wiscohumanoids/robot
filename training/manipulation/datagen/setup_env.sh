#!/usr/bin/env bash
# One-time (VM): Python env for manipulation datagen at ~/hsim_envs/datagen (MuJoCo 3.11.0 = the version that
# compiled the scene .mjb files on the Mac; a different MuJoCo version can't load them).
set -euo pipefail
UV=${UV:-$HOME/.local/bin/uv}
$UV venv --python 3.12 "$HOME/hsim_envs/datagen"
VIRTUAL_ENV="$HOME/hsim_envs/datagen" $UV pip install "mujoco==3.11.0" numpy pillow pyyaml
MUJOCO_GL=egl "$HOME/hsim_envs/datagen/bin/python" -c "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><light/><geom size=\".1\"/></worldbody></mujoco>'); r=mujoco.Renderer(m,64,64); d=mujoco.MjData(m); mujoco.mj_forward(m,d); r.update_scene(d); print('EGL render ok', r.render().shape, 'mujoco', mujoco.__version__); r.close()"
