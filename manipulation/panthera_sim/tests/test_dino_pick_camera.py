"""Compare the moving camera's calibration with RGB pixel positions."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dino_pick_camera import read_camera


def test_mounted_camera_projects_rgb_pixels_after_arm_motion():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <visual><global offwidth="320" offheight="240"/></visual>
      <asset>
        <material name="red" rgba="1 0 0 1" emission="1"/>
      </asset>
      <worldbody>
        <body pos="0 0 3">
          <joint type="slide" axis="1 0 0"/>
          <joint type="hinge" axis="0 1 0"/>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <camera name="wrist" fovy="45"/>
        </body>
        <geom type="sphere" pos=".32 .25 0" size=".045" material="red"/>
      </worldbody>
    </mujoco>
    """)
    sim = SimpleNamespace(data=mujoco.MjData(model))
    point = np.array([.32, .25, 0.])
    pixels = []
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        for qpos in ([0., 0.], [.2, -.1]):
            sim.data.qpos[:] = qpos
            mujoco.mj_forward(model, sim.data)
            frame, camera = read_camera(sim, renderer, "wrist")
            assert frame.shape == (240, 320, 3)
            assert frame.dtype == np.uint8
            rows, columns = np.nonzero((frame[..., 0] > 150) & (frame[..., 1] < 50))
            assert columns.size > 20
            xy = camera.project(point)
            np.testing.assert_allclose(xy, [columns.mean(), rows.mean()], atol=.45)
            np.testing.assert_allclose(camera.on_plane(xy, 0.), point, atol=1e-8)
            pixels.append(xy)
    assert np.linalg.norm(pixels[0] - pixels[1]) > 10
    with pytest.raises(ValueError, match="in front"):
        camera.on_plane(pixels[-1], 4.)
    with pytest.raises(ValueError, match="in front"):
        camera.project([0., 0., 4.])
