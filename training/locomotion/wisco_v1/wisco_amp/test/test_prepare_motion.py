"""Tests for the amp-rsl-rl motion converter (no simulator needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("amp_rsl_rl")

# wisco_amp lives under training/locomotion/wisco_v1; the test runs from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wisco_amp.prepare_motion import JOINTS, convert  # noqa: E402


def test_convert_npz_to_amp_npy(tmp_path: Path):
  frames = 4
  joints = len(JOINTS)
  npz = tmp_path / "walk.npz"
  np.savez(
    npz,
    fps=np.array([50.0]),
    joint_pos=np.zeros((frames, joints), np.float32),
    joint_vel=np.zeros((frames, joints), np.float32),
    body_pos_w=np.zeros((frames, 1, 3), np.float32),
    body_quat_w=np.tile(np.array([1, 0, 0, 0], np.float32), (frames, 1, 1)),
    body_lin_vel_w=np.zeros((frames, 1, 3), np.float32),
    body_ang_vel_w=np.zeros((frames, 1, 3), np.float32),
  )
  out = tmp_path / "walk.npy"
  convert(npz, out)
  data = np.load(out, allow_pickle=True).item()
  assert data["joints_list"] == JOINTS
  assert len(data["joint_positions"]) == frames
  assert len(data["root_position"]) == frames
  assert len(data["root_quaternion"]) == frames
  assert data["fps"] == 50.0
  # wxyz [1,0,0,0] -> xyzw [0,0,0,1]
  assert np.allclose(data["root_quaternion"][0], [0.0, 0.0, 0.0, 1.0])
