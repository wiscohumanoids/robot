"""Pure motion-loader tests; mjlab is not needed, but the package import is guarded."""
import numpy as np
import pytest

pytest.importorskip("mjlab")

from wisco_amp.motion import load_clip  # noqa: E402


def _write_clip(path, frames=5, joints=2, bodies=3):
  t = np.arange(frames, dtype=np.float32)
  np.savez(
    path,
    fps=np.array([50.0]),
    joint_pos=np.zeros((frames, joints), np.float32),
    joint_vel=np.ones((frames, joints), np.float32) * 0.1,
    body_pos_w=np.zeros((frames, bodies, 3), np.float32),
    body_quat_w=np.tile(np.array([1, 0, 0, 0], np.float32), (frames, bodies, 1)),
    body_lin_vel_w=np.zeros((frames, bodies, 3), np.float32),
    body_ang_vel_w=np.zeros((frames, bodies, 3), np.float32),
  )
  return t


def test_load_clip_shapes(tmp_path):
  path = tmp_path / "clip.npz"
  _write_clip(path)
  clip = load_clip(path)
  assert clip.num_frames == 5
  assert clip.num_joints == 2
  assert clip.num_bodies == 3
  assert clip.fps == 50.0


def test_load_clip_rejects_bad_shapes(tmp_path):
  path = tmp_path / "bad.npz"
  np.savez(
    path,
    fps=np.array([50.0]),
    joint_pos=np.zeros((5, 2), np.float32),
    joint_vel=np.zeros((4, 2), np.float32),
    body_pos_w=np.zeros((5, 3, 3), np.float32),
    body_quat_w=np.zeros((5, 3, 4), np.float32),
    body_lin_vel_w=np.zeros((5, 3, 3), np.float32),
    body_ang_vel_w=np.zeros((5, 3, 3), np.float32),
  )
  with pytest.raises(ValueError, match="joint arrays"):
    load_clip(path)
