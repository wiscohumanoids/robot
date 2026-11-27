"""Reference motion clips for AMP.

The files are the .npz format produced by training/tracking/motion_to_npz.py:

  fps             scalar
  joint_pos       (T, J)
  joint_vel       (T, J)
  body_pos_w      (T, B, 3)
  body_quat_w     (T, B, 4)   wxyz
  body_lin_vel_w  (T, B, 3)
  body_ang_vel_w  (T, B, 3)

Body and joint order are the training model's order, which is also the order
used by mjlab's entity data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MotionClip:
  """One retargeted reference clip, at the policy rate."""
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  body_pos_w: np.ndarray
  body_quat_w: np.ndarray
  body_lin_vel_w: np.ndarray
  body_ang_vel_w: np.ndarray
  fps: float = 50.0

  def __post_init__(self) -> None:
    jp, jv = self.joint_pos, self.joint_vel
    bp, bq = self.body_pos_w, self.body_quat_w
    bl, ba = self.body_lin_vel_w, self.body_ang_vel_w
    if jp.ndim != 2 or jv.shape != jp.shape:
      raise ValueError(f"joint arrays must be (T, J) and equal: {jp.shape}, {jv.shape}")
    if bp.ndim != 3 or bp.shape[-1] != 3:
      raise ValueError(f"body_pos_w must be (T, B, 3), got {bp.shape}")
    if bq.shape != (*bp.shape[:2], 4):
      raise ValueError(f"body_quat_w must be (T, B, 4), got {bq.shape}")
    if bl.shape != bp.shape or ba.shape != bp.shape:
      raise ValueError(f"body velocity shapes must match body_pos_w: {bl.shape}, {ba.shape}")
    if not np.isfinite(jp).all() or not np.isfinite(jv).all():
      raise ValueError("motion contains non-finite joint data")
    if not np.isfinite(bp).all() or not np.isfinite(bq).all():
      raise ValueError("motion contains non-finite body data")

  @property
  def num_frames(self) -> int:
    """Number of frames in the clip."""
    return int(self.joint_pos.shape[0])

  @property
  def num_joints(self) -> int:
    """Number of joints in the training model order."""
    return int(self.joint_pos.shape[1])

  @property
  def num_bodies(self) -> int:
    """Number of bodies in the training model order."""
    return int(self.body_pos_w.shape[1])


def load_clip(path: str | Path) -> MotionClip:
  """Load one .npz motion clip."""
  path = Path(path).expanduser()
  if not path.is_file():
    raise FileNotFoundError(path)
  with np.load(path, allow_pickle=False) as z:
    required = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    missing = [k for k in required if k not in z.files]
    if missing:
      raise ValueError(f"{path}: missing {missing}")
    fps = float(np.asarray(z["fps"]).reshape(-1)[0]) if "fps" in z.files else 50.0
    return MotionClip(
      joint_pos=np.asarray(z["joint_pos"], dtype=np.float32),
      joint_vel=np.asarray(z["joint_vel"], dtype=np.float32),
      body_pos_w=np.asarray(z["body_pos_w"], dtype=np.float32),
      body_quat_w=np.asarray(z["body_quat_w"], dtype=np.float32),
      body_lin_vel_w=np.asarray(z["body_lin_vel_w"], dtype=np.float32),
      body_ang_vel_w=np.asarray(z["body_ang_vel_w"], dtype=np.float32),
      fps=fps,
    )


def load_clips(paths: tuple[str, ...] | list[str]) -> list[MotionClip]:
  """Load and shape-check a list of clips against each other."""
  if not paths:
    raise ValueError("no AMP motion files given")
  clips = [load_clip(p) for p in paths]
  first = clips[0]
  for clip in clips[1:]:
    if clip.num_joints != first.num_joints or clip.num_bodies != first.num_bodies:
      raise ValueError(
        f"motion clips disagree on shape: {first.num_joints} joints/{first.num_bodies} bodies "
        f"vs {clip.num_joints}/{clip.num_bodies}")
  return clips
