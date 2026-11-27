"""AMP feature vectors for the simulated robot and reference clips.

The feature vector is deliberately training-only: the actor observation and the
deployed policy do not change. It contains the quantities that describe gait
style and are available in both the motion data and the simulated robot:

  joint positions, joint velocities,
  root linear/angular velocity in body frame,
  projected gravity,
  every body position relative to the root, in root frame.

Using every body avoids a separate hard-coded key-body list. The body order is
the training model order, which is also the order of body_pos_w in the .npz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .motion import MotionClip


def quat_to_matrix_np(q: np.ndarray) -> np.ndarray:
  """Rotation matrix from wxyz quaternions, shape (..., 4) -> (..., 3, 3)."""
  q = np.asarray(q, dtype=np.float32)
  q = q / np.linalg.norm(q, axis=-1, keepdims=True)
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.stack((
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ), axis=-1).reshape(q.shape[:-1] + (3, 3))


def rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vectors from world into the frame of wxyz quaternion q."""
  r = quat_to_matrix_np(q)
  return np.einsum("...ji,...j->...i", r, v)


def quat_to_matrix_torch(q: torch.Tensor) -> torch.Tensor:
  """Rotation matrix from wxyz quaternions, shape (..., 4) -> (..., 3, 3)."""
  q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return torch.stack((
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ), dim=-1).reshape(q.shape[:-1] + (3, 3))


def rotate_inverse_torch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate vectors from world into the frame of wxyz quaternion q."""
  r = quat_to_matrix_torch(q)
  return torch.einsum("...ji,...j->...i", r, v)


@dataclass(frozen=True)
class FeatureSpec:
  """Shape and root body of the AMP features."""
  num_joints: int
  num_bodies: int
  root_index: int = 0

  @property
  def dim(self) -> int:
    """Feature dimension: 2*J + 3 root velocities + 3 gravity + 3*B body positions."""
    return 2 * self.num_joints + 9 + 3 * self.num_bodies

  @classmethod
  def from_env(cls, env: Any) -> "FeatureSpec":
    """Read the joint/body counts from a ManagerBasedRlEnv."""
    robot = env.scene["robot"]
    data = robot.data
    num_joints = int(data.joint_pos.shape[1])
    num_bodies = int(data.body_link_pos_w.shape[1])
    names = tuple(robot.body_names)
    root_index = names.index("pelvis") if "pelvis" in names else 0
    return cls(num_joints=num_joints, num_bodies=num_bodies, root_index=root_index)


class FeatureBuilder:
  """Build AMP features from a simulated env or a reference clip."""

  def __init__(self, spec: FeatureSpec):
    """Store the feature shape."""
    self.spec = spec

  def from_env(self, env: Any) -> torch.Tensor:
    """Features for every environment, shape (N, F)."""
    data = env.scene["robot"].data
    rel = data.body_link_pos_w - data.root_link_pos_w[:, None, :]
    rel_b = rotate_inverse_torch(data.root_link_quat_w, rel)
    gravity = data.projected_gravity_b
    return torch.cat((
      data.joint_pos,
      data.joint_vel,
      data.root_link_lin_vel_b,
      data.root_link_ang_vel_b,
      gravity,
      rel_b.reshape(rel_b.shape[0], -1),
    ), dim=-1).float()

  def from_motion(self, clip: MotionClip) -> np.ndarray:
    """Features for one reference clip, shape (T, F)."""
    if clip.num_joints != self.spec.num_joints or clip.num_bodies != self.spec.num_bodies:
      raise ValueError(
        f"clip shape {clip.num_joints}/{clip.num_bodies} does not match "
        f"env {self.spec.num_joints}/{self.spec.num_bodies}")
    root = self.spec.root_index
    r = quat_to_matrix_np(clip.body_quat_w[:, root])
    rel = clip.body_pos_w - clip.body_pos_w[:, root:root + 1, :]
    rel_b = np.einsum("tji,tbj->tbi", r, rel)
    lin_b = np.einsum("tji,tj->ti", r, clip.body_lin_vel_w[:, root])
    ang_b = np.einsum("tji,tj->ti", r, clip.body_ang_vel_w[:, root])
    gravity = np.einsum("tji,j->ti", r, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    return np.concatenate((
      clip.joint_pos,
      clip.joint_vel,
      lin_b,
      ang_b,
      gravity,
      rel_b.reshape(rel_b.shape[0], -1),
    ), axis=1).astype(np.float32)


class FeatureNormalizer:
  """Per-feature mean/std fitted on reference motion."""

  def __init__(self, mean: np.ndarray, std: np.ndarray):
    """Store mean/std as float32 arrays."""
    self.mean = np.asarray(mean, dtype=np.float32)
    self.std = np.asarray(std, dtype=np.float32)

  @classmethod
  def fit(cls, features: np.ndarray) -> "FeatureNormalizer":
    """Fit on a (T, F) reference array."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    return cls(mean, np.maximum(std, 1e-4))

  def normalize_np(self, x: np.ndarray) -> np.ndarray:
    """Normalize a numpy feature array."""
    return ((x - self.mean) / self.std).astype(np.float32)

  def normalize_torch(self, x: torch.Tensor) -> torch.Tensor:
    """Normalize a torch feature tensor."""
    mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
    std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
    return (x - mean) / std

  def state_dict(self) -> dict[str, list[float]]:
    """Serializable checkpoint state."""
    return {"mean": self.mean.tolist(), "std": self.std.tolist()}

  def load_state_dict(self, state: dict[str, list[float]]) -> None:
    """Restore checkpoint state."""
    self.mean = np.asarray(state["mean"], dtype=np.float32)
    self.std = np.asarray(state["std"], dtype=np.float32)


def reference_pairs(clips: list[MotionClip], builder: FeatureBuilder,
                    normalizer: FeatureNormalizer) -> torch.Tensor:
  """Concatenated (s_t, s_t+1) reference transitions, shape (N, 2F)."""
  pairs = []
  for clip in clips:
    feat = normalizer.normalize_np(builder.from_motion(clip))
    if len(feat) < 2:
      raise ValueError(f"motion clip has {len(feat)} frames; need at least 2")
    pairs.append(np.concatenate((feat[:-1], feat[1:]), axis=1))
  return torch.from_numpy(np.concatenate(pairs, axis=0)).float()
