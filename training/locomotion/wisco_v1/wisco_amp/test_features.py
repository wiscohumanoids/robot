"""AMP feature tests, skipped when mjlab/torch is unavailable."""
import numpy as np
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from wisco_amp.features import FeatureBuilder, FeatureNormalizer, FeatureSpec, reference_pairs  # noqa: E402
from wisco_amp.motion import MotionClip  # noqa: E402


def _clip(frames=6):
  q = np.linspace(0.0, 0.2, frames * 2, dtype=np.float32).reshape(frames, 2)
  zeros_b = np.zeros((frames, 3, 3), np.float32)
  quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (frames, 3, 1))
  return MotionClip(
    joint_pos=q, joint_vel=np.zeros_like(q), body_pos_w=zeros_b, body_quat_w=quat,
    body_lin_vel_w=np.zeros_like(zeros_b), body_ang_vel_w=np.zeros_like(zeros_b))


def test_feature_shape_and_pairs():
  spec = FeatureSpec(num_joints=2, num_bodies=3, root_index=0)
  builder = FeatureBuilder(spec)
  clip = _clip()
  feat = builder.from_motion(clip)
  assert feat.shape == (clip.num_frames, spec.dim)
  normalizer = FeatureNormalizer.fit(feat)
  normalized = normalizer.normalize_np(feat)
  assert normalized.shape == feat.shape
  assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-5)
  pairs = reference_pairs([clip], builder, normalizer)
  assert pairs.shape == (clip.num_frames - 1, 2 * spec.dim)


def test_normalizer_round_trip():
  norm = FeatureNormalizer(mean=np.array([1.0, 2.0]), std=np.array([2.0, 4.0]))
  x = torch.tensor([[3.0, 10.0]])
  y = norm.normalize_torch(x)
  assert torch.allclose(y, torch.tensor([[1.0, 2.0]]))
  state = norm.state_dict()
  norm2 = FeatureNormalizer(mean=np.zeros(2), std=np.ones(2))
  norm2.load_state_dict(state)
  assert np.allclose(norm2.mean, norm.mean)
