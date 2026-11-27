"""AMP observation group for amp-rsl-rl.

The library expects an observation group named ``amp`` containing, in order:

  joint positions, joint velocities, base linear velocity (body), base angular velocity (body)

The expert loader builds the same vector from the motion dataset.
"""
from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def amp_features(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """The AMP observation vector for every environment."""
  data = env.scene[asset_cfg.name].data
  return torch.cat((
    data.joint_pos,
    data.joint_vel,
    data.root_link_lin_vel_b,
    data.root_link_ang_vel_b,
  ), dim=-1).float()
