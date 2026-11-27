"""Wisco V1 velocity task with the AMP observation group."""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from wisco_mjlab.env_cfg import wisco_flat_env_cfg

from .amp_obs import amp_features


def wisco_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The normal velocity task plus an ``amp`` observation group."""
  cfg = wisco_flat_env_cfg(play=play)
  cfg.observations["amp"] = ObservationGroupCfg(
    terms={
      "amp": ObservationTermCfg(
        func=amp_features,
        params={"asset_cfg": SceneEntityCfg("robot")},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  return cfg
