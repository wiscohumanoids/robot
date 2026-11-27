"""Wisco V1 AMP velocity task config.

AMP is a training-time style reward; the actor observation and deployed policy
are the normal velocity task. This wrapper exists so the task registry can list
a distinct task ID.
"""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from wisco_mjlab.task.env_cfg import wisco_flat_env_cfg


def wisco_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The velocity environment; AMP is handled by the runner."""
  return wisco_flat_env_cfg(play=play)
