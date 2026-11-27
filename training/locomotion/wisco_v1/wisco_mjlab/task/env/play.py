"""Play/evaluation overrides for the Wisco velocity task."""
from __future__ import annotations

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from ...terms import arms_free_terms
from .recipe import PUSH_SPEED, TRACK_ANG_STD, TRACK_LIN_STD, YAW_RANGE


def configure_play(cfg: ManagerBasedRlEnvCfg, twist_cmd) -> None:
  """Freeze the curriculum at a fixed level and open the command range for evaluation."""
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.curriculum = {}
  # ARMS-FREE: default full difficulty; set WISCO_PLAY_LEVEL / WISCO_PLAY_LEVEL2 to mirror a training level.
  cfg.curriculum["level"] = CurriculumTermCfg(func=arms_free_terms.fixed_level, params={
    "L": float(os.environ.get("WISCO_PLAY_LEVEL", "1.0")),
    "L2": float(os.environ.get("WISCO_PLAY_LEVEL2", "1.0")),
    "push_min": PUSH_SPEED[0], "push_max": PUSH_SPEED[1],
    "lin_std": TRACK_LIN_STD, "ang_std": TRACK_ANG_STD})
  twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
  twist_cmd.ranges.ang_vel_z = YAW_RANGE
