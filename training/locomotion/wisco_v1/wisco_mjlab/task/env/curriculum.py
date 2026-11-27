"""Performance-gated curriculum for the Wisco arms-free recipe."""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from ...terms import arms_free_terms
from .recipe import PUSH_SPEED, TRACK_ANG_STD, TRACK_LIN_STD


def add_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add the difficulty level and its read-only info term."""
  curr_params = {
    "push_min": PUSH_SPEED[0], "push_max": PUSH_SPEED[1],
    "lin_std": TRACK_LIN_STD, "ang_std": TRACK_ANG_STD,
  }
  cfg.curriculum["level"] = CurriculumTermCfg(func=arms_free_terms.difficulty_level, params=curr_params)
  cfg.curriculum["level_info"] = CurriculumTermCfg(func=arms_free_terms.level_info, params=curr_params)
