"""hsim training tasks for mjlab 1.6.0 (registered on import).

  Hsim-G1-Velocity-Flat-v2   G1 velocity v2: legs+waist policy, scripted arms, v2 recipe
                             (docs/runs/2026-11-10_overnight_plan.md)

mjlab's own `train` / `play` only import tasks inside the mjlab package, so use the launchers
next to this file (train.py, play.py): they import this package, then run mjlab's unchanged CLI.
"""
from mjlab.tasks.registry import register_mjlab_task

from .g1_velocity_v2 import g1_velocity_v2_env_cfg
from .rl_cfg import g1_velocity_v2_ppo_runner_cfg
from .runner import SubsetVelocityRunner

G1_VELOCITY_V2 = "Hsim-G1-Velocity-Flat-v2"

register_mjlab_task(
  task_id=G1_VELOCITY_V2,
  env_cfg=g1_velocity_v2_env_cfg(),
  play_env_cfg=g1_velocity_v2_env_cfg(play=True),
  rl_cfg=g1_velocity_v2_ppo_runner_cfg(),
  runner_cls=SubsetVelocityRunner,
)
