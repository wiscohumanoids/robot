"""hsim training tasks for mjlab 1.6.0 (registered on import).

  Hsim-G1-Velocity-Flat-ArmsFree   G1 velocity arms-free: legs+waist policy, scripted arms, arms-free recipe
                             (docs/runs/2026-11-10_overnight_plan.md)

mjlab's own `train` / `play` only import tasks inside the mjlab package, so use the launchers
next to this file (train.py, play.py): they import this package, then run mjlab's unchanged CLI.
"""
from mjlab.tasks.registry import register_mjlab_task

from .g1_velocity_arms_free import g1_velocity_arms_free_env_cfg
from .rl_cfg import g1_velocity_arms_free_ppo_runner_cfg
from .runner import SubsetVelocityRunner

G1_VELOCITY_ARMS_FREE = "Hsim-G1-Velocity-Flat-ArmsFree"

register_mjlab_task(
  task_id=G1_VELOCITY_ARMS_FREE,
  env_cfg=g1_velocity_arms_free_env_cfg(),
  play_env_cfg=g1_velocity_arms_free_env_cfg(play=True),
  rl_cfg=g1_velocity_arms_free_ppo_runner_cfg(),
  runner_cls=SubsetVelocityRunner,
)
