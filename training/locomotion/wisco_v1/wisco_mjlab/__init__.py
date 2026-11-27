"""Registers the Wisco V1 walking task with mjlab's task registry.

mjlab's own `train` / `play` commands only know tasks inside the mjlab package, so the
launchers next to this folder (train.py, play.py) import this package first, then hand
over to mjlab's unchanged train / play entry points.
"""
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .task.env_cfg import wisco_flat_env_cfg
from .task.rl_cfg import wisco_ppo_runner_cfg

TASK_ID = "Wisco-V1-Velocity-Flat"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=wisco_flat_env_cfg(),
  play_env_cfg=wisco_flat_env_cfg(play=True),
  rl_cfg=wisco_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
