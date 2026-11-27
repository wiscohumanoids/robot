"""Register the clean-room Wisco AMP task."""
from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import wisco_amp_env_cfg
from .rl_cfg import wisco_amp_runner_cfg
from .runner import AMPOnPolicyRunner

TASK_ID = "Wisco-V1-AMP-Velocity-Flat"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=wisco_amp_env_cfg(),
  play_env_cfg=wisco_amp_env_cfg(play=True),
  rl_cfg=wisco_amp_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)
