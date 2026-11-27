"""Register the Wisco AMP task on amp-rsl-rl."""
from __future__ import annotations

from amp_rsl_rl.runners import AMPOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import wisco_amp_env_cfg
from .rl_cfg import wisco_amp_runner_cfg

TASK_ID = "Wisco-V1-AMP-Velocity-Flat"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=wisco_amp_env_cfg(),
  play_env_cfg=wisco_amp_env_cfg(play=True),
  rl_cfg=wisco_amp_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)
