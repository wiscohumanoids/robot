"""Registers Wisco-V1-Tracking-Flat with mjlab's task registry (import wisco_mjlab first: robot config)."""
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfg import wisco_flat_tracking_env_cfg
from .rl_cfg import wisco_tracking_ppo_runner_cfg

TASK_ID = "Wisco-V1-Tracking-Flat"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=wisco_flat_tracking_env_cfg(),
  play_env_cfg=wisco_flat_tracking_env_cfg(play=True),
  rl_cfg=wisco_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
