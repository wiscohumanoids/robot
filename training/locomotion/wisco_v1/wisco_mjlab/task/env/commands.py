"""Velocity command sampling for the Wisco arms-free recipe."""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from ...terms import arms_free_terms
from .recipe import REL_YAW_ONLY, YAW_RANGE


def configure_commands(cfg: ManagerBasedRlEnvCfg) -> arms_free_terms.YawInPlaceVelocityCommandCfg:
  """Use the G1 velocity command plus pure turn-in-place samples and the full yaw range."""
  base_twist = cfg.commands["twist"]
  assert isinstance(base_twist, UniformVelocityCommandCfg)
  twist_cmd = arms_free_terms.YawInPlaceVelocityCommandCfg(
    entity_name=base_twist.entity_name,
    resampling_time_range=base_twist.resampling_time_range,
    rel_standing_envs=base_twist.rel_standing_envs,
    rel_heading_envs=base_twist.rel_heading_envs,
    rel_forward_envs=base_twist.rel_forward_envs,
    heading_command=base_twist.heading_command,
    heading_control_stiffness=base_twist.heading_control_stiffness,
    debug_vis=base_twist.debug_vis,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=base_twist.ranges.lin_vel_x, lin_vel_y=base_twist.ranges.lin_vel_y,
      ang_vel_z=YAW_RANGE, heading=base_twist.ranges.heading),
    rel_yaw_only_envs=REL_YAW_ONLY,
    yaw_only_range=(0.3, abs(YAW_RANGE[1])),
  )
  twist_cmd.viz.z_offset = 1.15
  cfg.commands["twist"] = twist_cmd

  # ARMS-FREE: the G1 velocity curriculum widens yaw only to 0.7; keep +-1.5 at every stage.
  for stage in cfg.curriculum["command_vel"].params["velocity_stages"]:
    stage["ang_vel_z"] = YAW_RANGE
    if "lin_vel_x" in stage:   # cap forward command at 1.5 m/s
      lo, hi = stage["lin_vel_x"]
      stage["lin_vel_x"] = (max(lo, -1.0), min(hi, 1.5))
  return twist_cmd
