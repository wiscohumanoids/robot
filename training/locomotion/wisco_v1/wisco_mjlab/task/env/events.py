"""Domain randomization, scripted arms and pushes for the Wisco velocity task."""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...robot import robot_cfg as RC
from ...robot.robot_cfg import P
from ...terms import arms_free_terms
from .recipe import ALPHA_RANGE, HAND_PAYLOAD_MAX_KG, PUSH_SPEED


def add_events(cfg: ManagerBasedRlEnvCfg) -> None:
  """Foot/torso event targets, startup randomization, scripted arms and pushes."""
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = P.FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = (P.TORSO_BODY,)        # WISCO

  # WISCO: per-part mass/inertia randomization at startup (masses are estimates).
  cfg.events["mass_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(".*",)), "alpha_range": ALPHA_RANGE},
  )

  # WISCO: motor parameter randomization around the datasheet values.
  def _joint_dr(func, names, ranges):
    return EventTermCfg(mode="startup", func=func, params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=names), "ranges": ranges, "operation": "scale"})
  cfg.events["armature_x8"] = _joint_dr(dr.joint_armature, RC.X8_JOINTS, RC.X8_ARMATURE_SCALE)
  cfg.events["armature_other"] = _joint_dr(dr.joint_armature, RC.OTHER_JOINTS, RC.OTHER_ARMATURE_SCALE)
  cfg.events["joint_friction"] = _joint_dr(dr.joint_friction, (".*",), RC.FRICTION_SCALE)
  cfg.events["joint_damping"] = _joint_dr(dr.joint_damping, (".*",), RC.DAMPING_SCALE)

  # ARMS-FREE: arms are not policy actions: moving targets (controls-stack stand-in) + payload in the hands.
  cfg.events["arm_targets_reset"] = EventTermCfg(
    mode="reset", func=arms_free_terms.arm_targets_reset,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=RC.ARM_JOINT_NAMES), "fracs": RC.ARM_TARGET_FRACS})
  cfg.events["arm_targets"] = EventTermCfg(
    mode="step", func=arms_free_terms.arm_targets_step,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=RC.ARM_JOINT_NAMES), "hold_prob": 0.3,
            "fracs": RC.ARM_TARGET_FRACS, "period_s": (1.0, 3.0), "speed_range": (0.3, 2.5)})
  # Hand payload resampled for all envs every 10-20 s (mass changes recompute model constants).
  cfg.events["hand_payload"] = EventTermCfg(
    mode="interval", interval_range_s=(10.0, 20.0), is_global_time=True, func=arms_free_terms.hand_payload,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=RC.HAND_BODIES), "max_kg": HAND_PAYLOAD_MAX_KG})
  # ARMS-FREE: kp/kd +/-10 % on every actuator.
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup", func=dr.pd_gains,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)), "kp_range": RC.PD_GAIN_SCALE,
            "kd_range": RC.PD_GAIN_SCALE, "operation": "scale"})
  # ARMS-FREE: pushes start at the G1 default and grow with the curriculum level.
  push = cfg.events["push_robot"].params["velocity_range"]
  push["x"], push["y"] = (-PUSH_SPEED[0], PUSH_SPEED[0]), (-PUSH_SPEED[0], PUSH_SPEED[0])
