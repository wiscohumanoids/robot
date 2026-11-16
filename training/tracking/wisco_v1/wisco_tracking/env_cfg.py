"""Wisco V1 motion tracking (BeyondMimic-style): a copy of mjlab 1.6.0's G1 flat tracking config
(tasks/tracking/config/g1/env_cfgs.py) with the robot swapped to Wisco. Lines that differ are marked "# WISCO:".

Whole body: all 29 joints are actions (head welded, as in the velocity task). Robot, actuators (DC motors with
torque-speed curve + 0-10 ms delay), gains and action scales come from wisco_mjlab.robot_cfg (helper A), so
tracking and velocity always share the same robot model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from wisco_mjlab import linkage_penalty
from wisco_mjlab import robot_cfg as RC

_INFO = json.loads((Path(RC.MODEL_DIR) / "train_model_info.json").read_text())
# Joint order of the retargeted CSV / motion npz = the training model's joint order.
WISCO_CSV_JOINTS = tuple(_INFO["policy_joints_in_model_order"])

ANCHOR_BODY = "torso"                                                       # WISCO: G1 "torso_link"
KEY_BODIES = (                                                              # WISCO: same 14 places as the G1 list
  "pelvis",
  "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
  "torso",
  "left_shoulder_roll_link", "left_elbow_link", "left_hand",                # WISCO: G1 left_wrist_yaw_link
  "right_shoulder_roll_link", "right_elbow_link", "right_hand",
)
END_EFFECTORS = ("left_ankle_roll_link", "right_ankle_roll_link", "left_hand", "right_hand")

MASS_SCALE = (0.85, 1.20)                                                   # WISCO: same as the velocity task (H1)
ALPHA_RANGE = (math.log(MASS_SCALE[0]) / 2, math.log(MASS_SCALE[1]) / 2)


def wisco_flat_tracking_env_cfg(has_state_estimation: bool = True, play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": RC.get_wisco_robot_cfg()}                  # WISCO

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = RC.ACTION_SCALE                                  # WISCO: all 29 joints

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = ANCHOR_BODY
  motion_cmd.body_names = KEY_BODIES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot_collision$"   # WISCO
  cfg.events["base_com"].params["asset_cfg"].body_names = (ANCHOR_BODY,)

  # WISCO: same model-uncertainty randomization as the Wisco velocity task (masses H1, motors H6/H10)
  cfg.events["mass_inertia"] = EventTermCfg(
    mode="startup", func=dr.pseudo_inertia,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(".*",)), "alpha_range": ALPHA_RANGE})

  def _joint_dr(func, names, ranges):
    return EventTermCfg(mode="startup", func=func, params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=names), "ranges": ranges, "operation": "scale"})
  cfg.events["armature_x8"] = _joint_dr(dr.joint_armature, RC.X8_JOINTS, RC.X8_ARMATURE_SCALE)
  cfg.events["armature_other"] = _joint_dr(dr.joint_armature, RC.OTHER_JOINTS, RC.OTHER_ARMATURE_SCALE)
  cfg.events["joint_friction"] = _joint_dr(dr.joint_friction, (".*",), RC.FRICTION_SCALE)
  cfg.events["joint_damping"] = _joint_dr(dr.joint_damping, (".*",), RC.DAMPING_SCALE)

  # WISCO: ankle linkage motor overload penalty (H3), same as the velocity task
  cfg.rewards["ankle_motor_overload"] = RewardTermCfg(
    func=linkage_penalty.ankle_motor_overload, weight=-1.0,
    params={"motor_limit": RC.P.X4_36_PEAK_NM})

  cfg.terminations["ee_body_pos"].params["body_names"] = END_EFFECTORS      # WISCO
  cfg.viewer.body_name = ANCHOR_BODY

  if not has_state_estimation:
    new_actor_terms = {k: v for k, v in cfg.observations["actor"].terms.items()
                       if k not in ["motion_anchor_pos_b", "base_lin_vel"]}
    cfg.observations["actor"] = ObservationGroupCfg(terms=new_actor_terms, concatenate_terms=True,
                                                    enable_corruption=True)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
