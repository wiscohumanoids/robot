"""Wisco V1 velocity (walking) task, arms-free recipe: a copy of mjlab 1.6.0's Unitree G1 flat velocity config
(tasks/velocity/config/g1/env_cfgs.py, unitree_g1_rough_env_cfg + unitree_g1_flat_env_cfg) with the robot swapped
(lines marked "# WISCO:") and the arms-free locomotion recipe of docs/runs/2026-11-10_overnight_plan.md (marked "# ARMS-FREE:"):
  1 legs + waist only (15 actions); 2 arms position-held with random moving targets; 3 0-1 kg hand payload;
  4 tracking std 0.25 / 0.35; 5 >= 15 % turn-in-place commands, yaw range +-1.5; 6 command delay 0-10 ms
  (robot_cfg.py); 7 pushes 0.8 m/s; 8 motor DR (armature, kp/kd, friction, damping) + mass DR.
Difficulty curriculum (coordinator fix after J3 run 1 learned to stand still; same design as the G1 arms-free task):
items 2, 3, 7 scale with a performance-gated level L (0 -> 1) and item 4 starts at the G1 defaults (0.5 / 0.707) and
tightens with a second level L2 after L = 1. See arms_free_terms.py. Mass and motor DR are on from the start.
"""
from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from . import linkage_penalty
from . import robot_cfg as RC
from . import arms_free_terms
from .robot_cfg import P, get_wisco_robot_cfg

# WISCO: mass uncertainty (HARDWARE_ISSUES H1). Each part's mass and inertia scaled together by a
# factor in [0.85, 1.20] (skewed up: the CAD masses leave out wiring). pseudo_inertia scales by e^(2*alpha).
MASS_SCALE = (0.85, 1.20)
ALPHA_RANGE = (math.log(MASS_SCALE[0]) / 2, math.log(MASS_SCALE[1]) / 2)

# WISCO: penalty weight for ankle motors above their peak torque (HARDWARE_ISSUES H3).
ANKLE_MOTOR_OVERLOAD_WEIGHT = -1.0
ANKLE_THERMAL_WEIGHT = -1.0          # fix 4: sustained ankle motor torque above rated (see below)

# ARMS-FREE: posture reward on the policy's joints only (legs + waist); arms are moved by the controls-stack stand-in.
POSE_STD_WALKING = {
  # Lower body (same values as the G1).
  r".*hip_pitch.*": 0.3,
  r".*hip_roll.*": 0.15,
  r".*hip_yaw.*": 0.15,
  r".*knee.*": 0.35,
  r".*ankle_pitch.*": 0.25,
  r".*ankle_roll.*": 0.1,
  # Waist (WISCO: our joints are torso_*, the G1's waist_*).
  r".*torso_yaw.*": 0.2,
  r".*torso_roll.*": 0.08,
  r".*torso_pitch.*": 0.1,
}
POSE_STD_RUNNING = {
  r".*hip_pitch.*": 0.5,
  r".*hip_roll.*": 0.2,
  r".*hip_yaw.*": 0.2,
  r".*knee.*": 0.6,
  r".*ankle_pitch.*": 0.35,
  r".*ankle_roll.*": 0.15,
  r".*torso_yaw.*": 0.3,
  r".*torso_roll.*": 0.08,
  r".*torso_pitch.*": 0.2,
}

# ARMS-FREE recipe numbers
TRACK_LIN_STD = (0.5, 0.25)          # G1 default at L2 = 0 -> arms-free target at L2 = 1 (decision 0004 deadband fix)
TRACK_ANG_STD = (0.5 ** 0.5, 0.35)   # 0.707 -> 0.35 rad/s
YAW_RANGE = (-1.5, 1.5)
REL_YAW_ONLY = 0.18           # x (1 - 10 % standing) = 16 % of samples are pure turns
PUSH_SPEED = (0.5, 0.8)              # push velocity range +-v, v from 0.5 (G1 default, L = 0) to 0.8 (L = 1)
HAND_PAYLOAD_MAX_KG = 1.0            # max added mass per hand at L = 1
# arm target ranges at L = 1: robot_cfg.ARM_TARGET_FRACS (kept away from the body; fix 2)


def wisco_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()

  # --- from unitree_g1_rough_env_cfg -------------------------------------------------------
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_wisco_robot_cfg()}                       # WISCO

  site_names = P.FOOT_SITES                                                     # WISCO: left_foot / right_foot
  geom_names = P.FOOT_GEOMS                                                     # WISCO: one box per foot

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",                # same body names as the G1
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # ARMS-FREE fix 2: self-collision counts only contacts where neither body is an arm link. The arms are moved by the scripted
  # controls-stack stand-in, so the policy can't avoid arm contacts (the G1 run's self_collisions went to -0.70 at full
  # difficulty). A contact sensor takes one secondary element, so the non-arm pairs are covered by one sensor per group:
  # left leg vs the whole right leg, and both legs vs the pelvis and each torso link (torso_yaw_link, torso_roll_link,
  # torso). Arm links appear in none of them. Head/torso pairs are adjacent (excluded by MuJoCo) or only touch after a fall.
  LEFT_LEG = r"^left_(hip_pitch|hip_roll|hip_yaw|knee|ankle_pitch|ankle_roll)_link$"
  LEGS = r"^(left|right)_(hip_pitch|hip_roll|hip_yaw|knee|ankle_pitch|ankle_roll)_link$"
  def _self_col(name, primary, secondary_mode, secondary):
    return ContactSensorCfg(
      name=name, primary=ContactMatch(mode="body", pattern=primary, entity="robot"),
      secondary=ContactMatch(mode=secondary_mode, pattern=secondary, entity="robot"),
      fields=("found", "force"), reduce="none", num_slots=1, history_length=4)
  self_col_sensors = (
    _self_col("selfcol_leg_leg", LEFT_LEG, "subtree", "right_hip_pitch_link"),
    _self_col("selfcol_legs_pelvis", LEGS, "body", "pelvis"),
    _self_col("selfcol_legs_torso_yaw", LEGS, "body", "torso_yaw_link"),
    _self_col("selfcol_legs_torso_roll", LEGS, "body", "torso_roll_link"),
    _self_col("selfcol_legs_torso", LEGS, "body", "torso"),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,) + self_col_sensors

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = RC.LOCOMOTION_ACTION_NAMES                   # ARMS-FREE: legs + waist only (15)
  joint_pos_action.preserve_order = True
  joint_pos_action.scale = RC.LOCOMOTION_ACTION_SCALE                            # WISCO

  cfg.viewer.body_name = P.TORSO_BODY                                            # WISCO: "torso"

  base_twist = cfg.commands["twist"]
  assert isinstance(base_twist, UniformVelocityCommandCfg)
  # ARMS-FREE: same command as the G1 recipe + a share of pure turns in place, wider yaw range
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
  # ARMS-FREE: the G1 velocity curriculum widens yaw only to 0.7; keep +-1.5 at every stage
  for stage in cfg.curriculum["command_vel"].params["velocity_stages"]:
    stage["ang_vel_z"] = YAW_RANGE
    if "lin_vel_x" in stage:   # J3i: cap forward command at 1.5 m/s (stock stages widen to 2 then 3 m/s at 5k / 10k iterations)
      lo, hi = stage["lin_vel_x"]
      stage["lin_vel_x"] = (max(lo, -1.0), min(hi, 1.5))

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = (P.TORSO_BODY,)        # WISCO

  # WISCO: per-part mass/inertia randomization at startup (H1)
  cfg.events["mass_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(".*",)), "alpha_range": ALPHA_RANGE},
  )

  # WISCO: motor parameter randomization around the datasheet values (see robot_cfg.py, model/robot_numbers.yaml)
  def _joint_dr(func, names, ranges):
    return EventTermCfg(mode="startup", func=func, params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=names), "ranges": ranges, "operation": "scale"})
  cfg.events["armature_x8"] = _joint_dr(dr.joint_armature, RC.X8_JOINTS, RC.X8_ARMATURE_SCALE)
  cfg.events["armature_other"] = _joint_dr(dr.joint_armature, RC.OTHER_JOINTS, RC.OTHER_ARMATURE_SCALE)
  cfg.events["joint_friction"] = _joint_dr(dr.joint_friction, (".*",), RC.FRICTION_SCALE)
  cfg.events["joint_damping"] = _joint_dr(dr.joint_damping, (".*",), RC.DAMPING_SCALE)

  # ARMS-FREE: arms are not policy actions: moving targets (controls-stack stand-in) + payload in the hands
  cfg.events["arm_targets_reset"] = EventTermCfg(
    mode="reset", func=arms_free_terms.arm_targets_reset,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=RC.ARM_JOINT_NAMES), "fracs": RC.ARM_TARGET_FRACS})
  cfg.events["arm_targets"] = EventTermCfg(
    mode="step", func=arms_free_terms.arm_targets_step,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=RC.ARM_JOINT_NAMES), "hold_prob": 0.3,
            "fracs": RC.ARM_TARGET_FRACS, "period_s": (1.0, 3.0), "speed_range": (0.3, 2.5)})
  # hand payload resampled for all envs every 10-20 s (mass changes recompute model constants, so not per reset)
  cfg.events["hand_payload"] = EventTermCfg(
    mode="interval", interval_range_s=(10.0, 20.0), is_global_time=True, func=arms_free_terms.hand_payload,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=RC.HAND_BODIES), "max_kg": HAND_PAYLOAD_MAX_KG})
  # ARMS-FREE: kp/kd +/-10 % on every actuator
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup", func=dr.pd_gains,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)), "kp_range": RC.PD_GAIN_SCALE,
            "kd_range": RC.PD_GAIN_SCALE, "operation": "scale"})
  # ARMS-FREE: pushes start at the G1 default and grow with L (curriculum below)
  push = cfg.events["push_robot"].params["velocity_range"]
  push["x"], push["y"] = (-PUSH_SPEED[0], PUSH_SPEED[0]), (-PUSH_SPEED[0], PUSH_SPEED[0])

  # ARMS-FREE: velocity tracking starts at the G1 defaults and tightens with L2 (curriculum below)
  cfg.rewards["track_linear_velocity"].params["std"] = TRACK_LIN_STD[0]
  cfg.rewards["track_angular_velocity"].params["std"] = TRACK_ANG_STD[0]
  cfg.rewards["track_angular_velocity"].func = arms_free_terms.track_yaw_rate   # J3g: yaw rate only (no roll/pitch in the error)
  # J3f: fixed-std copies for the curriculum gate (weight 1e-3: ~no effect on training)
  cfg.rewards["gate_track_lin"] = RewardTermCfg(func=mdp.track_linear_velocity, weight=1e-3,
                                                params={**cfg.rewards["track_linear_velocity"].params, "std": TRACK_LIN_STD[0]})
  cfg.rewards["gate_track_ang"] = RewardTermCfg(func=arms_free_terms.track_yaw_rate, weight=1e-3,
                                                params={**cfg.rewards["track_angular_velocity"].params, "std": TRACK_ANG_STD[0]})

  # ARMS-FREE curriculum: performance-gated difficulty level (Curriculum/level) + read-only info term
  curr_params = {"push_min": PUSH_SPEED[0], "push_max": PUSH_SPEED[1], "lin_std": TRACK_LIN_STD, "ang_std": TRACK_ANG_STD}
  cfg.curriculum["level"] = CurriculumTermCfg(func=arms_free_terms.difficulty_level, params=curr_params)
  cfg.curriculum["level_info"] = CurriculumTermCfg(func=arms_free_terms.level_info, params=curr_params)
  # ARMS-FREE: posture and joint-limit terms on the policy's joints only
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=RC.LOCOMOTION_ACTION_NAMES)
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=RC.LOCOMOTION_ACTION_NAMES)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = POSE_STD_WALKING                  # WISCO: our joint names
  cfg.rewards["pose"].params["std_running"] = POSE_STD_RUNNING

  cfg.rewards["upright"].params["asset_cfg"].body_names = (P.TORSO_BODY,)        # WISCO
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (P.TORSO_BODY,)   # WISCO

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  # ARMS-FREE fix 2/3: G1 uses -0.02 on whole-body angular momentum. Ours counts only the policy-controlled bodies (pelvis,
  # legs, torso links, welded head; not arms/hands): the whole-body value is dominated by the scripted arms + payload as
  # the curriculum scales them (J3c: -0.02 -> -1.10/s at level 0.35 even at weight -0.005). Weight -0.005 kept (fix 2:
  # Wisco is heavier and taller than the G1). The old whole-body value is logged as a metric for comparison.
  cfg.rewards["angular_momentum"] = RewardTermCfg(
    func=arms_free_terms.policy_body_angular_momentum_sq,
    weight=-0.005,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(".*",), preserve_order=True)},
  )
  cfg.rewards["angular_momentum"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=tuple(f"^{b}$" for b in RC.POLICY_BODIES), preserve_order=True)
  cfg.metrics["angmom_wholebody_sq"] = MetricsTermCfg(func=arms_free_terms.wholebody_angular_momentum_sq)
  # J3h: feet air time (0.05-0.5 s swings) whenever command > 0.1, so slow commands get short steps (same as G1 J1j)
  cfg.rewards["air_time"].weight = 0.25
  cfg.rewards["air_time"].params["command_threshold"] = 0.1

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=arms_free_terms.self_collision_cost_multi,
    weight=-1.0,
    params={"sensor_names": tuple(s.name for s in self_col_sensors), "force_threshold": 10.0},
  )

  # WISCO: ankle linkage penalty (H3) + a metric to watch the peak ankle motor torque
  cfg.rewards["ankle_motor_overload"] = RewardTermCfg(
    func=linkage_penalty.ankle_motor_overload,
    weight=ANKLE_MOTOR_OVERLOAD_WEIGHT,
    params={"motor_limit": P.X4_36_PEAK_NM},
  )
  cfg.metrics["ankle_motor_peak_nm"] = MetricsTermCfg(func=linkage_penalty.ankle_motor_peak, reduce="max")
  # Fix 4: sustained ankle motor torque above rated (thermal). Weight: with all 4 motors at 50 % over rated
  # (sqrt(EMA) = 15.75 N*m) the term is 4 x 0.25 = 1.0 -> -1.0/s, half the max linear-tracking reward (2.0/s).
  cfg.rewards["ankle_motor_thermal"] = RewardTermCfg(
    func=linkage_penalty.ankle_motor_thermal, weight=ANKLE_THERMAL_WEIGHT,
    params={"rated_nm": P.X4_36_RATED_NM, "tau_s": 2.0})
  cfg.metrics["ankle_motor_frac_above_rated"] = MetricsTermCfg(func=linkage_penalty.ankle_motor_frac_above_rated,
                                                               params={"rated_nm": P.X4_36_RATED_NM})
  cfg.metrics["ankle_motor_rms_nm"] = MetricsTermCfg(func=linkage_penalty.ankle_motor_rms)

  # --- from unitree_g1_flat_env_cfg --------------------------------------------------------
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    # ARMS-FREE: evaluate at a fixed difficulty: default full (L = L2 = 1); set WISCO_PLAY_LEVEL / WISCO_PLAY_LEVEL2 to mirror
    # the level training has reached (e.g. 0.2), so arm range and hand payload match what the policy has seen.
    import os
    cfg.curriculum["level"] = CurriculumTermCfg(func=arms_free_terms.fixed_level, params={
      "L": float(os.environ.get("WISCO_PLAY_LEVEL", "1.0")), "L2": float(os.environ.get("WISCO_PLAY_LEVEL2", "1.0")),
      "push_min": PUSH_SPEED[0], "push_max": PUSH_SPEED[1],
      "lin_std": TRACK_LIN_STD, "ang_std": TRACK_ANG_STD})
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = YAW_RANGE

  return cfg
