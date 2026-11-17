"""Hsim-G1-Velocity-Flat-ArmsFree: mjlab 1.6.0's Unitree G1 flat velocity task with the arms-free recipe.

The policy drives the legs and waist (15 joints) and must keep balance while the arms, which manipulation will
own on the robot, move on their own. Every change from mjlab's G1 task is marked `# ARMS-FREE:` below:

- Actions: legs + waist only; observations still see all 29 joints.
- Arms: position-held, following scripted random targets within body-safe offsets (`arm_ranges.py`); 0-1 kg payload
  on each hand.
- Commands: >= 15 % pure-yaw turns, 25 % standing, slow-command practice, forward speed capped at 1.5 m/s, yaw range
  +-1.5 rad/s, resampled every 2-6 s.
- Rewards: yaw-rate-only turn tracking; feet air time for any command above 0.1 (so small commands get short steps);
  stand-still, action-rate and joint-acceleration terms for smooth standing; self-collision counted only between the
  legs and the torso, other leg and pelvis; angular momentum of the policy-controlled bodies only. Optional terms
  (torso tilt, foot slip while standing, precise tracking) are switched on by environment variables.
- Realism: 0-10 ms command delay; pushes up to 0.8 m/s; armature and kp/kd randomized +-10 %.
- Curriculum: a performance-gated level L in [0, 1] scales the arm range, payload and pushes from zero; once L = 1, a
  second level L2 tightens the tracking stds from mjlab's defaults (0.5, 0.707) to 0.25 / 0.35. Levels rise only while
  episodes are long and tracking is good, measured with fixed-std gate terms so tightening cannot stall its own gate
  (`mdp.difficulty_curriculum`).

Environment overrides for variants: G1_START_LEVEL (resume at a curriculum level; > 1 sets L = 1, L2 = value - 1),
G1_YAW_W, G1_AIRTIME_W, G1_ACTRATE_W, G1_JACC_W, G1_STAND_W, G1_BODYANG_W, G1_FLAT_W, G1_SLIP_W, G1_PRECISE_W,
G1_TURN_FRAC.
"""
from __future__ import annotations

import dataclasses
import math
import os

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp import rewards as mdp_base
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from . import mdp as hmdp
from .arm_ranges import G1_ARM_OFFSETS
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

LEG_JOINTS = (
  ".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint",
  ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
)
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
POLICY_JOINTS = LEG_JOINTS + WAIST_JOINTS
ARM_JOINTS = (
  ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint",
  ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint",
)
HAND_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
POLICY_BODIES = (
  "pelvis", r"^(left|right)_(hip_pitch|hip_roll|hip_yaw|knee|ankle_pitch|ankle_roll)_link$",
  "waist_yaw_link", "waist_roll_link", "torso_link",
)
POLICY_POSE_KEYS = ("hip", "knee", "ankle", "waist")   # pose-reward std keys that apply to policy joints

STD_LIN_START = math.sqrt(0.25)   # mjlab G1 default 0.5
STD_ANG_START = math.sqrt(0.5)    # mjlab G1 default 0.707
PUSH_MIN, PUSH_MAX = 0.5, 0.8
CURRICULUM_WINDOW = 512           # finished episodes per curriculum decision
DELAY_MAX_STEPS = 2          # 2 x 5 ms physics = 10 ms
DELAY_UPDATE_PERIOD = 50     # resample the lag every 0.25 s (slowly varying latency)


def _robot_cfg_with_delay():
  """mjlab's G1 robot config with a 0-10 ms command delay on every actuator."""
  cfg = get_g1_robot_cfg()
  acts = tuple(
    dataclasses.replace(a, delay_min_lag=0, delay_max_lag=DELAY_MAX_STEPS,
                        delay_update_period=DELAY_UPDATE_PERIOD)
    for a in cfg.articulation.actuators
  )
  cfg.articulation = dataclasses.replace(cfg.articulation, actuators=acts)
  return cfg


def _policy_only(std: dict) -> dict:
  """Keep only the pose-reward stds that apply to policy joints."""
  return {k: v for k, v in std.items() if any(s in k for s in POLICY_POSE_KEYS)}


def g1_velocity_arms_free_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the task config; `play` drops the curriculum and turn-only commands for evaluation."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  cfg.scene.entities = {"robot": _robot_cfg_with_delay()}                                  # ARMS-FREE: command delay

  # ARMS-FREE: actions = legs + waist. The G1 action scale dict has entries for every joint
  # pattern; keep only the policy joints so the action term resolves cleanly.
  jp = cfg.actions["joint_pos"]
  assert isinstance(jp, JointPositionActionCfg)
  jp.actuator_names = POLICY_JOINTS
  jp.scale = {k: v for k, v in jp.scale.items()
              if any(s in k for s in ("hip", "knee", "ankle", "waist"))}

  # ARMS-FREE: scripted arm motion (event every env step + reset to default)
  cfg.events["arm_motion"] = EventTermCfg(
    mode="step",
    func=hmdp.random_arm_targets,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
      "resample_s": (1.0, 3.0),
      "hold_default_prob": 0.3,
      "range_frac": 0.0,          # set by the curriculum, 0 -> 1 x the safe offsets
      "offsets": G1_ARM_OFFSETS,  # body-safe arm target offsets
      "move_s": (0.3, 1.5),
    },
  )

  # ARMS-FREE: hand payload, resampled every episode: default + U(0, max) kg, max set by the curriculum
  # (0 -> 1 kg). "add" is relative to the model's default mass, so it never accumulates.
  cfg.events["hand_payload"] = EventTermCfg(
    mode="reset",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODIES),
      "ranges": (0.0, 0.0),
      "operation": "add",
    },
  )

  # ARMS-FREE: tracking stds start at the mjlab G1 defaults; the curriculum's L2 tightens them to 0.25 / 0.35
  cfg.rewards["track_linear_velocity"].params["std"] = STD_LIN_START
  cfg.rewards["track_angular_velocity"].params["std"] = STD_ANG_START
  yw = os.environ.get("G1_YAW_W")   # turn tracking weight (stock 2.0)
  if yw:
    cfg.rewards["track_angular_velocity"].weight = float(yw)
  cfg.rewards["track_angular_velocity"].func = hmdp.track_yaw_rate   # ARMS-FREE: yaw rate only (no roll/pitch in the error)
  # ARMS-FREE: reward feet air time (0.05-0.5 s per swing) whenever |v| + |w| command > 0.1, so small commands are
  # met with short steps instead of standing flat-footed (stock G1 has this term at weight 0, threshold 0.5).
  cfg.rewards["air_time"].weight = float(os.environ.get("G1_AIRTIME_W", "0.2"))
  cfg.rewards["air_time"].params["command_threshold"] = 0.1

  # ARMS-FREE: smoothness and standing still
  cfg.rewards["action_rate_l2"].weight = float(os.environ.get("G1_ACTRATE_W", "-0.2"))      # stock -0.1
  cfg.rewards["joint_acc"] = RewardTermCfg(func=mdp_base.joint_acc_l2, weight=float(os.environ.get("G1_JACC_W", "-2.5e-7")),
                                           params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINTS)})
  # Optional anti-rocking terms: torso roll/pitch rate (mjlab G1 -0.05), pelvis tilt, foot slip also while standing
  # (stock: only when moving).
  bw = os.environ.get("G1_BODYANG_W")
  if bw:
    cfg.rewards["body_ang_vel"].weight = float(bw)
  fw = float(os.environ.get("G1_FLAT_W", "0"))
  if fw:
    cfg.rewards["base_flat"] = RewardTermCfg(func=mdp_base.flat_orientation_l2, weight=fw,   # pelvis (root) tilt
                                             params={"asset_cfg": SceneEntityCfg("robot")})
  sw = os.environ.get("G1_SLIP_W")
  if sw:
    cfg.rewards["foot_slip"].weight = float(sw)
    cfg.rewards["foot_slip"].params["command_threshold"] = -1.0   # always (standing too)
  # Optional precise tracking (tight stds), so small commands are still worth starting for once the smoothness terms
  # make moving cost more. G1_PRECISE_W = 0 turns it off.
  pw = float(os.environ.get("G1_PRECISE_W", "0"))
  if pw:
    cfg.rewards["track_lin_precise"] = RewardTermCfg(func=cfg.rewards["track_linear_velocity"].func, weight=pw,
                                                     params={**cfg.rewards["track_linear_velocity"].params, "std": 0.1})
    cfg.rewards["track_yaw_precise"] = RewardTermCfg(func=hmdp.track_yaw_rate, weight=pw,
                                                     params={**cfg.rewards["track_angular_velocity"].params, "std": 0.2})
  cfg.rewards["stand_still"] = RewardTermCfg(func=hmdp.stand_still, weight=float(os.environ.get("G1_STAND_W", "-1.0")),
                                             params={"command_name": "twist", "sensor_name": "feet_ground_contact",
                                                     "command_threshold": 0.05, "foot_weight": 0.5})

  # Policy-only posture / joint-limit terms: the arms are driven by the scripted motion, not the policy.
  pose = cfg.rewards["pose"].params
  pose["asset_cfg"] = SceneEntityCfg("robot", joint_names=POLICY_JOINTS)
  pose["std_standing"] = {r".*(hip|knee|ankle|waist).*": 0.05}
  pose["std_walking"] = _policy_only(pose["std_walking"])
  pose["std_running"] = _policy_only(pose["std_running"])
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=POLICY_JOINTS)

  # ARMS-FREE: commands (turn-in-place, slow and standing shares), yaw range +-1.5 in every curriculum stage
  old = cfg.commands["twist"]
  fields = {f.name: getattr(old, f.name) for f in dataclasses.fields(old)}
  twist = hmdp.TurnInPlaceVelocityCommandCfg(**fields, rel_turn_envs=float(os.environ.get("G1_TURN_FRAC", "0.3")), turn_yaw_range=(0.2, 1.5),
                                              rel_small_envs=0.2,   # slow-command practice
                                              keep_standing=True)   # turns never replace standing commands
  twist.rel_standing_envs = 0.25                 # stock 0.1
  twist.resampling_time_range = (2.0, 6.0)       # stock 3-8 s: more stops and starts
  twist.ranges.ang_vel_z = (-1.5, 1.5)
  cfg.commands["twist"] = twist
  if "command_vel" in cfg.curriculum:
    for stage in cfg.curriculum["command_vel"].params["velocity_stages"]:
      if "ang_vel_z" in stage:
        stage["ang_vel_z"] = (-1.5, 1.5)
      if "lin_vel_x" in stage:   # cap forward speed at 1.5 m/s (stock stages go to 3 m/s)
        lo, hi = stage["lin_vel_x"]
        stage["lin_vel_x"] = (max(lo, -1.0), min(hi, 1.5))
    cfg.curriculum["command_vel"].params["velocity_stages"][0]["ang_vel_z"] = (-1.5, 1.5)

  # ARMS-FREE: pushes start at the G1 default 0.5 m/s; the curriculum raises them to 0.8 m/s
  if "push_robot" in cfg.events:
    vr = cfg.events["push_robot"].params["velocity_range"]
    vr["x"] = (-PUSH_MIN, PUSH_MIN)
    vr["y"] = (-PUSH_MIN, PUSH_MIN)

  # ARMS-FREE: motor randomization at startup (G1 masses are accurate: no mass DR)
  cfg.events["armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "ranges": (0.9, 1.1), "operation": "scale"},
  )
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)), "kp_range": (0.9, 1.1),
            "kd_range": (0.9, 1.1), "operation": "scale"},
  )

  # ARMS-FREE: self-collision penalty without the arms. Replace the G1's whole-body sensor with three
  # sensors that pair leg links with the torso, the other leg, and the pelvis.
  legs = r"^(left|right)_(hip_pitch|hip_roll|hip_yaw|knee|ankle_pitch|ankle_roll)_link$"
  left_leg = r"^left_(hip_pitch|hip_roll|hip_yaw|knee|ankle_pitch|ankle_roll)_link$"
  def _sc(name, primary, secondary, mode):
    return ContactSensorCfg(
      name=name, primary=ContactMatch(mode="body", pattern=primary, entity="robot"),
      secondary=ContactMatch(mode=mode, pattern=secondary, entity="robot"),
      fields=("found", "force"), reduce="none", num_slots=1, history_length=4)
  body_sensors = (
    _sc("selfc_leg_torso", legs, "torso_link", "body"),
    _sc("selfc_leg_leg", left_leg, "right_hip_pitch_link", "subtree"),
    _sc("selfc_leg_pelvis", legs, "pelvis", "body"),
  )
  cfg.scene.sensors = tuple(s_ for s_ in (cfg.scene.sensors or ()) if s_.name != "self_collision") + body_sensors
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=hmdp.body_self_collision_cost, weight=-1.0,
    params={"sensor_names": tuple(s_.name for s_ in body_sensors), "force_threshold": 10.0})

  # ARMS-FREE: angular momentum of the policy-controlled bodies only
  cfg.rewards["angular_momentum"] = RewardTermCfg(
    func=hmdp.policy_bodies_angular_momentum_penalty,
    weight=cfg.rewards["angular_momentum"].weight,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=POLICY_BODIES)},
  )

  # ARMS-FREE: performance-gated curriculum (logged as Curriculum/difficulty/*)
  cfg.curriculum["difficulty"] = CurriculumTermCfg(
    func=hmdp.difficulty_curriculum,
    params={
      "up_len_frac": 0.8, "up_track_frac": 0.5, "up_ang_frac": 0.4, "down_len_frac": 0.4,
      "up_ang_frac_l2": 0.0,    # no turn gate for L2: the G1 ignores slow in-place turns (< ~1 rad/s)
                                #   until L2 tightens the turn std, so a turn gate on L2 deadlocks it. Length gate still guards falls.
      "step": 0.05, "dwell_iters": 50, "steps_per_iter": 24,
      "window_episodes": CURRICULUM_WINDOW,
      "start_level": float(os.environ.get("G1_START_LEVEL", "0") or 0),
      "arm_range_max": 1.0, "payload_max_kg": 1.0,
      "push_min": PUSH_MIN, "push_max": PUSH_MAX,
      "std_lin_start": STD_LIN_START, "std_lin_end": 0.25,
      "std_ang_start": STD_ANG_START, "std_ang_end": 0.35,
      "arm_term": "arm_motion", "payload_term": "hand_payload", "push_term": "push_robot",
      "lin_reward_term": "track_linear_velocity", "ang_reward_term": "track_angular_velocity",
      "lin_gate_term": "gate_track_lin", "ang_gate_term": "gate_track_ang",
    },
  )
  # ARMS-FREE: curriculum gates measured with the fixed starting stds (tiny weight: no effect on training), so the L2
  # std tightening cannot stall its own gate.
  cfg.rewards["gate_track_lin"] = RewardTermCfg(func=cfg.rewards["track_linear_velocity"].func, weight=1e-3,
                                                params={**cfg.rewards["track_linear_velocity"].params, "std": STD_LIN_START})
  cfg.rewards["gate_track_ang"] = RewardTermCfg(func=cfg.rewards["track_angular_velocity"].func, weight=1e-3,
                                                params={**cfg.rewards["track_angular_velocity"].params, "std": STD_ANG_START})

  if play:
    twist.rel_turn_envs = 0.0
    cfg.curriculum.pop("difficulty", None)   # play: fixed parameters (set them explicitly if needed)
  return cfg


TRACK_STD = {"lin": 0.25, "ang": 0.35, "lin_v1": math.sqrt(0.25), "ang_v1": math.sqrt(0.5)}
