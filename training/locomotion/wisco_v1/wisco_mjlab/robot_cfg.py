"""Wisco V1 robot for mjlab: the counterpart of mjlab's asset_zoo/robots/unitree_g1/g1_constants.py.

All numbers come from model/robot_params.py (shared with the pre-training checks in robots/wisco_v1/checks/) and
model/train_model_info.json (written by model/build_training_model.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg

from .ankle_actuator import AnkleLinkageActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

MODEL_DIR = Path(__file__).resolve().parents[4] / "robots" / "wisco_v1" / "model"
sys.path.insert(0, str(MODEL_DIR))
import robot_params as P  # noqa: E402

INFO = json.loads((MODEL_DIR / "train_model_info.json").read_text())
PARAMS = P.joint_params()


def get_spec() -> mujoco.MjSpec:
  """The training model as an MjSpec."""
  return mujoco.MjSpec.from_file(str(P.TRAIN_XML))


# Command delay (motor bus latency), in physics steps of 5 ms: 0-10 ms, sampled per episode
# (robot_numbers.yaml "delay": estimate from PACE / Menlo measurements until our bus is measured).
SIM_DT = 0.005
DELAY_MAX_STEPS = int(round(P.command_delay_ms()[1] / 1000 / SIM_DT))

# One DC-motor actuator per joint: PD torque (gains from the G1 recipe, armature-derived),
# clipped to the motor's back-EMF torque-speed line, max torque = min(peak, sat * (1 - |w| / w_noload)).
# Values from model/robot_numbers.yaml (MyActuator datasheets, RP1-style gains) via robot_params.joint_params().
# Joint friction (datasheet backdrive torque) and viscous damping live in the model XML.
# The pre-training checks found this model stable with these gains at 5 ms, including a 10 ms delay.
ANKLE_JOINTS = ("left_ankle_pitch", "left_ankle_roll", "right_ankle_pitch", "right_ankle_roll")
ACTUATORS = tuple(
  DcMotorActuatorCfg(
    target_names_expr=(f"^{name}$",),
    stiffness=p["kp"],
    damping=p["kd"],
    effort_limit=p["effort"],
    saturation_effort=p["dc_saturation"],
    velocity_limit=p["dc_velocity_limit"],
    armature=p["armature"],
    delay_min_lag=0,
    delay_max_lag=DELAY_MAX_STEPS,
  )
  for name, p in PARAMS.items() if name not in ANKLE_JOINTS
)
# Ankles: one actuator for the 4 ankle joints that clamps each X4-36 MOTOR (peak + torque-speed line)
# through the rod linkage, like the real drivers (ankle_actuator.py). Armature stays per joint from the model XML
# (cfg armature is a single value). Base-class stiffness/saturation are placeholders; per-joint kp/kd below.
_x4 = P.motor("X4-36")
ACTUATORS = ACTUATORS + (
  AnkleLinkageActuatorCfg(
    target_names_expr=tuple(f"^{n}$" for n in ANKLE_JOINTS),
    stiffness=PARAMS["left_ankle_pitch"]["kp"],
    damping=PARAMS["left_ankle_pitch"]["kd"],
    effort_limit=max(PARAMS[n]["effort"] for n in ANKLE_JOINTS),
    saturation_effort=1.0e6,
    velocity_limit=1.0e6,
    armature=None,
    delay_min_lag=0,
    delay_max_lag=DELAY_MAX_STEPS,
    joint_kp={n: PARAMS[n]["kp"] for n in ANKLE_JOINTS},
    joint_kd={n: PARAMS[n]["kd"] for n in ANKLE_JOINTS},
    motor_peak=_x4["peak"],
    motor_saturation=P._torque_speed_slope("X4-36") * _x4["w_noload"],
    motor_velocity_limit=_x4["w_noload"],
  ),
)

# Randomization ranges (scale factors) around the nominal values, by how well each is known.
X8_JOINTS = tuple(f"^{n}$" for n, p in PARAMS.items() if p["motor"] == "X8-150")
OTHER_JOINTS = tuple(f"^{n}$" for n, p in PARAMS.items() if p["motor"] != "X8-150")
_x8 = next(p for p in PARAMS.values() if p["motor"] == "X8-150")
X8_ARMATURE_SCALE = (_x8["armature_range"][0] / _x8["armature"], _x8["armature_range"][1] / _x8["armature"])
OTHER_ARMATURE_SCALE = (0.9, 1.1)      # datasheet values
FRICTION_SCALE = (0.7, 1.3)            # datasheet backdrive torque is one number at standstill
DAMPING_SCALE = (0.0, 2.0)             # nominal = half the datasheet upper bound -> 0 .. upper bound

ACTION_SCALE: dict[str, float] = {f"^{name}$": p["action_scale"] for name, p in PARAMS.items()}   # all 29 joints (tracking)

# arms-free velocity policy: legs + waist only (15 actions); arms belong to the ROS controls stack (robot_numbers.yaml control).
LOCOMOTION_ACTION_JOINTS = tuple(P.LOCOMOTION_ACTION_JOINTS)
LOCOMOTION_ACTION_NAMES = tuple(f"^{n}$" for n in LOCOMOTION_ACTION_JOINTS)
LOCOMOTION_ACTION_SCALE: dict[str, float] = {f"^{n}$": PARAMS[n]["action_scale"] for n in LOCOMOTION_ACTION_JOINTS}
ARM_JOINTS = tuple(n for n in PARAMS if n not in LOCOMOTION_ACTION_JOINTS)
ARM_JOINT_NAMES = tuple(f"^{n}$" for n in ARM_JOINTS)
HAND_BODIES = ("left_hand", "right_hand")
ARM_BODY_EXCLUDE = (".*shoulder.*", ".*elbow.*", ".*forearm.*", ".*wrist.*", ".*hand.*")   # arm links (self-collision exclusion)
# Bodies the walking policy moves (angular momentum term): everything except arm links and hands.
import re as _re
_m = get_spec().compile()
POLICY_BODIES = tuple(_m.body(i).name for i in range(1, _m.nbody)
                      if not _re.search(r"shoulder|elbow|forearm|wrist|hand", _m.body(i).name))
del _m
# Random arm targets (controls-stack stand-in): per joint type, (fraction toward the lower soft limit, fraction toward the
# upper soft limit) around default, scaled by the curriculum level. Chosen with a MuJoCo contact sweep (3000 random poses,
# legs at home and walking-like): 60 % of the range on every arm joint touches the body in 5.3 % of poses; this in 0.0 %.
# shoulder roll only outward (inward hits the torso near -0.17 rad); shoulder yaw toward the upper limit swings the
# forearm into the torso (23 % of poses when moved alone) -> 0.1 there.
ARM_TARGET_FRACS = {
  "shoulder_pitch": (0.25, 0.6), "shoulder_roll": (0.0, 0.6), "shoulder_yaw": (0.25, 0.1), "elbow": (0.3, 0.6),
  "forearm_yaw": (0.3, 0.3), "wrist_pitch": (0.3, 0.3), "wrist_roll": (0.3, 0.3),
}
PD_GAIN_SCALE = (0.9, 1.1)             # driver kp/kd implementation uncertainty (units / loop rate not yet bench-checked; hardware_observations.md)

# Knees-bent home pose (the same posture as mjlab's G1 KNEES_BENT pose).
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, INFO["default_pelvis_height"]),
  joint_pos={f"^{name}$": p["default"] for name, p in PARAMS.items()},
  joint_vel={".*": 0.0},
)

# Same collision setup as the G1's FULL_COLLISION: everything collides (self-collision on);
# feet get 3D friction (condim 3), priority 1, friction 0.6; the rest condim 1.
FOOT_RE = r"^(left|right)_foot_collision$"
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={FOOT_RE: 3, ".*_collision": 1},
  priority={FOOT_RE: 1, ".*": 0},
  friction={FOOT_RE: (0.6,)},
)

ARTICULATION = EntityArticulationInfoCfg(actuators=ACTUATORS, soft_joint_pos_limit_factor=0.9)


def get_wisco_robot_cfg() -> EntityCfg:
  """A fresh config each call (mjlab mutates configs)."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )
