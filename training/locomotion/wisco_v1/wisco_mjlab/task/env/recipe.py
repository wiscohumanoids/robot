"""Tuning constants for the Wisco V1 velocity task.

These are recipe decisions, not physical robot values. Physical values live in
robots/wisco_v1/model/robot_numbers.yaml and are read through robot/robot_cfg.py.
"""
import math

# WISCO: mass uncertainty (robots/wisco_v1/hardware_observations.md, "Masses"). Each part's mass and
# inertia scale together by this factor (skewed up: the CAD masses leave out wiring). pseudo_inertia
# scales by e^(2*alpha).
MASS_SCALE = (0.85, 1.20)
ALPHA_RANGE = (math.log(MASS_SCALE[0]) / 2, math.log(MASS_SCALE[1]) / 2)

# WISCO: ankle motor limits (robots/wisco_v1/hardware_observations.md, "Ankle torque").
ANKLE_MOTOR_OVERLOAD_WEIGHT = -1.0
ANKLE_THERMAL_WEIGHT = -1.0          # sustained torque above rated (see rewards.py)

# ARMS-FREE: posture reward on the policy's joints only (legs + waist).
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

# ARMS-FREE recipe numbers.
TRACK_LIN_STD = (0.5, 0.25)          # G1 default at L2 = 0 -> arms-free target at L2 = 1
TRACK_ANG_STD = (0.5 ** 0.5, 0.35)   # 0.707 -> 0.35 rad/s
YAW_RANGE = (-1.5, 1.5)
REL_YAW_ONLY = 0.18                  # x (1 - 10 % standing) = 16 % of samples are pure turns
PUSH_SPEED = (0.5, 0.8)              # push velocity range +-v, v from 0.5 (G1 default, L = 0) to 0.8 (L = 1)
HAND_PAYLOAD_MAX_KG = 1.0            # max added mass per hand at L = 1
# arm target ranges at L = 1: robot/robot_cfg.ARM_TARGET_FRACS (kept away from the body)
