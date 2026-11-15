"""Random arm-target offsets from the default pose that keep the G1's arms away from its body
(J1c fix: the old "60 % of soft range" targets drove the arms into the legs/torso).

Offsets are in radians relative to the default joint position, for the LEFT arm; the right arm is
mirrored (roll / yaw offsets negated). Checked with hsim_tasks/tools/arm_contact_sweep.py.
  shoulder pitch: negative = raise forward
  shoulder roll : outward only (left +, right -)
  elbow         : bend more (hand forward / up) mostly; little extension (straight arm hits the thigh)
"""
G1_ARM_OFFSETS = {
  "shoulder_pitch_joint": (-0.9, 0.1),
  "shoulder_roll_joint": (0.0, 0.6),
  "shoulder_yaw_joint": (-0.3, 0.3),
  "elbow_joint": (0.0, 0.8),
  "wrist_roll_joint": (-0.6, 0.6),
  "wrist_pitch_joint": (-0.5, 0.5),
  "wrist_yaw_joint": (-0.5, 0.5),
}
MIRRORED = ("shoulder_roll_joint", "shoulder_yaw_joint", "wrist_roll_joint", "wrist_yaw_joint")


def offset_bounds(joint_name: str, table: dict = G1_ARM_OFFSETS) -> tuple[float, float]:
  """(lo, hi) offset for a full joint name like 'right_shoulder_roll_joint'."""
  for key, (lo, hi) in table.items():
    if joint_name.endswith(key):
      if joint_name.startswith("right_") and key in MIRRORED:
        return (-hi, -lo)
      return (lo, hi)
  raise KeyError(joint_name)
