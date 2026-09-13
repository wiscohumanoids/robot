# Joint-space position targets, CANONICAL_JOINT_ORDER (23) -- see
# humanoid_interfaces/config/canonical_joint_order.yaml and ARCHITECTURE.md.
#
# Published by: locomotion_runner (50 Hz) and manipulation_runner (10 Hz),
#   each on its own topic (/locomotion/joint_targets, /manipulation/joint_targets)
# Consumed by:  wbc_stub (500 Hz), which arbitrates between the two sources
#   and forwards a single stream downstream as JointCommand.
#
# This is a target-generation-layer message (arms/legs/waist coordination,
# task-level intent). It carries NO torque information on purpose -- torque
# feedforward + gains are a lowlevel_control concern (see JointCommand).

std_msgs/Header header

float64[23] positions    # desired joint position, rad
float64[23] velocities   # desired joint velocity feedforward, rad/s (0.0 if unused)

# Provenance tag for logging/debugging arbitration decisions in wbc_stub.
string source   # one of: "locomotion", "manipulation", "teleop_hold"
