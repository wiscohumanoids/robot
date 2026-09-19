# The exact observation vector the locomotion policy (locomotion_runner's
# ONNX slot) expects, at 50 Hz. Assembled by locomotion_runner itself from
# RobotState + the latest /cmd_vel Twist + its own last output: this
# message exists as an explicit, documented, loggable intermediate so the
# locomotion team can record/replay/diff exactly what a policy saw, rather
# than that vector only ever existing as an anonymous numpy array inside one
# node (which was impossible to debug in berkeley_humanoid's original
# training code).
#
# CANONICAL_JOINT_ORDER (23): see humanoid_interfaces/config/canonical_joint_order.yaml
# and ARCHITECTURE.md. Every array below is indexed 0-22 in that order.
#
# Reconciliation with berkeley_humanoid: G1Env's trained policy was built
# against a 69-dim / 29-DOF observation (1 behaviour flag + 33 pos [quat+29
# joints] + 35 vel [lin3+ang3+29 joints]). This message is 23-DOF-native
# (this repo's canonical hardware). locomotion_runner's ONNX-loading slot is
# responsible for the 23->29 (pad 6 held-fixed joints with zero pos/vel) and
# 29->23 (drop the same 6) conversions at the boundary: see
# INTEGRATION_POINTS.md, "Locomotion" section, for the exact index mapping.
# This message itself intentionally stays 23-dim so it does not leak
# berkeley_humanoid-specific padding into the rest of the stack.

std_msgs/Header header

# --- command (from the latest geometry_msgs/Twist on /cmd_vel: linear.x, linear.y, angular.z) ---
float64 cmd_vx
float64 cmd_vy
float64 cmd_vyaw

# --- base state (from RobotState.imu_*) ---
geometry_msgs/Quaternion base_orientation          # world/pelvis orientation, ROS xyzw order
geometry_msgs/Vector3    base_angular_velocity     # body-frame, rad/s, from IMU gyro
geometry_msgs/Vector3    base_linear_velocity       # body-frame, m/s, NOT directly measurable
                                                     # on real hardware; sim = ground truth,
                                                     # real hardware = state estimator output
                                                     # (no state estimator exists in this repo
                                                     # yet, see ARCHITECTURE.md gap list)
geometry_msgs/Vector3    projected_gravity           # gravity vector expressed in base frame,
                                                       # a common humanoid-RL obs term derived
                                                       # from base_orientation (kept explicit
                                                       # here so locomotion_runner doesn't have
                                                       # to recompute it, and so it's loggable)

# --- proprioception, CANONICAL_JOINT_ORDER ---
float64[23] joint_positions   # rad
float64[23] joint_velocities  # rad/s

# --- policy's own last output, CANONICAL_JOINT_ORDER ---
# many locomotion policies (including berkeley_humanoid's G1Env) include the
# previous action in the observation. Zero-filled on the first control tick.
float64[23] previous_action
