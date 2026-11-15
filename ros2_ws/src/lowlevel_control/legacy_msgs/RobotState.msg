# Aggregated robot proprioception, CANONICAL_JOINT_ORDER (23): see
# humanoid_interfaces/config/canonical_joint_order.yaml and ARCHITECTURE.md.
#
# Published by: lowlevel_control (re-publishing controller_manager's
#   joint_state_broadcaster output + IMU data as one typed, documented message,
#   instead of making every downstream node subscribe to raw sensor_msgs/JointState
#   + sensor_msgs/Imu and re-derive gravity itself).
# Consumed by: locomotion_runner (to build PolicyObservation), safety (fault checks).

std_msgs/Header header

float64[23] joint_positions   # rad
float64[23] joint_velocities  # rad/s
float64[23] joint_efforts     # Nm, measured/applied torque (state_interface "effort")

geometry_msgs/Quaternion imu_orientation           # xyzw
geometry_msgs/Vector3    imu_angular_velocity       # body frame, rad/s
geometry_msgs/Vector3    imu_linear_acceleration    # body frame, m/s^2
geometry_msgs/Vector3    gravity_vector             # gravity direction expressed in the IMU
                                                     # (pelvis) frame, derived from
                                                     # imu_orientation: kept explicit so
                                                     # every consumer doesn't reimplement the
                                                     # same quaternion-rotate-gravity math.
