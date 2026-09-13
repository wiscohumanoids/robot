# Body-frame planar velocity command for locomotion.
#
# Published by: teleop_input (~10 Hz) on /cmd_vel
# Consumed by:  locomotion_runner (resampled internally to its own 50 Hz loop)
#
# Frame: base_link, x-forward / y-left / z-up (REP-103). This is a distinct
# type from geometry_msgs/Twist on purpose: Twist carries 6 fields (3 of which
# -- linear.z, angular.x, angular.y -- are meaningless for a ground-locomoting
# humanoid and would be silently ignored), which is exactly the ambiguity this
# repo's message-contract policy exists to avoid (see ARCHITECTURE.md).

std_msgs/Header header

float64 vx      # forward velocity command   [m/s]
float64 vy      # lateral velocity command    [m/s]
float64 vyaw    # yaw rate command            [rad/s]
