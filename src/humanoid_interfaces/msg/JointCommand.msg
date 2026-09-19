# The final, torque-loop-ready command, CANONICAL_JOINT_ORDER (23).
#
# Published by: wbc_stub (500 Hz), from the arbitrated JointTargets it receives.
# Consumed by:  lowlevel_control's JointImpedanceController, which runs inside
#   controller_manager's real-time update() loop at 1000 Hz and computes, per
#   joint:
#       tau = effort_feedforward + Kp*(position_target - q) + Kd*(velocity_target - qdot)
#   then writes tau to the "effort" command_interface exported by whichever
#   hardware plugin is active (mujoco_ros2_control/MujocoSystem in sim,
#   ethercat_bridge/EthercatHardwareInterface on real hardware).
#
# This message is received via a non-real-time topic subscription
# and cached in a realtime_tools::RealtimeBuffer inside the controller (see
# lowlevel_control's controller source + RESEARCH_NOTES.md "Multi-rate
# architecture"): 500 Hz publishing into a 1000 Hz consumer is fine because
# the controller re-evaluates the PD law against the LATEST cached target on
# every 1kHz tick, not once per received message.

std_msgs/Header header

float64[23] position_target       # rad
float64[23] velocity_target       # rad/s
float64[23] effort_feedforward    # Nm: tau_ff term, 0.0 from wbc_stub today;
                                   # a real WBC would fill this with e.g. gravity
                                   # compensation or contact-consistent torques.
