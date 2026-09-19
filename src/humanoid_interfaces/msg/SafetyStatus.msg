# Published by: safety node, at 100 Hz, on /safety_status.
# Consumed by:  lowlevel_control's JointImpedanceController (zeros effort output
#   and refuses to re-enable until is_safe is true again), and any node that
#   wants to gate its own output on robot safety state (e.g. locomotion_runner
#   could refuse to arm the "walk" behaviour while estop_active).
#
# This is the ROS2-side half of the hardware E-stop interface: see
# safety/README.md and INTEGRATION_POINTS.md "Safety" section for how a
# physical wireless E-stop (separate firmware project, out of scope here) is
# expected to drive estop_active.

std_msgs/Header header

bool estop_active        # true = hardware or software E-stop asserted
bool is_safe              # true = safe to command nonzero effort (false whenever
                           # estop_active is true, or any fault_flags entry is set)
string[] fault_flags       # free-form fault identifiers, e.g. "joint_limit_exceeded",
                            # "ethercat_link_down", "overtemperature:left_knee_joint".
                            # Empty array = no faults.
