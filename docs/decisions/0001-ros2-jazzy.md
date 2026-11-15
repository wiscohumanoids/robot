# 0001 — ROS 2 Jazzy on Ubuntu 24.04 (not Humble)

**Decision:** Use ROS 2 Jazzy instead of the Humble in the original brief.

**Consequences**
- Ubuntu 24.04 and Python 3.12 on the CPU VM.
- Nav2 on Jazzy publishes `geometry_msgs/TwistStamped` for velocity commands, so the whole stack uses `TwistStamped` for `/cmd_vel*`.
- BehaviorTree.CPP v4 (the version Nav2 Jazzy uses).
- macOS isn't a supported Jazzy platform, so the laptop uses Docker or connects to the CPU VM. It doesn't run ROS natively.
