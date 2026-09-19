# slam_stub

**Layer:** 4 -- perception & state estimation. **Status:** STUB.
**Publishes:** `/map` (`nav_msgs/OccupancyGrid`, transient-local, re-sent every
5 s) and the TF edge `map -> odom` at 20 Hz -- **the only publisher of both.**

## What is fake

An empty (all-free) 10 m x 10 m map centered on the origin and an **identity**
`map -> odom`: "no drift, the map origin is where the robot started". No
mapping, no loop closure, no localization.

## What replaces it

ORB-SLAM3 (or `slam_toolbox`) wired into ROS. It owns `map -> odom` (the drift
correction) and `/map`; it must **not** publish `odom -> base_link`, which
belongs to state estimation. See the TF table in
[INTERFACE_CONTRACT.md](../../INTERFACE_CONTRACT.md).

```bash
ros2 launch bringup full_stack.launch.py slam:=external
```
