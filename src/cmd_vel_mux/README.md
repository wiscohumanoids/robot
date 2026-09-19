# cmd_vel_mux

**Layer:** 5 -- navigation & velocity arbitration. **Status:** REAL (small and complete).
**Frequency:** 50 Hz, always publishing.
**Subscribes:** `/cmd_vel_teleop`, `/cmd_vel_nav` (`geometry_msgs/Twist`).
**Publishes:** `/cmd_vel` (`geometry_msgs/Twist`) -- **this node is its only publisher.**

Priority `teleop > nav > zero`. A source that has been silent for `timeout_s`
(default 0.5 s) is ignored, so:

- a human touching the keyboard overrides navigation, and releasing (teleop goes
  quiet) hands control back to navigation;
- if *everything* goes quiet -- teleop crashed, Nav2 died -- `/cmd_vel` becomes an
  explicit zero within half a second instead of the last command latching forever.

Why it exists: Nav2 and every off-the-shelf teleop node publish `Twist` on
`/cmd_vel`. Having two publishers on one topic is exactly what the
[interface contract](../../INTERFACE_CONTRACT.md) forbids, so each gets its own
input topic and this node is the one owner of `/cmd_vel`. When real Nav2 arrives,
remap its velocity output to `/cmd_vel_nav`.

The selection logic is in `mux_logic.py` (no ROS imports, unit-tested in `test/`).

```bash
ros2 run cmd_vel_mux mux_node --ros-args -p timeout_s:=0.5
```
