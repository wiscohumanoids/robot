# nav_stub

**Layer:** 5 (navigation). **Status:** STUB standing in for Nav2.
**Subscribes:** `/robot_pose`.
**Action server:** `navigate_to_pose` (`nav2_msgs/NavigateToPose`), Nav2's own
action name and type, so the behavior tree's client is unchanged when Nav2 arrives.
**Publishes:** `/cmd_vel_nav` (`Twist`, 20 Hz, only while a goal is active),
**the only publisher.** `cmd_vel_mux` forwards it to `/cmd_vel`.

## What is fake

A proportional go-to-point controller (`nav_logic.py`, unit-tested including a
closed-loop convergence test): turn toward the goal, drive forward as the heading
error shrinks, stop within 0.10 m. No costmaps, no global/local planning, no
obstacle avoidance, and the goal *orientation* is ignored. Times out after 60 s.

## What replaces it

Nav2, configured (not built): map from `slam`, pose/TF from `state_estimation`.
Remap Nav2's velocity output to `/cmd_vel_nav`. It closes the loop through
`state_estimation`'s `/robot_pose`, so today's stub navigation "works" only
because that node also trusts the command.

```bash
ros2 launch bringup full_stack.launch.py navigation:=external
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.5}, orientation: {w: 1.0}}}}" --feedback
```
