# odom_adapter_stub

**Layer:** 5 (navigation). **Status:** STUB.
**Subscribes:** TF `odom -> base_link` (from `state_estimation`).
**Publishes:** `/odom` (`nav_msgs/Odometry`, 50 Hz), **the only publisher.**

## What is fake

Nothing in the stack publishes `/odom`, and Nav2's controller and `bt_navigator`
read the robot's current velocity from it. This node builds the message from the
`odom -> base_link` transform: the pose is the transform, and the velocity is the
change between two transforms, expressed in the body frame (`odom_logic.py`,
unit-tested, including a round trip against `state_estimation_stub`'s integrator).
Covariances are zero; Nav2 does not use them.

## What replaces it

State estimation publishing `/odom` itself (the IEKF or `robot_localization`).
When that exists, delete this node and give `/odom` to `state_estimation` in
`interface_contract.yaml`; consumers do not change.

```bash
ros2 launch bringup full_stack.launch.py sim:=false
ros2 topic echo /odom --once
ros2 launch bringup full_stack.launch.py odom_adapter:=external   # you publish /odom yourself
```
