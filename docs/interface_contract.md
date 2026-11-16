# Interface contract

Every topic, TF edge, action and service between layers of the stack, and which node owns it.

**Source of truth:** `ros2_ws/src/hsim_bringup/config/interface_contract.yaml`. The tables below are generated
from it by `python3 tools/gen_docs.py` (never edit them by hand; `tests/test_docs.py` fails if they are stale).
The YAML's topics and TF edges are checked against a running stack by
`ros2 run hsim_bringup contract_check --phase N`; `tests/test_contract.py` checks the YAML against the code
without ROS.

## Rules

- **One publisher per topic and one owner per TF edge.** Owners are ROS **node names**, so a stub and the real
  node that replaces it share a name and the contract does not change.
- Each topic has a declared message type and, where it matters, a rate. `required: no` means the topic is only
  checked when something is publishing it (teleop, Nav2 or the stereo camera may not be running).
- Nodes talk only through these interfaces, never by importing each other's code.
- All nodes use sim time in simulation (`use_sim_time: true`); `/clock` comes from `sim`.
- **Phase 1 topics, TF edges and `/sim/reset` are frozen (2026-11-08).** Changing a frozen entry needs a decision
  record in `docs/decisions/`. Later-phase entries freeze at the end of their phase.
- `/sim/ground_truth/*` is **privileged**: only graders (`tools/eval/`) and explicitly labeled stubs may read it,
  never the robot stack (decision 0007).

## Topics

<!-- BEGIN GENERATED:topics -->
| Topic | Type | Owner (only publisher) | Rate | Phase | Required |
|---|---|---|---|---|---|
| `/clock` | `rosgraph_msgs/msg/Clock` | `sim` | 200 Hz | 1 | yes |
| `/cmd_vel` | `geometry_msgs/msg/TwistStamped` | `twist_mux` | - | 1 | yes |
| `/cmd_vel_teleop` | `geometry_msgs/msg/TwistStamped` | `teleop` | - | 1 | no |
| `/imu` | `sensor_msgs/msg/Imu` | `sim` | 200 Hz | 1 | yes |
| `/joint_states` | `sensor_msgs/msg/JointState` | `sim` | 200 Hz | 1 | yes |
| `/locomotion/joint_targets` | `sensor_msgs/msg/JointState` | `locomotion` | 50 Hz | 1 | yes |
| `/locomotion/status` | `diagnostic_msgs/msg/DiagnosticArray` | `locomotion` | 1 Hz | 1 | yes |
| `/odometry/filtered` | `nav_msgs/msg/Odometry` | `state_estimation` | 50 Hz | 1 | yes |
| `/sim/ground_truth/odom` | `nav_msgs/msg/Odometry` | `sim` | 200 Hz | 1 | yes |
| `/sim/viewer/image_raw` | `sensor_msgs/msg/Image` | `sim` | - | 1 | no |
| `/map` | `nav_msgs/msg/OccupancyGrid` | `map_server`, `localization` | - | 2 | yes |
| `/battery` | `sensor_msgs/msg/BatteryState` | `sim` | 1 Hz | 3 | yes |
| `/camera/head/color/image_raw` | `sensor_msgs/msg/Image` | `stereo_camera` | - | 3 | no |
| `/camera/head/depth/image_raw` | `sensor_msgs/msg/Image` | `stereo_camera` | - | 3 | no |
| `/camera/head/points` | `sensor_msgs/msg/PointCloud2` | `depth_to_cloud` | - | 3 | no |
| `/cmd_vel_approach` | `geometry_msgs/msg/TwistStamped` | `behavior` | - | 3 | no |
| `/cmd_vel_nav` | `geometry_msgs/msg/TwistStamped` | `velocity_smoother` | - | 3 | no |
| `/leg_odometry` | `nav_msgs/msg/Odometry` | `leg_odometry` | 50 Hz | 3 | no |
| `/perception/detections` | `vision_msgs/msg/Detection3DArray` | `perception` | 5 Hz | 3 | yes |
| `/planner/command` | `std_msgs/msg/String` | `ask`, `teleop` | - | 3 | no |
| `/planner/reply` | `std_msgs/msg/String` | `planner` | - | 3 | yes |
| `/planner/status` | `std_msgs/msg/String` | `planner` | - | 3 | yes |
| `/robot/say` | `std_msgs/msg/String` | `behavior` | - | 3 | yes |
| `/safety/estop` | `std_msgs/msg/Bool` | `safety` | 10 Hz | 3 | yes |
| `/safety/estop_request` | `std_msgs/msg/Bool` | `operator` | - | 3 | no |
| `/safety/status` | `diagnostic_msgs/msg/DiagnosticArray` | `safety` | 2 Hz | 3 | yes |
| `/scan` | `sensor_msgs/msg/LaserScan` | `sim`, `depth_to_scan` | - | 3 | yes |
| `/sim/ground_truth/state` | `std_msgs/msg/Float64MultiArray` | `sim` | 20 Hz | 3 | yes |
| `/world/objects` | `vision_msgs/msg/Detection3DArray` | `world_model` | 2 Hz | 3 | yes |
| `/world/summary` | `std_msgs/msg/String` | `world_model` | 2 Hz | 3 | yes |
| `/perception/status` | `std_msgs/msg/String` | `perception` | - | 4 | no |
| `/arm/joint_targets` | `sensor_msgs/msg/JointState` | `manipulation` | - | 5 | no |
<!-- END GENERATED:topics -->

Shared topics (many publishers by ROS convention, not single-owner; wildcards allowed):

<!-- BEGIN GENERATED:shared -->
`/diagnostics`, `/bond`, `/*/transition_event`, `/cmd_vel_nav_raw`, `/local_costmap/*`, `/global_costmap/*`, `/plan`, `/plan_smoothed`, `/received_global_plan`, `/transformed_global_plan`, `/lookahead_point`, `/lookahead_arc`, `/lookahead_collision_arc`, `/particle_cloud`, `/amcl_pose`, `/speed_limit`, `/cost_cloud`, `/behavior_tree_log`, `/*/_action/*`, `/pose`, `/map_metadata`, `/map_updates`, `/localization/*`, `/footprint`, `/tracked_pose`, `/goal_pose`, `/unsmoothed_plan`, `/camera/head/*/camera_info`, `/rtabmap/*`, `/info`, `/mapData`, `/mapGraph`, `/cloud_map`, `/cloud_obstacles`, `/cloud_ground`, `/grid_prob_map`, `/octomap_*`, `/global_path*`, `/local_path*`, `/landmarks`, `/labels`, `/mapPath`, `/goal_reached`, `/elevation_map`
<!-- END GENERATED:shared -->

## TF tree

<!-- BEGIN GENERATED:tf -->
| Edge | Owner | Kind | Phase |
|---|---|---|---|
| `map` → `odom` | `localization` | static or live | 1 |
| `odom` → `base_footprint` | `state_estimation` | dynamic | 1 |
| `base_footprint` → `base_link` | `leg_odometry` | dynamic | 3 |
| `base_link` → `torso_link` | `leg_odometry` | dynamic | 3 |
| `base_footprint` → `lidar_link` | `lidar_mount` | static | 3 |
| `base_link` → `imu_link` | `imu_mount` | static | 3 |
| `torso_link` → `head_camera` | `camera_mount` | static | 3 |
| `head_camera` → `head_camera_optical` | `camera_optical_mount` | static | 3 |
<!-- END GENERATED:tf -->

Only the frames above are published today: the sensor mounts are static transforms placed from the robot profile's
`mounts` (`sim_stack.launch.py`). A `robot_state_publisher` for all robot links (from the URDF and `/joint_states`) is
planned (`ros2_ws/src/hsim_description/`). `base_link` is the pelvis on every robot; frame names are the same for every
robot profile (`ros2_ws/src/hsim_bringup/config/robots/`).

## Actions

<!-- BEGIN GENERATED:actions -->
| Name | Type | Server | Clients | Phase |
|---|---|---|---|---|
| `/behavior/execute_plan` | `hsim_interfaces/action/ExecutePlan` | `behavior` | `planner`, `tools/eval/run_plan.py` | 3 |
| `/navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | `bt_navigator` | `behavior` | 3 |
| `/spin` | `nav2_msgs/action/Spin` | `behavior_server` | `behavior` | 3 |
| `/skills/manipulate` | `hsim_interfaces/action/Manipulate` | `manipulation` | `behavior` | 5 |
<!-- END GENERATED:actions -->

## Services

<!-- BEGIN GENERATED:services -->
| Name | Type | Server | Clients | Phase |
|---|---|---|---|---|
| `/sim/reset` | `std_srvs/srv/Trigger` | `sim` | `teleop` | 1 |
| `/sim/set_body_pose` | `hsim_interfaces/srv/SetBodyPose` | `sim` | `tools/eval graders` | 3 |
<!-- END GENERATED:services -->

## Joint ownership (one writer per joint)

The simulator (and later the hardware backend) takes joint targets from exactly one topic,
`/locomotion/joint_targets`, published by `locomotion` at 50 Hz for **every** joint of the robot.

| Joints | Who decides the target | How it reaches the robot |
|---|---|---|
| Joints the locomotion policy drives (manifest `role: policy`; legs + waist/torso in the current policies) | the policy | `locomotion` publishes them |
| Held joints (manifest `role: held`; arms, hands, head) | `/arm/joint_targets` (from `manipulation`), else the manifest default pose | `locomotion` merges them by joint name and republishes |

So a joint has one writer at a time: the policy for policy joints, the latest `/arm/joint_targets` for held ones.
`locomotion` ignores names on `/arm/joint_targets` that are not held joints (warned once). Policies that drive every
joint (manifest v1, e.g. `g1_velocity_all_joints`) have no held joints and cannot be combined with manipulation; see
`docs/decisions/0002-two-locomotion-policies.md`.
