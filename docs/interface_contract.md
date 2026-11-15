# Interface contract — v1 (Phase 1 frozen, Phase 3 draft)

Status: **Phase 1 topics, TF edges and `/sim/reset` are frozen (2026-11-08).** Changing them needs a decision record in `docs/decisions/`. Later-phase entries are still drafts and freeze at the end of their phase.

All nodes use sim time (`use_sim_time: true`). `/clock` comes from hsim_sim.

Machine-readable copy: `ros2_ws/src/hsim_bringup/config/interface_contract.yaml`, checked live by `contract_check`. Owners there are **ROS node names** (`sim`, `locomotion`, `twist_mux`, `state_estimation`, `localization`, `teleop`). A stub and the real node it becomes use the same name.

## Topics

| Topic | Type | Rate | Owner (only publisher) | Phase | Notes |
|---|---|---|---|---|---|
| /clock | rosgraph_msgs/Clock | physics rate | hsim_sim | 1 | |
| /joint_states | sensor_msgs/JointState | 200 Hz | hsim_sim | 1 | all 29 G1 joints, names in manifest order |
| /imu | sensor_msgs/Imu | 200 Hz | hsim_sim | 1 | pelvis IMU, frame `imu_link` |
| /sim/ground_truth/odom | nav_msgs/Odometry | 200 Hz | hsim_sim | 1 | debug + temporary stand-in for the estimator. Never a TF publisher |
| /sim/viewer/image_raw | sensor_msgs/Image | ~10 Hz | hsim_sim | 1 | follow camera for Foxglove; optional |
| /cmd_vel_teleop | geometry_msgs/TwistStamped | on keypress | hsim_teleop (keyboard) | 1 | |
| /cmd_vel_nav | geometry_msgs/TwistStamped | 20 Hz | Nav2 velocity_smoother | 3 | Nav2 on Jazzy publishes stamped twists by default |
| /cmd_vel_approach | geometry_msgs/TwistStamped | 20 Hz while stepping up | hsim_behavior | 3 | navigate_to's final step to arm's reach, and the step back |
| /cmd_vel | geometry_msgs/TwistStamped | 50 Hz | twist_mux | 1 | teleop > approach > nav |
| /locomotion/joint_targets | sensor_msgs/JointState | 50 Hz | hsim_locomotion | 1 | `position` = targets; names make ordering explicit |
| /locomotion/status | diagnostic_msgs/DiagnosticArray | 1 Hz | hsim_locomotion | 1 | policy id, inference time, fell-over flag |
| /odometry/filtered | nav_msgs/Odometry | 200 Hz | hsim_state_estimation | 1 | stub passes ground truth through until the IEKF (Phase 3) |
| /map | nav_msgs/OccupancyGrid | latched | nav2 map_server | 2 | from the environment build (`assets/envs/<env>/build/map.yaml`); only with `env:=` |
| /camera/head/color/image_raw | sensor_msgs/Image | 15 Hz | hsim_sim | 4 | |
| /camera/head/depth/image_raw | sensor_msgs/Image | 15 Hz | hsim_sim | 4 | |
| /camera/head/color/camera_info | sensor_msgs/CameraInfo | 15 Hz | hsim_sim | 4 | |
| /planner/command | std_msgs/String | event | UI / CLI | 3 | natural-language input |
| /planner/plan | hsim_interfaces/SkillPlan (TBD) | event | hsim_planner | 3 | |
| /perception/objects | vision_msgs/Detection3DArray | ~2 Hz | hsim_perception | 4 | class text + 6-DOF pose |
| /arm/joint_targets | sensor_msgs/JointState | 20 Hz (manipulation) | hsim_manipulation (until hsim_arm_control exists); read by the locomotion node for held (arm) joints | 5 | arm + gripper joints only |
| /stack_monitor/status | diagnostic_msgs/DiagnosticArray | 1 Hz | each node (own namespace in `name`) | 6 | aggregated by hsim_monitor |

## Actions

| Action | Type | Server | Phase |
|---|---|---|---|
| /navigate_to_pose | nav2_msgs/action/NavigateToPose | Nav2 bt_navigator | 3 |
| /skills/find_object | hsim_interfaces/action/FindObject (TBD) | hsim_perception | 4 |
| /skills/manipulate | hsim_interfaces/action/Manipulate (instruction -> VLA loop at 20 Hz; replaces the planned Pick) | hsim_manipulation | 5 |

## TF edges

| Edge | Owner | Phase |
|---|---|---|
| map → odom | hsim_localization (static identity until Phase 3) | 1 |
| odom → base_link | hsim_state_estimation (stub fed by ground truth until Phase 3) | 1 |
| base_link → all robot links | robot_state_publisher (from /joint_states) | 1 |
| base_link → camera frames | robot_state_publisher (fixed joints in URDF) | 4 |

## Joint ownership (one writer per joint)

| Joints | Writer in Phases 1–4 | Writer in Phase 5 |
|---|---|---|
| legs + waist | hsim_locomotion | hsim_locomotion |
| arms (+ hands) | hsim_locomotion (policy holds them) | hsim_arm_control |

hsim_sim merges both target topics by joint name. A joint that gets targets from two topics is an error. Phase 5 needs a locomotion policy that doesn't control the arms; see `docs/decisions/0002`.

## Phase 3 additions (draft)

The machine-readable copy (`ros2_ws/src/hsim_bringup/config/interface_contract.yaml`) is the source of truth;
`contract_check` enforces it. Summary:

| Topic / action | Type | Owner |
|---|---|---|
| /scan | LaserScan | `sim` (lidar, 10 Hz) or `depth_to_scan` (stereo) |
| /battery | BatteryState | sim |
| /leg_odometry | Odometry | leg_odometry |
| /camera/head/{color,depth}/image_raw, /camera/head/points | Image, PointCloud2 | stereo_camera, depth_to_cloud (stereo only) |
| /perception/detections | Detection3DArray (camera frame) | perception (stub, or server client with `perception:=server`) |
| /perception/status | String (JSON: mode, latency, fallback) | perception (server client) |
| /world/objects, /world/summary | Detection3DArray (map), String (JSON; per object `views`, plus `tentative` count; objects committed after `confirm_views` sightings, 2 in perception:=server mode) | world_model |
| /safety/estop, /safety/status | Bool, DiagnosticArray | safety |
| /safety/estop_request | Bool (latched) | operator |
| /planner/command | String | ask |
| /planner/reply, /planner/status | String | planner |
| /robot/say | String | behavior |
| /behavior/execute_plan | hsim_interfaces/ExecutePlan (action) | behavior |
| /navigate_to_pose, /spin | Nav2 actions | Nav2 |
| /sim/set_body_pose | hsim_interfaces/SetBodyPose (TEST-ONLY) | sim |
| /sim/ground_truth/{odom,state} | PRIVILEGED: graders and labeled stubs only (decision 0007) | sim |

TF: `map → odom` (localization) → `base_footprint` (state_estimation) → `base_link` → `torso_link` (leg_odometry) →
`head_camera` (camera_mount) → `head_camera_optical`; `base_footprint → lidar_link`; `base_link → imu_link`.
