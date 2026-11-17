# Stack walkthrough: this repo, file by file

[end_to_end.md](end_to_end.md) explains the layers of a humanoid stack in general. This document follows the same
path through **this repo's code**: which launch file starts what, which file each message passes through, and where the
simulator ends and the hardware begins. Topic names, owners and rates are the contract's
([`docs/interface_contract.md`](../interface_contract.md)); the root `README.md` has the swap points.

## 1. Starting the stack

Everything starts from `ros2_ws/src/hsim_bringup/launch/`:

| Launch file | Starts |
|---|---|
| `stub_stack.launch.py` | every node as a lightweight stub (no physics): for checking wiring and the contract |
| `sim_stack.launch.py` | the robot: MuJoCo (`hsim_sim`), sensors, state estimation, the walking policy, `twist_mux`, safety, teleop |
| `autonomy.launch.py` | `sim_stack` plus localization and Nav2 (`hsim_navigation`), perception and the object memory (`hsim_perception`), the behavior tree (`hsim_behavior`), the planner (`hsim_planner`) |

Both real launch files read a **robot profile**, `hsim_bringup/config/robots/<robot>.yaml` (`robot:=wisco` by default):
which MuJoCo world, which walking policy folder, the body and sensor-mount names, the default sensors and the
environment build folder. Changing robots is choosing a different profile; no node has a robot name in its code.

`env:=<source>/<name>` picks a room from `assets/envs/`, already built into a single MuJoCo file plus a Nav2 map and a
list of named objects (`tools/env_build.py`, see `assets/envs/README.md`).

## 2. A request becomes skills

```
tools/ask.sh "walk to the sink"  ─► /planner/command
hsim_planner/planner_node.py      reads /world/summary, calls Claude with prompt.py's schema
                                  ─► list of skills (navigate_to, find_object, search_for, look_around, pick, place, say)
                                  ─► /behavior/execute_plan (action)
```

`prompt.py` holds the system prompt and the JSON schema; the planner never produces coordinates, only skill names and
object names the robot has actually seen. `ask.py` is the command-line client behind `tools/ask.sh`.

## 3. Skills become motion

`hsim_behavior/src/behavior_node.cpp` runs a BehaviorTree.CPP v4 tree for the plan, with a safety check every tick:

- `navigate_to` resolves the object from `/world/objects`, picks a staging pose clear of obstacles, sends Nav2's
  `/navigate_to_pose`, then does its own short step-in (`/cmd_vel_approach`) to end within arm's reach, facing the target;
- `find_object`, `search_for`, `look_around` turn the robot in place and watch the object memory;
- `pick` / `place` call the `/skills/manipulate` action (`hsim_manipulation`).

Nav2 (`hsim_navigation/config/nav2.yaml`) plans on `/map` and drives through Regulated Pure Pursuit and the velocity
smoother to `/cmd_vel_nav`. Its limits keep every command above the walking policy's deadband.

## 4. Velocity commands meet

`twist_mux` (`hsim_bringup/config/twist_mux.yaml`) merges `/cmd_vel_teleop` (keyboard or viewer, highest priority),
`/cmd_vel_approach` and `/cmd_vel_nav` into `/cmd_vel`, and locks everything while `/safety/estop` is true.
`hsim_safety/safety_monitor.py` owns that lock (operator request, fall, stale sensors, battery).

## 5. The walking policy

`hsim_locomotion/policy_node.py` (node `locomotion`) loads one folder from `models/locomotion/`:

- `policy_contract.py` reads the folder's `manifest.yaml` and checks it against the ONNX file: joint order, default
  pose, action scale, observation layout, gains, rate. A mismatch stops the node instead of walking badly.
- Every 20 ms (50 Hz) it builds the observation from `/joint_states`, `/imu` and `/cmd_vel`, runs the ONNX policy on the
  CPU and publishes **`/locomotion/joint_targets`**: one position target for every joint.
- With an arms-free policy, the joints it does not drive (arms, head) take their targets from `/arm/joint_targets`
  (published by manipulation) or hold the default pose.
- With a tracking policy (manifest v3, `training/tracking/`), it plays the motion clip carried in the ONNX file
  instead: the clip is aligned to where the robot stands (TF `odom → torso_link`) and advances one step per tick;
  `/cmd_vel` is ignored.

## 6. The robot backend

**In simulation** (`hsim_sim/mujoco_sim.py`, node `sim`): the joint targets go into MuJoCo's position actuators, which
apply the same PD (kp, kd from the robot's numbers file) at the 5 ms physics step, exactly as in training. The node
publishes `/joint_states`, `/imu`, `/clock` and the battery model; in *lockstep* it holds physics at each control
boundary until the policy's new targets arrive, so a slow laptop slows the world instead of starving the policy.
`stereo_camera.py` renders the head camera; `sensors.py` adds sensor noise; `/sim/ground_truth/*` exists only for
graders and labeled stubs.

**Through ros2_control in simulation** (`backend:=ros2_control`): `hsim_sim/joint_target_bridge.py` turns the named
targets into the forward position controller's command, and the upstream `mujoco_ros2_control` plugin runs the same
trained scene as hardware (files from `hsim_sim/ros2_control_backend.py`). Same PD law, no lockstep: the control path
the robot uses.

**On hardware**, the same `/locomotion/joint_targets` go to `ros2_control` on the RK3588
(`ros2_ws/src/wisco_ethercat_ros2_hw/`): legs, waist and head as MyActuator mode-5 targets with the trained kp and kd,
the arms through `ArmController` (gravity compensation + PD). How that layer works:
[hardware_interface.md](hardware_interface.md). The backends side by side are in the root `README.md` ("Robot
backends").

## 7. Knowing where the robot is

- `hsim_state_estimation/leg_odometry.py`: stance-foot kinematics from `/joint_states` → `/leg_odometry`; fused with
  `/imu` by `robot_localization`'s EKF (node `state_estimation`) → `/odometry/filtered` and the `odom → base_footprint`
  transform.
- `hsim_navigation/launch/navigation.launch.py`: RTAB-Map visual SLAM (default), slam_toolbox, or AMCL on the env's map
  → `map → odom` and `/map`.

## 8. Seeing objects

- `hsim_perception/perception_client.py` (node `perception`) sends color + depth frames to the GPU detection server
  (`services/perception_server/`: Grounding DINO + SAM 2.1 + depth) and publishes `/perception/detections`.
  `perception_stub.py` is the same node fed by line-of-sight checks in the simulator.
- `world_model.py` moves detections into the map frame, merges them across views and publishes `/world/objects` and the
  text `/world/summary` the planner reads.

## 9. Manipulation

`hsim_manipulation/manipulation_node.py` serves `/skills/manipulate`: `vla_client.py` streams camera frames and joint
state to the VLA server (`services/vla_server/`, a fine-tuned GR00T model trained in `training/manipulation/`) and
publishes the returned arm targets on `/arm/joint_targets`, which the walking policy passes through while it keeps
balance. The arm pipelines (teleop, datasets, ACT / pi0.5 training) are in `manipulation/`.

## 10. Where the models come from

| Artifact | Made by | Lives in |
|---|---|---|
| Robot model and numbers | `robots/<robot>/model/` scripts from CAD and datasheets; actuator values refined by `training/sim2real/pace.py` | `robots/<robot>/` |
| Simulation world of the robot | `tools/build_robot_world.py` from the training task | `assets/robots/` |
| Rooms | `tools/env_build.py` (and `tools/robocasa_export.py` for kitchens) | `assets/envs/<env>/build*/` |
| Walking and tracking policies | `training/train.sh` on the GPU machine, exported by `tools/vm.sh pull-policy` | `models/locomotion/` |
| Manipulation policy | `manipulation/finetune.sh` (ACT, pi0.5, GR00T over an arm profile) | served by `services/vla_server/` |

## 11. Checking it

- `python -m pytest tests ros2_ws/src/*/test training/sim2real manipulation/finetune`: the contract, docs, repo consistency, every package's ROS-free
  logic, the actuator fit and the pi0.5 trainer's checks.
- `tools/ros.sh test [stub|sim] [env] [robot] [backend]`: brings the stack up in the container, checks the live graph
  against the contract (`hsim_bringup/contract_check.py`) and drives the robot (`drive_check.py`);
  `tools/ros.sh test sim "" wisco ros2_control` runs it through the ros2_control backend.
- `tools/ros.sh autotest [name] [robot]`: navigation, behavior-tree and reach checks graded on simulator ground truth
  (`tools/autonomy_test.sh`, `tools/eval/`).
