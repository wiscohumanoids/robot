# Architecture

Status legend: ✅ built and tested · 🟡 stub (correct interface, fake inside) · ⬜ planned · ⏸ deferred
Last updated: 2026-11-15 (repos unified: simulation + autonomy stack, Wisco V1 robot model and training, EtherCAT
hardware interface, wireless e-stop, arm teleop and training pipelines).

The robot this stack is built around is **Wisco V1** (WiscoHumanoids' humanoid, `robots/wisco_v1/`). The Unitree G1
is kept as a reference robot: same stack, same interfaces, its own profile. For where each part stands right now
see [`STATUS.md`](../STATUS.md); for how the whole stack works conceptually see [`knowledge/`](knowledge/README.md).

## 1. The idea

You give the robot an instruction, by keyboard or in plain English. Each layer turns it into something more concrete
and passes it down: *"pick up the red block"* → skills → a path → velocity commands → joint angles → motor torques →
physics (or motors). Every layer is a separate ROS 2 node, and nodes talk only through ROS topics, services and
actions, never by importing each other's code. That is what lets a stub be swapped for the real node, and the
simulator for the real robot, without touching anything else.

```
 YOU ──keys──► viewer (Mac) ──► teleop ─────────────────┐
 YOU ──text──► LLM planner ──► behavior tree ──► Nav2 ──┤        (two ways to command the robot)
                                   │      │             ▼
                                   │      │        twist_mux  ── picks one: teleop wins
                                   │      │             │ /cmd_vel  "walk at this speed"
                                   │      │             ▼
                                   │      │     locomotion policy (ONNX, 50 Hz) ◄── /arm/joint_targets (held joints)
                                   │      │             │ /locomotion/joint_targets (every joint)
                                   │      ▼             ▼
                                   │  perception    robot backend: MuJoCo sim (200 Hz) today; ros2_control + EtherCAT on the robot (⬜)
                                   ▼  (GPU server)      │ sensors: joints, IMU, camera
                              manipulation ◄────────────┤
                              (VLA server)              ▼
                                               state estimation ─► localization ─► Nav2 / everyone
```

**Multi-rate cascade:** planner (on event) → behavior tree (20 Hz) → Nav2 (planner ~1 Hz, controller 20 Hz) →
locomotion policy (50 Hz) → physics + PD (200 Hz; on the robot the drivers run the PD at the EtherCAT cycle rate).
Each layer reads the latest message from the one above and never waits on it.

## 2. Where things run

| Machine | Runs | Status |
|---|---|---|
| Mac | MuJoCo viewer (`tools/viewer.sh`), standalone player (`tools/play.sh`), editing | ✅ |
| Docker on the Mac (Ubuntu 24.04, ROS 2 Jazzy, `infra/docker/`) | every ROS node | ✅ |
| GPU VM, g7e.2xlarge (RTX PRO 6000, 96 GB) | training (mjlab, GR00T), perception server, VLA server | ✅ |
| **On the robot:** Jetson Thor | full ROS 2: locomotion policy, perception, SLAM, planner, behavior tree; sends targets over DDS (50–100 Hz) | ⬜ |
| **On the robot:** RK3588 | real-time side: EtherCAT master, `ros2_control` + hardware interface, arm controller; e-stop receiver | 🟡 one motor on a bench (`ros2_ws/src/wisco_ethercat_ros2_hw/`) |
| Handheld ESP32-S3 | wireless e-stop transmitter (Wi-Fi heartbeat to the RK3588) | 🟡 tested with a laptop receiver (`hardware/estop/`) |

## 3. Layers, bottom up

### 3.1 Robot backend

**Simulation ✅** — `sim` (`hsim_sim/mujoco_sim.py`): MuJoCo 3.11, training physics (5 ms), position actuators = the
PD loop with the policy's kp/kd. Realistic sensor models (`sensors.py`): IMU with noise/bias and an onboard attitude
filter, head lidar (G1 only; 16-beam rays flattened to `/scan`), battery drained by motor power. `/sim/reset`,
test-only `/sim/set_body_pose`, privileged ground truth. `stereo_camera` (`sensors:=stereo`, Wisco's default): ZED X
color + depth with a stereo error model, in its own process.

**Real robot ⬜** — the same joint targets go to `ros2_control` on the RK3588, whose hardware plugin is
`ethercat_driver_ros2` (`ros2_ws/src/wisco_ethercat_ros2_hw/`: one MyActuator X4-10 runs on a bench in CST mode at
100 Hz without PREEMPT_RT). Planned drive modes (decided 2026-11-10, `robots/wisco_v1/CHANGES.md`):

| Joints | MyActuator EtherCAT mode | Who computes torque |
|---|---|---|
| Legs, waist, head | 5 "PVT/MIT": position target + kp + kd every cycle | the driver, with the same kp/kd the policy was trained with |
| Arms | 10 "CST": torque target | the arm controller (gravity compensation + impedance, Pinocchio), ⬜ its own package |

A ros2_control **simulation** backend (`mujoco_ros2_control`, ported, not built yet) will let the exact controller
and URDF configuration used on the robot run against MuJoCo first. See `docs/migration/porting_backlog.md`.

### 3.2 Walking: `locomotion` ✅
ONNX policy, 50 Hz, reads `models/locomotion/<policy>/manifest.yaml` (joint order, kp/kd, default pose, action scale,
observation layout). Arms-free policies (manifest v2) drive legs + waist/torso and hold the other joints at
`/arm/joint_targets` (or the default pose); all-joints policies (manifest v1) drive every joint.

### 3.3 Commands: `twist_mux` ✅ — teleop (100) > behavior tree's final approach (20) > Nav2 (10); the **e-stop lock**
(255) blocks everything.

### 3.4 Manual driving + viewer: `teleop`, `viewer_bridge` + Mac viewer ✅ (UDP side channels)
Follow camera with see-through walls, a "robot's view" panel (its own map, scan, object memory, path) and tasks typed
in the viewer's terminal. See `ros2_ws/src/hsim_teleop/README.md`.

### 3.5 State estimation ✅
- `leg_odometry`: body velocity from stance-foot kinematics (<2% speed error) + body TF (pelvis height, tilt, waist).
- `state_estimation` (robot_localization EKF, 50 Hz): fuses leg odometry + IMU → `/odometry/filtered`,
  `odom → base_footprint`.

### 3.6 Localization ✅ (node `localization`, owns `map → odom`; in `hsim_navigation`)
- `mapping:=rtabmap` (default): RTAB-Map visual SLAM from the head camera (plus the scan on the G1).
- `mapping:=slam`: slam_toolbox builds the map while working.
- `mapping:=map`: AMCL on the environment's prebuilt map, started at the spawn pose (tests and debugging).

### 3.7 Navigation: Nav2 ✅ — NavFn + Regulated Pure Pursuit (≥0.3 m/s, turns at 0.9 rad/s), live costmaps from the
scan (Wisco: depth-derived scan, obstacles trusted to 3 m), spin/backup/wait recoveries, velocity smoother →
`/cmd_vel_nav`. Avoids obstacles that are on no map (dropped-box test).

### 3.8 Perception ✅ + object memory ✅
- `perception` (server client, `perception:=server`, default): sends head-camera RGB-D to
  `services/perception_server` on the GPU VM (Grounding DINO base+tiny + SAM 2.1 + depth → 3D; SSH tunnel to
  localhost:8100), publishes `/perception/detections`. Measured: 3D error 1–4 cm median, 0.4–0.8 s/frame; mislabels
  are the weak spot. Details: `docs/runs/2026-11-10_worklog_E.md`.
- `perception` (stub, `perception:=stub`, tests): honest line-of-sight detections, privileged positions.
- `world_model`: places detections on the map with the robot's own pose, tracks and merges sightings, relations
  ("on"), `/world/objects` + `/world/summary`.

### 3.9 Safety ✅ — `safety` node (operator e-stop, fall, battery, stale sensors) → `/safety/estop` lock +
`/safety/status`. On the robot this sits on top of the hardware e-stop: the handheld transmitter cuts the 48 V
contactor through the RK3588 (`hardware/estop/`), independent of ROS.

### 3.10 Task behavior tree ✅ — `behavior` (BehaviorTree.CPP v4): `/behavior/execute_plan`, ReactiveSequence with
IsSafe, skills navigate_to / find_object / search_for / look_around / say; pick/place become one `/skills/manipulate`
goal. Groot2 on port 1667. navigate_to ends within arm's reach, facing the target: picks a side it can stand at on its
own map, Nav2 to a staging spot, then its own short step-up on `/cmd_vel_approach`.

### 3.11 Planner ✅ — `planner` (Claude, decision 0006): `/planner/command` → plan → behavior tree; replans on failure.
`tools/ask.sh "walk to the table"` from the Mac.

### 3.12 Manipulation
| Track | Where | Status |
|---|---|---|
| ROS skill: `/skills/manipulate` runs a VLA (GR00T) on the arms at 20 Hz while the legs keep balancing | `ros2_ws/src/hsim_manipulation/`, `services/vla_server/` | 🟡 built, smoke-tested in isolation; not yet run in the full sim |
| G1 + Dex1 grippers in sim: scripted demos → LeRobot dataset → GR00T N1.6 fine-tune | `training/manipulation/` | ✅ 52 % closed-loop on 50 held-out layouts |
| Real arm: Panthera-HT arms, gravity assist, Quest 3S teleop, human-in-the-loop recording, remote pi0.5 inference | `manipulation/panthera_teleop/` | ✅ on the bench arms |
| Arm in sim: keyboard teleop, scripted IK datasets, ACT and pi0.5 training, bimanual IK, actuator-assembly task | `manipulation/panthera_sim/` | ✅ |
| Wisco V1 arm control on the robot (gravity compensation + impedance, CST torque) | its own package | ⬜ |

## 4. How to run

Defaults are realistic: robot Wisco V1, RTAB-Map SLAM from scratch (no prebuilt map) and real detection on the VM (no
privileged perception). Mac first: `ssh -f -N -L 8100:localhost:8100 hsim-gpu` (perception tunnel).

    tools/ros.sh up            # once: build the image and start the container
    tools/ros.sh shell
    ros2 launch hsim_bringup autonomy.launch.py env:=robocasa/kitchen_l1_s1 [robot:=wisco|g1|g1_arms_free] [sensors:=lidar|stereo]
    # Mac:   tools/viewer.sh --robot <same> --env robocasa/kitchen_l1_s1     tools/ask.sh "walk to the sink"
    # stop:  tools/ros.sh stop (from the Mac)     e-stop: ros2 run hsim_safety estop on|off

| robot:= | Body + walking policy | Sensors (default first) |
|---|---|---|
| `wisco` (default) | Wisco V1, arms-free policy (legs + torso; arms and head held) | stereo (ZED X, pitched 25° down: assumed, see `robots/wisco_v1/HARDWARE_ISSUES.md` H8) |
| `g1` | G1, all-joints policy | lidar (+ D435i-like head camera), stereo |
| `g1_arms_free` | G1, arms-free policy (legs + waist; arms held) | lidar (+ head camera), stereo |

Test/debug only (privileged): `mapping:=map` (AMCL on the prebuilt map), `perception:=stub` (simulator positions).
Tests: `tools/ros.sh test [stub|sim] [env] [robot]` (smoke) and `tools/ros.sh autotest [name] [robot]` (autonomy);
static checks without ROS: `python -m pytest tests ros2_ws/src/*/test`.

## 5. Interface

The machine-readable contract is `ros2_ws/src/hsim_bringup/config/interface_contract.yaml`, checked live by
`contract_check` and statically by `tests/test_contract.py`. The tables are generated into
[`interface_contract.md`](interface_contract.md). Rules: one publisher per topic and per TF edge, the declared
message type and rate, and owners named by ROS node name, so a stub and the real node share a name.

## 6. Offline pipelines

| Pipeline | Tech | Where | Status |
|---|---|---|---|
| Robot model | CAD export + MyActuator datasheets → URDF/MJCF, linkage analysis, pre-training checks | `robots/wisco_v1/` (Mac) | ✅ |
| Walking training | mjlab 1.6.0, MuJoCo-Warp, RSL-RL PPO, 4,096 parallel robots → ONNX + manifest | `training/locomotion/` (GPU VM) | ✅ Wisco V1 arms-free; G1 all-joints + arms-free |
| Motion tracking (LAFAN1) | same, "match the clip" reward | `training/tracking/` (GPU VM) | ✅ trained on single clips (G1, Wisco V1); not used by the stack |
| Environment build | `tools/env_build.py`: robot world + room → `world.mjb`, Nav2 map, object list, previews (decision 0005) | Mac | ✅ test room, RoboCasa kitchens |
| RoboCasa export | `tools/robocasa_export.py`: kitchen + task objects, robot removed | Mac | ✅ |
| Manipulation (sim G1) | scripted demos → LeRobot → GR00T N1.6 | `training/manipulation/` (GPU VM) | ✅ |
| Manipulation (Panthera arm) | teleop / scripted demos → LeRobot → ACT, pi0.5 | `manipulation/` | ✅ |
| Current-draw estimate | policy in plain MuJoCo + datasheet motor model → per-motor and bus current | `robots/wisco_v1/tools/` | ✅ |
| Phone scan, generated rooms | extensions to the environment library | `scan_pipeline/` | ⬜ later |

**Train → deploy contract:** mjlab embeds joint order, PD gains, default pose, action scale and observation layout
in the ONNX file. `export_manifest.py` turns that into `manifest.yaml`, which is all the deploy side reads. The CPU
deployment was checked against mjlab's own evaluation (`eval_commands.py`), and the two agree closely.

## 7. Code map

| Path | Contents |
|---|---|
| `robots/wisco_v1/` | Wisco V1 robot: URDF/MJCF + meshes (generated from the CAD export in `original_from_team/`), `model/` (the numbers file, datasheets, generators, training MJCF), `checks/`, `HARDWARE_ISSUES.md`, current-draw tool + report |
| `ros2_ws/src/hsim_*` | the ROS stack, one package per layer (each has a README) |
| `ros2_ws/src/wisco_ethercat_ros2_hw/` | EtherCAT hardware configuration (slave/PDO maps, ros2_control test bench) |
| `ros2_ws/src/lowlevel_control/`, `mujoco_ros2_control/` | ported ros2_control controller and MuJoCo plugin (not built yet) |
| `hardware/estop/` | wireless e-stop: ESP32 transmitter firmware, RK3588 receiver |
| `manipulation/` | Panthera arm pipelines: `panthera_teleop/` (real arms), `panthera_sim/` (sim, IK, ACT/pi0.5) |
| `training/` | `locomotion/` (G1 tasks + `wisco_v1/`), `tracking/` (G1 + `wisco_v1/`), `manipulation/` (GR00T pipeline) |
| `models/locomotion/` | deployed policies: `policy.onnx` + `manifest.yaml` + training configs |
| `assets/` | `robots/` (mjlab-exported robot worlds), `envs/` (environments) |
| `services/` | GPU inference servers (perception, VLA) |
| `tools/`, `tools/eval/` | dev scripts; graders (the only code allowed to read ground truth) |
| `tests/` | static checks (contract vs code, docs, robot profiles, manifests) |
| `infra/` | container, GPU VM setup, overnight job runner |
| `docs/` | this file, interface contract, decisions, knowledge base, academy, hardware notes, run logs |
