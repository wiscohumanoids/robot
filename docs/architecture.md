# Architecture

Status legend: ✅ built and tested · 🟡 stub (correct interface, fake inside) · ⬜ planned · ⏸ deferred
Last updated: 2026-11-08 (Phases 0–2 done, Phase 3 autonomy built: realistic sensing, estimation, localization, Nav2, perception stub + memory, BT, safety, Claude planner)

## 1. The idea

You give the robot an instruction, by keyboard now and in plain English later. Each layer turns it into something more concrete and passes it down: *"pick up the red block"* → skills → a path → velocity commands → joint angles → motor torques → physics. Every layer is a separate ROS 2 node, and nodes talk only through ROS topics, services and actions, never by importing each other's code. That's what lets a stub be swapped for the real node without touching anything else.

```
 YOU ──keys──► viewer (Mac) ──► teleop ─────────────────┐
 YOU ──text──► LLM planner ──► behavior tree ──► Nav2 ──┤        (two ways to command the robot)
                                   │      │             ▼
                                   │      │        twist_mux  ── picks one: teleop wins
                                   │      │             │ /cmd_vel  "walk at this speed"
                                   │      │             ▼
                                   │      │     locomotion policy (ONNX, 50 Hz)
                                   │      │             │ joint targets
                                   │      ▼             ▼
                                   │  perception    MuJoCo sim (physics, 200 Hz) ──► viewer (Mac)
                                   ▼  (GPU VM)          │ sensors: joints, IMU, pose
                              manipulation ◄────────────┤
                              (VLA on GPU VM)           ▼
                                               state estimation ─► localization ─► Nav2 / everyone
```

**Multi-rate cascade:** planner (on event) → behavior tree (20 Hz) → Nav2 (planner ~1 Hz, controller 20 Hz) → locomotion policy (50 Hz) → physics + PD (200 Hz). Each layer reads the latest message from the one above and never waits on it.

## 2. Where things run

| Machine | Runs | Status |
|---|---|---|
| Mac | MuJoCo viewer (`tools/viewer.sh`), standalone player (`tools/play.sh`), editing | ✅ |
| Docker on the Mac (Ubuntu 24.04, ROS 2 Jazzy) | every ROS node | ✅ |
| GPU VM, g7e.2xlarge (RTX PRO 6000, 96 GB) | training (mjlab now, VLA later); later the perception + VLA servers | ✅ training / ⬜ servers |
| CPU VM | terminated; not needed while the Mac handles the stack | — |

## 3. Layers, bottom up

### 3.1 World + sensors (simulator side) ✅
- `sim` (`hsim_sim/mujoco_sim.py`): MuJoCo 3.11, training physics (5 ms), position actuators = PD loop. Realistic
  sensor models (`sensors.py`): IMU with noise/bias and an onboard attitude filter, head lidar (16-beam rays flattened
  to `/scan`), battery drained by motor power. `/sim/reset`, test-only `/sim/set_body_pose`, privileged ground truth.
- `stereo_camera` (`sensors:=stereo`): ZED X-like color + depth with a stereo error model, in its own process.

### 3.2 Walking: `locomotion` ✅ — ONNX policy, 50 Hz (unchanged; body velocity now comes from the real estimator).

### 3.3 Commands: `twist_mux` ✅ — teleop (100) beats Nav2 (10); **e-stop lock** (255) blocks everything.

### 3.4 Manual driving + viewer: `teleop`, `viewer_bridge` + Mac viewer ✅ (UDP side channels)
Follow camera with see-through walls, a "robot's view" panel (its own map, lidar, object memory, path)
and tasks typed in the viewer's terminal. See `ros2_ws/src/hsim_teleop/README.md`.

### 3.5 State estimation ✅
- `leg_odometry`: body velocity from stance-foot kinematics (<2% speed error) + body TF (pelvis height, tilt, waist).
- `state_estimation` (robot_localization EKF, 50 Hz): fuses leg odometry + IMU → `/odometry/filtered`, `odom → base_footprint`.

### 3.6 Localization ✅ (node `localization`, owns `map → odom`)
- `mapping:=map`: AMCL on the env's known map, started at the env spawn pose (operator-given).
- `mapping:=slam`: slam_toolbox builds the map while working (also publishes `/map`).
- `mapping:=rtabmap`: RTAB-Map visual SLAM from the stereo camera.

### 3.7 Navigation: Nav2 ✅ — NavFn + Regulated Pure Pursuit (≥0.3 m/s, turns at 0.9 rad/s), live lidar costmaps,
spin/backup/wait recoveries, rotation-aware progress checker, velocity smoother → `/cmd_vel_nav`. Avoids obstacles
that are on no map (dropped-box test).

### 3.8 Perception 🟡 + object memory ✅
- `perception` (stub): honest line-of-sight detections (FOV, range, min size, occlusion), relative to the camera, noisy.
- `perception` (server client, `perception:=server`, needs `sensors:=stereo`, 2026-11-10): sends ZED X RGB-D to
  `services/perception_server` on the GPU VM (Grounding DINO base+tiny + SAM 2.1 + depth -> 3D; SSH tunnel to
  localhost:8100), publishes the same `/perception/detections`; prompts = default kitchen list + env labels + planner
  targets; falls back to the stub when the server is unreachable. Status on `/perception/status`. Measured: 3D error
  1-4 cm median, 0.4-0.8 s/frame, precision 0.41 (mislabels are the weak spot). Details: docs/runs/2026-11-10_worklog_E.md.
- `world_model`: places detections on the map with the robot's own pose, tracks/merges sightings, relations ("on"),
  `/world/objects` + `/world/summary`.

### 3.9 Safety ✅ — `safety` node (operator e-stop, fall, battery, stale sensors) → `/safety/estop` lock + `/safety/status`.

### 3.10 Task behavior tree ✅ — `behavior` (BehaviorTree.CPP v4): `/behavior/execute_plan`, ReactiveSequence with
IsSafe, skills navigate_to / find_object / search_for / look_around / say (pick/place: Phase 5). Groot2 on port 1667.
navigate_to ends within arm's reach, facing the target: picks a side it can stand at on its own map, Nav2 to a
staging spot, then its own short step-up on `/cmd_vel_approach` (twist_mux: teleop 100 > approach 20 > nav 10).

### 3.11 Planner ✅ — `planner` (Claude, decision 0006): `/planner/command` → plan → BT; replans on failure.
`tools/ask.sh "walk to the table"` from the Mac.

### 3.12 Manipulation ⬜ (Phase 5), real perception ⬜ (Phase 4).

### How to run
Defaults are realistic (2026-11-11): RTAB-Map SLAM from scratch (no prebuilt map) and real detection on the VM
(no privileged perception). Mac first: `ssh -f -N -L 8100:localhost:8100 hsim-gpu` (perception tunnel).

    tools/ros.sh shell
    ros2 launch hsim_bringup autonomy.launch.py robot:=<g1|g1_v2|wisco> env:=robocasa/kitchen_l1_s1 [sensors:=lidar|stereo]
    # Mac:   tools/viewer.sh --robot <same> --env robocasa/kitchen_l1_s1     tools/ask.sh "walk to the sink"
    # stop:  tools/ros.sh stop (from the Mac)     e-stop: ros2 run hsim_safety estop on|off

| robot:= | body + walking policy | sensors (default first) |
|---|---|---|
| g1 | G1, v1 policy (all joints) | lidar (+ D435i-like head camera), stereo (ZED X) |
| g1_v2 | G1, v2 policy (legs + waist, arms held) | lidar (+ head camera), stereo |
| wisco | Wisco V2, v2 policy (legs + torso, arms/head held) | stereo (ZED X, pitched 25 deg down) |

Test/debug only (privileged): `mapping:=map` (AMCL on the prebuilt map), `perception:=stub` (simulator positions).
Tests: `tools/ros.sh autotest [name] [robot]` (uses mapping:=map perception defaults per check).

## 4. Interface

The machine-readable contract is `ros2_ws/src/hsim_bringup/config/interface_contract.yaml`, checked live by `contract_check`. The table with notes is in `docs/interface_contract.md`. Rules: one publisher per topic and per TF edge, the declared message type and rate, and owners named by ROS node name, so a stub and the real node share a name.

## 5. Offline pipelines

| Pipeline | Tech | Where | Status |
|---|---|---|---|
| Walking training | mjlab 1.6.0, MuJoCo-Warp, RSL-RL PPO, 4,096 parallel robots → ONNX + manifest | GPU VM | ✅ v1 |
| Motion tracking (LAFAN1) | same, "match the clip" reward | GPU VM | ⏸ |
| Environment build | `tools/env_build.py`: G1 world + room → `world.mjb`, Nav2 map, object list, previews (decision 0005) | Mac | ✅ test room, RoboCasa kitchens |
| RoboCasa export | `tools/robocasa_export.py` (RoboCasa's own env): kitchen + task objects, robot removed | Mac | ✅ |
| Phone scan, generated rooms | extensions to the environment library | Mac | ⬜ later |
| VLA fine-tuning | LeRobot, sim demos → SmolVLA | GPU VM | ⬜ Phase 5 |

**Train → deploy contract:** mjlab embeds joint order, PD gains, default pose, action scale and observation layout in the ONNX file. `export_manifest.py` turns that into `manifest.yaml`, which is all the deploy side reads. The CPU deployment was checked against mjlab's own evaluation (`eval_commands.py`), and the two agree closely.

## 6. Code map

| Path | Contents |
|---|---|
| `ros2_ws/src/hsim_sim` | MuJoCo sim node + kinematic stub |
| `ros2_ws/src/hsim_locomotion` | ONNX policy node + stub |
| `ros2_ws/src/hsim_teleop` | viewer bridges: keys + typed tasks in, robot's perception out |
| `ros2_ws/src/hsim_state_estimation` | state estimation stub |
| `ros2_ws/src/hsim_bringup` | launch files (`sim_stack`, `autonomy`, `stub_stack`), contract, configs, `contract_check`, `drive_check` |
| `ros2_ws/src/hsim_navigation` | Nav2, AMCL, slam_toolbox, RTAB-Map configs + launch |
| `ros2_ws/src/hsim_perception` | perception stub, world model |
| `ros2_ws/src/hsim_behavior` | task behavior tree (C++) |
| `ros2_ws/src/hsim_planner` | Claude planner, `ask` CLI |
| `ros2_ws/src/hsim_safety` | safety monitor, e-stop CLI |
| `ros2_ws/src/hsim_interfaces` | Skill, SkillPlan, ExecutePlan, SetBodyPose |
| `tools/eval/` | graders and debug tools (the only code allowed to read ground truth) |
| `tools/` | `viewer.sh`, `play.sh`, `ros.sh` (container: up/shell/build/test/stop/down), VM sync, policy pull |
| `training/locomotion/` | mjlab train / eval / export scripts (GPU VM) |
| `models/locomotion/` | policies, manifests, model cards |
| `assets/robots/g1/` | G1 training world from mjlab |
| `assets/envs/` | environments (`env.yaml` + built `world.mjb`, map, objects) |
| `infra/` | Dockerfile, compose, GPU VM setup |
| `docs/` | this file, interface contract, infra, Phase 0 write-up, decisions 0001–0004 |
