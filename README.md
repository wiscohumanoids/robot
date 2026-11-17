# WiscoHumanoids robot software

The complete software stack for the WiscoHumanoids humanoid, **Wisco V1**: from a spoken or typed command down to the
motor drives, in simulation and on the robot. This repository holds the code, the robot models, the training
pipelines, the hardware interface, and the knowledge base that explains how it all works. It is the single source of
truth for every software team.

**Compute.** On the robot, an **RK3588 (Rockchip)** runs the real-time side: the EtherCAT master, `ros2_control` and
the joint controllers. A **Jetson Thor or Jetson Orin NX** is the central computer: policies, perception, mapping and
planning. They talk over ROS 2 (DDS). During development the whole stack runs in a ROS 2 Jazzy container on a laptop,
and training runs on a cloud GPU machine.

**New here?** Read [How the stack works](#how-the-stack-works), find your team in
[Where each team works](#where-each-team-works), then go to that folder's README. For the concepts behind every layer
(planning, SLAM, reinforcement learning, manipulation, ros2_control, EtherCAT), read [`docs/knowledge/`](docs/knowledge/README.md)
or open the interactive [academy](docs/academy/README.md).

---

## How the stack works

```
 "walk to the sink" ──► planner (Claude) ──► behavior tree ──► Nav2 ──┐
 keyboard / viewer ──────────────────────────────────────────────────┤ /cmd_vel (twist_mux: teleop wins, e-stop locks)
                                                                     ▼
 perception + object memory ──► planner          walking policy (ONNX, 50 Hz) ◄── /arm/joint_targets ◄── manipulation (VLA)
 state estimation + SLAM ──────► Nav2, policy                  │ joint targets for every joint
                                                               ▼
              robot backend:  MuJoCo simulation   |   ros2_control ─► controllers ─► EtherCAT ─► motor drives
```

Each layer is a ROS 2 node and talks to the others only through topics, actions and services listed in one contract
([`docs/interface_contract.md`](docs/interface_contract.md)). That is what lets any layer be replaced (a stub by the
real thing, the simulator by the robot) without touching the others. Rates step down as the layers get more abstract:
motor drives at the EtherCAT cycle, physics and PD at 200 Hz in simulation, the walking policy at 50 Hz, Nav2's
controller at 20 Hz, the behavior tree at 20 Hz, the planner once per request.

How one command flows through this repo's code, file by file:
[`docs/knowledge/stack_walkthrough.md`](docs/knowledge/stack_walkthrough.md); the same layers explained in general:
[`docs/knowledge/end_to_end.md`](docs/knowledge/end_to_end.md).

## Repository map

| Folder | What it holds |
|---|---|
| [`ros2_ws/src/`](ros2_ws/README.md) | the ROS 2 stack: one package per layer (`hsim_*`), and the hardware interface and controllers (`wisco_ethercat_ros2_hw`) |
| [`robots/`](robots/README.md) | one folder per robot we build: [`wisco_v1/`](robots/wisco_v1/README.md) (full body) and [`arm01/`](robots/arm01/README.md) (7-DOF test arm): models, numbers with their sources, calibration, hardware observations |
| [`training/`](training/README.md) | GPU training: walking policies, motion tracking, humanoid manipulation (one entry point, `training/train.sh`); sim-to-real (`sim2real/`) |
| [`manipulation/`](manipulation/README.md) | arm pipelines: real-arm teleop and recording, arm simulation, imitation learning and VLA fine-tuning (one entry point, `manipulation/finetune.sh`, over arm profiles) |
| [`models/`](models/locomotion/README.md) | deployed policies (ONNX + manifest) |
| [`assets/`](assets/README.md) | simulation worlds: robot worlds exported from training, rooms and kitchens |
| [`services/`](services/README.md) | GPU inference servers: object detection, VLA |
| [`hardware/`](hardware/README.md) | the robot's computers and buses, the wireless e-stop |
| [`tools/`](tools/README.md), [`tests/`](tests/README.md) | developer scripts and graders; repository checks |
| [`infra/`](infra/README.md) | the development container, the GPU machine, the training job queue, the RK3588 real-time setup |
| [`docs/`](docs/README.md) | interface contract, knowledge base, academy |

## Where each team works

| Team | Main folders | Start with |
|---|---|---|
| Planning and behavior | `ros2_ws/src/hsim_planner`, `hsim_behavior`, `hsim_interfaces` | `hsim_behavior/README.md` |
| Perception | `ros2_ws/src/hsim_perception`, `services/perception_server` | `hsim_perception/README.md` |
| State estimation, mapping, navigation | `ros2_ws/src/hsim_state_estimation`, `hsim_navigation` | `hsim_navigation/README.md` |
| Locomotion | `training/locomotion`, `training/tracking`, `training/sim2real`, `models/locomotion`, `ros2_ws/src/hsim_locomotion` | `training/README.md` |
| Manipulation | `manipulation/`, `training/manipulation`, `services/vla_server`, `ros2_ws/src/hsim_manipulation` | `manipulation/README.md` |
| Controls and hardware interface | `ros2_ws/src/wisco_ethercat_ros2_hw`, `robots/arm01`, `hardware/`, `infra/rk3588` | `ros2_ws/src/wisco_ethercat_ros2_hw/README.md` |
| Robot model (with mechanical and electrical) | `robots/wisco_v1` | `robots/wisco_v1/README.md` |
| Simulation and infrastructure | `ros2_ws/src/hsim_sim`, `hsim_bringup`, `assets/`, `tools/`, `infra/` | `ros2_ws/src/hsim_bringup/README.md` |

## Swap points

The stack is built so the parts most often changed can be swapped from the command line or a config file.

| What you swap | How | Options today |
|---|---|---|
| **Robot** | `robot:=<profile>` on any launch file; profiles in `ros2_ws/src/hsim_bringup/config/robots/` | `wisco` (default), `g1`, `g1_arms_free` |
| **Walking policy** | `policy:=<folder>` (else the profile's) | any folder in `models/locomotion/` |
| **Room** | `env:=<source>/<name>` | `test/table_room`, `robocasa/kitchen_l<N>_s<N>` |
| **Sensors** | `sensors:=stereo\|lidar` | ZED X stereo (Wisco's default), head lidar (G1's) |
| **Localization** | `mapping:=rtabmap\|slam\|map` | RTAB-Map visual SLAM (default), slam_toolbox, AMCL on a known map |
| **Perception / estimation** | `perception:=server\|stub`, `estimator:=ekf\|truth` | real detector or line-of-sight stub; EKF or ground truth (debug) |
| **Robot backend (low level)** | `backend:=sim\|ros2_control` | MuJoCo simulation node, or ros2_control + MuJoCo; ros2_control + EtherCAT on hardware |
| **Training job** | `infra/vm_jobs/jobs.yaml` + `tools/vm.sh jobs launch <job>` | Wisco / G1 walking, tracking, GR00T fine-tune |

**Robot backends.** The layer under the walking policy and the arm controller can be:

| Backend | What runs | What it tests | State |
|---|---|---|---|
| MuJoCo simulation node (`hsim_sim`) | joint targets go straight into MuJoCo, which applies the PD and the physics | everything above the controllers | ✅ default |
| `ros2_control` + MuJoCo plugin (upstream `mujoco_ros2_control`, `backend:=ros2_control`) | the robot's own receivers (`JointTargetController` for legs and arms: hold, ramp-in, limits, timeout) under a real controller manager, with the world (kitchens included) as the hardware; MuJoCo's position actuators apply the trained kp / kd at the physics step, the law the drives run in mode 5 | the robot's control path under full autonomy, without lockstep | ✅ walks to targets in a kitchen with SLAM and perception |
| `ros2_control` + simulated EtherCAT drives on MuJoCo (`backend:=ethercat_sim`) | the robot computer's controller chain (`JointTargetController` → `ParallelLinkageController` for the ankles and waist) on a simulated drive per motor (CiA402 state machine, mode 5 PD in the drive, one bus cycle each way), MuJoCo as the robot, kitchens included | the robot's control path down to the PDOs, the two-motor joints included | ✅ walks; found the ankles' mode-5 gains too far off the trained PD (`hsim_sim/README.md`) |
| `ros2_control` + mock EtherCAT drives (tests) | the controllers and the drive protocol against the same simulated drives on a toy joint | drive state machine, units, watchdogs, faults | ✅ in `wisco_ethercat_ros2_hw` tests |
| `ros2_control` + EtherCAT | the controllers on real MyActuator drives (`ethercat_driver_ros2`) | the robot | 🟡 arm01 on the bench |

## Quick start (laptop with Docker)

```bash
cp .env.example .env                       # add ANTHROPIC_API_KEY for the planner
tools/ros.sh up                            # build the ROS 2 Jazzy image and start the container
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements-mac.txt
.venv/bin/python tools/env_build.py assets/envs/test/table_room   # build the test room for Wisco V1
tools/ros.sh test sim test/table_room      # Wisco V1 walking in MuJoCo: contract check + drive test
tools/ros.sh shell                         # inside: ros2 launch hsim_bringup autonomy.launch.py env:=test/table_room
```

Watch with `tools/viewer.sh --env test/table_room`; talk to it with `tools/ask.sh "walk to the table"`. Real object
detection needs the perception server on the GPU machine and a tunnel (`ssh -f -N -L 8100:localhost:8100 hsim-gpu`);
without it use `perception:=stub`.

## Training on the GPU machine

`training/train.sh <robot> <task>` is the one entry point for walking and tracking runs, and
`manipulation/finetune.sh <method> <arm profile>` for manipulation fine-tunes. `tools/vm.sh sync` copies the repo to
the GPU machine; jobs are defined in `infra/vm_jobs/jobs.yaml` and launched with
`tools/vm.sh jobs launch <job>`; `tools/vm.sh pull-policy <run> <id>` exports a trained walking policy into
`models/locomotion/<id>/`, where the stack can load it. Machine setup: [`infra/README.md`](infra/README.md).

## Decisions this repo has made

- **ROS 2 Jazzy on Ubuntu 24.04**, so velocity commands are `geometry_msgs/TwistStamped` everywhere and behavior trees
  use BehaviorTree.CPP v4. macOS runs ROS in the container.
- **One publisher per topic and one owner per TF edge**, named by ROS node name, so a stub and its replacement share a
  name. The contract is enforced live (`contract_check`) and statically (`tests/`).
- **The robot stack only uses what real sensors give it.** Simulator ground truth is for graders and labeled stubs.
- **One numbers file per robot** (`robots/<robot>/model/robot_numbers.yaml`), every value with its source; the URDF,
  MuJoCo model and training model are generated from it.
- **Walking policies are trained in mjlab (MuJoCo-Warp) with PPO** and deployed as ONNX plus a manifest that fixes joint
  order, gains, default pose and observation layout, so training and deployment cannot disagree.
- **Two walking recipes:** *all-joints* (the policy drives every joint) and *arms-free* (legs and waist only, arms held
  or driven by manipulation; trained with random arm motion and payloads). Wisco V1 uses arms-free. Motion-tracking
  policies are trained separately; the stack's walking runs on the velocity policy.
- **Nav2 commands stay above the walking policy's deadband** (minimum speed while moving, fast in-place turns), because
  learned policies ignore very small velocity commands.
- **Autonomy:** Claude plans from the robot's own object memory and returns skills, never coordinates; a BehaviorTree.CPP
  tree runs the skills with a safety check every tick; Nav2's own tree handles path planning and recovery.
  `navigate_to` ends within arm's reach, facing the target (Nav2 to a staging spot, then the tree's own step-in).
- **Safety in layers:** a hardware e-stop that cuts motor power independent of software; the `safety` node's e-stop,
  which locks every velocity command in `twist_mux`; the behavior tree's safety check; the controllers' own limits and
  watchdogs.
- **Heavy models run as servers on the GPU machine** (object detection, VLA), called by thin ROS clients.
- **Environments come from a library** (hand-built rooms, RoboCasa kitchens), each built into the same set of files.
- **Drive modes:** legs, waist and head use MyActuator mode 5 (position + kp + kd every cycle, PD in the drive, the same
  gains the policy was trained with); the arms use the drive's position loop plus a gravity torque offset, or torque
  mode with the PD on the host (the arm controller).

## Working in this repo

- **Branch per change**, pull request into `main`; CI runs the static checks and a container build with smoke tests.
- **Interfaces are shared, internals are yours.** Change a topic, type, rate, owner, TF edge, action or service in
  `ros2_ws/src/hsim_bringup/config/interface_contract.yaml`, run `python3 tools/gen_docs.py`, and update the stub and
  every consumer in the same change.
- **Adding a node:** keep decision logic in a ROS-free module with plain pytest tests; publish topic names as string
  literals; add its interfaces to the contract with its node name as owner; start it from a launch file in
  `hsim_bringup`; give the package a README.
- **Generated files are never hand-edited:** robot models (`robots/*/model/`), training models, the contract tables.
- **Checks:** `python -m pytest tests ros2_ws/src/*/test training/sim2real manipulation/finetune` (no ROS needed); `tools/ros.sh test [stub|sim] [env] [robot] [backend]`
  (stack smoke test); `tools/ros.sh autotest [name] [robot]` (autonomy checks).
- **Code and docs describe the work, not who did it**; history and authorship live in git.

## Known limitations

- Wisco V1 has no lidar: in simulation its stereo camera needs a built environment (`env:=`).
- The deployed Wisco V1 walking policy is trained up to 1.0 m/s forward.
- The strict reach grader (0.15–0.40 m from a surface, square and centered) is sometimes missed by a few centimetres;
  improving the final approach is open work in `hsim_behavior`.
- Manipulation in the full humanoid simulation needs the gripper-equipped robot model and the training camera in
  `hsim_sim`; the skill runs today against the VLA server on its own.
- Hardware findings from simulation (masses, ankle motor sizing, calibration, IMU axes, drive gains and delay, supply
  voltage) are tracked in [`robots/wisco_v1/hardware_observations.md`](robots/wisco_v1/hardware_observations.md).
