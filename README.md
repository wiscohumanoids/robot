# WiscoHumanoids robot

The single source of truth for the WiscoHumanoids software stack: every layer from a spoken command down to the
motors, for our humanoid **Wisco V1**, in simulation today and on the robot next. Code, robot model, training,
hardware interface, and the knowledge base that explains how all of it works.

```
 "walk to the sink" ─► Claude planner ─► behavior tree ─► Nav2 ─┐
 keyboard / viewer ───────────────────────────────────────────┤ /cmd_vel
                                                               ▼
        perception + SLAM + state estimation ─────► walking policy (ONNX, 50 Hz) ◄── arm targets (manipulation)
                                                               │ joint targets
                                                               ▼
                                       MuJoCo (now)  |  ros2_control + EtherCAT ─► MyActuator drives (next)
```

**Stack:** ROS 2 Jazzy · MuJoCo 3.11 / mjlab 1.6 (RSL-RL PPO) · Nav2 · RTAB-Map · BehaviorTree.CPP · Claude ·
Grounding DINO + SAM 2.1 · GR00T / LeRobot · ros2_control + `ethercat_driver_ros2`

## Start here

| If you want to... | Read |
|---|---|
| Know **where things stand**: what's real, what's simulated, what's next, known issues | [`STATUS.md`](STATUS.md) |
| **Understand the whole stack** from first principles (planning, perception, RL, manipulation, ros2_control, EtherCAT) | [`docs/knowledge/`](docs/knowledge/README.md) and the interactive [academy](docs/academy/README.md) |
| See **how this repo's stack fits together**: layers, rates, machines, code map | [`docs/architecture.md`](docs/architecture.md) |
| Know **every topic, action and TF edge** and who owns it | [`docs/interface_contract.md`](docs/interface_contract.md) |
| Work on the **robot itself**: model, motors, numbers, hardware findings | [`robots/wisco_v1/`](robots/wisco_v1/README.md), [`hardware/`](hardware/README.md) |
| Train **walking** or **motion tracking** | [`training/`](training/README.md) |
| Work on **arms and hands** | [`manipulation/`](manipulation/README.md) |
| **Contribute**: workflow, interfaces, tests | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

## Quick start (Mac with Docker Desktop)

```bash
cp .env.example .env                     # add ANTHROPIC_API_KEY for the planner
tools/ros.sh up                          # build the ROS 2 Jazzy image and start the container (first time ~10-20 min)
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements-mac.txt   # Mac-side MuJoCo (viewer, env builder)
.venv/bin/python tools/env_build.py assets/envs/test/table_room      # build the test room for Wisco V1 (build_wisco/)
tools/ros.sh test sim test/table_room    # MuJoCo + Wisco V1 walking policy: contract check + drive test
tools/ros.sh shell                       # then: ros2 launch hsim_bringup autonomy.launch.py env:=test/table_room
```
Watch it with `tools/viewer.sh --env test/table_room`; talk to it with `tools/ask.sh "walk to the table"`. The full
autonomy defaults (RTAB-Map, perception on the GPU VM) and other robots (`robot:=g1`, `g1_arms_free`) are in
`docs/architecture.md` §4. Static checks, no ROS needed: `python -m pytest tests ros2_ws/src/*/test`.

## Repository layout

| Folder | What it holds |
|---|---|
| `robots/wisco_v1/` | the robot: URDF/MJCF + meshes (from CAD), every number with its source, pre-training checks, hardware findings |
| `ros2_ws/src/` | ROS 2 packages, one per layer (`hsim_*`), the EtherCAT hardware interface, ported ros2_control packages |
| `hardware/` | wireless e-stop (ESP32 + RK3588); overview of the real-robot side |
| `manipulation/` | arm pipelines: real Panthera arms (teleop, recording, pi0.5) and their simulation (IK, ACT, VLAs) |
| `training/` | GPU training: locomotion (Wisco V1, G1), motion tracking, humanoid manipulation (GR00T) |
| `models/` | deployed policies (ONNX + manifest) |
| `assets/` | robot worlds exported from mjlab, environments (test rooms, RoboCasa kitchens) |
| `services/` | GPU inference servers: perception, VLA |
| `tools/`, `tests/` | dev scripts and graders; static checks |
| `infra/` | container, GPU VM setup, overnight job runner |
| `docs/` | architecture, interface contract, decisions, knowledge base, academy, hardware notes, run logs, migration notes |

History: this repo merges the simulation/autonomy stack, the Wisco robot model and training, the EtherCAT hardware
interface, the wireless e-stop and the arm pipelines (2026-11-15). The pre-unification scaffold and the
planner/BT/Nav2 branch are kept as tags (`legacy-scaffold-v0`, `archive/task-planner`); renamed things are listed in
[`docs/migration/renames.md`](docs/migration/renames.md).
