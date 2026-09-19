# robot

The canonical ROS2 Humble humanoid software stack for WiscoHumanoids: **a frozen
interface contract, a stub for every layer of the stack, and the real low-level
infrastructure underneath**, so several teams can build their own piece in
parallel and swap it in without waiting on anyone else.

## Start here

| If you want to... | Read |
|---|---|
| Know **where the project stands**: what's real, what's stubbed, who owns what, what to do next | [`STATUS.md`](STATUS.md) |
| Know **every topic, action and TF edge** between layers, and who publishes it | [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md) |
| Understand **how the stack fits together** (both halves, rates, data flow, gaps) | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Find **where your team plugs in** (which stub you replace, what you must honor) | [`INTEGRATION_POINTS.md`](INTEGRATION_POINTS.md) |
| **Contribute**: workflow, changing an interface, adding a node, running the checks | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| See the EtherCAT / `ros2_control` / real-time research (and what's unverified) | [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) |

Every package has its own `README.md` stating its layer, status (real / stub /
scaffold), rate and interfaces. Read those before assuming anything works.

## How it works

Nodes never call each other; they publish/subscribe (or call an action) through
interfaces listed in the contract. Each topic and each TF edge has **exactly one
owner**. Every layer has a stub that honors its interface, so:

```bash
ros2 launch bringup full_stack.launch.py locomotion:=external   # whole stack, minus your stub
ros2 run my_pkg my_locomotion_node                              # your real node takes its place
ros2 run bringup check_contract.py                              # prove you honored the contract
```

**What stubs prove:** data flows through the right topics at the right rates
with the right types, and everyone can develop in parallel. **What they can't:**
real-time timing, dynamics, whether a policy balances the real robot. Those only
show up on hardware.

## The four-repo context

This repo does not contain a trained locomotion or manipulation policy, an LLM
planner, a real perception system, or a real EtherCAT master. It provides the
glue and the slots they plug into:

1. **`berkeley_humanoid`**: MuJoCo + Stable-Baselines3 PPO locomotion RL,
   sim-only, no ROS. Will eventually export `policy.onnx`, which drops into
   this repo's `locomotion_runner` slot. See `INTEGRATION_POINTS.md`
   "Locomotion" for the (nontrivial) 29-DOF-to-23-DOF reconciliation this
   requires.
2. **`bipedal_nav`**: a real ROS2 Humble workspace. Its `g1_description`
   (URDF+MJCF with `ros2_control` bindings) and `mujoco_ros2_control` bridge
   are ported into this repo as the canonical robot model and sim backend (see
   `src/g1_description/README.md` for the small, disclosed changes made). Its
   SLAM/state-estimation/planner nodes were non-functional stubs and were **not**
   brought over; this repo's stubs for those layers are new.
3. **`lerobot_alohamini`**: a LeRobot fork for bimanual manipulation,
   ZMQ-based, with no ROS dependency. Will eventually be wrapped as the
   `ExecuteManipulation` action server this repo's `manipulation_runner`
   already implements as a stub. See `INTEGRATION_POINTS.md` "Manipulation".
4. **`ros2_hub`**: an empty ROS2 tutorial workspace. Not used; superseded by this repo.

## Repo structure

```
robot/
  README.md  STATUS.md  INTERFACE_CONTRACT.md  ARCHITECTURE.md
  INTEGRATION_POINTS.md  CONTRIBUTING.md  RESEARCH_NOTES.md
  tools/gen_docs.py            regenerates the tables in STATUS.md / INTERFACE_CONTRACT.md from the YAML
  tests/                       static contract + consistency tests (pytest, no ROS needed)
  docker/                      ROS2 Humble + MuJoCo dev container (x86_64 and arm64, headless-capable)
  .github/workflows/ci.yml     static checks, then colcon build + stub-stack contract check + end-to-end task
  src/
    humanoid_interfaces/       THE contract: messages, action, canonical joint order, interface_contract.yaml
    bringup/                   full_stack / lowlevel_test launch files, controllers.yaml, check_contract.py, smoke_task.py

    # task stack (all stubs today)
    task_planner_stub/         /user_intent -> /skill_sequence
    behavior_tree_stub/        /skill_sequence -> navigate_to_pose + execute_manipulation, /task_status
    perception_stub/           /object_poses (hardcoded cube)
    slam_stub/                 /map, map->odom
    state_estimation_stub/     /robot_pose, odom->base_link (dead-reckons /cmd_vel)
    nav_stub/                  navigate_to_pose action server (stands in for Nav2) -> /cmd_vel_nav

    # velocity arbitration (real)
    teleop_input/              keyboard/joystick -> /cmd_vel_teleop
    cmd_vel_mux/               teleop > nav > zero -> /cmd_vel (sole publisher)

    # control stack
    locomotion_runner/         50 Hz stub gait + ONNX-policy slot
    manipulation_runner/       10 Hz stub ExecuteManipulation action server
    wbc_stub/                  500 Hz pass-through arbitration (stands in for a real WBC)
    lowlevel_control/          1000 Hz REAL joint impedance controller (ros2_control plugin)
    ethercat_bridge/           1000 Hz REAL SCAFFOLD ros2_control hardware_interface
    safety/                    100 Hz E-stop/fault monitor

    # platform
    g1_description/            canonical URDF/MJCF (ported from bipedal_nav; adds base_link)
    mujoco_ros2_control/       vendored MuJoCo<->ros2_control bridge (unmodified, MIT)
    mujoco_sim/                G1-specific MuJoCo sim bringup
```

## Prerequisites

Docker is **optional**. ROS 2 Humble is Ubuntu 22.04 software, so pick by what you have:

| You have | Use |
|---|---|
| **Ubuntu 22.04** (workstation, lab PC, the robot's computer) | Native setup below, no Docker |
| macOS, Windows, or another Linux | The dev container, an Ubuntu 22.04 VM, or WSL2 on Windows |
| Just Python, no ROS | Enough to run the static tests and doc tooling (see [Test](#test)) |

**Native (Ubuntu 22.04):** install [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
first, then:

```bash
./scripts/setup_ubuntu.sh      # apt/pip deps, MuJoCo into ~/.local/mujoco, rosdep (uses sudo)
./scripts/build.sh             # colcon build
source install/setup.bash
export LD_LIBRARY_PATH="$HOME/.local/mujoco/mujoco-3.2.7/lib:${LD_LIBRARY_PATH:-}"   # for the sim; add to ~/.bashrc
```
(That is a wrapper around `rosdep install --from-paths src --ignore-src -y` and
`colcon build --symlink-install`; `docker/Dockerfile` does the same steps.)

**Dev container** (macOS/Windows/any Docker host; x86_64 or Apple silicon):

```bash
./docker/build.sh      # builds the workspace inside the image (a few minutes; needs network)
./docker/run.sh        # shell with your working copy mounted; Python edits are live
```
On macOS there is no X11 forwarding, so run the sim with `headless:=true`. The
image builds MuJoCo with OSMesa so headless needs no display at all.

## Build by hand (what `scripts/build.sh` wraps)

```bash
cd robot
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
source install/setup.bash
```

## Run

```bash
ros2 launch bringup full_stack.launch.py                  # MuJoCo sim + every stub
ros2 launch bringup full_stack.launch.py sim:=false       # stubs only, no MuJoCo, runs on a laptop
ros2 launch bringup full_stack.launch.py headless:=true   # sim with no GUI window

ros2 run teleop_input teleop_node                         # keyboard, in a SECOND terminal
ros2 topic pub --once /user_intent std_msgs/msg/String "{data: 'pick up the cube'}"
ros2 topic echo /task_status                              # RUNNING 1/2 navigate_to ... SUCCEEDED
```
Launch switches (per-layer `stub|external`, `sim`, `headless`, ...) are in
`bringup/README.md`. The EtherCAT-scaffold milestone harness:
`ros2 launch bringup lowlevel_test.launch.py [use_hardware:=true]`.

## Test

```bash
pip install pytest pyyaml xacro
python3 tools/gen_docs.py --check     # generated doc tables match the contract
python3 -m pytest tests src/*/test    # static contract/consistency + pure-logic unit tests (no ROS)

ros2 run bringup check_contract.py [--no-sim]   # live: one publisher per topic, types, rates, TF, actions
ros2 run bringup smoke_task.py                  # live: intent -> plan -> navigate -> manipulate -> SUCCEEDED
```
CI (`.github/workflows/ci.yml`) runs all of the above. `STATUS.md` records what has
actually been built and launched on a ROS machine and what has not; check it before
assuming anything works.
