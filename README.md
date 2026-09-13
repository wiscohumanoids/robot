# robot

The canonical ROS2 Humble humanoid software stack for WiscoHumanoids. This
repo unifies four previously disconnected prototypes into one coherent
system with a real message contract, a stubbed-but-correctly-interfaced
vertical control stack, and a real low-level ROS2-to-EtherCAT bridge
scaffold.

**Start here:** `ARCHITECTURE.md` (the full stack, frequencies, real-vs-stub
status, canonical joint order) and `INTEGRATION_POINTS.md` (exactly where
each team's work plugs in). `RESEARCH_NOTES.md` documents the
EtherCAT/`ros2_control`/real-time research the `ethercat_bridge` and
`lowlevel_control` packages are built on, with citations.

## The four-repo context

This repo does not contain a trained locomotion or manipulation policy, and
does not contain a real EtherCAT master. What it provides is the glue:

1. **`berkeley_humanoid`** — MuJoCo + Stable-Baselines3 PPO locomotion RL,
   sim-only, no ROS. Will eventually export `policy.onnx`, which drops into
   this repo's `locomotion_runner` slot. See `INTEGRATION_POINTS.md`
   "Locomotion" for the (nontrivial) 29-DOF-to-23-DOF reconciliation this
   requires.
2. **`bipedal_nav`** — a real ROS2 Humble workspace. Its `g1_description`
   (URDF+MJCF with `ros2_control` bindings) and `mujoco_ros2_control` bridge
   are ported into this repo verbatim as the canonical robot model and sim
   backend (see `src/g1_description/README.md` for the small, disclosed
   changes made). Its SLAM/state-estimation/planner nodes were non-functional
   stubs and were **not** brought over — this repo doesn't have a navigation
   or SLAM stack, by design (see `ARCHITECTURE.md`'s gap list).
3. **`lerobot_alohamini`** — a LeRobot fork for bimanual manipulation,
   ZMQ-based, no ROS by design. Will eventually be wrapped as the
   `ExecuteManipulation` action server this repo's `manipulation_runner`
   already implements as a stub. See `INTEGRATION_POINTS.md` "Manipulation"
   for the real transport/joint-mapping gaps that wrapping requires.
4. **`ros2_hub`** — an empty ROS2 tutorial workspace. Not used as a source
   of anything; superseded by this repo.

## Repo structure

```
robot/
  README.md                    you are here
  ARCHITECTURE.md               the stack, frequencies, canonical joint order, data flow
  INTEGRATION_POINTS.md         exactly where each team plugs in
  RESEARCH_NOTES.md             cited EtherCAT/ros2_control/real-time research
  docker/                       optional ROS2 Humble + MuJoCo dev container
  src/
    g1_description/             canonical URDF/MJCF (ported from bipedal_nav)
    mujoco_ros2_control/        vendored MuJoCo<->ros2_control bridge (unmodified, MIT)
    mujoco_sim/                 G1-specific MuJoCo sim bringup
    humanoid_interfaces/        THE message/action contract (real, consumed)
    teleop_input/                10 Hz keyboard/joystick -> /cmd_vel (real)
    locomotion_runner/           50 Hz stub gait + real ONNX-policy slot
    manipulation_runner/         10 Hz stub ExecuteManipulation action server
    wbc_stub/                    500 Hz pass-through arbitration (stands in for a real WBC)
    lowlevel_control/            1000 Hz REAL joint impedance controller (ros2_control plugin)
    ethercat_bridge/             1000 Hz REAL SCAFFOLD ros2_control hardware_interface
    bringup/                     launch files + shared controllers.yaml
    safety/                      100 Hz E-stop/fault monitor
```

Every package has its own `README.md` stating its status (real / stub /
real scaffold), frequency, and interfaces — read those before assuming
anything works or doesn't.

## Prerequisites (native install, Ubuntu 22.04 + ROS2 Humble)

```bash
sudo apt install \
  libglfw3-dev libeigen3-dev libyaml-cpp-dev \
  ros-humble-control-toolbox ros-humble-effort-controllers \
  ros-humble-joint-state-broadcaster ros-humble-robot-state-publisher \
  ros-humble-xacro
pip3 install xacro onnxruntime   # onnxruntime only needed to use a real locomotion policy
```

Plus **MuJoCo** itself: download a release from
https://github.com/google-deepmind/mujoco/releases and either let CMake's
`find_package(mujoco)` find it, or set `MUJOCO_DIR` to its extracted path
before building. See `docker/Dockerfile` for a complete working example of
this exact setup, or use it directly (`./docker/build.sh && ./docker/run.sh`).

## Build

```bash
cd robot
rosdep install --from-paths src --ignore-src -y
colcon build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
source install/setup.bash
```

## Run the full sim stack

```bash
ros2 launch bringup full_stack.launch.py
```

Drives the simulated G1 in MuJoCo end-to-end: keyboard teleop ->
`locomotion_runner`'s stub gait -> `wbc_stub` -> `lowlevel_control`'s real
impedance controller -> `mujoco_ros2_control`. See `bringup/README.md`.

## Run the low-level milestone test

```bash
ros2 launch bringup lowlevel_test.launch.py                    # sim path
ros2 launch bringup lowlevel_test.launch.py use_hardware:=true # EtherCAT scaffold path
```

See `bringup/README.md` for the full worked example (publish a
`JointCommand`, watch `/robot_state` respond) and `ethercat_bridge/README.md`
for exactly what the hardware path does and does not do today.
