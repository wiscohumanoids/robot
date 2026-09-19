# mujoco_sim

**Status:** Real; this is the working sim path.
**Contains:** `launch/g1_mujoco.launch.py`, `rviz/g1.rviz`.

Brings up `g1_description`'s robot model (with `hardware_plugin` fixed to
`mujoco_ros2_control/MujocoSystem`), the vendored `mujoco_ros2_control` node,
`robot_state_publisher`, and loads `joint_state_broadcaster` +
`lowlevel_control`'s `JointImpedanceController`: the same pattern
`bipedal_nav`'s `unitree_ros2_control/launch/unitree_g1.launch.py` used
(`Node` + `xacro` + delayed `ros2 control load_controller`), reused rather
than reinvented.

Run standalone:

```bash
ros2 launch mujoco_sim g1_mujoco.launch.py
```

`bringup/launch/lowlevel_test.launch.py` (which `full_stack.launch.py` includes for
`sim:=true`) does **not** include this launch file: it re-implements the same
sim bring-up so that it can also switch to the EtherCAT hardware plugin
(`use_hardware:=true`). The two are kept parallel; if you change how
the sim is brought up, change both. `g1_mujoco.launch.py` is the minimal
"just the sim + controllers" entry point; use `bringup`'s launch files for
anything involving the rest of the stack, and see `bringup/README.md`.

Both accept `headless:=true` (no GUI window). The Docker image is built with
OSMesa so headless works with no display server at all.

## Prerequisites (see top-level README.md for the full list)

- MuJoCo built/installed with either `find_package(mujoco)` working or
  `MUJOCO_DIR` set.
- `libglfw3-dev`, `libeigen3-dev` (for `mujoco_ros2_control`'s rendering).
- `ros-humble-control-toolbox`, `ros-humble-effort-controllers`,
  `ros-humble-joint-state-broadcaster`.
