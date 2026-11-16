# mujoco_ros2_control (vendored)

**Layer:** robot backend (simulation side of the ros2_control path). **Status:** ported, **not built**
(`COLCON_IGNORE`). It was used with ROS 2 Humble in the pre-unification scaffold (tag `legacy-scaffold-v0`) and
has not been built on Jazzy yet.

A `ros2_control` hardware plugin (`mujoco_ros2_control/MujocoSystem`) that runs a MuJoCo model and exposes its
joints and IMU sensors through the standard state/command interfaces. It lets the exact controllers, joint YAML and
URDF `<ros2_control>` tags used on the real robot run against MuJoCo instead: on the robot, the only difference is
the `<plugin>` tag (the EtherCAT driver, `ros2_ws/src/wisco_ethercat_ros2_hw/`).

Upstream: an MIT-licensed `mujoco_ros2_control` package (see `LICENSE` and the `package.xml` maintainer),
vendored unmodified.

## What it supports

| | |
|---|---|
| State interfaces | `position`, `velocity`, `effort` per joint; IMU orientation, angular velocity, linear acceleration |
| Command interfaces | `position` (with a PID from `control_toolbox`, or set directly), `velocity` (PID or set), `effort` (clamped to the URDF effort limit) |
| Rate | the controller_manager `update_rate` |

## Gaps before it can be the sim backend

- **Builds on Jazzy:** untested. If the upstream ros-controls MuJoCo plugin covers our needs, prefer it over this copy.
- **MIT-style command:** the legs, waist and head are planned to run MyActuator mode 5 (position target + kp + kd
  per cycle, PD computed in the driver; see `robots/wisco_v1/CHANGES.md`). This plugin has no
  position + kp + kd interface; one has to be added so sim and hardware run the same control law.
- **PD rate:** the walking policies are trained with PD applied at the physics step (5 ms). Here a PID runs at the
  controller update rate; the two must match, or the difference must be measured, before trusting sim results.

See `docs/migration/porting_backlog.md`.
