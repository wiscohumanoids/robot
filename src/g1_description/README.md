# g1_description

**Purpose:** the single source of truth for the robot model used by every package
in this repo. Ported verbatim (geometry, kinematics and inertials
unchanged) from `bipedal_nav/src/g1_description`, variant `g1_23dof_rev_1_0`.

**Frequency:** N/A (static description, loaded once at bringup).

## What changed vs. the bipedal_nav original

- Added a massless `base_link` root link and a fixed identity `base_link_to_pelvis` joint (the Unitree model's own root is `pelvis`), so the TF contract's `odom -> base_link` frame and Nav2's default `robot_base_frame` work without renaming anything. No geometry/inertia touched.
- File renamed `g1_23dof_rev_1_0.urdf` -> `g1_23dof.urdf.xacro` and wrapped with
  `xmlns:xacro` + one `<xacro:arg name="hardware_plugin" .../>` so the
  `<ros2_control><hardware><plugin>` value can be chosen at launch time instead of
  being hard-coded (read with `$(arg hardware_plugin)`, `${...}` would look up a *property*, and an earlier version of this file used it, which made the URDF fail to expand and would have crashed every launch file; `tests/test_repo_consistency.py` now guards this). Default is `mujoco_ros2_control/MujocoSystem` (sim). Pass
  `hardware_plugin:=ethercat_bridge/EthercatHardwareInterface` to target real
  hardware. No link/joint/geometry/inertial data was touched.
- The `<ros2_control>` block itself was rewritten: every joint now exports
  `command_interface="effort"` instead of the original custom `position_pid`
  interface (which only `mujoco_ros2_control`'s built-in PID understood and a real
  EtherCAT torque-mode drive has no equivalent for). PD/PID tracking now happens
  one layer up in `lowlevel_control`, identically for sim and hardware. See
  ARCHITECTURE.md, "Sim/hardware symmetry".
- Added a `<sensor name="imu_imu">` resource to the `<ros2_control>` block,
  exporting the standard IMU state interfaces (`orientation.{x,y,z,w}`,
  `angular_velocity.{x,y,z}`, `linear_acceleration.{x,y,z}`) so
  `lowlevel_control` can actually populate `RobotState`'s IMU fields. Doing
  this required renaming one MJCF sensor, `imu_orientation` -> `imu_quat`
  (in `g1_23dof_rev_1_0.xml`), because `mujoco_ros2_control`'s
  `register_sensors()` requires an exact `<prefix>_quat` name match and the
  original name didn't satisfy it; this is why `bipedal_nav`'s IMU was
  documented as working but was never actually reachable through
  `ros2_control` (confirmed in the original repo audit). No other MJCF
  content changed.
- Trimmed from 26 URDF/MJCF variants down to just the 23-DOF one this repo
  standardizes on, to remove ambiguity about which file is canonical. The other
  variants (29-DOF, dual-arm, Inspire-hand, etc.) are still available in
  `bipedal_nav/src/g1_description` if a future project needs them. Do not add
  them here without also updating `humanoid_interfaces/config/canonical_joint_order.yaml`
  and every message/node that assumes 23 joints.

## Files

- `g1_23dof.urdf.xacro`: kinematic/visual/collision/inertial description +
  `<ros2_control>` tag. Loaded as the `robot_description` parameter (via
  `xacro` command substitution in launch files) for `robot_state_publisher`
  and for `hardware_interface::parse_control_resources_from_urdf()`.
- `g1_23dof_rev_1_0.xml`: the matching native MuJoCo MJCF used only by the sim
  path (`mujoco_sim`). Differs from bipedal_nav's copy only by the sensor rename described above
  (and punctuation in two comments). Confirmed to contain
  zero `<actuator>` elements (torque is applied directly via `qfrc_applied`) and
  `<option timestep="0.001">`, i.e. physics already steps at exactly 1 kHz,
  matching this repo's `lowlevel_control`/`ethercat_bridge` rate 1:1.
- `meshes/`: STL visual/collision meshes referenced by both files above.

## Canonical joint order

See `humanoid_interfaces/config/canonical_joint_order.yaml` and
`ARCHITECTURE.md`. The `<ros2_control>` block in the xacro file carries the
same order as an inline comment; if you ever reorder joints here, you must
update that YAML file in lockstep.
