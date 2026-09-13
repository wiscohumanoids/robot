# bringup

Launch files and the shared `config/controllers.yaml` that every
`controller_manager` instance in this repo (sim or hardware) loads.

## `full_stack.launch.py` — the whole thing, sim path

```bash
ros2 launch bringup full_stack.launch.py
```

Starts everything in `ARCHITECTURE.md`'s vertical stack diagram against the
MuJoCo sim: `teleop_input`, `locomotion_runner`, `manipulation_runner`,
`wbc_stub`, `lowlevel_control` (+ `mujoco_ros2_control`), `safety`. Drive the
robot with the keyboard (see `teleop_input/README.md`) and watch
`/robot_state` and the MuJoCo viewer respond.

Drop in a real locomotion policy:
```bash
ros2 launch bringup full_stack.launch.py policy_onnx_path:=/path/to/policy.onnx
```

## `lowlevel_test.launch.py` — the milestone harness

**This is the "prove a command produces motion" test** the low-level team
should run after every change to `ethercat_bridge`.

```bash
# sim path (default)
ros2 launch bringup lowlevel_test.launch.py

# EtherCAT scaffold path (will NOT move a real motor until
# ethercat_bridge's TODOs are filled in -- see its README)
ros2 launch bringup lowlevel_test.launch.py use_hardware:=true
```

Then, in a second terminal, publish a step command directly (bypassing the
whole upper stack) and confirm the robot responds:

```bash
ros2 topic pub /joint_command humanoid_interfaces/msg/JointCommand "
position_target: [-0.2, 0.0, 0.0, 0.4, -0.2, 0.0, -0.2, 0.0, 0.0, 0.4, -0.2, 0.0, 0.0, 0.3, 0.15, 0.0, 0.5, 0.0, 0.3, -0.15, 0.0, 0.5, 0.0]
velocity_target: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
effort_feedforward: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
" --rate 100
```

(This is exactly the nominal pose from `locomotion_runner`'s
`NOMINAL_POSE` — publishing it holds the robot standing. Perturb a couple of
entries to see individual joints move.)

```bash
ros2 topic echo /robot_state
```

In the sim path you should see the MuJoCo viewer's G1 track the commanded
pose and `/robot_state.joint_positions` converge to it. In the hardware
path today (per `ethercat_bridge`'s current stub state) you will see
`/robot_state` stay static and the terminal log
`write(): STUB -- would pack ... into RxPDO TargetTorque ... No frame sent.`
— that log line, not silence, is the expected and correct result until
`ethercat_bridge`'s TODOs are filled in.

## `config/controllers.yaml`

The single controller_manager config shared by both paths: `update_rate:
1000`, `joint_state_broadcaster`, and `lowlevel_control`'s
`joint_impedance_controller` with its `joints`/`kp`/`kd` parameters. See
`ARCHITECTURE.md`'s canonical joint order table before editing the `joints`
list — it must stay index-aligned with `kp`/`kd` and with
`g1_description`'s `<ros2_control>` block.
