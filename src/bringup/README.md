# bringup

Launch files, the shared `config/controllers.yaml`, and the two ROS-level
checks that enforce the interface contract.

## `full_stack.launch.py`: the whole stack, stubs included

```bash
ros2 launch bringup full_stack.launch.py                 # MuJoCo sim + every stub
ros2 launch bringup full_stack.launch.py sim:=false      # stubs only, no MuJoCo
ros2 launch bringup full_stack.launch.py headless:=true  # sim without a GUI window (Docker/macOS/CI)
```

It starts one node per layer of [INTERFACE_CONTRACT.md](../../INTERFACE_CONTRACT.md):
`task_planner`, `behavior_tree`, `perception`, `state_estimation`, `slam`,
`nav`, `cmd_vel_mux`, `locomotion`, `manipulation`, `wbc`, `safety`, plus, for
`sim:=true`, `mujoco_ros2_control` + `lowlevel_control` + `robot_state_publisher`
(via `lowlevel_test.launch.py`). Who owns what and what is fake: [STATUS.md](../../STATUS.md).

### Swapping a stub for your real node

Every stub has a launch switch. `:=stub` (default) launches the repo's stub;
`:=external` launches nothing so **you** run the real node, which then becomes
that topic's only publisher:

```bash
ros2 launch bringup full_stack.launch.py perception:=external
ros2 run my_perception_pkg my_perception_node
ros2 run bringup check_contract.py          # did you honor the contract?
```

| Switch | Values | Default | Launches |
|---|---|---|---|
| `planner` `behavior_tree` `perception` `state_estimation` `slam` `navigation` `locomotion` `manipulation` `wbc` | `stub` \| `external` | `stub` | that layer's stub |
| `teleop` | `true` \| `false` | `false` (needs its own terminal, see below) | `teleop_input` |
| `cmd_vel_mux` `safety` | `true` \| `false` | `true` | the real node |
| `sim` | `true` \| `false` | `true` | MuJoCo + ros2_control + lowlevel_control |
| `headless` | `true` \| `false` | `false` | (sim option) no GUI window |
| `policy_onnx_path` | path | empty | forwarded to `locomotion_runner` |

A typo (`planner:=real`) is an error, not a silently empty launch.

### Driving it

```bash
ros2 run teleop_input teleop_node          # keyboard, in its OWN terminal (ros2 launch gives nodes no TTY)
ros2 topic pub --once /user_intent std_msgs/msg/String "{data: 'pick up the cube'}"
ros2 topic echo /task_status               # RUNNING 1/2 navigate_to ... SUCCEEDED
```

## The checks

```bash
ros2 run bringup check_contract.py [--no-sim]   # live graph vs interface_contract.yaml
ros2 run bringup smoke_task.py                  # end-to-end: intent -> plan -> navigate -> manipulate -> SUCCEEDED
```

`check_contract.py` verifies one publisher per topic, the right type, minimum
rates, action servers, and that every contract TF edge is published with no frame
having two parents. Pass `--no-sim` when the stack was launched with
`sim:=false`. The static (no-ROS) counterpart is `pytest tests/` in the repo root.
CI runs all of it (`.github/workflows/ci.yml`).

## `lowlevel_test.launch.py`: the "prove a command produces motion" harness

**This is the test the low-level team should run after every change to
`ethercat_bridge`.**

```bash
# sim path (default)
ros2 launch bringup lowlevel_test.launch.py [headless:=true]

# EtherCAT scaffold path (will NOT move a real motor until
# ethercat_bridge's TODOs are filled in: see its README)
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

(This is the nominal pose from `locomotion_runner`'s `NOMINAL_POSE`;
publishing it holds the robot standing. Perturb a couple of entries to see
individual joints move. Don't run this while `wbc` is also running: it is
`/joint_command`'s owner.)

```bash
ros2 topic echo /robot_state
```

In the sim path you should see the MuJoCo viewer's G1 track the commanded
pose and `/robot_state.joint_positions` converge to it. In the hardware
path today (per `ethercat_bridge`'s current stub state) you will see
`/robot_state` stay static and the terminal log
`write(): STUB, would pack ... into RxPDO TargetTorque ... No frame sent.`
That log line, not silence, is the expected result until
`ethercat_bridge`'s TODOs are filled in.

## `config/controllers.yaml`

The single controller_manager config shared by both paths: `update_rate:
1000`, `joint_state_broadcaster`, and `lowlevel_control`'s
`joint_impedance_controller` with its `joints`/`kp`/`kd` parameters. See
`ARCHITECTURE.md`'s canonical joint order table before editing the `joints`
list: it must stay index-aligned with `kp`/`kd` and with
`g1_description`'s `<ros2_control>` block (`tests/` checks this).
