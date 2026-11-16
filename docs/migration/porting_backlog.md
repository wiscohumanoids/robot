# Porting backlog

What was carried over from the pre-unification repos but is not finished, and what was retired but still
has pieces worth porting. Every retired file is still available from its tag:

- `legacy-scaffold-v0`: the ROS 2 Humble stub scaffold (old `main`)
- `archive/task-planner`: the planner / behavior tree / Nav2 branch

Read a retired file with `git show <tag>:<path>`, e.g. `git show archive/task-planner:src/task_planner/README.md`.

## Carried over, not yet working

| Path | From | State | To finish |
|---|---|---|---|
| `ros2_ws/src/lowlevel_control/` | `legacy-scaffold-v0:src/lowlevel_control` | `COLCON_IGNORE` | Build on Jazzy. Replace `humanoid_interfaces` msgs (copies in `legacy_msgs/`) with `hsim_interfaces` equivalents. Becomes the torque (CST) path for the arms and the PD reference for the ros2_control sim backend. |
| `ros2_ws/src/mujoco_ros2_control/` | `legacy-scaffold-v0:src/mujoco_ros2_control` (vendored, MIT) | `COLCON_IGNORE` | Build on Jazzy, or switch to the upstream ros-controls MuJoCo plugin if it covers our needs. Must support position + kp + kd command interfaces (driver MIT mode) as well as effort. |
| `docs/hardware/research_notes_ethercat_rt.md` | scaffold `RESEARCH_NOTES.md` | moved as-is | Cross-check against the EtherCAT hardware package and the driver mode decision (MIT mode 5 legs / CST arms). |

Done (2026-11-15): `tools/gen_docs.py`, `tests/`, `conftest.py` and `.github/workflows/ci.yml` were rewritten for the
hsim contract and layout (the scaffold versions checked the retired 23-joint G1 stub stack).

## Retired, with pieces still to port

| Source (tag:path) | What to port | Into |
|---|---|---|
| `archive/task-planner:src/task_planner` | Intent queue, `/planner_command` queue editing, DONE only on SUCCEEDED, heartbeat / lost-tree handling, one replan with failure context | `hsim_planner` |
| `archive/task-planner:src/behavior_tree` | Cancel-then-wait before retry, retry backoff, plan-id de-duplication | `hsim_behavior` (BT.CPP) |
| `archive/task-planner:src/nav2_config/maps`, `scripts/avoidance_test.py`, `tools/nav2_avoidance_report*` | Obstacle test maps, avoidance test, HTML report | `tests/nav2_avoidance/` |
| `archive/task-planner:src/integration_tests`, `tools/run_planner_stress.sh` | Scenario harness and stress driver | `tests/integration/` |
| `legacy-scaffold-v0:src/ethercat_bridge/README.md` | CiA402 state machine notes, stub-vs-real checklist | docs for the EtherCAT hardware package |
| `legacy-scaffold-v0:CONTRIBUTING.md` | Workflow for changing an interface / adding a node | new `CONTRIBUTING.md` |

## Retired outright (superseded by the sim stack)

`perception_stub`, `slam_stub`, `state_estimation_stub`, `nav_stub`, `odom_adapter_stub`, `locomotion_runner`
(sine gait), `manipulation_runner`, `wbc_stub`, `cmd_vel_mux` (twist_mux), `teleop_input`, `safety`, `bringup`,
`mujoco_sim`, `g1_description` (23-DOF G1), `humanoid_interfaces`, `ethercat_bridge` (superseded by the
`ethercat_driver_ros2`-based hardware package), the Humble `docker/` and `scripts/`, the stub-era
`ARCHITECTURE.md`, `INTEGRATION_POINTS.md`, `INTERFACE_CONTRACT.md`, `STATUS.md`.
