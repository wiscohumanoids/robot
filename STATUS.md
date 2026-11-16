# Status

*Snapshot 2026-11-15 (repos unified). Read this first: what is real, what is simulated, what is planned, what is
known to fail, and what is next. Update it when any of that changes.*

## At a glance

Legend: ✅ works · 🟡 partial / stub · ⬜ planned. "Sim" = in the MuJoCo simulation; "Robot" = on Wisco V1.

| Layer | Sim | Robot | Where |
|---|---|---|---|
| Robot model (URDF/MJCF from CAD + datasheets) | ✅ | — | `robots/wisco_v1/` |
| Walking policy (Wisco V1, arms-free) | ✅ 50 Hz ONNX | ⬜ | `models/locomotion/`, `training/locomotion/wisco_v1/` |
| State estimation (leg odometry + EKF) | ✅ | ⬜ (needs the IMU driver) | `hsim_state_estimation` |
| Localization / SLAM (RTAB-Map default) | ✅ | ⬜ | `hsim_navigation` |
| Navigation (Nav2) | ✅ | ⬜ | `hsim_navigation` |
| Perception (Grounding DINO + SAM 2.1 server) + object memory | ✅ | ⬜ (ZED X driver) | `hsim_perception`, `services/perception_server/` |
| Behavior tree, Claude planner | ✅ | — (runs the same) | `hsim_behavior`, `hsim_planner` |
| Safety node (e-stop lock, fall, battery, stale sensors) | ✅ | ⬜ | `hsim_safety` |
| Manipulation skill (`/skills/manipulate`, GR00T) | 🟡 smoke-tested alone, not yet in the full sim | ⬜ | `hsim_manipulation`, `training/manipulation/` |
| Arm pipelines (Panthera: teleop, recording, pi0.5, ACT) | ✅ | ✅ on the bench arms | `manipulation/` |
| Wisco V1 arm control (gravity comp + impedance, CST) | ⬜ | ⬜ | its own package, planned |
| ros2_control backend | ⬜ (`mujoco_ros2_control` ported, not built) | 🟡 one X4-10 on a bench, CST, 100 Hz, no PREEMPT_RT | `ros2_ws/src/wisco_ethercat_ros2_hw/`, `lowlevel_control` |
| Hardware e-stop (ESP32 → RK3588 → 48 V contactor) | — | 🟡 end to end with a laptop receiver | `hardware/estop/` |
| Robot TF from URDF (`robot_state_publisher`) | ⬜ (mount frames only) | ⬜ | `hsim_description` |

## Test results (this snapshot)

Static checks (`python -m pytest tests ros2_ws/src/*/test`): all pass. Build: 13 packages. Smoke tests
(`tools/ros.sh test`: contract check + drive test): stub ✅; table room with Wisco V1 ✅, G1 ✅, G1 arms-free ✅.

Autonomy checks (`tools/ros.sh autotest "" <robot>`, graded on true geometry). Passes out of 3 full runs (one on
the pre-unification folder, two on this repo; the results are equivalent), no falls in any run:

| Check | Wisco V1 | G1 |
|---|---|---|
| `nav_known_map` (AMCL, 4 m goal) | 3/3 | 3/3 |
| `nav_slam` (SLAM from scratch) | 2/2 (since the SLAM-check fix below) | 3/4 |
| `dropped_box` (obstacle not on the map) | 3/3 | 3/3 |
| `bt_plan` (behavior tree plan) | 3/3 | 2/3 |
| `viewer_idle` (viewer open + teleop takeover) | 3/3 | 3/3 |
| `planner_flow` (scripted Claude) | 3/3 | 2/3 |
| `reach_table` | 3/3 | 1/3 |
| `reach_block` | 3/3 | 1/3 |
| `kitchen_fridge` (known map) | 0/3 (strict grader, see below) | 0/3 (strict grader) |
| `slam_fridge` | 4/4 (since the SLAM-check fix below) | 2/4 |
| `kitchen_sink` | 2/3 (2/9 with six more runs) | 1/3 |

## Known issues

**Fixed 2026-11-16: the SLAM checks were configured for a lidar robot.** `nav_slam` and `slam_fridge` failed for Wisco V1
in every run, before and after the unification, but neither was a SLAM or navigation problem:
- They ran slam_toolbox (`mapping:=slam`), a 2D laser SLAM written for the G1's lidar. Wisco has no lidar, only a scan
  derived from its stereo depth. Wisco's SLAM is RTAB-Map (the stack's default), which passes.
- `nav_slam` sent a goal 4 m away straight after startup. In SLAM mode Nav2 does not plan into unexplored space (a
  deliberate setting, `training/locomotion/wisco_v1/WORK_LOG.md`), and Wisco's forward-only camera had not seen the goal
  area yet, so Nav2 refused to plan and the robot never moved. The G1's all-round lidar sees it at once.

`tools/autonomy_test.sh` now follows the robot's sensors: stereo robots use RTAB-Map and look around before the goal,
as in the real flow ("look_around, navigate_to: ..."); lidar robots (G1) run the checks unchanged. Result: Wisco
`nav_slam` 2/2, `slam_fridge` 4/4.

Open items:

1. **Reach precision against the strict grader (pre-existing, tracked as a precision goal).** The robot reaches the
   table, block, fridge and sink in every run, without falls, but the strict grader (furniture 0.15–0.40 m from the
   surface, fridge 0.3–0.6 m, square within 10°, centered on the front) is sometimes missed by a few centimetres or by
   centering: the G1 stops short of the table (0.4–0.9 m) or can't turn because it is too close (29 cm); Wisco ends too
   close to the sink (0.10–0.12 m); both robots arrive at the fridge square-on but not centered. These grades failed
   the same way in the original repo (work log 2026-11-11: "Wisco 0/4 but near misses", "strict grade 2/5"; and runs
   of the untouched pre-unification folder on 2026-11-15). The limits stay as they are; improving the final approach
   in `hsim_behavior` (`NavigateTo`, `StepBack`) to meet them is the goal.
2. **Wisco V1 needs an environment in simulation.** It has no lidar; its stereo camera renders the built world, so
   `tools/ros.sh test sim` without `env:=` fails for Wisco ("sensors:=stereo needs an env"). Use
   `tools/ros.sh test sim test/table_room`, or `robot:=g1` for the bare world.
3. **Manipulation in the full sim** needs the G1 with Dex1 grippers and the training ego camera in `hsim_sim`, and an
   arms-free walking policy, before `/skills/manipulate` can run end to end (`hsim_manipulation/README.md`).
4. **The deployed Wisco policy was trained up to 1.0 m/s forward**; a 1.5 m/s step command from standing makes it fall
   (ramped, it holds).

Hardware findings for the mechanical and electrical team (masses, ankle motor sizing, calibration, IMU axes, drive
gains and delay, supply voltage): `robots/wisco_v1/HARDWARE_ISSUES.md`.

## Next

In order; each item names where the work goes.

1. **Real-robot path in simulation first.** One joint configuration (order, signs, offsets, limits, gains, drive mode;
   grow `robots/wisco_v1/model/robot_numbers.yaml` into it); build `mujoco_ros2_control` + a position/kp/kd command
   interface on Jazzy and run the walking policy through ros2_control in sim (`ros2_ws/src/`, `docs/migration/porting_backlog.md`).
2. **EtherCAT on the robot computer.** Mode 5 (PVT/MIT) PDO mapping for legs/waist/head, mode 10 for arms, the full
   slave chain, PREEMPT_RT on the RK3588 and 1 kHz jitter tests (`ros2_ws/src/wisco_ethercat_ros2_hw/`, `hardware/`).
3. **Safety layer on the robot.** Soft joint limits, torque limits, missed-frame watchdog, stale-target watchdog for the
   Thor → RK3588 link, fall detection → damping, the hardware e-stop on the RK3588 GPIO (`hardware/estop/`).
4. **Bring-up tools.** IMU driver (EVAL_HI05, board-to-pelvis rotation), calibration (encoder zeros, sign checks, ankle
   crank offsets), logging of commands / feedback / IMU with a sim-vs-real overlay.
5. **Wisco V1 arm control** (its own package) and IK for the Wisco arms.
6. **Sim-to-real infrastructure**: actuator model (torque limits, delay, armature, friction as config), system
   identification of the motors, lessons from other open humanoid projects, documented in `docs/knowledge/`.
7. **Reach precision** (open item 1): final approach in `hsim_behavior` meeting the strict grader.
8. **Planner and behavior-tree ports** from the planner/BT/Nav2 branch (intent queue, queue editing, retry backoff,
   Nav2 test maps, stress tests): `docs/migration/porting_backlog.md`.
