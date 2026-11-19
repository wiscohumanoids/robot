# hsim_sim — the simulated world and the robot's sensors

| Executable | Node | What |
|---|---|---|
| `mujoco_sim` | `sim` | MuJoCo 3.11 world of the robot profile (`robot:=`, Wisco V1 by default) at 200 Hz (training physics). Sensors with realistic models (`sensors.py`): IMU (noise, bias, onboard attitude filter), head lidar → `/scan` (10 Hz, `sensors:=lidar`; the G1's default, Wisco has none), battery → `/battery`. Services `/sim/reset`, test-only `/sim/set_body_pose`. Privileged `/sim/ground_truth/*` for graders only. |
| `stereo_camera` | `stereo_camera` | ZED X-style color + depth (`sensors:=stereo`, Wisco's default), stereo depth-error model, own process, ~2.5–5 Hz on CPU |
| `sim_stub` | `sim` | kinematic stand-in (no physics) for testing upper layers |
| `backend_supervisor` | `backend_supervisor` | `backend:=ros2_control` / `ethercat_sim` only: serves `/robot_description`; standing start (pause → reset to the keyframe → controllers re-hold → unpause); `/sim/reset` |
| `ros2_control_backend` | (CLI) | writes the ros2_control backend's files for a world (below) |
| `ethercat_sim_backend` | (CLI) | writes the ethercat_sim backend's files for a world (below) |

## ros2_control backend

`backend:=ros2_control` (on `sim_stack.launch.py` and `autonomy.launch.py`) runs the robot through its own control
layer instead of the sim node:

```
planner, behavior tree, Nav2, RTAB-Map, perception, state estimation, walking policy      (unchanged)
                     │ /locomotion/joint_targets (50 Hz)
                     ▼
controller manager:  leg_targets, arm_targets   JointTargetController (wisco_ethercat_ros2_hw), as on the RK3588:
                                                hold, ramp-in, joint limits, timeout
                     joint_state_broadcaster, imu_sensor_broadcaster   /joint_states, /imu
                     ▼ position commands
MuJoCo (upstream mujoco_ros2_control plugin):   position actuators apply the trained kp/kd every 5 ms physics step
                                                (MyActuator mode 5's law); GroundTruthPlugin (hsim_mujoco_plugins)
                                                publishes the world state for the camera and the Mac viewer
```

```bash
ros2 launch hsim_bringup autonomy.launch.py env:=robocasa/kitchen_l1_s1 backend:=ros2_control   # full autonomy
ros2 launch hsim_bringup sim_stack.launch.py backend:=ros2_control viewer:=false              # bare floor
```

**Worlds.** Bare: the profile's training scene, converted at launch (`robot/` prefix removed, an IMU added; checked
equal to the original: gains, masses, geometry, keyframe, timestep). With `env:=`: the env's
`world_ros2_control/world.xml` (+ `assets/`), which `tools/env_build.py` writes next to `world.mjb`: the same world
as MJCF with a copy of every mesh and texture, because the plugin bundles its own MuJoCo (3.12) and a compiled `.mjb`
only loads in the version that wrote it. Masses and inertias are written explicitly, so the two files match (checked:
masses, inertias, geometry, friction, joints, actuators; one second of physics agrees to 1e-7). Rebuild older envs
once (`env_build.py`) to get the folder.

**Startup.** The simulation runs from the moment the plugin loads, and a position hold alone does not keep a humanoid
up while the controllers spawn. `backend_supervisor` waits until every controller is active, pauses, resets to the
standing keyframe, has the receivers re-hold the pose they now measure (`~/hold`), and unpauses; the policy (on sim
time) then starts with the robot standing and the receivers ramp in. `/sim/reset` (the Mac viewer's reset) does the
same.

**Differences from the sim node:** no lockstep with the policy (as on the robot); no lidar (use `sensors:=stereo`);
no battery model; IMU and joint states are noise-free. The receivers' timeout counts control-loop time, here
simulated time, so a busy computer slows the world and the policy together without tripping it. Gains live in the
model, so on timeout the receivers hold rather than damp. The ankles and waist are plain joints in the model, so
`ParallelLinkageController` is not exercised here; `backend:=ethercat_sim` (below) does that.

**Tested** in `kitchen_l1_s1` with full autonomy: all four controllers active, the robot standing, live RTAB-Map map
from the stereo camera, perception detecting the stove, microwave, cabinets and counter, and "walk to the stove"
planned and walked (about 2.6 m). In `kitchen_l5_s5`, "walk to the sink" fails the same way on both backends (the
planner reports the way blocked), so that is the kitchen, not this backend.

## ethercat_sim backend: the EtherCAT layer simulated

`backend:=ethercat_sim` goes one layer further down: MuJoCo is the robot's *hardware*, behind simulated EtherCAT
drives, and the controllers are the robot computer's chain, with the ankles and the waist's roll/pitch as two motors
each:

```
planner, behavior tree, Nav2, RTAB-Map, perception, state estimation, walking policy      (unchanged)
                     │ /locomotion/joint_targets (50 Hz)
                     ▼
controller manager:  leg_targets, arm_targets   JointTargetController: position + kp + kd (the trained gains)
                     parallel_joints            ParallelLinkageController: ankle / waist pitch-roll -> 2 motors each
                     joint_states               ChainedJointStateBroadcaster: /joint_states in the robot's joints
                     drive_states               /drive_states/joint_states, /drive_states/dynamic_joint_states
                     imu_sensor_broadcaster     /imu
                     ▼ command interfaces of each motor (as EcCiA402Drive exports them)
VirtualRobotSystem (hsim_mujoco_plugins):  the master's CiA402 logic -> RxPDO -> a simulated RMD-X drive per motor
                                           (state machine, mode 5 PD in the drive) -> torque at every physics step
                                           -> MuJoCo (two-motor joints through the rod geometry, Jᵀτ)
```

```bash
ros2 launch hsim_bringup autonomy.launch.py env:=robocasa/kitchen_l1_s1 backend:=ethercat_sim   # full autonomy
ros2 launch hsim_bringup sim_stack.launch.py env:=test/table_room backend:=ethercat_sim           # the robot alone
ros2 topic echo /drive_states/dynamic_joint_states       # each drive's status word, mode, error code
```

Same worlds, startup and `/sim/reset` as the ros2_control backend (the supervisor talks to the `virtual_robot` node
instead), and the same differences from the sim node. On top of it: one bus cycle of delay each way (a command
reaches the drive a cycle after it is written, a reading the controllers a cycle after it is taken), the drives'
torque limits, and the linkage. The profile's `ethercat_sim` section (`hsim_bringup/config/robots/wisco.yaml`) names
the robot computer's `ParallelLinkageController` file and the rod geometry; the G1 has none, so every joint is a drive.
`update_rate:=` is the bus rate: a whole number of the model's 5 ms physics steps, or faster, which shortens the
physics step to the bus cycle.

**What it found.** In the test room at 0.5 m/s commanded, after the first 3 s (all at real time), and after
commands stop (`drive_check`):

| Path | Speed | Lowest pelvis | After the stop |
|---|---|---|---|
| ros2_control backend (MuJoCo's PD) | 0.55 m/s | 0.67 m | stands |
| ethercat_sim, a drive on every joint (no linkage) | 0.54 m/s | 0.67 m | |
| ethercat_sim, `parallel_joints` in `pd_in_drive` (the robot computer's file) | 0.38 m/s | 0.68 m | steps in place, sidesteps, falls at ~2 s |
| the same, linkage motors strong enough never to saturate | 0.33 m/s | fell | |
| ethercat_sim, `parallel_joints` in `torque` (joint-space PD on the host; what the profile runs) | 0.54 m/s | 0.67 m | stands |

So the bus delay, the drives' PD and the scaling cost nothing, and the ankle's mode-5 gains do. A per-motor PD can only
apply the diagonal of the motor-space stiffness `J⁻ᵀKJ⁻¹`, and at the ankle the coupling is about 70 % of the
diagonal (about [[38, 27], [27, 38]] Nm/rad for the trained 35 / 35): the ankle ends up about 124 Nm/rad stiff in
pitch and 20 in roll. No choice of per-motor gains fixes that, since the ratio is set by the rod geometry. The waist's
coupling is small (about 7 %). The profile therefore sets `ethercat_sim.command_mode: torque`; delete it to simulate
the file's `pd_in_drive`. Which mode the robot computer runs is open (wisco_ethercat_ros2_hw README, "Parallel joints").

