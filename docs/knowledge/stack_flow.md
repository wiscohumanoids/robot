# Stack flow: what each layer does at setup time and at run time

One-line descriptions of every layer, split into what you configure once ("Training (Setup)") and what runs
continuously ("Inference Time"). The narrative versions are [end_to_end.md](end_to_end.md),
[perception.md](perception.md), [locomotion_rl.md](locomotion_rl.md) and [manipulation.md](manipulation.md).

## The whole stack at a glance

```text
User intent
  "Listening for joystick, text, or voice input..."
  "Parsing raw input into structured command..."

Task planner (High-Level Planner)
  "Breaking goal into ordered skill sequence..."
  "Consulting remote LLM agents for novel situations..."

Behavior tree (Low-Level Planner)
  "Ticking skills reactively at 20Hz, observing downstream events..."
  "Watching  for failures and obstacles, adapting as needed..."

Perception / ORB-SLAM3
  "Extracting features, tracking landmarks, resolving loops..."
  "Localizing robot and building live map using terrain builder and simulation..."

State estimation / IEKF
  "Integrating IMU forward at 1kHz..."
  "Correcting drift from foot contact at 500Hz..."

RL locomotion policy
  [PPO — stored separately in RRL_PPO_FLOW.txt]

Manipulation policy
  [ACT — stored separately in MANIP_FLOW.txt]

WBC
  "Reconciling locomotion and manipulation demands..."
  "Solving whole-body QP, projecting through null space..."

ros2_control
  "Running joint impedance loop at 1kHz..."
  "Writing torque commands to hardware interface..."

ethercat_driver_ros2
  "Mapping commands to PDO frames and packaging messages..."
  "Transmitting EtherCAT packet down the chain..."

Motors
  "Receiving torque commands over EtherCAT..."
  "Actuating 25 joints, hardware doing its thing..."
```

## High level: intent, planning, perception, navigation

```text
User Intent

Inference Time

"Listening for joystick, keyboard, voice, or text input..."
"Teleop path: mapping input directly to v_cmd, bypassing planners..."
"Autonomous path: transcribing voice through Whisper to text string..."
"Publishing goal to /user_intent, waking the task planner..."

Task Planner (High-Level Planner)

Training (Setup)

"Writing skill library YAML: defining every skill with preconditions and postconditions..."
"Configuring LLM prompt template and structured JSON output schema..."
"Connecting world state sources: robot pose, object poses, gripper, battery topics..."

Inference Time

"Assembling world state snapshot from live topics..."
"Calling LLM with goal + world state + skill library..."
"Generating ordered skill sequence: navigate_to → pick → navigate_to → place..."
"Validating preconditions chain, retrying on malformed output..."
"Publishing skill sequence to /skill_sequence, returning to idle..."

Behavior Tree (Low-Level Planner)

Training (Setup)

"Defining BT XML structure, or generating it from the skill sequence via LLM..."
"Registering condition nodes and action node clients..."
"Wiring blackboard subscribers to state topics at their native rates..."

Inference Time

"Ticking tree from root at 20Hz..."
"ReactiveSequence checking safety guards every tick: battery, upright, faults..."
"Reading world state from blackboard: robot_pose 500Hz, object_poses 30Hz..."
"Dispatching active skill to its ROS action server, awaiting feedback..."
"On failure: running recovery routine, retrying, escalating to replan..."

Perception / ORB-SLAM3

Training (Setup)

"Calibrating camera intrinsics and hand-eye extrinsics..."
"Loading object meshes for FoundationPose 6-DOF estimation..."

Inference Time

"Extracting ORB features, tracking landmarks, resolving loop closures..."
"Localizing robot and building live map, projecting into occupancy grid..."
"Estimating object poses via FoundationPose, publishing to /object_poses..."

State Estimation / IEKF

Training (Setup)

"Calibrating IMU noise parameters and filter covariances..."
"Configuring contact detection thresholds per foot..."

Inference Time

"Integrating IMU forward at 1kHz on SE_k(3)..."
"Correcting drift from foot contact and kinematics at 500Hz..."
"Publishing fused base pose, velocity, and joint state to observation..."

Nav2 (bridge to locomotion)

Training (Setup)

"Configuring costmap layers, robot footprint, inflation radius..."
"Setting velocity limits to match RL policy training distribution..."

Inference Time

"Global planner computing path through costmap at ~1Hz..."
"Local controller following path, emitting /cmd_vel at ~20Hz..."
"Handing velocity command to locomotion policy, both paths converge here..."
```

## Locomotion policy (PPO)

```text
RL locomotion policy: PPO training

Spinning up N parallel MuJoCo environments...

Starting rollout: stepping physics until max timesteps...

Constructing observation vector from simulated sensors
obs_t ∈ ℝ⁸⁴: q[25], q̇[25], ω_base, imu_gravity, v_cmd, a_{t-1}

Feeding actor, sampling action for each joint
a_t = μ + σ · ε,   ε ~ N(0,I)

Logging action probability and computing reward
r_t = r_vel_tracking − r_torque − r_action_rate + r_air_time

Feeding asymmetric critic to estimate future return

V(s_t) from full privileged state including contact forces

Computing TD error and advantage across timesteps
A_t = Σ_k (γλ)^k · δ_{t+k},   δ_t = r_t + γV(s_{t+1}) − V(s_t)

Shuffling rollout buffer into minibatches of 256...

Rerunning actor and critic, computing PPO clipped loss
L = −min(r·A, clip(r, 1−ε, 1+ε)·A)

Backpropagating, averaging gradients, running Adam optimizer step...

Randomizing mass, friction, motor lag for next rollout...

Iteration N of 2000, reward climbing...

Training complete — exporting actor → policy.onnx, critic discarded

RL locomotion policy: inference

Receiving velocity command from behavior tree...
v_cmd = [vx, vy, vyaw] in robot body frame

Assembling observation vector from state estimator...
obs_t ∈ ℝ⁸⁴: q[25], q̇[25], ω_base, imu_gravity, v_cmd, a_{t-1}

Running actor forward pass — ONNX runtime...
obs_t → MLP 512→256→128 → μ[25]

Taking mean only, no sampling at inference...
q_target = q_default + 0.25 · μ

Forwarding joint targets to WBC...
```

## Manipulation policy (ACT)

```text
Manipulation policy: ACT training

Loading 200 teleoperated demonstration episodes...

Each episode: camera frames + joint positions recorded at 50Hz

Encoding camera frames through ResNet-18 backbone...
f(I_t) → visual feature tokens per camera

Compressing ground truth action sequence into latent style code...
z = μ + σ · ε,   ε ~ N(0,I)   via CVAE encoder

Transformer decoder attending to visual tokens + latent z...

Predicting 100-step action chunk in parallel

Computing training loss and updating weights...
L = ||â − a||₁ + β · KL(N(μ,σ) || N(0,I))

Epoch N of 200 — loss converging...

Saving weights → act_manip.pt

Manipulation policy: inference

FoundationPose localizing cube from camera image + mesh...
T_cube_world = T_camera_world · T_cube_camera

Computing pre-grasp target 10cm above cube...
p_target = T_cube_world[:3,3] + [0, 0, 0.10]

IK solver running — iterating until convergence...
Δθ = Jᵀ(JJᵀ)⁻¹ · e,   e = p_target − p_ee_current

Arm reaching pre-grasp position — handing off to ACT...

Encoding wrist camera frame through ResNet-18...
f(I_t) → visual tokens, z ~ N(0,I) sampled

Transformer decoder predicting 100-step grasp trajectory...
â_{t:t+100} = decoder(visual tokens, z)

Executing action chunk with temporal ensembling...
â_t = Σ_i e^{−0.01·i} · â_t^(i)

Forwarding joint targets to WBC...
```

## Low level: ros2_control, EtherCAT, motors

```text
ros2_control

Training (Setup)

"Writing URDF ros2_control tags: mapping joint names to hardware interfaces..."

"Configuring controller gains YAML: tuning Kp, Kd per joint group..."

"Setting joint limits: position, velocity, effort bounds per joint..."

"Launching controller manager, verifying all 25 joints claim their interfaces..."

Inference Time

"Running joint impedance loop at 1kHz..."
"τ_cmd = τ_ff + Kp(q_d − q) + Kd(q̇_d − q̇)"

ethercat_driver_ros2

Training (Setup)

"Scanning EtherCAT bus, detecting all slaves in daisy chain..."

"Writing slave config YAML: vendor IDs, PDO register addresses per motor drive..."

"Configuring RxPDO mapping (TargetTorque address per slave)..."

"Configuring TxPDO mapping (ActualPosition, ActualVelocity, ActualTorque addresses)..."

"Enabling distributed clocks, synchronizing all slaves to under 1 microsecond..."

"Verifying round-trip latency (ex: 25 slaves approximately 85 microseconds per cycle)..."

Inference Time

"Packing τ_cmd[25] into EtherCAT RxPDO frames every 1ms..."

"Transmitting down the chain, reading ActualPosition, ActualVelocity, ActualTorque back..."

Motors

Training (Setup)

"Verifying sign convention per joint (ex: positive torque direction matches URDF)..."

"Confirming FOC gains and encoder resolution inside each drive..."

"No programming needed for native EtherCAT drives, firmware handles FOC internally..."

Inference Time

"Receiving signed torque command, sign determines spin direction..."

"FOC running at 30kHz (ex: Clarke, Park, current PI loops, SVPWM)..."

"Encoder reading actual position back up the chain every cycle..."
```
