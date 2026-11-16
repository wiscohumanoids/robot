# RL locomotion (PPO): pipeline outline

Step-by-step outline of training and running a PPO walking policy. The narrative version is [locomotion_rl.md](locomotion_rl.md).

## Training

```text
1. Environment setup
   Load robot mesh + model into MuJoCo → mjModel + mjData
   Robot initializes to nominal standing pose
   N parallel environments running simultaneously

2. Rollout collection (everything learnable is frozen here)
   Loop until max timesteps reached:

   2a. mj_step() — advance physics one timestep → new mjData

   2b. Observation construction
       Pull from mjData only what's available on a real robot:
       q[25] joint positions, q̇[25] joint velocities,
       ω_base[3], imu_gravity[3], v_cmd[3], a_{t-1}[25]
       Add Gaussian noise (simulates real sensor noise)
       → obs_t ∈ ℝ⁸⁴

   2c. Actor forward pass
       obs_t → MLP 512→256→128 → μ[25] + σ[25] per joint

   2d. Action sampling
       a_t = μ + σ · ε,   ε ~ N(0,I)
       Clip to joint limits
       Write to mjData as next control

   2e. Log probability calculation
       log π(a_t|s_t) = Σ log N(a_t^i; μ^i, σ^i)
       One scalar per timestep — used later in PPO update

   2f. Reward calculation
       Based on state AND action from this timestep:
       r_t = r_vel_tracking − r_torque − r_action_rate
             + r_air_time − r_collision

   2g. Critic forward pass (asymmetric — sees full privileged state)
       Full mjData state (including contact forces, true friction)
       → V(s_t): estimated total future return from here

   2h. Check episode termination
       Robot fell? Max steps reached? → reset to nominal pose

   2i. Store in rollout buffer:
       (obs_t, a_t, r_t, log π(a_t|s_t), V(s_t))

3. Advantage calculation (after rollout completes)

   3a. TD error per timestep:
       δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
       "How wrong was the critic's prediction?"
       δ > 0: better than expected
       δ < 0: worse than expected

   3b. Advantage via GAE:
       A_t = Σ_k (γλ)^k · δ_{t+k}
       γ=0.99, λ=0.95
       "Was this action better or worse than average?"
       A > 0: do more of this
       A < 0: do less of this

   3c. Return per timestep:
       R_t = A_t + V(s_t)
       "What was the actual total return from this state?"

4. Minibatch updates — K epochs over the rollout buffer

   Shuffle buffer → minibatches of 256 rows
   For each minibatch, for each of K epochs:

   4a. Actor track — rerun policy on stored obs:
       New μ, σ → resample action → new log probability
       Ratio: r = exp(log π_new − log π_old)
       "How much has the policy changed?"
       Unclipped: u = r · A
       Clipped:   c = clip(r, 1−ε, 1+ε) · A,  ε=0.2
       Actor loss: L_actor = −min(u, c)

   4b. Critic track — rerun value function on stored obs:
       New V(s_t)
       Critic loss: L_critic = (V(s_t) − R_t)²
       "Make predictions match actual returns"

   4c. Entropy calculation:
       H = Σ_joints log(σ_j · √(2πe))
       Added as bonus to prevent σ from collapsing too fast
       Keeps exploration alive during early training

   4d. Total loss per row:
       L = L_actor + w1·L_critic − w2·H

   4e. Full minibatch loss:
       Sum L across all 256 rows

   4f. PyTorch autograd:
       Backprop through computation graph
       ∂L/∂θ per weight — averaged across 256 rows
       Averaging reduces noise in gradient direction

   4g. Adam optimizer step:
       Tracks m = momentum of gradient
       Tracks v = variance of gradient
       w ← w − α · m̂ / (√v̂ + ε)
       Each weight gets adaptive learning rate
       Consistent gradient direction → big step
       Noisy gradient direction → small step

5. Domain randomization before next rollout
   mass ±20%, friction [0.1, 2.0], motor lag [5–40ms]
   random pushes every ~10s

6. Repeat from step 2 — ~2000 iterations

7. Export actor → policy.onnx — critic discarded
```

## Inference

```text
1. Receive velocity command from behavior tree
   v_cmd = [vx, vy, vyaw] in robot body frame

2. Construct observation ∈ ℝ⁸⁴
   q[25], q̇[25] from encoders via state estimator
   ω_base[3], imu_gravity[3] from IMU
   v_cmd[3] from behavior tree
   a_{t-1}[25] last sent action

3. Actor forward pass — ONNX runtime
   obs → MLP 512→256→128 → μ[25]
   Take μ only — no sampling, no σ

4. Output scaling
   q_target = q_default + 0.25 · μ
   Clip to joint limits

5. Forward q_target[25] → WBC
   Repeat at 50Hz
```

## Sizes in our stack

The outline uses a 25-joint example (`q[25]`). For Wisco V1's arms-free policy: the observation has 85 values (29
joint positions and velocities, 15 last actions, 3 + 3 + 3 body terms, 3 command), the action has 15. See
[locomotion_rl.md](locomotion_rl.md), "In our stack".

