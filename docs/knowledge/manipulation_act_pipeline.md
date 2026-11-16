# Manipulation (ACT): pipeline outline

Step-by-step outline of training and running an ACT imitation policy. The narrative version is [manipulation.md](manipulation.md).

## Training

```text
1. Environment setup
   Load robot URDF + MuJoCo model → mjModel + mjData
   Robot initializes to nominal standing pose
   N parallel environments running simultaneously
   Each environment independently randomized

2. Demonstration collection
   200 teleoperated episodes recorded on real ALOHA arm
   Human operates leader arm → follower arm mirrors motion
   At each timestep record:
     camera frames from wrist camera + overhead camera
     q[7] joint positions of follower arm
     gripper state (open/closed)
   One episode: arm at rest → approach cube → grasp → lift → place
   ~300 timesteps per episode at 50Hz = ~6 seconds of motion

3. Dataset construction
   For each episode, for each timestep t:
     observation: (camera_frames_t, q_current_t)
     action sequence: q[t:t+100] — next 100 joint positions

   Total dataset: 200 episodes × ~300 timesteps
   = ~60,000 (observation, action_sequence) pairs
   Shuffle into minibatches of 64

4. CVAE encoder forward pass (training only)
   Input: current observation + full ground truth action sequence
     camera frames → ResNet-18 backbone → spatial softmax
     → 64-dim feature vector per camera
     joint positions → linear layer → joint tokens
     action sequence → flattened → transformer input
   4 transformer encoder layers attend over all inputs
   Output: μ ∈ ℝ³² and σ ∈ ℝ³² — parameters of latent distribution
   Sample latent style code:
     z = μ + σ · ε,   ε ~ N(0,I)
   z captures "what style of motion am I about to do"
   e.g. approach from left vs right, fast vs slow grasp

5. Transformer decoder forward pass
   100 learnable position queries (one per future timestep)
   Cross-attends to:
     visual tokens from ResNet-18
     joint tokens from current q
     latent z from CVAE encoder
   7 transformer decoder layers
   Output: predicted action chunk â ∈ ℝ^{100×7}
   100 future joint positions predicted in parallel

6. Loss calculation per minibatch row
   Reconstruction loss (L1 not L2 — more robust to outliers):
     L_recon = ||â − a_ground_truth||₁
   KL divergence loss (keeps latent space regular):
     L_KL = KL(N(μ,σ) || N(0,I))
          = Σ (σ² + μ² − log σ − 1) / 2
   Total loss:
     L = L_recon + β · L_KL
     β = 10 (weight balancing reconstruction vs regularization)

7. Full minibatch loss
   Sum L across all 64 rows in minibatch

8. PyTorch autograd
   Backprop through computation graph
   ∂L/∂θ per weight — averaged across 64 rows

9. Adam optimizer step
   Tracks m = momentum of gradient
   Tracks v = variance of gradient
   w ← w − α · m̂ / (√v̂ + ε)
   Each weight gets adaptive learning rate

   Repeat steps 4-9 for each minibatch
   Repeat for 200 epochs

10. Validation
    Run policy on held-out episodes
    Check if predicted trajectories match ground truth
    Loss should converge — reconstruction loss dropping toward 0

11. Save weights → act_manip.pt
    CVAE encoder discarded at deployment
    Only transformer decoder + ResNet-18 backbone kept
```

## Inference

```text
1. FoundationPose localizes cube
   Input: wrist camera RGB-D frame + cube mesh (box geometry)
   Generate pose hypotheses → render each → compare to image
   Best match: T_cube_camera (4x4 pose matrix)
   Convert to robot world frame:
     T_cube_world = T_camera_world · T_cube_camera
   cube_pos = T_cube_world[:3, 3]   ← [x, y, z] in robot base frame
   Updates at 30Hz continuously

2. Compute pre-grasp target
   p_target = cube_pos + [0, 0, 0.10]   ← 10cm above cube center
   R_target = pointing_down rotation matrix
   T_target (4x4) = [R_target | p_target]
                    [0        | 1       ]

3. IK solver runs — arm navigating to pre-grasp position
   Loop until ||e|| < 0.005m:

   3a. Forward kinematics from current encoder readings
       T_ee = T_1(θ₁) · T_2(θ₂) · ... · T_7(θ₇)
       p_ee = T_ee[:3, 3]     ← current EE position
       R_ee = T_ee[:3, :3]    ← current EE orientation

   3b. Compute error
       e_pos = p_target − p_ee                  ← 3 numbers
       e_rot = log(R_target · R_eeᵀ)            ← 3 numbers
       e = [e_pos, e_rot] ∈ ℝ⁶

   3c. Compute Jacobian at current joint angles
       For each joint j:
         column j top 3 = ẑⱼ × (p_ee − pⱼ)    ← linear velocity
         column j bot 3 = ẑⱼ                   ← angular velocity
       J ∈ ℝ⁶ˣ⁷

   3d. Pseudoinverse and joint update
       Δθ = Jᵀ(JJᵀ)⁻¹ · e                     ← 7 joint updates
       θ ← θ + 0.5 · Δθ

   Send updated θ to arm at each iteration
   Arm converges to pre-grasp position in ~20 iterations

4. ACT policy takes over at pre-grasp
   CVAE encoder gone — sample z ~ N(0,I) directly
   z = [0,0,...,0] ∈ ℝ³²   ← mean of prior, no sampling needed

   4a. Encode current observation
       Wrist camera frame → ResNet-18 → spatial softmax
       → 64-dim visual feature vector
       Current joint positions → joint tokens

   4b. Transformer decoder forward pass
       100 position queries cross-attend to visual tokens + z
       Output: â ∈ ℝ^{100×7} — next 100 joint positions
       Inference: <5ms on laptop GPU

   4c. Temporal ensembling
       Execute first 25 steps of chunk
       Re-query with updated camera frame + joint positions
       New chunk overlaps with current chunk
       Blend overlapping predictions:
         â_t = Σ_i e^{−0.01·i} · â_t^(i)
       Recent chunk weighted higher, older chunks decay

   4d. Grasp execution
       Arm descends toward cube following predicted trajectory
       Gripper closes at predicted timestep
       Lift and carry

5. Forward joint targets → WBC
   Repeat ACT inference at 10Hz during grasp execution
   IK handles navigation, ACT handles contact and grasp
```
