# RL Locomotion, End to End — The Whole Rant

Okay so you want to teach a humanoid to walk using reinforcement learning. Here's the entire thing start to finish, and I'm going to build it up so each piece connects to the next.

## Starting point: the robot model

You start with the robot's blueprint — the URDF, or in MuJoCo's case usually MJCF, which is its native XML format (you often convert URDF to MJCF). This file isn't just a shape like a CAD file — it's a physics-ready description. It has the kinematic tree: every link (the rigid bodies — thighs, shins, torso, arms), every joint connecting them (with its rotation axis, its limits, whether it's a hinge or slide or ball), the mass and inertia of each link, the collision geometry, the visual meshes. The key thing that makes it different from CAD is that it carries the physics the simulator needs — masses, inertias, joint dynamics, friction — so a physics engine can actually simulate forces and motion, not just render a picture.

You load that into MuJoCo and now you have a simulatable robot.

## The setup: thousands of parallel worlds

You don't train one robot — you spin up thousands of identical simulated robots in parallel, each in its own little world. 4096 is a common number. Why? Because you collect experience 4096 times faster. Each of these is an environment, running independently, each with slightly randomized physics (more on that later).

Now the terminology, because people mangle this constantly:

An environment is one simulated robot in one world.
An episode is one attempt. It starts with the robot reset to a standing pose and ends when the robot either falls over (termination) or hits a maximum step count (truncation) — whichever comes first. If it falls early, the episode ends early and that env immediately resets to start a fresh episode.
A rollout is one chunk of collected experience. This is the word people get wrong — a rollout is NOT the whole training run. It's this: you let all 4096 envs run for some fixed number of steps (say 24 each), collect all that data, and that collection is the rollout. So one rollout is 4096 × 24 ≈ 98,000 transitions of experience.
An iteration is one full cycle of (collect a rollout) then (train on it). You do maybe 2000 of these.
The whole training run is all 2000 iterations.

So the nesting is: training run → 2000 iterations → each iteration collects one rollout then trains on it. And during a rollout, each environment is continuously running episodes that begin and end independently. One env might finish three short episodes because the robot kept falling, while another is still in the middle of one long episode. Doesn't matter — all the data gets pooled together.

## The two networks

There are two neural networks in play:

The actor (the policy) — this is the thing that decides what to do. It survives to deployment. Given what the robot senses, it outputs an action.
The critic (the value function) — this is a helper used only during training, then thrown away. Its job is to predict how good a situation is.

## Data collection phase — networks frozen, just gathering experience

During collection nothing learns — the weights are frozen. You're just running the current policy and recording what happens. Each timestep, in each env:

1. Build the observation. MuJoCo knows the full true state of everything, but you don't hand all of it to the actor. You build a specific observation vector from only what a real robot could actually sense: joint positions, joint velocities, base angular velocity (from the IMU), the projected gravity vector (which way is down, expressed in the robot's own frame), the velocity command (what you're telling it to do), and the last action it took. Stack all that and you get the observation vector — for a 25-DoF robot it worked out to 84 numbers.

That projected gravity vector deserves a special mention — it's the single most important element, because it tells the robot which way is down in its own body frame, which means it can feel itself tilting before it falls. It's the early-warning signal for balance.

2. Actor forward pass. The observation goes into the actor network — an MLP, something like 512 then 256 then 128 neurons. But here's the important part: the actor doesn't output a single action. It outputs, per joint, a mean and a standard deviation — it's describing a probability distribution over actions, a little Gaussian for each joint.

3. Sample the action. You draw a random sample from that distribution: action = mean + std × random noise. Why randomness? Because this is exploration. If the policy always did exactly one deterministic thing, it could never discover that a slightly different action is better. By sampling around the mean, it tries variations — some turn out better, some worse — and the learning step later shifts the mean toward the better ones. The std usually starts large (explore a lot early) and shrinks as the policy gets confident.

4. Apply and step. The sampled action (usually scaled joint position targets) gets written into the sim, MuJoCo steps physics forward one tick, you get the new state.

5. Compute the reward. Based on what just happened — did it track the commanded velocity, stay upright, use reasonable torque — you compute a single scalar reward for this step.

6. Critic forward pass. The critic takes the state and outputs one number: its estimate of the value of this state. Note it does NOT compute advantage here — it just outputs V(s). Advantage comes later.

7. Record everything. Store the transition: observation, action taken, the log-probability of that action under the current policy, the reward, the value estimate. All into the rollout buffer.

Repeat for every env, every step, until the rollout is full.

## The reward vs return vs value cluster — the conceptual heart

This is the part people struggle with, so slow down here. The trap is confusing reward and return.

Reward (r_t) is the single scalar at one timestep. "Did I do well this instant."
Return (R_t) is the total discounted future reward from this timestep onward: r_t + γ·r_{t+1} + γ²·r_{t+2} + … The whole future rolled into one number. That γ (gamma, ~0.99) is the discount — future rewards count slightly less than immediate ones.
Value V(s) — this is what the critic predicts, and it predicts the return, not the reward. Given a state, the critic says "I expect about this much total future reward from here." A good critic's V(s) closely matches the actual return you end up getting.

How does the critic "look ahead"? Not by simulating the future. By having learned, from millions of past states, what tends to happen next. It's pattern recognition — the same way you can look at someone mid-stumble and predict they'll fall without doing physics in your head. The looking-ahead is baked into the learned weights.

TD error — this measures how wrong the critic's return prediction was for one step:

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
```

Read it as: "I predicted this state's return was V(s_t). But I actually got reward r_t and landed in a state the critic values at V(s_{t+1}). So a better estimate is r_t + γ·V(s_{t+1}). The gap between that better estimate and my original guess is the TD error." Positive means the state turned out better than the critic thought; negative means worse. It's the one-step correction to the critic's return prediction.

Advantage — this answers "was this action better or worse than what the policy would do on average from this state." Positive = pleasant surprise, do more of it. Negative = worse than expected, do less. This is the signal that actually drives the policy update.

GAE (Generalized Advantage Estimation) — how you actually compute the advantage. A single TD error is a short-sighted, noisy estimate. Summing all rewards to the end of the episode is unbiased but wildly noisy. GAE blends them: it's a discounted sum of future TD errors.

```
A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + …
```

The λ (lambda) knob controls how far ahead you weight things. λ near 0 → basically just the one-step TD error (low variance, more biased). λ near 1 → sum far into the future (less biased, more variance). λ ≈ 0.95 is the sweet spot. The crucial insight: you sum TD errors not to rebuild the return, but because that sum literally is the advantage — it's a mathematical identity, and it gives you one advantage number per timestep that blends short-sighted and far-sighted views.

In practice you compute it backward through the rollout, because each row's advantage is just its own TD error plus a discounted version of the next row's advantage: A_t = δ_t + γλ·A_{t+1}. One backward pass, every row gets its advantage. And critically — at episode boundaries (robot fell, reset), you STOP the recursion. You don't let advantage from a fresh episode leak backward into the previous one, because those future rewards belong to a totally different attempt. You zero the bootstrap at done-flags. Getting this wrong subtly breaks training.

And one more piece: the critic's training target, the return, comes out as R_t = A_t + V(s_t).

So after the rollout you have all the rewards and all the value estimates → compute TD errors → sum them via GAE into advantages (backward pass, cut at episode boundaries) → get returns as advantage plus value. Now you have everything to train.

## Training phase — shuffle, minibatch, epochs

Take the whole rollout buffer — all ~98,000 transitions — and shuffle them. Why shuffle? Because consecutive samples are correlated (they're from the same episode moments apart), and correlated data makes gradient updates biased and unstable. Shuffling decorrelates them.

Chop the shuffled data into minibatches (a few hundred each). Then:

One epoch = you've passed through every transition once, in minibatches. PPO does several epochs (say 5) over the same rollout data before throwing it away. This data reuse is part of what makes PPO efficient — multiple updates squeezed out of each rollout. And it's exactly why PPO needs clipping (next), because you're updating multiple times on data collected by an older version of the policy, so you have to prevent the policy from drifting too far from the one that generated the data.

## The PPO clipped objective — the actor's loss

```
r = π_new(a|s) / π_old(a|s)
u = r · A
c = clip(r, 1−ε, 1+ε) · A
L_actor = −min(u, c)
```

The ratio r measures how much the policy has changed. π_old collected the data (frozen), π_new is the current policy being updated. r = 1 means no change; r = 1.3 means the new policy thinks this action is 30% more likely than the old one did.

You want to push up the probability of positive-advantage actions and push down negative ones — that's just r·A. But if you let r grow without limit, one update could shove the policy way too far based on stale data and training blows up. So PPO clips r to [1−ε, 1+ε] (ε ≈ 0.2, so [0.8, 1.2]) and takes the minimum of clipped and unclipped. The effect: you're allowed to improve an action's probability, but only up to a capped amount per update — beyond that the gradient goes flat and there's no incentive to push further. The minus sign is just because optimizers minimize while you want to maximize the objective.

The one sentence that captures all of PPO: take multiple gradient steps on a batch of data without letting the policy run away from the one that collected it.

## The critic loss and the total loss

```
L_critic = (V(s_t) − R_t)²
```

Dead simple — squared error between the critic's prediction and the actual return. Train the critic to predict returns accurately. Plain regression.

```
L = L_actor + w1·L_critic − w2·entropy
```

The total loss combines three things: the actor loss (improve the policy), the critic loss (improve value predictions, weighted ~0.5), and an entropy bonus (weighted ~0.01). Entropy measures how spread-out the policy's distribution is — high entropy means still exploring, low means confident and near-deterministic. The minus sign rewards high entropy, which stops the policy from collapsing to a narrow deterministic distribution too early and killing its own exploration. So this one loss number trains both networks and keeps exploration alive, all at once.

## Backprop and weight updates — the mechanical heart

The loss L is a single number. The network has weights all through it. Backprop answers: how much would L change if I nudged each individual weight a hair? That's the gradient ∂L/∂w. If it's positive, increasing that weight increases loss, so you decrease the weight; if negative, increase it. That's gradient descent — step each weight opposite its gradient.

Backprop computes all these gradients efficiently with the chain rule, working backward layer by layer. Concretely: say the last layer computes output = w·h + b, where h is the input from the previous layer. And say you've already got the gradient flowing back into this layer, ∂L/∂output, call it g. Then:

```
∂L/∂w = g · h (loss-change-per-weight = incoming gradient × the input that weight multiplied)
∂L/∂b = g (the bias just adds, so its gradient is just g)
∂L/∂h = g · w (loss-change-per-input = incoming gradient × the weight)
```

That last one, ∂L/∂h, is the gradient you pass backward to the previous layer, which then does the exact same thing for its own weights and passes gradient further back. Layer by layer to the front. Each layer: receive gradient from ahead, compute its own weight gradients (incoming gradient × its inputs), pass gradient backward (incoming gradient × its weights). That's the whole algorithm.

Then the update. Plain gradient descent is w ← w − α·(∂L/∂w). Adam is the smarter version: per weight it keeps a running average of the gradient (m — momentum, "which way has this weight consistently been going") and of the squared gradient (v — "how noisy have this weight's gradients been"), and updates roughly as w ← w − α · m/(√v + ε). The effect: weights with a consistent gradient direction take big confident steps, weights with jumpy noisy gradients take small cautious steps — each weight gets its own adaptive learning rate. You can black-box Adam's internals when teaching, but keep the one-liner: momentum for direction, variance for caution, per weight.

So the full learning chain: loss → backprop computes every weight's gradient via the chain rule → Adam turns gradients into actual weight updates → repeat for every minibatch, every epoch. That's learning.

## Domain randomization — the sim-to-real secret

Between rollouts you randomize the physics — friction, link masses, motor strength, control latency, random shoves. Why? The sim is never a perfect copy of reality. Train on one fixed set of physics and the policy overfits to that exact sim and falls over on the real robot where friction and mass are a little different. Train across a range of randomized physics and the policy is forced to learn a robust strategy that works across many conditions — and the real robot is then just one more condition it already knows how to handle. This is the single most important trick for sim-to-real transfer.

## Inference — running on the real robot

Now flip to deployment. A few things change.

Where does the velocity command come from? From higher up the stack. In teleoperation, straight from a joystick or keyboard. In autonomous mode, from Nav2, which the behavior tree invokes. Either way it arrives as [vx, vy, vyaw] in the robot's body frame — vx is forward relative to where the robot faces, vy is sideways, vyaw is turn rate. The body-frame detail matters: "go north" in world frame has to be rotated into "go forward-left" in body frame depending on which way the robot currently faces, and forgetting that rotation makes the robot walk sideways.

Building the observation from the state estimator. In sim, MuJoCo hands you perfect state. On the real robot you don't have perfect state — you have noisy sensors: IMU, joint encoders. The state estimator's job is to fuse those noisy readings into a clean estimate of the actual state (base orientation, velocities, joint positions). That's the IEKF — it runs filters to clean up and fuse the sensor data. Then you build the exact same observation vector the policy trained on, but now from estimated state instead of ground-truth. The state estimator is its own real subsystem — not free from ROS, someone implements or configures it — but for understanding the locomotion policy you can mostly black-box it, as long as you hold onto the one critical rule: the observation at deployment must exactly match the format and meaning of the observation at training, or the policy sees garbage.

The asymmetric critic thing. During training, the critic is allowed to see "privileged" information the real robot can't sense — true contact forces, true friction, true masses. The actor only ever sees the deployable observation. Why let the critic cheat? Because it's discarded after training, so it's fine — and a better-informed critic gives better value estimates, better advantages, better training for the actor. And the actor never becomes dependent on the privileged info because it never sees it. So it's not that the actor's observation is "masked" — it's that the critic gets extra info. At inference the critic is gone entirely, so none of it matters; you just run the actor on its normal observation.

Forward pass, take the mean. Observation into the actor, out comes the per-joint mean and std. You take the mean and ignore the std — no sampling, because at deployment you want the best known action deterministically, not exploration.

What does the policy output — targets or torques? Almost always joint position targets — specifically small offsets added to a default pose: q_target = q_default + scale · action. NOT torques directly. Those position targets get converted to torques by a low-level controller below the policy — either a simple fast PD controller (the common v0: torque = Kp(q_target − q) + Kd(−q̇)), or a full WBC if you're doing whole-body control. For a v0 you can skip WBC entirely and let a PD controller on each joint track the targets. Either way the position-target-to-torque conversion happens below the policy, and the policy is blissfully unaware of it. So the chain is: policy → position targets → (PD or WBC) → torques → motors.

## The sim-vs-hardware insight that ties it together

In sim, MuJoCo is the entire low-level stack. There's no EtherCAT, no motor drives, no FOC, no real-time kernel — because there's no hardware. MuJoCo has built-in actuators: you write a control value (position target or torque) to the actuator and MuJoCo's physics computes the resulting motion directly. So the policy reads state straight from MuJoCo's data and writes commands straight to MuJoCo's actuators. The entire hardware chain — WBC, ros2_control, EtherCAT, FOC, motors — collapses into "MuJoCo applies your control to the joint."

This is exactly why you can build the whole locomotion policy in sim before any of the low-level hardware work exists. And it's why the low-level team's real job is to replace MuJoCo's built-in actuation with the real hardware chain — same interface (policy outputs position targets, something makes the joints track them), completely different implementation underneath (real torque control on real motors instead of MuJoCo doing it for you). The policy doesn't know or care which is underneath as long as that interface holds. That interface-preserving swap is the entire secret to why sim-to-real works at all.

## The three things to lock in above all else
Rollout = one chunk of data collection, not the whole training run. Iteration = collect + train. Run = all iterations.
Reward vs return vs value. Reward is one step. Return is the discounted future sum. The critic predicts the return. TD error is the one-step correction to the return prediction. Advantage (via GAE, summing TD errors backward, cut at episode boundaries) is "was this action better than average," and it drives the policy update.
The policy outputs position targets, and a PD controller or WBC below it turns those into torques — the policy does not output torques directly. And in sim, MuJoCo replaces that whole low-level chain, which is why the same policy transfers to hardware once the real chain is swapped in behind the same interface.

## Part 1 — What you actually touch in RL/PPO code

You're right that in practice nobody re-derives GAE. The framework (rsl_rl, Stable-Baselines3) owns all the math you learned. What you actually write and tune falls into four buckets. Here's the honest list.

## Bucket 1 — The environment (this is 90% of your real work)

The framework handles PPO. You write the environment, and this is where all your effort goes. Concretely you define:

The observation function — what goes into the obs vector each step. You pull from the sim state and assemble the vector: joint positions, joint velocities, base angular velocity, projected gravity, the velocity command, last action. You decide exactly what's in it and in what order. This is a real design choice, not a knob — get it wrong (wrong ordering, missing projected gravity) and nothing learns.

The reward function — the single thing you'll iterate on most. You write each term and its weight. Typical starting set:

```
tracking_lin_vel      + weight ~1.0    (match commanded velocity)
tracking_ang_vel      + weight ~0.5
orientation penalty   − weight ~1-5    (stay upright)
torque penalty        − weight ~1e-5   (efficiency)
action rate penalty   − weight ~0.01   (smoothness)
feet air time         + weight ~1.0    (real gait, not shuffling)
```

The numbers matter enormously and you'll tune them by watching behavior. The rule you already have in your onboarding doc — start minimal, add a term only when you can name the bad behavior it fixes — is exactly right.

The termination conditions — when does an episode end. Robot fell (base height below threshold, or tilt beyond some angle), or max episode length hit.

The action scaling — the policy outputs roughly [-1, 1]; you scale it (the action_scale, ~0.25) and add it to a default pose to get joint targets.

The domain randomization ranges — the actual min/max for friction, mass, motor strength, push magnitude, control latency. These are numbers you set.

## Bucket 2 — The PPO hyperparameters (you tune, don't implement)

These are config values the framework reads. You'll adjust these but never code them:

```
learning_rate        ~1e-3  (often adaptive, KL-targeted)
gamma (discount)      0.99
lam (GAE lambda)      0.95
clip range (epsilon)  0.2
entropy coef          0.01
value loss coef       0.5
num_envs              4096   (or whatever your hardware handles)
steps per rollout     24
```

num epochs per update 5

```
minibatch size        ~a few hundred
```

Most of these you'll leave at defaults. The ones you actually touch: num_envs (match your hardware), learning_rate (if training is unstable), and rollout length. The rest are well-established and you rarely move them.

## Bucket 3 — The network architecture (small choices)

```
hidden layer sizes    [512, 256, 128]  (actor)
                      [1024, 512, 256] (critic, often wider)
activation            ELU
```

You pick these but the defaults from any legged locomotion repo are fine. Bigger isn't better here — these sizes are proven.

## Bucket 4 — Training logistics

Number of iterations to run (~2000+), checkpoint frequency, W&B logging config, whether you're on CPU or GPU. Practical, not conceptual.

## How you actually do it, start to finish
1. Clone a working repo (unitree_rl_gym or your berkeley_humanoid).
   Do NOT start from scratch. Ever.

2. Get their existing example training running unchanged first.
   Confirm the robot renders, training launches, reward logs to W&B.
   This proves your environment works before you touch anything.

3. Swap in your robot's URDF/MJCF. Fix joint names and counts.
   Get it to render and stand.

4. Write your observation function and action scaling for your robot.

5. Write a minimal reward (velocity tracking + upright + torque penalty).

6. Train. Watch the reward curve and episode length climb in W&B.
   Watch a "play" rollout in the viewer to see actual behavior.

7. Iterate on the reward. See a bad behavior → add one term → retrain.

8. Once it walks, add domain randomization ranges, retrain for robustness.

9. Export the trained policy to ONNX for deployment.

The whole game is: framework owns PPO, you own the environment and the reward. Ninety percent of your time is reward shaping and debugging why the robot does something dumb. The PPO math you learned is what lets you diagnose those problems (is my advantage leaking across episodes, is my value function garbage), but you never write it.
