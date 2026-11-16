# Manipulation, End to End — The Whole Rant

Okay so you want to teach a robot to use its arms and hands — pick things up, place them, interact with the world. Unlike locomotion, where there was basically one dominant approach (RL with PPO), manipulation has a whole ladder of approaches, each one solving a problem the previous one couldn't. So I'm going to walk up that ladder: IK first because it's the foundation, then the two imitation learning methods, then VLAs, then where the frontier actually is. Each rung adds a capability the one below it lacked, and — this is the punchline I'll keep coming back to — the very top of the ladder loops right back to the RL concepts you already learned for walking.

## Rung 1: IK — geometry, no learning

Start with the most foundational thing, which involves no machine learning at all: inverse kinematics. This is pure geometry and it's the bedrock everything else sits on.

Here's the setup. Your arm has a bunch of joints, and each joint angle contributes to where the hand ends up. The easy direction is forward kinematics: if you know all the joint angles, you can chain the transforms together — each joint contributes a rotation and a translation — and compute exactly where the hand is. You just multiply the transformation matrices down the chain from the base out to the end effector, and out pops the hand's position and orientation. Every joint already has an encoder reporting its angle, so this is always available.

The hard direction is inverse kinematics: you want the hand at a specific spot, and you need to figure out what joint angles get you there. There's no clean formula in general, because multiple joint configurations can put the hand in the same place, and sometimes there's no solution at all. So you solve it iteratively.

The key object is the Jacobian. It's a matrix that relates joint motion to hand motion — "if I spin each degree of freedom at this rate, how fast and in what direction does the hand move." Quick clarification on a thing that trips people up: a joint and a degree of freedom aren't quite the same. A simple hinge joint has 1 DOF; a ball joint is one joint with 3 DOFs. The Jacobian has one column per DOF, not per joint. For a 7-DOF arm controlling a 6-dimensional pose (3 position, 3 orientation), the Jacobian is 6×7. When all your joints are simple hinges the two line up and people use the words interchangeably, but technically a column tracks a DOF.

The IK loop itself: compute where the hand is now (forward kinematics), compute the error between where it is and where you want it (a 6-vector: position error plus orientation error), compute the Jacobian at the current angles, then use the Jacobian pseudoinverse to turn that Cartesian error into a joint-angle update. And here's a correction worth burning in — it doesn't nudge one joint. The pseudoinverse computes a coordinated nudge across all the DOFs at once, picking the smallest overall joint motion that moves the hand toward the target. All seven joints move a little, together, in the most efficient combination. Take a small step, recompute, repeat until the error is tiny.

Now the sensor question, because people get this wrong. IK needs joint encoders (always — every motor has them, they tell you the current angles) and a known base pose. For a fixed arm bolted to a table, the base never moves, so its pose is just a known constant — no IMU needed at all. Forward kinematics chains from that fixed base using only the encoders. You only need an IMU (really, a full state estimator) when the base itself floats and you don't otherwise know where it is — like a humanoid whose torso moves as it walks. And even then it's not "an IMU for IK" — it's that the whole robot needs state estimation anyway, and IK just consumes the base pose that state estimation already produces. So: fixed arm, no IMU; floating base, you get the base pose from the same state estimator we discussed in locomotion.

The camera is separate. A pose estimator like FoundationPose looks at a camera image plus a 3D model of the object and tells you where that object is in space. That's not for finding the arm — it's for finding the target. So the two sensing needs are: encoders tell you where the arm is, the camera tells you where the target is, and IK computes the joint motion to close the gap.

The limitation of pure IK: it gets the hand to a pose, but it doesn't know how to grasp. The actual contact — closing fingers at the right moment, adjusting for shape, handling slip — is genuinely hard to specify with geometry. That's the exact gap the learning methods fill. And this sets up the pattern for the whole rest of manipulation: IK generalizes anywhere (it's just math, works for any object position) but can't do dexterity; learning does dexterity but struggles to generalize. The best systems combine them.

## Rung 2: ACT — imitation learning, the simple version

Now we start learning. The whole premise of imitation learning is different from RL: instead of designing a reward and letting the robot discover behavior through trial and error, you just show it the behavior through demonstrations and it learns to copy. You teleoperate the robot through the task — often with leader arms, where you move a small master arm and the real one mirrors you — and record, at every timestep, the camera images and the joint positions. One demonstration is one full run of the task; you collect fifty to a couple hundred.

ACT stands for Action Chunking with Transformers, and both halves of that name matter.

"Chunking" first, because it solves a real problem. If you train a policy to predict just the next single action from the current observation, tiny errors compound — the robot drifts slightly off the demonstrated path, lands in a state it never saw in training, and falls apart. Human demos also have pauses and jitter that single-step prediction amplifies into shaky motion. ACT's fix: predict a whole chunk of the next 100 actions at once, not just one. Committing to a coherent multi-step plan instead of reacting jitterily step to step is what tames the compounding error.

Now the architecture, which is a CVAE — a conditional variational autoencoder — built out of transformers. This is the part that confuses everyone, so go slow. A CVAE has an encoder and a decoder, they do different jobs, and only the decoder survives to deployment.

During training, both run. The encoder looks at the actual demonstrated action sequence — the 100 actions the human really performed — plus the current state, and squeezes them into a small latent vector z, maybe 32 numbers. Think of z as a compression of the style of this particular motion: was it a fast confident grab, a slow careful one, an approach curving left? That intent gets baked into z. Then the decoder takes the current observation — camera images through a ResNet, plus joint positions — and that latent z, and tries to reproduce the action chunk.

Here's the question that tangles everyone, and it tangled you: why compute z at all, if at inference you're just going to zero it out? The answer is that z exists during training to make the decoder's job learnable. Human demonstrations are inconsistent — the same task done a little differently each time. If you tried to train the decoder to map observation directly to action chunk, it would choke on that inconsistency: same input, different demonstrated outputs, so it produces mush. By handing the decoder the extra hint z ("this particular demo was the fast-left style"), you let it cleanly explain each demonstration — the variation gets absorbed into z instead of confusing the decoder. The decoder learns "given this observation and this style code, here's the precise chunk."

The loss has two parts:

```
Training loss = reconstruction + β · KL
  reconstruction = |predicted_chunk − demonstrated_chunk|   (L1, absolute difference)
  KL = how far z's distribution strays from a standard normal
```

The reconstruction loss just asks: did the predicted chunk match the demonstrated one? They use L1 (absolute difference) rather than L2 (squared) deliberately, because L1 averages less aggressively and gives crisper actions. The KL loss forces z to stay close to a standard normal distribution — and that's the trick that makes inference work.

Because at inference, you don't have a demonstration to compute z from — producing the action is the goal. So you set z to zero, the center of that standard-normal distribution the KL loss enforced. Zero means "give me the average, typical style." The decoder, fed z=0 plus the current observation, produces the average-style action. So to correct the thing you half-remembered: z wasn't "never needed." It was essential during training to absorb demonstration inconsistency. You just don't have it at inference, so you substitute the mean — and that substitution is exactly what makes ACT average.

Which is ACT's defining weakness. At deployment it always asks for the average style. If all your demos approach the object the same way, great. But if half approach from the left and half from the right — both perfectly valid — ACT averages them and drives straight down the middle, which might miss entirely. This is the multimodality problem. ACT handles small continuous variation fine (8cm vs 12cm approach just averages to 10cm, which works), but it fails when there are genuinely distinct valid strategies. Hold that thought, because it's the entire reason the next rung exists.

## Rung 3: Diffusion Policy — imitation learning that doesn't average

Same demonstrations, same imitation-learning goal, but it fixes the averaging. The idea is borrowed straight from image generation — you know how image diffusion models start with pure random noise and iteratively "denoise" it into a coherent picture? Diffusion Policy does exactly that, except the "image" is an action trajectory. Start with a random noisy trajectory, iteratively denoise it into a coherent action sequence, conditioned on what the camera sees. And yes — the denoiser is a neural network (usually a U-Net); everything here is neural nets, diffusion is just a particular way of training and running one.

Training, with a subtlety to get right. You do not start from pure noise and run a full denoising chain during training. Instead, per example: take a clean demonstrated chunk, pick a random noise level, add that much noise to corrupt it, and train the network to predict the noise that was added.

```
q_τ = √ᾱ_τ · q₀ + √(1−ᾱ_τ) · ε        (corrupt clean chunk q₀ with noise ε at level τ)
Loss = ‖ε − ε_θ(q_τ, τ, image)‖²        (predict that noise; squared error)
```

The random noise level per example is the crucial bit — the network sees corruption at every level, from barely-noisy to pure static, so it learns to denoise at any stage. And during training it's always a single prediction step, never the full chain, because you already know the answer (the noise you added), so you can train directly.

Inference is where the chain runs and where your "10x" instinct was right. At deployment you have no clean chunk — producing one is the goal — so you start from pure random noise and run the denoiser about 10 times (using DDIM, a fast sampling scheme), each pass stripping off a bit of predicted noise, conditioned on the current camera image. After ~10 passes the random noise has been sculpted into a clean, sensible action chunk. That's ~10 network evaluations per chunk, which is why diffusion is slower than ACT's single forward pass.

And here's the payoff — why it doesn't average. Because you start from fresh random noise each time and denoise, different starting noise leads to different valid trajectories. The model learned the whole distribution of good actions, not just the mean. So if half your demos went left and half went right, diffusion produces a clean left chunk on one run and a clean right chunk on another — it picks one coherent mode instead of averaging into a crash. The one-line contrast to keep: ACT does one forward pass and gives you the average action; diffusion does ~10 denoising passes and gives you a sample from the full distribution of good actions. You buy multimodality at the cost of ~10× the inference compute.

But — and this is the ceiling on this whole rung — both ACT and diffusion are still just imitation learning. They copy demonstrations. Neither generalizes beyond what it saw. If you only ever demonstrated grabbing a cube at position A, neither will grab it at position C where it never saw it. This is why, in practice, the smart move is the hybrid: FoundationPose finds the object wherever it is, IK moves the arm to a consistent pre-grasp pose relative to it, and then the learned policy handles just the final grasp from that consistent starting point. The learned part only has to generalize over the small variation in the grasp itself, not over the entire workspace. Rung 1 and Rung 2/3 working together — geometry for the "get there," learning for the "grasp." That's how industry actually ships pick-and-place today.

## Rung 4: VLAs — bringing in world knowledge

Here's the leap. ACT and diffusion are trained from scratch on your demos only — they know nothing except the task you showed them. A VLA — vision-language-action model — instead starts from a massive pretrained vision-language model, the same kind of model that powers image-understanding chatbots, already trained on enormous internet image and text data. That backbone already knows what a cup is, what "pick up the red one" means, how objects relate in space. You get, say, 3 billion parameters of world understanding for free, then connect an action-generating mechanism and fine-tune on robot demos. The knowledge flow is: internet-scale understanding → fine-tuned to also output robot actions. That's why VLAs generalize where ACT can't — they bring common sense no amount of your own demos could teach.

The problem is a VLM natively outputs text tokens, and you need joint commands. Bridging that gap is done two ways, and understanding both is what lets you tell the companies apart.

Approach A — tokenize actions (the original way, RT-2). Discretize each action dimension into bins — chop "shoulder target" into 256 buckets, treat each bucket as a new word in the vocabulary. Now actions are just more tokens, and the model predicts them exactly like predicting the next word, reusing all the machinery it already has.

```
[image tokens] [text: "pick up the cube"] → model → [action token, action token, ...]
                                                       ↓ de-tokenize
                                                    continuous joint commands
```

Elegant — you change almost nothing, the model just learns some new "words" that happen to be robot actions. Downside: chopping smooth continuous motion into discrete buckets is lossy and clunky, rough for fast precise control.

Approach B — a separate action expert (π0's modern way). Keep the pretrained VLM for understanding — it processes image and language into a rich representation of the scene and task — but bolt on a separate small network, the action expert, that produces continuous actions using flow matching (diffusion's close cousin). The big VLM thinks about what's going on; the action expert translates that understanding into smooth continuous motor commands. No discretization, so smooth high-frequency control.

```
[image] [text] → VLM backbone → rich understanding → action expert (flow matching) → continuous actions
```

Mechanically, fine-tuning works like this: feed the image and instruction through the model, and train so its action output matches the demonstrated action. For Approach A the loss is token-prediction (did you predict the right action tokens, exactly like language modeling); for Approach B it's a flow-matching loss on the continuous action (the same "move from noise toward the real action" idea as diffusion). Gradients flow back and tune the weights to produce your robot's actions for your tasks, while the backbone keeps all its general understanding. It can be done with LoRA (freeze the giant backbone, train small adapters — cheap) or fuller fine-tuning. The shared DNA with LLM fine-tuning is real — adapt a big pretrained model to a specialized task, keep its general knowledge — the difference is you're adding a whole new output modality, actions, not just adjusting text.

Now you can actually see what the three big players did differently, because they're all the same core object — pretrained VLM plus action mechanism, fine-tuned on robot data — organized differently:

NVIDIA GR00T splits the brain explicitly into two systems: System 2, a slow VLM that reasons about the task, and System 1, a fast diffusion action expert running at ~10ms latency. Two modules, each at its own speed — which is exactly the high-level/low-level frequency split from the whole-stack discussion, but living inside the manipulation policy. They bet hard on data scale: 20,000+ hours of human egocentric video plus sim.
Google Gemini Robotics thinks in language first, then acts — the model reasons explicitly in language about what to do, then translates that plan into action, leaning on its language-reasoning strength. It unifies data across different robot types via motion transfer.
Physical Intelligence (π series) runs one flow-based stream, generating actions through diffusion/flow matching (they pioneered Approach B for VLAs).

The three differ on how sharply they separate thinking from acting (NVIDIA most explicit, PI most unified), whether they reason in language first (Google yes), and how they escape the bottleneck of expensive robot demos (all three racing toward video, sim, and cross-robot transfer). But peel back the branding and it's one architecture: understanding model plus action generator, fine-tuned on robot data.

## Rung 5: The frontier — learning from experience

And here's where it all loops back to locomotion. The fundamental ceiling on everything so far — ACT, diffusion, even a fine-tuned VLA — is that imitation learning can only be as good as its demonstrations. It copies. It can't exceed the demos, and it can't learn from failure, because failures aren't in the demonstration set.

The fix that just emerged (Physical Intelligence's π*0.6, late 2025) is to bolt reinforcement learning on top of the imitation-learned VLA. The robot deploys in the real world, collects its own successes and failures, and improves from them — using a method called RECAP that incorporates demonstrations, the robot's own on-policy collection, and human interventions when it gets stuck. And notice the machinery: it improves the policy using advantage — the exact same advantage concept from PPO you learned in locomotion, the "was this action better or worse than average" signal. There's even a technical wrinkle that ties the two worlds together: flow-matching VLAs can't use standard PPO directly (the flow-based action generation doesn't fit the standard RL math cleanly), so they had to invent advantage-conditioning instead — but conceptually it's the same idea, push the policy toward good actions and away from bad, learned from real deployment rather than only from demos.

So the whole field is converging on locomotion's playbook: pretrain on demonstrations, then improve with RL from real experience. The advantage function, learning from collected experience, exploration versus exploitation — the same concepts underlie both the walking policy and the cutting edge of manipulation. They were never separate worlds.

A few other frontier threads worth knowing the words for: memory (standard VLAs are memoryless, reacting only to the current frame; new work adds memory for long-horizon tasks that require remembering earlier steps), force/tactile awareness (pure vision misses contact forces, which matter enormously for delicate manipulation), and mixture-of-experts (route different sub-tasks to specialized sub-networks to scale capacity efficiently).

## The ladder, top to bottom

So here's the whole thing in one view. IK: pure geometry, generalizes to any object position, but can't do dexterous contact — needs encoders always, a camera for the target, an IMU only if the base floats. ACT: imitation learning with a CVAE, learns dexterity from demos, predicts action chunks to fight compounding error, but averages across demonstrations because it zeros the style latent at inference — so it breaks on genuinely multimodal tasks. Diffusion: imitation learning that denoises from random noise, samples from the full distribution instead of averaging, handles multimodality, but costs ~10× inference compute. Both ACT and diffusion still can't generalize past their demos, which is why the practical system is the hybrid — FoundationPose plus IK to get there, learned policy for the grasp. VLAs: a pretrained vision-language model connected to an action head (tokenized actions or a flow-matching expert), fine-tuned on robot data, which finally brings generalization to novel objects and instructions. And the frontier: VLAs plus RL-from-experience, which lets the robot exceed its demonstrations and recover from mistakes — using the exact advantage machinery from the locomotion rant.

Each rung adds what the one below it lacked. And the top of the ladder reaches back down to where you started with walking. That's manipulation, start to finish.
