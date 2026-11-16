# Controls, End to End: from position targets to torques

The layer between the policies and the hardware: how a joint position target becomes a torque, why the legs and the
arms get different controllers, what Pinocchio and CRISP are for, and where a whole-body controller would fit.
Extracted from the academy's controls chapter (`docs/academy/src/ch_ctl.js`; the interactive widgets are only in the
academy). How it maps onto Wisco V1 is at the end.

## The control layer

*Between the policies and ros2_control there’s a layer that turns target positions into torques. ros2_control, EtherCAT and the drives move data around, and this layer decides what torque to send.*

### Where it sits

The locomotion policy and the manipulation policy both output joint position targets. Nothing below them wants position targets, because a motor drive takes torque. The control layer sits in between. It takes the targets and the current joint state and computes a torque for every joint, quickly and smoothly and without hurting anything.

ros2_control, the EtherCAT driver and the drives are the transport. Libraries handle most of that, and the hardware chapter covers it. The control layer is the math that runs on top of that plumbing and makes the joints move where the policy asked. It’s its own piece of software, and the robot doesn’t work without it.

*(Interactive in the academy: Both policies, both controllers.)*

### Two policies, two controllers

The arm and the legs get different controllers, and the reason is the main point of this chapter. The arm sticks out into free space with gravity pulling it down and nothing to push against, so its controller has to know the arm’s physics. The legs are touching the ground through the whole gait, and the locomotion policy was trained in a simulator that already had gravity and contact forces, so a plain fast PD loop underneath is enough.

- **Arm:** impedance control with gravity and Coriolis compensation. The arm lessons in this chapter cover it.
- **Legs:** a fast PD controller. That’s the next lesson.

### Two rates

Both controllers run much faster than the policies feeding them. A policy puts out a new target every 10 to 20 ms (50 to 100 Hz). The controllers run at 1 to 2 kHz, which is 10 to 40 times faster, and keep recomputing torque from the latest target and the current joint state. It’s the same slow-brain, fast-reflex pattern as the rest of the stack.

### Joint-level control now, a whole-body controller later

What this layer does is joint-level torque control, and it’s the ancestor of a whole-body controller (WBC). A WBC solves one optimization across the whole body at once, coordinating balance, several contacts and competing priorities under the full coupled dynamics. That’s more than a first version needs. Both controllers are always running, each on its own joints. What the first version doesn’t attempt is coordinating them, like reaching out while balancing on one foot. A real WBC can take over the same slot later, and the last lesson in this chapter covers when.

**Practice questions**

- What does the control layer take in and put out? — **Position targets from the policies and the current joint state in, a torque per joint out** The drive takes torque, so something has to convert the policies’ position targets into torques.
- Why do the arm and the legs get different controllers? — **The arm is in free space and needs a controller that knows its dynamics, while the leg policy already learned the dynamics in simulation** The leg policy was trained in a simulator with gravity and contact, so a plain PD is enough underneath it.
- How much faster do the controllers run than the policies? — **About 10 to 40 times** Policies run at 50 to 100 Hz and the controllers at 1 to 2 kHz.

**Review cards**

- What does the control layer do and where does it sit? — It turns the policies’ joint position targets into torques. It sits below both policies and above ros2_control.
- Which controller does each side use? — **Arm**: impedance control with gravity and Coriolis compensation. **Legs**: a plain fast PD controller.
- What’s the rate structure of the control layer? — Policies output a target every 10-20 ms (50-100 Hz). The controllers run at 1-2 kHz, 10 to 40 times faster, using the latest target and the current joint state.
- How does this layer relate to a whole-body controller? — It’s joint-level torque control, the ancestor of a WBC. A WBC would take over the same slot later and replace both controllers.

## Legs: a plain PD controller

*The locomotion papers the notes mention all use the same low-level controller, a plain PD loop with no dynamics model at all.*

### The whole controller

```text
τ = Kp·(q_target − q) + Kd·(0 − q̇)
```

That’s all of it. There’s no gravity compensation, no Coriolis term and no dynamics library. The 0 in the damping term is the desired velocity, since the policy only asks for positions. The team’s lowlevel_control package already computes the general form τ = τ_ff + Kp(q_d − q) + Kd(q̇_d − q̇). For locomotion you set τ_ff = 0 and leave the desired velocity at 0, and you have the standard locomotion PD law.

### Why this is enough

The policy was trained inside a physics simulator that already had gravity, inertia and contact forces. It learned to output joint targets that account for all of that. So the gravity compensation lives in the network’s weights, and the PD controller only has to follow whatever positions the policy commands. The legs are also in contact with the ground through the whole gait, so the dynamics are dominated by contact forces the policy learned to use, and not by the free-space sag an arm gets.

An arm needs an explicit physics model because a scripted or IK motion knows nothing about the arm’s dynamics. The leg policy is that physics model, learned.

### Two rates

The policy runs at 50 to 100 Hz and the PD loop runs underneath it at 1000 to 2000 Hz, 10 to 40 times faster. The policy drops a new target every 10 to 20 ms. Between those updates the inner loop keeps recomputing torque from the latest target and the current joint state, which is what keeps the joints tracking smoothly.

*(Interactive in the academy: Slow targets, fast torque.)*

### Gains have to match training

During training the policy assumed particular Kp and Kd values. It commands positions expecting the joints to respond the way those gains make them respond. Use different gains on the real robot and the learned behavior doesn’t transfer. Treat the gains like the observation format and copy them from the training config.

Those are typical starting values from the papers, and they’re close to what the repo’s defaults already are. The pattern is the same as the arm’s: stiff on the big joints and soft on the distal ones.

### What’s left to do

The controller itself is nearly finished in the scaffold. The remaining work is matching the gains to the trained policy and running the loop at 1 kHz.

> **Common mistake:** Kp and Kd in the deployed controller have to be the same values used in training. If the policy was trained with one set and the hardware runs another, the policy is commanding positions for a robot that doesn’t exist.

**Practice questions**

- Why does a locomotion policy not need gravity compensation in its controller? — **It was trained in a simulator with gravity and contact, so the compensation is already in its weights** The PD controller only has to track the positions the policy commands.
- Roughly how do the two loops’ rates compare? — **Policy 50-100 Hz, PD 1000-2000 Hz** The PD loop runs 10 to 40 times faster and fills in between the policy’s updates.
- What happens if the deployed PD gains differ from the training gains? — **The joints respond differently than the policy expects and the behavior doesn’t transfer** The policy learned to command positions assuming a specific response.

**Review cards**

- What’s the leg control law? — **τ = Kp·(q_target - q) + Kd·(0 - q̇)**. No gravity compensation, no Coriolis, no dynamics library. In lowlevel_control it’s the general form with τ_ff = 0.
- Why is a plain PD enough for the legs? — The policy was trained in a simulator with gravity, inertia and contact, so it already outputs targets that account for them. The compensation is in the weights. The legs are also in contact with the ground through the gait.
- What are the two rates for the legs? — Policy: 50-100 Hz, a new target every 10-20 ms. PD loop: 1000-2000 Hz, 10 to 40 times faster.
- Why must deployed gains match training gains? — The policy learned to command positions assuming a specific joint response. Different gains mean different behavior.
- Typical starting leg gains? — Hips Kp 100 / Kd 2.5. Knees 200 / 5. Ankles 20 / 0.1. Waist 400 / 5. Stiff on big joints, soft on distal ones.

## Arm: impedance control

*An arm sticks out into free space, so a plain spring toward the target sags under the arm’s own weight. The arm controller adds terms that come from the arm’s physics.*

### Why plain PD isn’t enough

Gravity pulls the arm down and there’s nothing to push against. Command “go to 45 degrees” with only a spring and the arm sags below 45, and the controller has to fight gravity constantly, which feels stiff and twitchy. The arm needs a richer control law, a sum of terms that each do one job.

```text
τ = g(q)                  gravity compensation
  + C(q, q̇)·q̇             Coriolis
  + Kp·(q_des − q)        position spring
  + Kd·(q̇_des − q̇)        velocity damper
  + barrier(q)            soft push away from joint limits
```

- **g(q)** is the torque needed just to hold the arm up at this pose.
- **C(q, q̇)·q̇** covers the velocity-dependent forces when the arm moves fast.
- **Kp·(q_des − q)** is the spring pulling toward the target.
- **Kd·(q̇_des − q̇)** is the damper that prevents overshoot.
- **barrier(q)** is a soft repulsive push near the joint limits, for safety.

Where each term comes from matters. Gravity and Coriolis are pure physics, so you don’t tune them. You compute them, exactly, from the arm’s dynamics model. Kp and Kd can’t be computed. You tune them by hand on the real hardware, watching for oscillation on one side and sluggishness on the other.

*(Interactive in the academy: Gravity compensation on one joint.)*

> **Common mistake:** The gravity term is only as good as the arm’s model. If the mass or center of mass in the URDF is off, the controller cancels the wrong amount of gravity and no choice of Kp and Kd fully fixes it. The next lesson covers where the model comes from.

> **Note:** Like the leg controller, this one takes joint position targets from a policy (or from IK) and runs at 1 to 2 kHz. It sits in the same slot in the stack. What differs is that it adds the arm’s own physics, because a free-space arm has no learned policy absorbing gravity for it.

**Practice questions**

- Why does a plain spring sag an arm? — **The spring has to carry the arm’s weight, so a nonzero error remains** At equilibrium Kp times the error equals the gravity torque, so the arm rests below the target.
- Which terms are computed and which are tuned? — **Gravity and Coriolis are computed from the dynamics model, and Kp and Kd are tuned by hand** Gravity and Coriolis are pure physics. Kp and Kd depend on the hardware and get tuned.
- What does the barrier term do? — **Gives a soft repulsive push near the joint limits for safety** It keeps the arm away from its hard limits.

**Review cards**

- Write the arm control law. — **τ = g(q) + C(q,q̇)q̇ + Kp(q_des - q) + Kd(q̇_des - q̇) + barrier(q)**: gravity compensation, Coriolis, spring, damper, and a soft joint-limit barrier.
- Why does the arm need gravity compensation? — It’s in free space with gravity pulling it down. A plain spring sags below the target and fights gravity constantly.
- Which arm terms are computed and which are tuned? — g(q) and C(q,q̇)q̇ are computed from the dynamics model. Kp and Kd are tuned by hand on the hardware.
- What does the barrier term do? — A soft repulsive push near the joint limits, for safety.

## Pinocchio, CRISP and the URDF

*Gravity and Coriolis torques come from a dynamics library, and a controller that uses one already exists. What’s left for you is correct inertials, gains and a teach-mode test.*

### Where gravity and Coriolis come from

You don’t derive the dynamics yourself. A dynamics library does it. Pinocchio is the modern one and KDL is the older, simpler one. You give Pinocchio the robot’s URDF and the current joint angles, and it runs an algorithm called RNEA (Recursive Newton-Euler) that returns g(q) and C(q, q̇)·q̇.

> **Common mistake:** Pinocchio is only as right as the inertial parameters in your URDF: the mass, center of mass and inertia tensor of every link. If those are wrong, or left as placeholder zeros, which happens a lot, the gravity compensation is garbage and no amount of gain tuning saves it. Getting them right is mechanical and CAD work upstream of the software, and it’s the real prerequisite.

### Pinocchio and CRISP

People often treat these as alternatives, but one lives inside the other. Pinocchio is only a math engine. It knows nothing about ROS or controllers. CRISP (and the Cartesian-Impedance-Controller it builds on) is a complete ros2_control controller that implements the whole sum of terms and calls Pinocchio internally for the gravity and Coriolis terms.

*(Interactive in the academy: Two ways to get the arm controller.)*

So there are two paths. Path A is to adopt CRISP: give it your URDF, tune the gains, and never touch Pinocchio directly because it’s already in there. Path B is to write your own controller and call Pinocchio yourself, which is more work and more learning. Start with Path A and fall back to B only if CRISP doesn’t fit.

### Tuning Kp and Kd

Kp and Kd can’t be computed. You tune them by hand on real hardware, watching for oscillation versus sluggishness. It takes patience. Like the legs, the big proximal joints get stiff gains and the distal ones get soft gains.

### Teach mode: the first milestone

Before tuning a single gain, run the controller with gravity compensation only, so Kp = 0 and Kd = 0. If the URDF inertials are correct, the arm becomes weightless. You can grab it, move it anywhere by hand, let go, and it stays where you left it, holding against gravity with no sag. That one test checks your whole gravity model. If the arm sags or drifts, the inertials are wrong and you fix those before anything else.

Teach mode is also how you hand-guide the arm to collect teleoperation demonstrations for the manipulation policy. The widget in the previous lesson has a teach-mode setting where you can see what a wrong mass does.

> **On a real robot:** Do teach mode before any policy or gain tuning. It’s the cheapest test there is, and it fails loudly if the URDF is wrong.

**Practice questions**

- What does Pinocchio compute for the arm controller? — **g(q) and C(q, q̇)·q̇, from the URDF and the current joint angles, using RNEA** Pinocchio is only the dynamics engine.
- How are Pinocchio and CRISP related? — **CRISP is a full ros2_control controller that calls Pinocchio internally** Adopting CRISP means never touching Pinocchio directly.
- What does teach mode test? — **The URDF inertials, through the whole gravity model** With Kp = Kd = 0 a correct model makes the arm weightless. Sagging or drifting means the inertials are wrong.

**Review cards**

- How do you get g(q) and C(q,q̇)q̇? — From a dynamics library (Pinocchio, or the older KDL). It takes the URDF and the joint angles and runs RNEA (Recursive Newton-Euler).
- What does Pinocchio need to be correct? — The URDF’s inertial parameters: mass, center of mass and inertia tensor of every link. Placeholder zeros make gravity compensation garbage. Fixing them is upstream mechanical and CAD work.
- How do Pinocchio and CRISP relate? — Pinocchio is only the math engine. CRISP is a complete ros2_control controller that implements the whole law and calls Pinocchio inside. Path A: adopt CRISP. Path B: write your own controller and call Pinocchio.
- What’s teach mode and why do it first? — Run with gravity compensation only (Kp = Kd = 0). With correct inertials the arm is weightless and stays wherever you leave it. It validates the whole gravity model, and it’s how you hand-guide the arm for teleoperation demos.

## Where a whole-body controller fits

*The two controllers are enough while the robot walks and manipulates one after the other. A whole-body controller takes over when it has to do the two together.*

### What joint-level control is

What this chapter built is joint-level torque control. Each joint’s torque comes from that joint’s own target, with an arm controller and a leg controller running side by side. It’s the ancestor of a whole-body controller.

### What a WBC does differently

A whole-body controller solves one optimization (typically a QP) across the entire body at once. It coordinates balance, multiple contacts and competing priorities under the full coupled dynamics, and it can push lower-priority tasks through the null space of higher-priority ones.

You need that when the tasks compete inside a single solve, like reaching out while balancing on one foot, where “keep my balance” and “extend my arm” pull against each other.

### Why the first version doesn’t need one

A first version locomotes, then manipulates, and never does both at the same time. It never reaches while balancing on one foot, and it never has to trade off balance against arm extension in a single solve. So it doesn’t need the QP, MPC or contact optimization. Two simpler controllers side by side are enough: impedance control for the arm, because the arm is in free space, and plain PD for the legs, because the RL policy already absorbed the dynamics.

*(Interactive in the academy: The same slot, two ways.)*

### Growing into one

When the humanoid has to do real loco-manipulation, with arm and legs coordinating, a WBC goes in this same slot and replaces both controllers. The policies above it and ros2_control and the drives below it stay as they are.

> **Note:** Whichever controller sits in this slot, the layer takes position targets in and puts torques out, so the policies above it and ros2_control below it never need to know which one it is.

**Practice questions**

- What does a whole-body controller do that the two joint-level controllers don’t? — **Solves one optimization across the whole body, coordinating balance, contacts and competing priorities** The joint-level controllers each handle their own joints and don’t coordinate.
- Why does the first version get by without one? — **It locomotes and then manipulates, and never reaches while balancing, so nothing has to be traded off in one solve** Trade-offs between balance and arm extension only come up when both happen at once.

**Review cards**

- What does a whole-body controller do? — Solves one optimization (typically a QP) across the whole body, coordinating balance, multiple contacts and competing priorities under the full coupled dynamics.
- Why doesn’t the first version need a WBC? — It locomotes and then manipulates, never both at once, and never reaches while balancing. So nothing has to be traded off in one solve. Arm impedance control plus leg PD is enough.
- Where does a WBC go when the robot needs one? — In the same slot, between the policies and ros2_control, replacing both joint-level controllers. Nothing above or below changes.

## In our stack (2026-11)

| Concept above | Wisco V1 |
|---|---|
| Leg controller (plain PD) | runs **inside the MyActuator driver** (EtherCAT mode 5, "PVT/MIT"): every cycle the computer sends the target position plus kp and kd, and the driver computes τ = kp (q_des − q) + kd (q̇_des − q̇) + τ_ff. The gains are the ones the walking policy was trained with (`robots/wisco_v1/model/robot_numbers.yaml`, `control.gains`), so sim and robot run the same law. Same for the waist and head |
| Arm controller (impedance + gravity/Coriolis) | on the RK3588, Pinocchio dynamics, torque sent in CST mode (EtherCAT mode 10). Its own package, planned |
| In simulation | MuJoCo position actuators apply the same PD at the physics step (5 ms); held arm joints get the sim stand-in gains from the same numbers file |
| Rates | walking policy 50 Hz; PD in the driver at the EtherCAT cycle (1 kHz planned) |
| Whole-body controller | none: the walking policy keeps balance while the arms move (the arms-free recipe trains it with random arm motion and hand payloads) |

Why the gains matter so much: a policy is trained against one PD law; if the driver's gains or units differ, the robot
behaves differently from the simulation (`robots/wisco_v1/HARDWARE_ISSUES.md` H11).
