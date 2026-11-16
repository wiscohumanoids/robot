# Knowledge base

How the whole humanoid software stack works, from a spoken command down to motor current, written to be read
start to finish. The interactive version of the same material is the academy in [`../academy/`](../academy/)
(open `index.html`).

| Document | Covers |
|---|---|
| [end_to_end.md](end_to_end.md) | The full stack, command to motor: teleop vs autonomy, planner, behavior tree, Nav2, `/cmd_vel`, policy, ros2_control, EtherCAT, FOC, CAN bridges |
| [perception.md](perception.md) | SLAM, loop closure, placing objects in the map, TF ownership, build-vs-use, stubbing |
| [locomotion_rl.md](locomotion_rl.md) | RL walking end to end: parallel envs, PPO, rewards, domain randomization, deployment |
| [locomotion_rl_pipeline.md](locomotion_rl_pipeline.md) | Step-by-step outline of PPO training and inference |
| [manipulation.md](manipulation.md) | The manipulation ladder: IK, ACT, diffusion policy, VLAs, RL fine-tuning |
| [manipulation_act_pipeline.md](manipulation_act_pipeline.md) | Step-by-step outline of ACT training and inference |
| [controls.md](controls.md) | From position targets to torques: PD for the legs, impedance control for the arms, Pinocchio and CRISP, where a whole-body controller fits |
| [hardware_interface.md](hardware_interface.md) | ros2_control, EtherCAT, CiA 402, MyActuator drive modes and units, real-time Linux, safety layers |
| [stack_walkthrough.md](stack_walkthrough.md) | This repo, file by file: how a run of the stack flows through the code, sim and hardware |
| [stack_flow.md](stack_flow.md) | One line per layer: what you configure once vs what runs continuously |

## Status

The explanations are written to stay true in general; every document ends with an **"In our stack"** section that
maps the concepts onto Wisco V1 and this repo. Where a general example in the text differs from our robot (a
25-joint example, ORB-SLAM3, an IEKF, FoundationPose, ACT on an ALOHA arm), the "In our stack" section and
[stack_walkthrough.md](stack_walkthrough.md) describe what we actually run. `stack_flow.md` is the one-line-per-layer
version of the same picture.

Still to write: sim-to-real for walking (actuator modelling, delay, system identification, lessons from other humanoid
projects).
