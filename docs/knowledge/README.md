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
| [stack_flow.md](stack_flow.md) | One line per layer: what you configure once vs what runs continuously |

## Status

**These are the 2026-09 texts, converted to markdown unchanged (2026-11-15).** They explain the concepts well, but
several specifics predate the current stack and are being updated. Until that update lands, where this folder and
the code disagree, the code and `docs/architecture.md` win. Known differences:

| The text says | The stack now |
|---|---|
| 25 joints | Wisco V2 model: 31 joints (the current walking policy drives 15: legs + torso; G1: 29) |
| Every joint gets a torque command through 0x6071 (CST) | Planned: legs, waist, head in MyActuator mode 5 (position + kp + kd, PD in the driver); arms in CST torque from the arm controller |
| ORB-SLAM3 | RTAB-Map (default), slam_toolbox, or AMCL on a known map; ZED X stereo camera, no lidar on Wisco |
| IEKF at 1 kHz / 500 Hz | Leg odometry + `robot_localization` EKF at 50 Hz (an IEKF remains a possible upgrade) |
| FoundationPose for object poses | Grounding DINO + SAM 2.1 server, depth back-projection to 3D |
| A generic LLM planner | Claude planner + BehaviorTree.CPP skills |
| ACT as the manipulation policy | GR00T N1.6 fine-tune in sim; ACT and pi0.5 pipelines for the Panthera arm (`manipulation/`) |
| MuJoCo + Stable-Baselines3 / rsl_rl examples | mjlab (MuJoCo-Warp) + RSL-RL PPO, exported to ONNX + `manifest.yaml` |

Missing so far: a controls document (the academy's controls chapter, `../academy/src/ch_ctl.js`, has the content),
the hardware specifics (MyActuator objects, drive modes, calibration) and sim-to-real.
