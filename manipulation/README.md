# Manipulation

New to the team? Start with [`GETTING_STARTED.md`](GETTING_STARTED.md): robust block pick-and-place, in simulation and
on the real arms.

Everything about using arms and hands: on a real arm, in simulation, and as a skill of the humanoid stack. The
concepts (IK, ACT, diffusion policy, VLAs, RL fine-tuning) are explained in
[`docs/knowledge/manipulation.md`](../docs/knowledge/manipulation.md).

| Where | What | Runs on |
|---|---|---|
| [`panthera_teleop/`](panthera_teleop/README.md) | **Real arms.** Two HighTorque Panthera-HT 6-DoF arms: gravity-assist hand guiding (vendor SDK gravity model, 150 Hz), Meta Quest 3S passthrough teleop, three-camera recording, LeRobot export and postprocessing, remote pi0.5 inference with a dashboard, human-in-the-loop correction collection, experimental EXPO-FT RL fine-tuning | Linux x86_64 laptop + the arms over USB |
| [`panthera_sim/`](panthera_sim/README.md) | **The same arm in MuJoCo.** Keyboard teleop and recording, scripted IK datasets, ACT training and RL fine-tuning (three-block stacking), VLA-Adapter and pi0.5 fine-tuning, contact-based gripper physics, a bimanual actuator-assembly task built from the WiscoHumanoids actuator CAD | Mac / Linux; GPU (or Colab) for training |
| [`../training/manipulation/`](../training/manipulation/README.md) | **Humanoid in sim.** G1 + Dex1 grippers in RoboCasa kitchens: scripted demonstrations → LeRobot dataset → GR00T N1.6 fine-tune (52 % closed-loop success on 50 held-out layouts) | GPU VM |
| [`../ros2_ws/src/hsim_manipulation/`](../ros2_ws/src/hsim_manipulation/README.md) | **The humanoid's skill.** Action server `/skills/manipulate`: runs the VLA (`services/vla_server/`) at 20 Hz on the arms while the walking policy keeps balancing; called by the behavior tree for `pick` / `place` | ROS stack |
| Wisco V1 arm control | gravity compensation + impedance control (Pinocchio) producing CST torque for the Wisco V1 arms on the RK3588 | ⬜ its own package, planned |

## Fine-tuning: one entry point

```bash
manipulation/finetune.sh --list                                   # arm profiles and the methods each supports
manipulation/finetune.sh <act|pi05|groot> <profile> [trainer arguments]
```

An **arm profile** ([`profiles/`](profiles/README.md)) states what a policy for that arm consumes and produces:
cameras, state and action sizes, rate, task and the control contract written into checkpoints. The methods:

| Method | Trainer | Profiles |
|---|---|---|
| `act` | `panthera_sim/train_act.py` (LeRobot ACT, closed-loop rollouts during training) | `panthera_sim` |
| `pi05` | [`finetune/pi05_full.py`](finetune/pi05_full.py): full pi0.5 fine-tune through the pinned LeRobot trainer, with strict weight loading, full-model and gradient checks, dataset and resume checks from the profile, and the deployment contract in every checkpoint | `panthera_sim`, `panthera_bimanual` |
| `groot` | `training/manipulation/groot/finetune.sh` (GR00T N1.6 / N1.7) | `g1_dex1` |

A new arm is a new profile; a method that does not apply to it yet fails before training starts.
`training/train.sh` covers locomotion and tracking the same way.

How they connect: the Panthera pipelines are where the data-collection, teleop, and training methods are worked out
on real hardware; the GR00T pipeline and `hsim_manipulation` are the same ideas on the humanoid in simulation. The
Panthera arm is not the Wisco V1 arm (different kinematics, motors and SDK), so trained Panthera policies do not
transfer directly; the pipelines and the tooling do.
