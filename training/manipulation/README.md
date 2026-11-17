# training/manipulation (GPU VM)

Standing pick-and-place for the humanoid in simulation: a G1 with Dex1-1 grippers at a RoboCasa kitchen counter.
Scripted demonstrations → LeRobot dataset → GR00T fine-tune → served by `services/vla_server/` → run by the ROS skill
`/skills/manipulate` (`ros2_ws/src/hsim_manipulation/`). Overview of all manipulation work: `manipulation/README.md`.

| Folder | What |
|---|---|
| [`sim/`](sim/) | the manipulation scene: G1 + Dex1 model build (`build_g1_dex1.py`), kitchen counter scenes and object sets (`scene.py`, `catalog.yaml`), the environment (`env.py`), IK and a scripted expert (`ik.py`, `expert.py`), episode format (`episode_io.py`) |
| [`datagen/`](datagen/README.md) | parallel scripted + MimicGen-style episode generation → raw episodes |
| [`groot/`](groot/README.md) | raw → LeRobot (GR00T modality config), fine-tune (`finetune.sh`), closed-loop evaluation through the VLA server |

Result so far: GR00T N1.6, 20,000 steps, final checkpoint **26/50 = 52 %** success on held-out layouts
(checkpoint 15,000: 50 %, 10,000: 27 %, 5,000: 0 %); server latency about 96 ms.

Built scenes (`sim/build/`), datasets and checkpoints are not in the repo (regenerate, or copy from the VM: `~/hsim_data/`,
`~/hsim_runs/`).
