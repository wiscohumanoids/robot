# training/manipulation/sim

The humanoid manipulation scene used to generate GR00T training data: a G1 with Dex1-1 grippers standing (pelvis
fixed) at a RoboCasa kitchen counter, with objects to pick and targets to place them on.

| File | What |
|---|---|
| `build_g1_dex1.py` | builds the G1 + Dex1-1 MJCF from Unitree's URDF → `assets/robots/g1_dex1/` |
| `scene.py`, `catalog.yaml` | compiles a scene: kitchen + robot + every object of a dataset version (object sets in the catalog) → `build/` |
| `env.py` | the environment: reset with a random layout, step, cameras, success check |
| `ik.py` | damped least-squares IK for one 7-joint arm to a tool pose |
| `expert.py` | scripted pick-and-place expert (knows true poses; it generates demonstrations, it is not a policy) |
| `episode_io.py` | raw episode format, read by `../groot/` to build the LeRobot dataset |

Scenes need the RoboCasa asset checkout (`~/hsim_third_party`, see `assets/envs/README.md`). Data generation:
[`../datagen/`](../datagen/README.md); dataset, fine-tune and evaluation: [`../groot/`](../groot/README.md).
