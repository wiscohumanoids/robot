# hsim_tasks

The G1 arms-free walking task for mjlab 1.6.0, `Hsim-G1-Velocity-Flat-ArmsFree`: mjlab's G1 flat velocity task with
the policy driving only the legs and waist, while the arms move to random targets and carry payloads. Every change
from mjlab's G1 task is marked `# ARMS-FREE` in `g1_velocity_arms_free.py`. The Wisco V1 task
(`training/locomotion/wisco_v1/`) follows the same recipe.

| File | What |
|---|---|
| `__init__.py` | registers the task with mjlab on import |
| `g1_velocity_arms_free.py` | the task: action joints, observations, rewards, randomization, curriculum |
| `mdp.py` | custom terms: in-place-turn commands, random arm targets, hand payloads, extra rewards |
| `arm_ranges.py` | per-joint arm-target offsets that keep the arms clear of the body; `tools/arm_contact_sweep.py` checks them |
| `rl_cfg.py`, `runner.py` | PPO settings (mjlab's G1 values); a runner that records which joints the policy drives in the ONNX metadata |
| `train.py`, `play.py` | launchers: register the task, then run mjlab's own train / play with the same flags |

Run from `training/locomotion/`: `scripts/train_g1_velocity_arms_free.sh` (GPU machine), or
`python hsim_tasks/train.py Hsim-G1-Velocity-Flat-ArmsFree --env.scene.num-envs 4096 --log-root <dir>`.
