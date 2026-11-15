# g1_velocity_flat_v1

Unitree G1 velocity-command walking policy. **Default policy for the stack.**

| | |
|---|---|
| Task | `Mjlab-Velocity-Flat-Unitree-G1` (mjlab 1.6.0, MuJoCo 3.11, rsl-rl PPO) |
| Run | `2026-11-08_05-06-48`, W&B `smukherjee39-university-of-wisconsin-madison/mjlab/ectoec48` |
| Training | 10,000 iterations × 4,096 envs × 24 steps ≈ 0.98 B env steps, 1 h 27 min on 1× RTX PRO 6000 Blackwell (g7e.2xlarge), ~185k steps/s, 2.4 GB GPU memory |
| Final | mean reward 59.8, episode length 1000/1000 (never falls within 20 s) |
| Checkpoint | `model_9999.pt` (ONNX exported at the same save) |
| Policy | 99 inputs → 29 joint position targets, 50 Hz; 0.01 ms per step on a Mac CPU |
| World | `assets/robots/g1/mjlab_1.6.0/g1_velocity_flat.xml` |

## Evaluation (2026-11-08), measured body velocity vs command, 2nd half of each segment

| Command (vx, vy, wz) | mjlab on GPU (`eval_commands.py`) | MuJoCo on CPU (`play.sh --headless`) |
|---|---|---|
| stand | 0.00, 0.00, -0.01 | 0.00, 0.00, 0.00 |
| forward 0.5 | 0.49, 0.00, -0.05 | 0.46, -0.02, -0.05 |
| forward 1.0 | 0.99, 0.01, -0.03 | 0.97, -0.03, -0.05 |
| backward 0.4 | -0.33, 0.04, -0.08 | -0.35, 0.00, -0.06 |
| turn in place 0.5 | 0.00, 0.00, 0.03 | 0.00, 0.00, 0.04 |
| sidestep 0.3 | 0.00, 0.00, 0.00 | 0.00, 0.00, -0.01 |
| turn in place 1.0 | | 0.72 from standing, 0.56 right after walking |
| sidestep 0.6 | | -0.03, 0.54, -0.06 |
| walk 0.5 + turn 0.5 | | 0.46, 0.00, 0.43 |
| diagonal 0.5 / 0.3 | | 0.47, 0.22, -0.04 |

GPU training env and CPU deployment agree closely → the sim-to-sim transfer is faithful.

## Known behavior: small-command deadband, slow turning in place
Pure turns below ~0.6 rad/s, pure sidesteps below ~0.4 m/s, and very slow walking with a turn are answered by standing still. Learned from the reward balance, not a deployment bug (it happens in mjlab too). Turning in place also undershoots: about 55–70% of the commanded rate at 1.0 rad/s. Turning while walking tracks well. Details and how to handle it: `docs/decisions/0004-small-command-deadband.md`.

Also: training's command curriculum widened the speed range at iteration 5,000 (visible as an entropy bump and a reward dip) and would have widened again at 10,000, where the run ended. Speeds above ~2 m/s were never practiced.
