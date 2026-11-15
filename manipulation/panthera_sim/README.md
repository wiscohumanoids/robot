# RobotArmLearning

Keyboard teleoperation and demonstration recording for a simulated HighTorque
Panthera-HT 6-DoF arm in MuJoCo.

## Setup

```bash
pip install -r requirements.txt
python teleop/keyboard.py
```

The keyboard drives a commanded end-effector pose. In the default world-frame
mode, movement is relative to the robot base:

```text
w / s   forward / back (+x / -x)      u / j   pitch
a / d   left / right   (+y / -y)      i / k   yaw
SPACE   up                             n / m   roll
SHIFT   down                           o / l   gripper open / close
[ / ]   slower / faster               c       re-centre target
r       reset arm near home           e       save take
x       discard take                  q       quit
1-4     focus a view                  g       show all four views
f       toggle Minecraft-style mouse look
side-scroll   roll the gripper (right / left = + / -)
```

Recording begins automatically when a movement or gripper command is made.
Press `e` to save the current take under `data/episode_NNN/`. The default
60-second limit can be changed with `--max-duration`, and `--no-video` records
only arrays.

The arm starts at a random reachable position spanning 10–40 cm in height, and
`r` samples a new position. Use `--arm-start-range X0 X1 Y0 Y1 Z0 Z1` to tune
the distribution. Run `python teleop/keyboard.py --help` for speed, workspace,
window, view, and mouse-look options.

## Views and mouse-look

The window shows four views:

| Key | View | Purpose |
| --- | --- | --- |
| `1` | shoulder | fixed overview behind the robot |
| `2` | wrist | first-person alignment with the jaws |
| `3` | overhead | table position and depth |
| `4` | chase | overview following the wrist |

The interactive wrist view is roll-stabilized, so rolling the gripper does not
spin the operator's view. This affects only teleoperation: the wrist images
rendered for model training retain the camera's physical roll.

Press a number again or `g` to return to the grid. Drag and vertical-scroll to
orbit and zoom the movable shoulder and chase views. Horizontal side-scrolling
rolls the gripper; it also works while Minecraft-style mouse look is active.

With `f` or `--minecraft`, the mouse aims the gripper and WASD moves on its
level heading. SPACE and SHIFT remain world-up and world-down; mouse buttons
close and open the gripper. ESC releases the cursor before quitting.

## Episode format

Each saved episode contains:

- `data.npz`: time, joint state, controls, achieved and target end-effector
  poses, object poses, gripper command, and IK residuals.
- `meta.json`: robot, scene, recording settings, object-column names, and
  frame conventions.
- `sim.mp4`: the displayed view or four-view mosaic, unless `--no-video` was
  used.

Replay a take:

```bash
python teleop/replay.py data/episode_000
```

Render several episodes into a grid video:

```bash
python teleop/grid_replay.py
```

Render shoulder/wrist observations for the VLA dataset pipeline:

```bash
python teleop/render_vla_dataset.py
```

## LeRobot ACT experiment

### Original ACT transfer-cube benchmark

To test our training code on the original task, `train_act.py --reference-act`
loads the authors' **published 50 simulated transfer-cube HDF5 demonstrations**
directly. This is the ALOHA bimanual benchmark, with 14 state/action dimensions
and one 480×640 top camera at 50 Hz. It is separate from Panthera stacking.
Source: [tonyzhaozh/act](https://github.com/tonyzhaozh/act), pinned to
`742c753c0d4a5d87076c8f69e5628c79a8cc5488`.

```bash
uv pip install --python .venv-act/bin/python -r requirements-act-reference.txt
.venv-act/bin/python tools/prepare_act_reference.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-act/bin/python train_act.py --reference-act --full-val-at-end
```

The downloader preserves every array and attribute, losslessly compresses each
episode, records the original download SHA-256 and Drive ID, and resumes completed
episodes. The complete dataset occupies about 700 MiB instead of 17.2 GiB.
No resizing, JPEG conversion, action shifting, or Panthera dataset export is used.

The preset matches the reference's 40/10 episode split (split seed 1), one random
frame per training episode per epoch, sample-standard-deviation normalization
over all 50 episodes with a 0.01 floor, ImageNet image normalization, 100-action
chunks, batch size 8, model seed 0, KL weight 10, hidden width 512, feedforward
width 3200, AdamW learning rates 1e-5, weight decay 1e-4, and FP32 training without
gradient clipping. The 10,000-update schedule equals 2,000 epochs with five
updates each. LeRobot's default one effective decoder layer matches the original
implementation's first-layer output behavior.

This remains our LeRobot ACT implementation, not a bit-for-bit upstream training
reproduction. Our validation uses deterministic held-out frames, zero latent,
and no future actions as input; `best/` is selected by this deployed inference
loss. The upstream trainer instead selects by its posterior-conditioned VAE loss.
Routine validation covers 1,024 evenly spaced held-out frames every 1,000 updates;
`--full-val-at-end` additionally checks all 4,000. Training disables early stopping.
Checkpoints retain the latest two periodic saves plus best and final models.

For the original simulator, use an isolated environment with its original MuJoCo:

```bash
uv venv .venv-act-sim --python .venv-act/bin/python --system-site-packages
uv pip install --python .venv-act-sim/bin/python -r requirements-act-reference-sim.txt
# Reuse the training environment's large torch installation.
.venv-act/bin/python - <<'PY'
import sysconfig
from pathlib import Path
shared = sysconfig.get_paths()['purelib']
target = Path('.venv-act-sim/lib/python3.10/site-packages/act_shared.pth')
target.write_text(shared + '\n')
PY
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-act-sim/bin/python tools/evaluate_act_reference.py \
  --checkpoint outputs/act/reference_transfer_cube/best \
  --output outputs/act/reference_transfer_cube/evaluation --episodes 50 --video
```

The pinned upstream checkout lives under `outputs/act_reference/upstream` with its
MIT license. Preparation removes only unused IPython debugger imports from its
`sim_env.py` and `utils.py` for Python builds without sqlite3; physics, rendering,
reset distribution and reward rules are unchanged. Evaluation uses 400 control
steps, 100-action queues, seeds 1000–1049, and the upstream reward-4 success rule.

To train and automatically evaluate the best checkpoint when training completes:

```bash
.venv-act/bin/python tools/run_act_reference.py
```

Progress is in `outputs/act/reference_transfer_cube/training.log`,
`run_status.json`, and `benchmark_status.json`; final task performance is in
`evaluation/summary.json`. Run the supervisor with a persistent service or
terminal session for unattended work. Resume with `tools/run_act_reference.py
--resume outputs/act/reference_transfer_cube/checkpoints/step_XXXXXXXX`.
Improved imitation loss alone does not establish successful manipulation.

This is offline imitation training: there is no training `num_envs` setting.
`tools/run_act_reference.py --workers 4 --prefetch-factor 2` configures the
data loaders. It also accepts `--amp` for mixed precision and
`--cudnn-benchmark` to autotune convolution kernels for the fixed image shape,
and `--channels-last` to use a GPU-friendly image/backbone memory layout.
These performance options can be combined with `--resume`; optimizer state and
the validation baseline are retained. Mixed precision changes the original FP32
recipe, so it remains opt-in. Benchmark the actual checkpoint before changing
settings with `tools/benchmark_act_reference.py --checkpoint PATH --output JSON`.

For two-block stacking with an identical arm start on every episode:

```bash
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/collect_scripted.py \
  --output data/scripted_two_block_fixed --blocks 2 --arm-start fixed \
  --episodes 500 --workers 4 --min-free-gb 2
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/validate_scripted_dataset.py \
  --input data/scripted_two_block_fixed --workers 4
```

This uses `sim/panthera/scene_two_blocks.xml`, containing only red and green.
The arm resets to the same IK solution for end-effector position
`[0.38, 0.0, 0.24]` metres and 55-degree pitch, with the gripper open.
Joint angles and the target pose are saved in each episode's metadata.
Block positions/yaws, movement speed and clearance vary. Each demonstration
stacks red onto green once, releases, retreats, and holds a valid stack for one
second. Green must remain on the table and horizontal misalignment must be at
most 12 mm. Collection uses physical pad contacts and interpolated arm commands.
The compact dataset contains 30 Hz state/action/object arrays; camera images
can be rendered later. Replay, the renderer, grid preview and scripted exporter
read the scene from episode metadata. With sufficient disk space, export using:

```bash
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/export_scripted_dataset.py \
  --input data/scripted_two_block_fixed \
  --rendered outputs/two_block_fixed_rendered_30hz \
  --output outputs/lerobot/panthera_two_block_fixed_30hz --workers 4
```

Train a fresh ACT model after the export writes its `COMPLETE` marker:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-act/bin/python train_act.py \
  --dataset outputs/lerobot/panthera_two_block_fixed_30hz \
  --output outputs/act/panthera_two_block_fixed_30hz \
  --steps 100000 --batch-size 12 --chunk-size 30 \
  --action-steps 30 --rollout-action-steps 30 \
  --eval-freq 10000 --val-batches 256 --full-val-at-end \
  --checkpoint-freq 2000 --keep-checkpoints 2 \
  --rollout-eval-freq 10000 --rollout-eval-episodes 8 --rollout-eval-seconds 30
```

The default episode split holds out 50 of the 500 demonstrations. Fixed-start
scene, task, joint angles and gripper settings are recovered from all source
episodes and stored in `deployment.json`. Rollout uses those settings with new
random block layouts. Two-block success requires red above green, green on the
table, alignment within 12 mm, and one second released. `best/` tracks held-out
imitation loss; `best_rollout/` tracks task success. Allow space for both the
compressed image dataset and its generated Arrow cache, plus checkpoints.

Scripted demonstrations can be collected separately from the teleop recordings:

```bash
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/collect_scripted.py --episodes 1000 --workers 6
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/validate_scripted_dataset.py
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/export_scripted_dataset.py --workers 4
```

The collector writes `data/scripted_stack/episode_NNNN/{data.npz,meta.json}`
using the existing native episode schema, including measured finger states and
simulation timestamps. It uses smooth Cartesian waypoints and bounded IK joint
targets at 30 Hz: approach, descend, close, lift, transfer, place, release, and
retreat. The consistent task order is red onto green, then blue onto red. Cube
positions/yaws, arm starts, speeds (0.10–0.16 m/s), and approach clearances
(6–8 cm) vary. Starts use a reachable elevated region (18–28 cm), so this dataset
does not establish performance over the entire teleop start distribution.
Only demos with both physical grasps/lifts and a released three-stack held for
one second with at most 12 mm adjacent horizontal misalignment are saved.
New recordings use the `contact-v2` dynamics described below. Rejected seeds and
reasons are retained in `attempts.jsonl`; `status.json` reports progress.
Re-running the collector resumes the same collection; changed code or scene
settings require a separate output directory.

The exporter produces `outputs/lerobot/panthera_scripted_stack_30hz` with the
same seven-dimensional state/action contract and 256×256 shoulder/wrist images
as the teleop dataset. It embeds the original rendered JPEG bytes in Parquet,
then removes its disposable image cache to save space. Rendered trajectories
and provenance remain in `outputs/scripted_stack_rendered_30hz`; the usual
renderer can regenerate the JPEGs. A `COMPLETE` file marks a finished export.
Both tools stop safely below 5 GiB free; exporter outputs must be fresh paths.
Do not build the optional decoded-image cache for this large dataset without
checking disk capacity first. The first training run also creates a Hugging Face
Arrow cache approximately the size of the compressed dataset; allow additional
space for that cache and checkpoints.

Start a fresh ACT run for this dataset (do not resume a teleop optimizer):

```bash
.venv-act/bin/python train_act.py \
  --dataset outputs/lerobot/panthera_scripted_stack_30hz \
  --output outputs/act/panthera_scripted_stack_30hz \
  --steps 100000 --action-steps 30 --rollout-action-steps 30 \
  --eval-freq 10000 --val-batches 256 --full-val-at-end \
  --checkpoint-freq 2000 \
  --rollout-eval-freq 10000 --rollout-eval-seconds 40
```

For this larger dataset, routine validation uses 3,072 evenly spaced held-out
frames every 10,000 updates. A separate full validation runs on the final model
after its checkpoint is saved, and writes `full_validation.json`. Validation logs
batch progress. When resuming a run with a changed validation sample, add
`--reset-validation-best` so checkpoint selection starts with the new sample.

The same demonstrations can be converted to LeRobot 0.4.4 and used to train
an ACT policy locally. The converter uses both shoulder and wrist images,
seven-dimensional joint/gripper-command state, and the next **30 Hz** joint
target as the action. The corrected dataset is `outputs/lerobot/panthera_stack_30hz`;
older datasets and checkpoints remain separate. Build it from the current raw takes:

```bash
uv venv .venv-act --python 3.10
uv pip install --python .venv-act/bin/python lerobot==0.4.4 mujoco
uv pip uninstall --python .venv-act/bin/python opencv-python-headless
uv pip install --python .venv-act/bin/python opencv-python==4.12.0.88
.venv-act/bin/python teleop/build_lerobot_dataset.py
.venv-act/bin/python tools/lerobot_image_cache.py outputs/lerobot/panthera_stack_30hz
.venv-act/bin/python tools/run_act_experiment.py --steps 100000
python rollout_act.py --checkpoint outputs/act/panthera_stack_30hz/best --steps 450 --no-display --no-realtime --video outputs/act/rollout.mp4
```

Rendering fingerprints each raw recording, its metadata, rendering code, scene,
and export settings. Changed inputs invalidate cached episodes. New teleop
recordings include actual finger positions and simulation timestamps. Legacy
recordings have their simulation clock and finger dynamics reconstructed; these
are estimates, recorded in each rendered episode's `source.json`. Continuous
poses are interpolated on a uniform simulation-time grid, with quaternion SLERP.
The final observation is dropped because it has no demonstrated successor action.

ACT predicts a one-second chunk (30 actions). The experiment runner deploys
all 30 actions before the next query: controlled comparisons found that frequent
replanning often stalls this policy. `inference.json` stores the tested execution
settings separately from the training/data contract. Use `--action-steps N` to
compare queue lengths, or `--temporal-ensemble` to explicitly enable averaging.
Checkpoint `deployment.json` stores the dataset identity and FPS; rollout and RL
read that rate and reject mismatches. A fractional physics clock keeps 30 Hz
accurate against the simulator timestep. Joint targets respect joint limits;
there is no extra per-step clipping unless `--max-joint-step` is explicitly set.

Validation uses `policy.eval()` with a zero latent and no future actions as
inputs, masks padding, and evaluates **all** held-out frames by default. Optional
`--val-batches` limits sample evenly across the split. Every 2,000 steps the
trainer evaluates and saves an improved imitation model under `best/`.
Offline-loss early stopping is disabled by default: it previously stopped at 28k
and did not track grasping reliably. The experiment runner additionally evaluates
actual rollouts every 10k steps, saving `best_rollout/` by stable full-stack rate,
then two-stack, lift, and grasp rate. Resumed experiments evaluate and preserve an
incumbent baseline so a continuation cannot silently replace it with a worse
checkpoint on the selection scenes.
Training uses AMP gradient scaling and retains only three periodic checkpoints.
The experiment runner then evaluates `best_rollout/` on eight additional fixed random seeds,
reporting grasp, lift, and released three-stack success held for 0.5 seconds.
Read `run_status.json`, `validation.jsonl`, and `evaluation/summary.json` in the
run directory for progress and results.

Resume a checkpoint only against the same dataset and control contract:

```bash
.venv-act/bin/python train_act.py \
  --resume outputs/act/panthera_stack_30hz/checkpoints/step_00002000 \
  --output outputs/act/panthera_stack_30hz --steps 100000
python rollout_act.py --checkpoint outputs/act/panthera_stack_30hz/best
.venv-act/bin/python tools/evaluate_act.py
.venv-act/bin/python tools/replay_act_actions.py --episodes 8
```

`r` resets the interactive rollout and `q` quits. `--no-temporal-ensemble`
compares with the checkpoint's saved action queue. Historical 10 Hz models
require their original dataset, e.g. `--dataset outputs/lerobot/panthera_stack`.
Do not resume their optimizer on the new 30 Hz dataset.

To continue a run with task-based selection:

```bash
.venv-act/bin/python tools/run_act_experiment.py \
  --resume outputs/act/panthera_stack_30hz \
  --baseline outputs/act/panthera_stack_30hz/best \
  --output outputs/act/panthera_stack_30hz_rollout_selected \
  --steps 100000 --action-steps 30
```

Full stacking is still unreliable; changing execution mode is not evidence that
training has solved the task. `tools/diagnose_act_control.py` compares checkpoints
and queue lengths on repeatable starts and saves trajectories for inspection.

The first-pickup experiment in `tools/run_pickup_experiment.py` compares matched
absolute and relative arm-target models using demonstrations cropped through
their first sustained lift. Both run for 30k updates, select checkpoints on 32
scenes, and test on 64 separate scenes. Relative targets are anchored to the
joint pose at the start of each chunk; `rollout_act.py` reads the checkpoint's
action representation automatically. Pickup success requires the same block to
remain grasped and elevated for 0.5 seconds. The experiment writes progress to
`outputs/act/pickup_ablation_20260923/run_status.json` and its final paired results
to `comparison.json` in that directory.

## RL fine-tuning ACT for three-block stacking

`train_act_rl.py` continues from the imitation checkpoint with conservative
PPO. It freezes ACT's visual backbone and transformer, temporally ensembles the
overlapping decoder features during both training and deployment, and updates
the shared ACT action head while penalizing drift from the imitation policy. A
privileged state critic is used only while training; the exported policy still
consumes the same shoulder image, wrist image, and seven-dimensional robot
state as before. Exploration is deliberately small and anneals during the run.
Half of training episodes begin from snapshots sampled across the two-cube
stage of the recorded demonstrations; the other half retain fully randomized
starts. This teaches final-block completion without hiding whether the policy
can still solve earlier stages.

The shaped reward uses discounted potential differences for reaching,
grasping, lifting, a supported two-cube pair, and a table-supported three-cube
chain. Command velocity and especially command reversal are penalized in
normalized action units. A three-cube stack must remain valid for five control
ticks before the episode succeeds, and the last cube has to be released.
Training prints rolling exploratory grasp/two-stack/three-stack rates, height,
throughput, and PPO diagnostics. Every 50 updates it evaluates the deterministic
policy on the same 24 random-start and 24 demo-start episodes. `best.json`
advances only for real task milestones (random-start success, demo-start
success, then two-stack, grasp, and height), never shaped return. Stats go to
JSONL and resumable ACT checkpoints are saved every ten updates:

```bash
python train_act_rl.py \
  --checkpoint outputs/act/panthera_stack_30hz/best \
  --output outputs/act_rl/panthera_stack_v3 \
  --num-envs 12 --env-workers 4

# Resume the most recent checkpoint shown in latest.json.
python train_act_rl.py \
  --resume outputs/act_rl/panthera_stack_v3/checkpoint_000100 \
  --output outputs/act_rl/panthera_stack_v3 \
  --num-envs 12 --env-workers 4

# RL checkpoints use temporally ensembled receding-horizon inference.
python rollout_act.py \
  --checkpoint outputs/act_rl/panthera_stack_v3/checkpoint_000100
```

`--env-workers` runs independent MuJoCo/EGL worker processes and exchanges
camera frames, actions, rewards, and critic state through shared memory. Four
workers is the default; each owns an even share of `--num-envs`. Pass
`--env-workers 1` for the original serial loop. Other useful overrides are
`--rollout-steps`, `--episode-steps`, `--checkpoint-freq`, and `--updates`.

The simulator backend is MuJoCo, not MJX. Physical grasp state is measured from
pad contact forces, and both ACT cameras are rendered by MuJoCo. Moving physics
alone to MJX would still require porting those measurements and rendering.

`tools/benchmark_act_batch.py` measures both GPU-only and end-to-end training
throughput. On the RTX 4060 Laptop GPU used for this experiment, batch 12 was
the fastest sustained end-to-end size; larger batches used the GPU more
efficiently but lost that gain while decoding the embedded camera images.
The optional decoded-image cache is about 22 GiB for the 30 Hz dataset. It preserves
the exact RGB pixels, is memory-mapped rather than loaded into RAM, and lets
training bypass PNG decoding and the large embedded Parquet image columns.
Training discovers the default cache automatically; pass `--no-image-cache`
to compare against the original path.

## Model rollout

`rollout.py` runs both full-model and legacy LoRA VLA-Adapter checkpoints in
the two-camera MuJoCo environment. It detects the dataset from the checkpoint:
the IK three-block model uses absolute joint targets, gripper opening in metres,
and 30 Hz control with training-matched random start poses; legacy teleop models
use end-effector deltas at 10 Hz. It accepts an extracted checkpoint, the
notebook's `.tar.gz`, or a Google Drive `.zip`. With no `--checkpoint` it selects
the newest `robot-arm-learning*` or `panthera-ik3-30hz-full*` artifact in
`~/Downloads`. Archive extraction skips optimizer state and downloaded Python
code. Full-model inference uses the saved model and processor directly and
does not download a separate base model.

The script automatically restarts itself in the project's existing
`~/venvs/vla-adapter` environment, so it can be launched with plain `python`.
Install the simulation dependencies into that environment once:

```bash
~/venvs/vla-adapter/bin/python -m pip install -r requirements.txt
```

Then run an interactive rollout. It continues until you press `q`; press `r` to
reset the arm and start with a newly randomized cube layout:

```bash
python rollout.py
```

For an extracted full-model checkpoint:

```bash
python rollout.py --checkpoint /path/to/checkpoint
```

If Drive downloaded the model separately from the ZIP, pass both explicitly:

```bash
python rollout.py --checkpoint /path/to/checkpoint.zip \
  --model-weights ~/Downloads/model-001.safetensors
```

The script assembles these into its inference cache and prints the resulting
directory, which can be reused with `--checkpoint`. The standalone weights
are installed as `model.safetensors`; `training_state--latest_checkpoint.pt`
is not needed. A full checkpoint needs the model, config/processor/tokenizer
files, dataset statistics, action head, and proprio projector. The notebook's
`validation_split.json` records the action contract and sample rate. Passing a
different `--hz` or scaling absolute joint targets is rejected for IK models.

Save a 30-second simulated rollout of the full model:

```bash
python rollout.py --checkpoint /path/to/checkpoint \
  --steps 900 --no-display --no-realtime --video outputs/vla/rollout.mp4
```

Simulation and video use 30 Hz; wall-clock speed depends on inference time.
`--open-loop 1` re-queries every control step instead of executing all eight
predicted actions. Legacy LoRA checkpoints may download the 2.5 GiB base model
and its Qwen/DINO/SigLIP backbones on first use. Legacy examples (10 Hz):

```bash
# Save a headless 20-second rollout using a particular artifact.
python rollout.py \
  --checkpoint ~/Downloads/robot-arm-learning-colab-*.zip \
  --steps 200 --no-display --video rollouts/test.mp4

# Re-query every control step instead of executing all 8 predicted actions.
python rollout.py --open-loop 1

# Run a finite 60-second interactive rollout instead of running indefinitely.
python rollout.py --steps 600
```

## Simulation

The model has six revolute joints, a parallel gripper, a table, and three free
cubes. `PantheraSim` provides reset, stepping, object-pose access, and damped
least-squares IK with joint-limit handling, a home-posture nullspace bias, and
a per-call joint-motion limit.

The commanded target is allowed to enter unreachable parts of its configured
box so failures remain visible: the orange target separates from the gripper
and the HUD reports the IK residual. Press `c` to place the target back on the
arm.

Model utilities:

```bash
python sim/prep_meshes.py
python sim/make_mjcf.py
python sim/panthera_env.py
python -m mujoco.viewer --mjcf sim/panthera/scene.xml
```

## Layout

```text
sim/      prep_meshes.py  make_mjcf.py  panthera_env.py  panthera/
teleop/   keyboard.py  episode.py  replay.py  grid_replay.py
          render_vla_dataset.py
data/     episode_NNN/{data.npz, meta.json, sim.mp4}
```

## Full-model VLA-Adapter on the IK three-block task

[`notebooks/robot_arm_learning_finetune_colab.ipynb`](notebooks/robot_arm_learning_finetune_colab.ipynb)
uses the same 1,000-demo scripted IK three-block dataset as the pi0.5 notebook,
at its native **30 Hz**. It downloads a pinned Parquet export and converts the
embedded shoulder/wrist JPEGs directly to RLDS; no MuJoCo rendering is required.
Actions stay as six absolute next-step joint targets in radians plus gripper
opening in metres, with no additional time shift. Proprio adds a zero padding
element before the gripper to fit VLA-Adapter's 8-D input. Whole episodes are
held out for validation, and normalization uses training episodes only.

Start a fresh run for this dataset. The notebook rejects resuming teleop/EEF
runs and uses separate dataset caches and checkpoint directories. These joint
control checkpoints use `rollout.py`'s full-model joint-control path (see above).
Use an A100-class GPU and at least 100 GiB free runtime disk for conversion,
training dependencies, and the model, plus persistent checkpoint storage.

## Full-model pi0.5 on the IK three-block task

Open [`notebooks/pi05_ik_three_block_full_finetune.ipynb`](notebooks/pi05_ik_three_block_full_finetune.ipynb)
on the **RTX PRO 6000 with 96 GB VRAM**. It downloads the published
[1,000-demo IK dataset](https://huggingface.co/datasets/FoxNerdSaysMoo/panthera-ik-three-block-stack-30hz)
and matching simulator runtime; a local project checkout is not required.
Budget roughly 100–150 GiB total storage, depending on caches and checkpoint
retention. Setup reports free space without a fixed 300 GiB blocker. The smoke
run saves no model copy; training retains the latest two completed checkpoints
(configurable to one) and reports their actual sizes.

The notebook trains the vision encoder, language backbone, action expert, and
projections with no LoRA or frozen parameters. It uses both shoulder/wrist cameras,
30 Hz absolute joint/gripper actions, training-only quantile statistics, an
episode-level validation split, and strict pretrained-weight loading. Defaults
are batch size 8, bfloat16, gradient checkpointing, and 30,000 updates. A four-update
smoke run verifies gradient flow and reports peak VRAM before the long run.

The rollout sections show browser-playable two-camera videos and measure success
over fixed fresh seeds, separately for scripted-region and broader teleop starts.
Success requires the released green–red–blue stack to remain aligned for one
second. Reports include per-seed results, grasp/lift/two-stack rates, errors, and
95% success-rate confidence intervals. Full checkpoints can be resumed or selected
for evaluation using the notebook settings.

W&B is enabled by default for training and held-out loss. Evaluation logs success
rates, confidence intervals, and saved rollout videos in runs grouped by the
training run ID. API keys are entered securely; model checkpoint uploads are
disabled. Select offline mode or turn off video uploads in the settings if desired.

Google Drive backup is enabled by default in Colab. The mount/access prompt comes
before package installation and downloads. Every completed training checkpoint is
copied to `MyDrive/pi05_ik3/<RUN_NAME>/`, including optimizer/RNG state; only the
latest complete Drive backup is retained, after its replacement finishes copying.
Set `RESUME_FROM_DRIVE=True` with the same run name to restore and verify the latest
backup automatically. Data and caches remain on the runtime's local disk.

Supporting tools: `tools/train_pi05_full.py`, `tools/evaluate_pi05.py`,
`tools/publish_pi05_dataset.py`, `tools/bundle_pi05_runtime.py`, and
`tools/build_pi05_notebook.py`. The notebook pins the runtime bundle and data to
an immutable Hugging Face revision. See `reports/PI05_FULL_FINETUNE_20260925.md`
for validation and the distinction between local smoke checks and full GPU training.

## Gripper physics and recording compatibility

The default simulator dynamics are `contact-v2`. Each `sim.step(n)` interpolates
from the previously applied arm command to the new target across the `n` physics
steps. The 30 Hz action format and 2 ms physics timestep are unchanged. Pad
contacts hold the blocks with the existing 3 N actuator limit (about 1.5 N per
finger); `impratio=100` suppresses slow contact creep. Grasp welds stay inactive.
`sim.grasp_flags()` reports bilateral pad loads sustained for 20 ms and clears on
contact loss. Lift and stack metrics still require sustained elevation/release.

After directly restoring a simulator snapshot, call `sim.sync_control_state()`;
when initializing a teleported arm pose, use `sim.set_arm_ctrl(q, immediate=True)`.
Ordinary action updates should use `sim.set_arm_ctrl(q)` followed by `sim.step(n)`.
The stored `ctrl` remains the command endpoint, so action labels stay compatible.

New native recordings, rendered sources, exports, and trained checkpoints carry
`simulation_dynamics`. Replay tools interpret untagged historical recordings as
`weld-v1` and use their original stepped commands, weld assistance, and impedance.
Mixed physics versions are rejected during export. Use a fresh collection/output
directory for new demonstrations. Historical data and checkpoints are not rewritten.
ACT, pi0.5, and VLA rollouts default to the corrected physics; pass
`--dynamics weld-v1` to reproduce the old behavior. ACT/pi0.5 reports distinguish
runtime dynamics from the checkpoint's training dynamics. RL optimizer resumes
across physics versions are rejected; `--checkpoint` starts a fresh fine-tuning run.

The [investigation](reports/GRIPPER_INVESTIGATION_20260926.md) and
[implementation validation](reports/GRIPPER_IMPLEMENTATION_20260926.md) contain
measurements and reproducible commands. `tools/investigate_gripper.py --variants production` exercises the production controller; its other variants preserve
the investigation's historical baselines.

## Onshape actuator assembly experiment

The [actuator environment](sim/actuator/README.md) imports the actual Onshape
carrier and planetary-gear meshes into a separate Panthera scene, excluding
the stator and rotor. Parts start at fixed stations with ±2 mm / ±2° variation.
The contact-only IK controller seats the large carrier and all four gears with
`tools/try_actuator_assembly.py --trial-only --all-gears`. The assembly layout,
rim grasps, and servo tracking correction keep the motions within the arm's
limits and the tight gear clearance.
Use `--with-pins` instead of `--all-gears` to also pick, align, and seat all four
pins in the fixture with physical depth stops. The original gear-only scene and
recordings remain compatible.
See the [measured results](reports/actuator_assembly.md)
for physical trials, limitations, and replay instructions.

## Bimanual place-then-push IK

The [bimanual controller](tools/bimanual_actuator_ik.py) follows the method in
the kept `Desktop/Panthera-Teleop` sessions: the right arm holds the carrier
with a 45° wrist lean, while the left arm partially inserts a gear, pushes it
into place, adds the pin, and presses it down. The separate scene has two
physical arms and exactly three cameras: overhead, left wrist, and right wrist.
Loose gears start directly on the table; this mode has no pickup stands.

```bash
OPENBLAS_NUM_THREADS=1 .venv-assembly/bin/python tools/bimanual_actuator_ik.py \
  --seed 0 --output outputs/bimanual_ik/seed_0 --video

# Render a saved rollout at 4× playback speed.
OPENBLAS_NUM_THREADS=1 .venv-assembly/bin/python tools/replay_bimanual_actuator.py \
  outputs/bimanual_ik/seed_0 --stride 4
```

Starts vary by ±2 mm and ±2°; `--placement-error-mm .35` adds an approximate
lateral placement target before the pushes. `--holder-angle-deg` adjusts the
support wrist lean (0–45°). Quintic motion and joint command bounds keep speed
below 0.6 rad/s and acceleration below 2 rad/s². Seating pushes use contact
loads to slow advancement and a small lateral search. The shallow rim pinch
supports the gear during its first push; a free rim push and downward settling
press follow release. Success requires physical gear and pin seating at the
final supported hold, including engagement through both bores.

`rollout.npz` records both arms as 14-value state/action arrays, raw full-model
qpos/controls, stage IDs, contact loads, and tracking errors at 20 Hz.
Simulated gripper values are **opening in metres**; the real kept recordings
use **motor radians** and measured next-pose actions. These contracts are not
interchangeable. The real USB wrist-to-arm assignments remain unverified, so
synthetic streams use explicit arm names. Scene placement and camera poses are
approximations, and the small carrier retains its physical fixture. The IK
uses privileged object/grasp geometry; it is a simulation demonstrator, not a
camera-only policy or hardware controller.

See [the session review and rollout measurements](reports/bimanual_actuator_ik.md).
