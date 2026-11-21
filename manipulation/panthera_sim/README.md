# panthera_sim

The arm-learning pipeline in simulation: a HighTorque Panthera-HT 6-DoF arm with a parallel gripper on a table with
coloured blocks in MuJoCo. You can teleoperate it or script it to record demonstrations, convert those to LeRobot
datasets, train ACT (then optionally fine-tune it with RL) or fine-tune pi0.5, and evaluate the policies in closed
loop. Where it fits in the manipulation stack: [`../README.md`](../README.md).

Run everything from this folder. `pip install -r requirements.txt` covers the simulator and teleop. Training,
export and evaluation use their own Python 3.10 environment from `requirements-train.txt` (`.venv-act`: run them as
`.venv-act/bin/python <script>`; `rollout_act.py` and `train_act_rl.py` switch to it on their own).

For RGB perception on the two-arm wood bench, see [Identify Mega Bloks, corners, edges and studs with DINOv3](perception/README.md).

## Try image-guided pick and place

Use the DINO environment and trained Mega Bloks adapters from the [perception setup](perception/README.md). Run this command from `manipulation/panthera_sim` with a display and Tk available.

```bash
.venv-dino/bin/python -m tools.dino_pick_place --interactive --seed 109 \
    --output outputs/my_pick_trial
```

Click a block inside a white detection box, then an empty spot inside the cyan workspace outline. The picker checks room around the destination and looks for an arm that can approach, grasp, carry and release there. A rejected click stays in the picker with an explanation. Press **Run pick and place** to watch the overhead and active wrist cameras in the same window. The final screen shows the scored outcome. Close the window during playback to let the simulation finish headless. Repeating the command saves a new numbered directory beside the previous attempt.

The controller uses the bench and camera calibration, nominal block dimensions, and arm feedback. Stud rows and square stud grids give the jaw orientation. The wrist camera matches the selected region by color and position. Corresponding studs in both images refine the grasp center. When wrist landmarks are incomplete, the controller can retain a consistent overhead estimate. An ambiguous match stops the descent. At the destination, the fingers open 12 mm above the nominal grasp height and the block settles on the same bench plane.

For a headless comparison, run both commands on the same seeds. Automatic selection favors a reachable block near the left arm and a bench point with at least 105 mm clearance from detected block centers. Manual placement uses the estimated footprint of the selected block and its neighbors.

```bash
.venv-dino/bin/python -m tools.dino_pick_place --seed 100 --trials 32 \
    --output outputs/pick_wrist
.venv-dino/bin/python -m tools.dino_pick_place --seed 100 --trials 32 \
    --overhead-only --output outputs/pick_overhead
```

Each seed directory contains the overhead selection (green source and magenta destination), available wrist and lift images, the final overhead frame, and `trial.json`. To replay a selection, use the same seed and block count and pass `--source-pixel X Y --destination-pixel X Y` in the 512 × 512 overhead image. The origin is at the upper left.

A separate evaluator uses simulator poses after the actions to score the intended block. Success requires a completed release and retreat, a lift of at least 40 mm with no other block lifted at the lift check, and a settled block on the bench within 25 mm of the destination XY and 6 mm of its initial center height. `summary.json` includes stopped attempts in the total. These trials measure one transfer on a level bench with fixed camera calibration.

### Calibrate cameras during a rollout

Add `--auto-calibrate` to a flat pick, platform transfer, or stack. For example, run this from `manipulation/panthera_sim`:

```bash
.venv-dino/bin/python -m tools.dino_pick_place --auto-calibrate --stack \
    --seed 100 --output outputs/calibrated_stack
```

The cameras use RGB, the robot's measured joint angles, its link geometry, and the specified lens field of view. The estimator starts near the nominal mount, assumes a centered principal point, and fits position, orientation, and one radial lens-distortion coefficient when the observations constrain it. If the robot is outside the initial overhead view, the open gripper moves into view for calibration and returns before block selection. The tabletop can be plain.

Before grasping, the wrist cameras also match DINO stud detections to simultaneous overhead observations. The flat pick uses the known resting stud height. On elevated supports, the measured image spacing and the block's 32 mm stud pitch provide a height estimate. Multiple viewpoints of a stationary stud pattern can refine the mount together. During transport the robot links and jaws continue to supply landmarks.

Each camera pools observations from up to 24 recent frames with matches. New estimates compete against the current mount on those observations. Triangulation receives rectified RGB and the corresponding camera pose. The wrist pose follows the measured gripper through occlusions. If a required camera never gets a usable fit, the trial records `camera_calibration_unobserved_<camera>`. Each trial starts fresh. Its `trial.json` records per-camera robot and stud match counts, accepted updates, pixel residuals, and the fitted mount.

### Track a block between platforms at different heights

Use `--platforms` for a transfer between two plain horizontal supports. Run from `manipulation/panthera_sim` in the DINO environment. If you already installed the flat-bench environment, rerun the perception requirements install to add OpenCV.

```bash
uv pip install --python .venv-dino/bin/python -r perception/requirements.txt
.venv-dino/bin/python -m tools.dino_pick_place --platforms --interactive --nudge \
    --seed 0 --output outputs/tracked_transfer
```

Select the colored block and click inside the opposite platform outline. The right arm moves its camera to a side view and holds there. The left wrist surveys the work area before grasping. The window shows the active wrist and overhead RGB, the tracked block position, and the estimated heights in the robot frame. With `--nudge`, the simulator moves the block 20 mm sideways during the survey. Omit that flag to keep the block stationary until pickup.

The three calibrated RGB cameras supply the geometry:

- DINO identifies the block and studs. Matching the stud pattern across views gives a 3D grasp estimate. During the survey and approach, the overhead and parked right cameras match fresh DINO stud detections to track the block's displacement. The controller checks the target again at the grasp hover.
- The controller locates the two gray support outlines against the wood bench in overhead RGB. It uses the camera poses to align wrist crops to the overhead view at candidate heights from −50 to 230 mm in robot coordinates. The DINOv3 backbone compares descriptors around the surface boundaries. Agreement across wrist views gives local heights for fitting the plane. The clicked pixel's ray intersects that plane at the landing point. When too much of the outline is occluded, the height can remain unobserved.
- After lift, the cameras observe the block at the start, midpoint and end of transport. The controller fits its visible silhouette or studs in the moving gripper frame and adjusts the release pose when the measurements agree. If the fit is inconclusive, it uses the nominal grasp offset. These trials assume an upright Mega Blok of known dimensions and a nearly level landing surface.

For a paired headless test, run the same seeds with and without `--single-estimate`. This comparison freezes the first usable pair of source and destination estimates and uses the nominal gripper offset through release. Even seeds request uphill transfers and odd seeds request downhill transfers.

```bash
.venv-dino/bin/python -m tools.dino_pick_place --platforms --nudge --seed 80 --trials 12 \
    --output outputs/tracked_nudges
.venv-dino/bin/python -m tools.dino_pick_place --platforms --nudge --single-estimate \
    --seed 80 --trials 12 --output outputs/frozen_nudges
```

Each `trial.json` records the accepted estimates, missed observations, physical motion, and release correction. The evaluator checks the selected object's lift and the final landing against the original clicked ray and the actual destination support. A successful transfer must settle on the other platform within 25 mm of the requested point. Source and destination height errors are reported separately in meters.

### Stack two blocks

Run from `manipulation/panthera_sim` in the DINO environment with a display and Tk available.

```bash
.venv-dino/bin/python -m tools.dino_pick_place --stack --interactive \
    --seed 0 --output outputs/stack
```

Click the block to lift, then click the other block to use as the base. The two 2×1 blocks have sockets that fit over the studs when their long axes line up. During the wrist’s three-quarter ellipse, the controller captures four sets of synchronized overhead and wrist images. A background worker runs DINO on those snapshots while the arm moves. Triangulating each two-stud row gives the block’s center, height, and heading.

Once both blocks have a verified pose, the gripper approaches the source and rounds the corner into its descent. It holds position to close the jaws, lifts to transport height, then turns the block toward the base during the carry. The cameras take two more sets of views during the crossing. As the arm descends, the controller uses the latest verified base estimate and the measured grasp offset to align the sockets. A refinement that arrives after this decision is recorded in `trial.json` with its capture and processing times. The gripper holds position again while opening its jaws.

For a headless trial, omit `--interactive`. To replay a particular selection, pass both pixel arguments with the same seed. The evaluator checks whether only the selected block lifted and whether it seated on the selected base while the base stayed upright on the bench near its initial pose. Success also requires the released stack to stay settled through the final hold. `trial.json` records the lift, base displacement, alignment, contact, and hold checks.

### Build a base and tower

Run this from `manipulation/panthera_sim` for a four-block build. The default layout for three or four blocks has two blocks on the bench. With four blocks, the arm places two more layers above the anchor.

```bash
.venv-dino/bin/python -m tools.dino_pick_place --stack --blocks 4 \
    --seed 0 --output outputs/grid_build
```

The anchor stays on the bench. Its stud midpoint defines the grid origin and its stud row defines the heading. The planner tries tabletop positions at whole stud pitches and checks the pickup and placement IK before starting. Set `--base-blocks 1` for a single tower, or `--base-blocks 3` with four blocks for a longer base and one upper block. Add `--interactive` to click the first block to move, then the anchor. Headless runs choose both automatically. You can also select just the anchor with `--destination-pixel X Y`.

The overhead and wrist cameras triangulate every block during the initial survey and refine visible resting blocks during each transfer. After release, the arm reacquires the moved block near its planned cell. It surveys again if the first view misses, and stops if it cannot confirm the placement. The next tower layer uses the measured studs of its support. Each `trial.json` records the initial measured grid coordinates, planned cells, per-block observations, actual pickups, and final contact and alignment checks for every layer. Grid X and Y are in stud pitches and Z is in layer heights.

### Assemble a stud blueprint

From `manipulation/panthera_sim`, run the four-block castle preset.

```bash
.venv-dino/bin/python -m tools.dino_pick_place --blueprint castle --seed 0 --output outputs/castle
```

The arm places a second pier three stud pitches from the anchor, turns a four-stud lintel across the two piers, and seats a cap on the lintel. The `pyramid` preset bridges two adjoining base blocks with a third block. The `one-stud` preset turns a three-stud row across the anchor so its middle socket engages one stud. Each trial saves a side view of the settled build in `build.png` and the per-block pickup, support, and alignment checks in `trial.json`.

The [five-block gatehouse blueprint](examples/gatehouse.json) uses two three-stud parapets. Each one sits across a single end stud of the lintel, leaving the center open between them. Pass the example file to the same command.

```bash
.venv-dino/bin/python -m tools.dino_pick_place --blueprint examples/gatehouse.json --output outputs/gatehouse
```

For a custom layout, pass a JSON file to `--blueprint`. Supply two to six blocks, each one stud wide and two, three, or four studs long. The first block stays on the bench with its first stud at `[0, 0, 0]`. For each block, `cell` locates its **first stud**, with X along the anchor row, Y across it, and Z in seated layers. One horizontal unit is a 32 mm stud pitch. The optional `turn` counts counterclockwise quarter-turns from +X. Values 0, 1, 2, and 3 point along +X, +Y, −X, and −Y respectively. Save the example below as `cross.json` to center a three-stud block across the third stud of a four-stud base.

```json
{
  "blocks": [
    {"length": 4, "cell": [0, 0, 0]},
    {"length": 3, "cell": [2, 1, 1], "turn": 3}
  ]
}
```

```bash
.venv-dino/bin/python -m tools.dino_pick_place --blueprint cross.json --output outputs/cross
```

The planner matches the observed rows to the requested lengths and checks the IK for each pickup and release. An elevated block needs at least one occupied stud directly below it. For a bridge, the controller refines the release against the observed studs on both supports. The evaluator checks load-bearing contact and alignment after release, including whether an overhang stayed seated through the final hold.

## Layout

```
sim/
  panthera/            MJCF model and scenes (scene.xml: three blocks; scene_two_blocks.xml: red and green)
  make_mjcf.py         builds the MJCF from the arm description; prep_meshes.py prepares its meshes
  panthera_env.py      PantheraSim: reset, step, object poses, damped least-squares IK (joint limits, home-posture
                       nullspace bias, per-call motion limit), pad-contact grasp detection
  dynamics.py          the simulator dynamics versions (contact-v2 default; weld-v1 for old recordings)
  stack_task.py        stacking task: block layouts, grasp / lift / stack checks, success rule
teleop/
  keyboard.py          keyboard (and mouse-look) teleoperation; records episodes
  episode.py           native episode format (data.npz + meta.json + sim.mp4)
  replay.py, grid_replay.py, render_vla_dataset.py
  build_lerobot_dataset.py      recordings → LeRobot dataset (shoulder + wrist images, 7-D state/action, 30 Hz)
  build_panthera_ik_rlds.py, build_robot_arm_learning_rlds.py   recordings → RLDS (VLA fine-tuning)
  dataset_contract.py  the shared state/action/camera contract every exporter and policy checks
tools/
  collect_scripted.py, validate_scripted_dataset.py, export_scripted_dataset.py   scripted IK demonstrations → LeRobot
  lerobot_image_cache.py        optional memory-mapped decoded-image cache for faster ACT training
  act_scene.py, act_pickup.py   ACT rollout scenes and the first-pickup task
  evaluate_act.py, replay_act_actions.py      closed-loop ACT evaluation; open-loop replay of dataset actions
  prepare_act_reference.py, act_reference_data.py, evaluate_act_reference.py   the original ACT transfer-cube benchmark
  train_pi05_full.py, evaluate_pi05.py, publish_pi05_dataset.py, pi05_checkpoint_backup.py   pi0.5 full fine-tune
train_act.py           ACT imitation training (LeRobot)
train_act_rl.py        PPO fine-tuning of a trained ACT policy
rollout_act.py         run an ACT checkpoint in the simulator (interactive or headless video)
rollout.py             run a VLA checkpoint (pi0.5-style joint targets or end-effector deltas) in the simulator
tests/                 pytest suite for the task, exporters, ACT pipeline and rollouts
```

Output folders (`data/`, `outputs/`) are created at run time and are not kept in the repo.

## 1. Collect demonstrations

**Teleop**: `python teleop/keyboard.py` (`--help` for speed, workspace, views, `--minecraft` mouse look).

```
w / s   forward / back (+x / -x)      u / j   pitch
a / d   left / right   (+y / -y)      i / k   yaw
SPACE   up                             n / m   roll
SHIFT   down                           o / l   gripper open / close
[ / ]   slower / faster               c       re-centre target
r       reset arm near home           e       save take
x       discard take                  q       quit
1-4     focus a view (shoulder, wrist, overhead, chase)      g   all four views
```

Recording starts on the first command; `e` saves `data/episode_NNN/` with `data.npz` (time, joint state, controls,
achieved and target end-effector poses, object poses, gripper command, IK residuals), `meta.json` (scene, settings,
conventions) and `sim.mp4`. `python teleop/replay.py data/episode_000` replays a take.

**Scripted IK**: smooth Cartesian waypoints (approach, descend, close, lift, transfer, place, release, retreat) through
bounded IK at 30 Hz. Only demonstrations that end in a released stack held for one second, aligned within 12 mm, are
kept; rejected seeds and reasons go to `attempts.jsonl`. Re-running resumes the same collection.

```bash
python tools/collect_scripted.py --episodes 1000 --workers 6                      # three blocks: red on green, blue on red
python tools/collect_scripted.py --output data/scripted_two_block_fixed --blocks 2 --arm-start fixed --episodes 500
python tools/validate_scripted_dataset.py --input <collection>
```

## 2. Build a LeRobot dataset

```bash
python teleop/build_lerobot_dataset.py                                   # teleop takes → outputs/lerobot/panthera_stack_30hz
python tools/export_scripted_dataset.py --input <collection> --output outputs/lerobot/<name> --workers 4
```

State is the six joint angles plus the gripper command; the action is the next 30 Hz joint target; images are the
shoulder and wrist cameras at 256×256. Exports fingerprint their inputs, write a `COMPLETE` marker when done, and reject
mixing simulator dynamics versions. Both tools stop below 5 GiB free disk.

## 3. Train

**ACT** (imitation):
```bash
python train_act.py --dataset outputs/lerobot/<name> --output outputs/act/<name> \
  --steps 100000 --action-steps 30 --rollout-action-steps 30 --eval-freq 10000 --full-val-at-end \
  --rollout-eval-freq 10000 --rollout-eval-episodes 8
```
ACT predicts a one-second chunk (30 actions) and executes the whole chunk before querying again (frequent replanning
stalled this policy in comparisons; `--temporal-ensemble` and `--action-steps N` change that). `best/` is chosen by
held-out imitation loss, `best_rollout/` by task success in closed-loop rollouts. `deployment.json` in each checkpoint
records the dataset, rate and start settings; rollout and RL reject mismatches. Resume with `--resume <checkpoint>`
against the same dataset only.

**ACT + RL**: `train_act_rl.py` continues from an ACT checkpoint with conservative PPO: the backbone and transformer
are frozen, the action head is updated with a penalty for drifting from the imitation policy, and a privileged state
critic is used only in training. Half the episodes start from mid-task snapshots of the demonstrations.
```bash
python train_act_rl.py --checkpoint outputs/act/<name>/best --output outputs/act_rl/<name> --num-envs 12 --env-workers 4
```

**pi0.5** (full fine-tune, no LoRA; needs a large GPU, about 96 GB at batch 8):
`../finetune.sh pi05 panthera_sim <LeRobot flags>` (or `tools/train_pi05_full.py`, the same trainer with this arm's
profile), evaluated with `tools/evaluate_pi05.py`; `tools/publish_pi05_dataset.py` uploads a dataset
to the Hugging Face Hub and `tools/pi05_checkpoint_backup.py` keeps the latest checkpoint backed up.

**ACT reference benchmark**: `python tools/prepare_act_reference.py` then `python train_act.py --reference-act` trains on
the original ACT authors' 50 simulated transfer-cube demonstrations, to check the training code against a known
result; `tools/evaluate_act_reference.py` evaluates it in the original simulator.

## 4. Evaluate

```bash
python rollout_act.py --checkpoint outputs/act/<name>/best                                   # interactive: r reset, q quit
python rollout_act.py --checkpoint <ckpt> --steps 450 --no-display --no-realtime --video out.mp4
python tools/evaluate_act.py                                                                 # success rates over fixed seeds
python rollout.py --checkpoint <vla checkpoint>                                              # VLA policies
```
Success means the released stack stays aligned for the required hold time; reports break results into grasp, lift,
two-stack and full-stack rates.

## Simulator dynamics

The default dynamics (`contact-v2`) hold blocks by pad contact under the 3 N gripper limit and interpolate each arm
command across the physics steps of a control tick (30 Hz control, 2 ms physics). `sim.grasp_flags()` reports grasps
from sustained two-sided pad loads. After restoring a snapshot call `sim.sync_control_state()`; to teleport the arm use
`sim.set_arm_ctrl(q, immediate=True)`. Recordings, datasets and checkpoints carry `simulation_dynamics`; untagged
recordings are treated as `weld-v1`, and `--dynamics weld-v1` reproduces it in rollouts.

## Tests

`python -m pytest tests` (needs lerobot and torch; rendering tests need EGL or a display).
