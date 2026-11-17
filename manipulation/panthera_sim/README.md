# panthera_sim

The arm-learning pipeline in simulation: a HighTorque Panthera-HT 6-DoF arm with a parallel gripper on a table with
coloured blocks in MuJoCo. You can teleoperate it or script it to record demonstrations, convert those to LeRobot
datasets, train ACT (then optionally fine-tune it with RL) or fine-tune pi0.5, and evaluate the policies in closed
loop. Where it fits in the manipulation stack: [`../README.md`](../README.md).

Run everything from this folder. `pip install -r requirements.txt` covers the simulator and teleop. Training,
export and evaluation use their own Python 3.10 environment from `requirements-train.txt` (`.venv-act`: run them as
`.venv-act/bin/python <script>`; `rollout_act.py` and `train_act_rl.py` switch to it on their own).

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
