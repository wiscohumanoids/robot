# Getting started in manipulation: robust block pick-and-place

For new members of the manipulation team. You can start today, on a laptop (simulation) or on the real Panthera arms
(hardware), and work all the way up to a trained policy. The lead runs the GPU training; you get the model back and
find out how good it really is.

## The goal

**Pick up a DUPLO-style block from any spot on the table and place it at any other spot, reliably.**

That one skill, done robustly, is the building block for something much bigger: a planner on top (a language model
that reasons about the scene) breaks "build a house" into a sequence of *move this block from here to there* steps,
and our policy executes each one. The planner is easy to swap; the hard part, and the part this team owns, is a
pick-and-place that works from *anywhere to anywhere*, every time.

```
"build a small house"  ──►  planner (LLM)  ──►  place red 2x2 at C3 · place blue 2x2 at C5 · stack yellow on C3 · ...
                                                          │ one step at a time
                                                          ▼
                                      our policy (pi0.5): camera images + "place the red block at C3"
                                                          │
                                                          ▼
                                                     Panthera arm
```

**Why a baseplate grid.** "Any x, y" is easiest to teach and to check when positions are discrete: a DUPLO baseplate
gives a grid of studs (16 mm apart; a 2×2 block is about 32 × 32 × 19 mm). Goals become cells like `C3`, which a
planner can name, a prompt can carry ("place the red block at C3"), and a script can verify (did the block end up on
C3, square, within a few mm?). Stacking is the same skill with a height: "place the yellow block on top of C3".

## How the work flows

```
 you (sim or hardware)                        lead                       you again
 ─────────────────────                        ────                       ─────────
 1. set up a task      ─►  2. collect demos  ─►  3. LeRobot dataset  ─►  4. train pi0.5 on   ─►  5. evaluate: success rate
    (scene, prompts,         (scripted or         on Hugging Face          the GPU machine         by start/goal cell,
    success check)           teleoperated)        ($HF_NAMESPACE/...)      (finetune.sh)           failure videos, fixes
```

Everything up to step 3 is yours; step 4 is one command for the lead
(`manipulation/finetune.sh pi05 <profile> ...`); step 5 is yours again, and it feeds the next round of data.

## Track A: simulation (laptop, no hardware)

The arm is the same Panthera-HT as on the real rig, simulated in MuJoCo with contact-based gripper physics, in
[`panthera_sim/`](panthera_sim/README.md). What exists today: a table with three coloured 4.5 cm blocks, keyboard
teleop, a scripted demonstration generator (inverse kinematics through waypoints), exporters to LeRobot datasets, ACT
and pi0.5 training, and closed-loop evaluation, all for **stacking**. The first members turn that into the any-to-any
pick-and-place task.

**Day 1: set up and look around**
```bash
cd manipulation/panthera_sim
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python teleop/keyboard.py                 # drive the arm (w/a/s/d, space/shift, o/l gripper; e saves a take)
.venv/bin/python teleop/replay.py data/episode_000  # replay what you recorded
.venv/bin/python tools/collect_scripted.py --episodes 20 --workers 2 --output data/try_scripted   # scripted demos
```
Read `sim/stack_task.py` (how a stacking task is defined and judged) and `sim/panthera_env.py` (the arm, the IK).

**First contributions (pick one, they fit together)**

| Piece | What you build | Done when |
|---|---|---|
| Baseplate scene | a scene with a DUPLO-sized baseplate (cell grid with labels) and 2×2 / 2×4 blocks in a few colours, next to the existing `scene.xml` | blocks rest stably on the plate; the gripper can close on a 32 mm block |
| Task definition | like `stack_task.py`: sample a block, a start cell and a goal cell (any to any, plus "on top of"); write the prompt ("place the red block at C3"); a success check (on the goal cell, within a few mm, upright, released, held for 1 s) | a random task can be sampled, prompted and judged |
| Scripted demonstrations | extend `collect_scripted.py`: approach, grasp, lift, carry, place on the goal cell, release, retreat; reject failed attempts, as the stacking collector does | hundreds of clean demos across the whole grid, including the edges and far corners |
| Variety | randomized lighting, block colours and small camera shifts, so the policy does not overfit to one look | the same task succeeds under a different lighting seed |
| Evaluation | like `tools/evaluate_pi05.py`, for the new task: success rate per start/goal region, near versus far, stacked versus flat, with failure videos | one command scores a checkpoint and writes a report |

**Make the dataset and hand it over**
```bash
.venv/bin/python tools/validate_scripted_dataset.py --input data/<collection>
.venv/bin/python tools/export_scripted_dataset.py --input data/<collection> --output outputs/lerobot/<name> --workers 4
.venv/bin/python tools/publish_pi05_dataset.py --source outputs/lerobot/<name> --repo-id $HF_NAMESPACE/<name> --push
```
`HF_NAMESPACE` is the team's Hugging Face account, from the repo's `.env`. Then tell the lead the dataset name and the
arm profile it matches (`manipulation/profiles/`; a new task gets a new profile: cameras, state and action sizes, rate,
and the prompt style). Teleoperated takes go through `teleop/build_lerobot_dataset.py --task "<prompt>"` instead.

## Track B: real Panthera arms (hardware)

The real rig is two Panthera-HT arms with three cameras (overhead and one per wrist), in
[`panthera_teleop/`](panthera_teleop/README.md). What exists today: gravity-assisted hand guiding (you move the arm
by its handle while it compensates its own weight), optional Meta Quest teleoperation, three-camera recording at
20 Hz, episode review, export to LeRobot datasets, upload to Hugging Face, a pi0.5 model served from the GPU machine,
a live dashboard to run it on the arms, and human-in-the-loop correction recording. It was all built on a gear-and-pin
assembly task; DUPLO pick-and-place reuses every piece.

**Before you touch the arms**: do Track A's day 1 first, then run the dashboard with no motors:
```bash
cd manipulation/panthera_teleop
# one-time setup: README "1. Install the software" (SDK, Python environment in work/)
./start.sh --demo --data-dir /tmp/panthera-demo        # simulated arms, real dashboard: http://127.0.0.1:8090
```
Work with someone who has run the rig before for your first live session. Support both arms before stopping anything.

**Set up the DUPLO task on the table**
- A DUPLO baseplate taped at a fixed place in the overhead camera's view; mark which cell is A1.
- A small set of 2×2 and 2×4 blocks in distinct colours.
- A written protocol for each demonstration: which block, start cell, goal cell, the exact prompt, and how to reset.
  Spread starts and goals over the *whole* plate: the policy only learns the positions it sees.

**Record**
```bash
./start.sh        # dashboard on http://127.0.0.1:8090; guide the arm by its handle
```
Each episode starts when an arm moves; **K** keeps it, **R** flags it for review, **D** discards it, and the arms
return to a raised neutral pose. Keep demos smooth and complete (grasp, carry, place, release, back off). A few
hundred good episodes beat a thousand sloppy ones.

**Review, export and upload**
```bash
python3 scripts/review_runs.py                                # trim and approve flagged runs: http://127.0.0.1:8091
work/postprocess-venv/bin/python scripts/postprocess_kept.py --source data/kept \
  --annotations config/postprocessing-template.json \
  --output data/processed/<name>-20hz --trimmed-output data/processed/<name>-full-rate \
  --repo-id $HF_NAMESPACE/<name>
work/postprocess-venv/bin/python scripts/verify_processed.py data/processed/<name>-20hz   # checked against the source
work/postprocess-venv/bin/python scripts/publish_processed.py data/processed/<name>-20hz
```
The task prompt of every episode goes into the dataset; it must be exactly the prompt the policy will be given.

**Try a trained model on the arms.** When the lead deploys your model to the GPU machine, the dashboard runs it live
over an SSH tunnel (`panthera_teleop/infra/inference/README.md`): pick a start and goal, give the prompt, watch,
score. Where it fails, record corrections with the human-in-the-loop flow
(`panthera_teleop/docs/human-in-the-loop.md`): you take over at the mistake, finish the task, and that becomes the
most valuable training data there is.

## What the lead does with your dataset

```bash
manipulation/finetune.sh pi05 <profile> --dataset.repo_id=$HF_NAMESPACE/<name> --dataset.root=<local copy> ...
```
on the GPU machine: a full pi0.5 fine-tune with checks that the dataset, cameras, prompt and action sizes match the
profile (`manipulation/README.md`, "Fine-tuning: one entry point"). The checkpoint comes back with its
`deployment.json` (what it expects), for simulation evaluation or for the inference server on the real arms.

## How we judge "robust"

Report numbers, not impressions:

- **Success rate over the whole grid**, broken down by distance (adjacent cells versus opposite corners), by block
  size and colour, and flat versus stacked.
- **Held-out positions**: some start/goal pairs never appear in the training data. Success there is what "any x, y"
  really means.
- **Placement error** in mm and angle on success, and **failure types** (missed grasp, dropped, wrong cell, knocked
  other blocks) with a few videos each.
- **Repeatability**: the same task ten times in a row.

A good first milestone: 80 % over the full grid in simulation, then on the real arms. The next ones: held-out cells,
stacking three high, and a five-block "house" from a list of steps.

## Ideas to take on once the basics work

- **Sim and real side by side**: the same arm and task in both. Train in sim, test on the real arm, measure the gap,
  and tune the simulator to match (the arm toolkit's PACE fit, `ros2_ws/src/wisco_ethercat_ros2_hw/arm_stack/`).
- **The planner on top**: a script that turns "build a house" into a list of cell-to-cell moves, executes them with
  the policy, and checks each step from the overhead camera.
- **Recovery**: deliberately disturbed demos (a block knocked off its cell) so the policy learns to re-grasp.
- **Two arms**: hand a block from one arm to the other to reach the far side of the plate.

## Rules of the road

- Work on a branch; open a pull request; keep large data out of git (datasets go to Hugging Face, recordings stay in
  `data/`, which git ignores).
- Accounts, hosts and keys live in the repo's `.env`, never in code.
- Name datasets so they say what they are: `panthera-duplo-pickplace-sim-30hz-<N>`,
  `panthera-duplo-pickplace-real-20hz-<N>`.
- Write down your collection protocol with the dataset; the next person needs to reproduce it.
