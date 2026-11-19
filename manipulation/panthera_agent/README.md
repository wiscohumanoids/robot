# panthera_agent

An LLM agent that does tabletop pick-and-place with the Panthera arms by calling tools, in the MuJoCo sim
(`../panthera_sim`) and on the real rig (`../panthera_teleop`'s arms and cameras). The model never sends motor
commands. It looks at camera images, decides what to do next, and calls one tool at a time: find an object, pick it,
place it, check the result. Each tool is ordinary geometry and inverse kinematics. Learned skills come in only where
the geometry fails, and the test results say where that is.

**Rule of thumb: the LLM chooses, perception models measure, geometry computes.** The LLM picks the object, the part
and the next step, and judges whether a step worked. Detection and segmentation find pixels. Geometry turns pixels
into positions, grasp poses and joint targets. The LLM is never asked for coordinates it would have to read off pixels.

## The pipeline

```
instruction ("stack red on green") + overhead image
        │
        ▼
   LLM (Claude) ──► find("red cube")              segmenter: text -> one mask per match, drawn numbered on the image
        │            └─ result: #1 red cube at (0.41, 0.12) m, 46 x 45 mm, yaw 12°
        ├──────────► pick(1)                       geometry + IK, no LLM inside:
        │              1. pre-grasp      overhead ray ∩ plane z = table + object height -> x, y; rectangle fit -> yaw;
        │                                8 cm above the grasp point (config)
        │              2. move           IK to the pre-grasp pose (up, across, down)
        │              3. refine         wrist camera: re-segment, re-intersect, correct x, y, yaw; repeat until < 2 mm
        │              4. grasp          descend, close
        │              5. check          gripper width (closed fully = missed; on the rig, current too), lift
        │            └─ result: held, width 44 mm, refine moved 3.1 mm
        ├──────────► look("overhead")              fresh image: did it work?
        ├──────────► place_on(2)                   the same against the target's top face (the held block hides the
        │                                          target from the wrist camera, so the overhead estimate is corrected by
        │                                          what the wrist refines have measured: see "Calibration error")
        ├──────────► find(...) / look(...)         verify
        └──────────► done(success, summary)
```

So the LLM runs at the start, after each step and on failures, never inside a motion.

## What we agreed, and why

| Decision | Choice | Why |
|---|---|---|
| Planner | Claude through the Anthropic API (`ANTHROPIC_API_KEY` in the repo's `.env`). Default model `claude-opus-5-5`, configurable. | It takes images in tool results, and it's the API we already pay for. The connector is one file (`llm/claude.py`), so another provider is a swap. |
| Tool granularity | Few, semantic tools (`find`, `pick`, `place_on`, `place_at`, `look`, `open_gripper`, `home`, `done`), one call per turn | Published agent harnesses find semantic tools beat low-level ones (88.6 vs 73.3 for a frontier model, 22.9 vs 4.8 for a 4B one), and one call per turn beats batching several. |
| Which object | Numbered mask outlines on the image; the LLM answers by number | Much more reliable than asking a VLM for pixel coordinates. |
| Depth | None. Ray ∩ known plane (table top + object height), then a wrist-camera refine | The rig's ELP cameras are monocular. The table is a known plane and the objects have known sizes, so this is geometry, not an estimate. The wrist refine cancels most calibration error. |
| Footprint | The mask's outline projected onto the top-face plane and onto the base plane; their overlap, fitted with a minimum-area rectangle | Seen at an angle, an object's side faces enlarge its mask; on the two planes they push the outline in opposite directions, so the overlap is the true footprint (centre error from 13 mm to under 0.1 mm in the wrist view). A stacked object's level is the one whose expected size the measurement matches. |
| Grasp orientation | Rectangle yaw, folded by the object's symmetry toward the bearing from the arm's base (a radial approach), at the sim collector's 55° pitch | Radial halves the unreachable poses compared with folding toward the forward axis. GraspGen later, for odd shapes: a drop-in, since the step's output is only "a grasp pose". |
| Clearance | Before a pick or place, each equivalent yaw (four for a cube) is checked for reach (IK) and for the open gripper's boxes (`grasp.gripper_boxes`) against every other known object; the most radial one that clears is used, else the tool fails and names what is in the way | The descent is straight down, so the final pose's boxes cover it. Seen in the sim: an unchecked pick next to a tower knocked it over. Refusing and naming the obstacle hands the decision (move it, or work elsewhere) to the LLM. |
| Look pose | Home is high and near the base (0.20, 0, 0.30 m) | From over the table, the arm hid part of a block from the overhead camera in 7 of 36 placements; from here, in none. |
| Motion | IK through safe waypoints (the sim's damped least-squares IK; on the rig the same solver on a MuJoCo copy of the arm kept in sync with the real joints) | Enough for an uncluttered table. MoveIt 2 later, for clutter. Same interface: go to this pose. |
| Grasp | Close the gripper; check the width (and current on the rig). No learned policy. | Picking a block is easy. A learned policy is for when the tests show the scripted grasp failing. |
| Place | Descend to just above seated (`place.gap`), release | May be too coarse for DUPLO studs (about 1–2 mm and a press). If tests show that, the first ACT skill goes there (see "Later"). |
| Calibration error | Each wrist refine of an overhead estimate measures the overhead camera's error at that spot; the median of the last five is added to overhead estimates | A small tilt error in the overhead camera shifts everything it measures by nearly the same amount (1° from 0.85 m is about 15 mm). Places need this, because the held block hides the target from the wrist camera. |
| Segmentation | Pluggable: ground truth in sim; on real images, the existing GPU-VM perception server (Grounding DINO + SAM 2.1, `services/perception_server`, port 8100, no depth needed) | Reuses proven infrastructure. SAM 3 is gated on Hugging Face (access has to be requested). It can be added to the server later without changing this code. |
| Cameras | Rig: ELP USB modules (two wrists, one centre), calibrated. Assumed wide-angle, not fisheye, with the centre camera looking roughly straight down, until we look. Insta360s: recording only. | ELPs are plain UVC webcams with a fixed image model. Action cameras stabilise electronically, which warps every frame differently and breaks calibration. |
| Arm choice (two arms) | A rule: the nearest arm whose IK reaches. Not the LLM. | A geometric fact, not a judgement. |
| Logging | Every LLM call, image, tool call and result per run in `runs/` | For debugging, and as training data for a smaller model later. |

**Sim vs. DUPLO.** The sim has 45 mm cubes (red, green, blue), not DUPLO. The geometry only needs each object's
height and footprint (`config/sim.yaml`, `objects:`), so DUPLO bricks are new entries there plus a scene with bricks.
DUPLO dimensions: 19.2 mm per brick height (body, without studs), studs 16 mm apart, 2×2 footprint 31.8 mm square.

## Layout

```
README.md            this plan
requirements.txt     Python 3.10+: mujoco, numpy, opencv-contrib (ChArUco), pyyaml, anthropic, pytest
config/
  sim.yaml           the sim: cameras, table, objects, grasp / place / refine / motion settings
  rig.yaml           the real rig: cameras and calibration files, DUPLO objects, gripper, speed and safety limits
                     (values marked R0 are measured at calibration)
core/
  camera.py          pinhole / fisheye camera (OpenCV convention): pixel -> ray, point -> pixel, MuJoCo or calibrated
  geometry.py        ray ∩ plane, mask -> footprint (centre, size, yaw), grasp rotation and yaw folding
  world.py           the found objects (numbered), their types (config `objects:`)
  skills.py          find, pick, place_on, place_at, look, open_gripper, home; the wrist refine and the overhead
                     correction; a trace of every estimate and move
  tasks.py           sim test tasks: instruction, set-up, success from ground truth (panthera_sim's thresholds)
  calibration.py     ChArUco intrinsics, hand-eye, fixed-camera pose, plane fit (tested on synthetic data)
  build.py           config, .env, segmenter and skills construction for the tools
perception/
  base.py            Segmenter interface: (image, query) -> instances (mask, label, score)
  sim_truth.py       ground-truth masks from MuJoCo's segmentation render (S1)
  server_client.py   the GPU VM's perception server (Grounding DINO + SAM 2.1) over an SSH tunnel (S2, rig)
  overlay.py         numbered mask outlines, and the table grid in cm, on images for the LLM
robots/
  base.py            Robot interface: observe, ee_pose, move_linear, set_gripper, gripper_width, home
  sim_robot.py       the Panthera MuJoCo sim (../panthera_sim), with S3's camera-pose error option
  real_robot.py      the real arm through the vendor SDK, OpenCV cameras, calibration, safety box (not yet run on
                     hardware)
llm/
  claude.py          Anthropic API connector: model, effort, one tool call per turn, prompt caching, refusal fallback
  tools.py           the tool schemas (strict) and their dispatch to the skills; results as text + images
  prompts.py         the system prompt
  loop.py            the agent loop (append-only conversation), logging every turn and image
tools/
  render_views.py    S0: what each sim camera sees
  run_scripted.py    S1-S3: scripted pick-and-place through the skills, per-stage error against ground truth
  run_agent.py       S4: one instruction through the LLM (sim, or --robot rig)
  evaluate.py        S5: tasks x seeds through the LLM, success from ground truth
  calibrate_intrinsics.py   R0: camera intrinsics from ChArUco views
  calibrate_extrinsics.py   R0: wrist hand-eye, centre camera pose and table plane from one board on the table
  touch_test.py      R1: centre camera -> table point -> TCP above it; measure the miss
tests/               pytest (no GPU, API key or rig): geometry against MuJoCo's cameras, segmentation, the skills in
                     the sim (stacking, calibration error, unstacking), calibration math, the agent loop with a
                     scripted stand-in for the model, the API request
runs/                run logs (not kept in the repo)
```

## Setup

```bash
cd manipulation/panthera_agent
uv venv --python 3.10 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q tests                      # no API key or GPU needed
```

The LLM needs `ANTHROPIC_API_KEY` in the repo's `.env` (already listed in `.env.example`). The tools read it from
there or from the environment. The real-image segmenter needs the perception server running on the VM and a tunnel:
`ssh -N -L 8100:localhost:8100 $VM_HOST` (see `services/perception_server/README.md`; `PERCEPTION_SERVER_URL`
overrides the address). On the rig's computer, install `requirements.txt` into panthera_teleop's environment (the
adapter uses its SDK set-up).

## Steps

### Sim first

| Step | What | Done when | Run |
|---|---|---|---|
| **S0** | Cameras as on the rig (placement, field of view, resolution); segmentation rendering | Sim and rig images look comparable | `tools/render_views.py` |
| **S1** | The skills with ground-truth masks: ray ∩ plane position, rectangle yaw, move above, refine, descend, grasp, check, place | Scripted pick-and-place works with perfect perception; position error per stage about 0 | `tools/run_scripted.py` |
| **S2** | Real segmentation on rendered images (the perception server) | Detection rate and position error per stage measured | `tools/run_scripted.py --segmenter server` |
| **S3** | Calibration noise: the overhead camera's estimated pose off by ±5 mm / ±1° | The wrist refine brings the error back under about 2 mm | `tools/run_scripted.py --camera-noise 0.005 1.0` |
| **S4** | The LLM loop with the tools | Multi-step tasks run: stack, unstack, sort, make a row | `tools/run_agent.py "stack the red cube on the green cube"` |
| **S5** | A fixed test set: N seeds × tasks, success from the simulator's state, every call logged | A success rate to track; the logs are future training data | `tools/evaluate.py` |

The sim is one arm. Two-arm selection comes with the rig.

### Then the real Panthera

| Step | What |
|---|---|
| **R0** | Mount the cameras rigidly and pin which wrist camera is on which arm (`../panthera_teleop` still lists this as unverified). Intrinsics with a ChArUco board (`tools/calibrate_intrinsics.py`). Then one board lying on the table gives the rest (`tools/calibrate_extrinsics.py`): the arm visits poses over it for the wrist camera's hand-eye (OpenCV `calibrateHandEye`), the board's pose in the base frame follows and the centre camera's from it, and the board's plane is the table's. Also: the gripper's motor range and jaw width, and how the centre image is oriented (`config/rig.yaml`). |
| **R1** | Touch test (`tools/touch_test.py`): the centre camera finds a printed marker, the arm puts its TCP 2 cm above the predicted point, measure the miss. Target under 3–5 mm before the wrist refine. |
| **R2** | The skills on the real arms with safety limits: workspace box, z floor, speed caps |
| **R3** | The same perception code on real images, one pick-and-place, then the agent loop |
| **R4** | The test set on the rig; the failures decide where the first ACT skill goes |

The rig's computer is Linux x86_64 (the arms' SDK is a Linux-only wheel). The perception server runs on the GPU VM
because the rig's laptop has no NVIDIA GPU.

## Status (sim)

`tools/run_scripted.py`, seeds 0-19, ground-truth masks (S1) and with the overhead camera's pose off by 5 mm / 1° (S3):

| Run | Success | Notes |
|---|---|---|
| S1 red on green | 20/20 | stack within 0.1-1.6 mm |
| S1 three-block tower | 18/20 | within 0.2-6.2 mm. The 2 misses are refusals: the blue block stands beside the finished tower and every reachable approach would put the open gripper into it, so pick says so and names the tower (before the clearance check, these picks knocked it over). Building elsewhere or moving the blue block first is the LLM's call. |
| S3 red on green | 20/20 | overhead estimates 5-18 mm off; the wrist refine brings each pick to about 0.1 mm; stacks within 0.7-4.3 mm (the overhead correction removes the shift, not the rotation part of the error) |
| S3 three-block tower | 18/20 | within 1.0-6.8 mm; the same two refusals |

Not yet run: S2 (needs the perception server; the VM is off) and S4/S5 (need API credit: about 10-20 turns per task,
each with one or two images). The agent loop is tested with a scripted stand-in for the model.

## Later (not built yet)

- **GraspGen** (NVIDIA, on a point cloud) in place of the rectangle fit, for shapes a top-down rectangle does not
  describe. Its released models cover the Franka and Robotiq parallel grippers. Check its licence (reported to be
  non-commercial research only) and the Panthera jaw width.
- **MoveIt 2** (CPU, ROS 2 Jazzy; runs on the rig laptop now, the humanoid's computer later) in place of IK
  waypoints, for clutter.
- **SAM 3** in the perception server once access to the gated weights is granted.
- **Depth**: a wrist depth camera, or triangulation from two wrist poses.
- **A smaller planner**: the `runs/` logs from successful tasks are supervised fine-tuning data for a small local
  model, then RL with rewards the simulator can check. Worth it for speed, cost and offline use, after the big model
  works.
- **Learned contact skills** (ACT), where the tests show the scripted grasp or place failing. Notes below.

### Training an ACT contact skill (when the tests call for one)

- **What a skill is:** a short episode that starts at the pre-grasp or pre-place pose this pipeline produces and ends
  after release and retract (5–10 s). It is not a whole-task policy.
- **Recording:** start every demonstration from that handoff pose, deliberately 1–2 cm or a few degrees off, and
  correct during the demonstration. Absorbing the pipeline's error is the skill's job. Include a few that misalign and
  then recover.
- **Vary:** object colour and size, target position, angle and lighting. Keep the wrist camera in the inputs; it is
  what sees the studs.
- **How many:** 50 per skill to start (the ACT paper used 50 per task for 64–96 % success), growing to about 100.
  Train on the VM (`../finetune.sh`, `../panthera_sim/train_act.py`), evaluate, then add demonstrations where it
  fails.
- **Skill families:** one ACT per family (place_on_studs, insert, pour), reused for every object in it. pi0.5 only for
  one model across many skills by prompt, or for dexterous hands.
- **How it plugs in:** a new tool (`place_on_studs(object_id)`) that runs this pipeline to the handoff pose, then
  hands control to the policy. The LLM's side does not change.

## Open questions

- The ELP lenses (wide-angle or fisheye) and the centre camera's tilt: assumed wide-angle and roughly straight down
  until checked. The calibration handles either (`config/rig.yaml`, `distortion_model`).
- The rig's arm base frames relative to each other and to the table: measured at R0.
