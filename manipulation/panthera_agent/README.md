# panthera_agent

An LLM agent that does tabletop manipulation with the Panthera arms by calling tools, in the MuJoCo sim
(`../panthera_sim`) and on the real rig (`../panthera_teleop`'s arms and cameras). The model never sends motor
commands. It looks at camera images, decides what to do next, and calls one tool at a time: find an object, pick it,
place it, check the result. Each tool is ordinary geometry and inverse kinematics. Objects it has never seen are
learned on first sight: the LLM chooses numbered points on them, and local DINOv3 features plus two camera views
measure their shape. Contact work the geometry cannot finish (a peg into a hole) goes to a small learned local
policy, which improves with practice: supervised in the sim, labelled by Claude, retrained overnight on the GPU VM.

**Rule of thumb: the LLM chooses, perception models measure, geometry computes, small policies finish.** The LLM
picks the object, the part and the next step, and judges whether a step worked. Detection and segmentation find
pixels. Geometry turns pixels into positions, grasp poses and joint targets. The LLM is never asked for coordinates
it would have to read off pixels: it answers by number. A learned policy takes only the last centimetres of contact.

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

An object it has not seen ("purple mug"): `find` says so, and the LLM learns it.

```
   LLM ──► propose_points()          DINOv3 patch features on this machine, compared with the empty table's photo:
        │                            numbered points on every thing on the table, a region per thing, parts inside
        │                            (propose_points(detail="B"): dense points over one region, to mark a part)
        ├──► learn("purple mug", points=[4, 5], parts=[{handle: 6}])
        │       1. the chosen regions -> the object's exact pixels (what differs from the empty table, shadows not)
        │          -> DINO examples and colours, saved in the seen-objects list (memory/)
        │       2. the wrist camera looks from up to three poses (predicted to show all of it, clear of everything
        │          else); in each, the footprints from above and from the wrist agree at one height only: the median
        │          is its height; outline, size and shape class (round, box, long, irregular, flat) follow; parts
        │          become offsets on the object
        ├──► find("purple mug")         dino_local: similarity to its examples minus the table's, gated by its
        │                               colours (shadow-tolerant), sharpened to pixels; then the same geometry
        ├──► pick(n, part=null|"handle"|"top")   grasps from the outline: across faces, widths that fit, fingers
        │                               clear of the object itself; tall things at mid-height; "top" for inserting;
        │                               flat or soft things pinched straight down; in-hand check by sight
        ├──► place_on / place_at        then measured from the wrist camera: offset, turn, seated or not
        ├──► insert(n)                  the handoff (just above the hole), then a local controller: the trained
        │                               policy (wrist image + pose since the handoff, 10 Hz) or a scripted search
        ├──► pull_with_tool(n)          the tool's learned outline and "hook" part plan where it goes behind n
        └──► pour_into(n)               tip about the horizontal axis the arm reaches furthest, mouth over n
```

So the LLM runs at the start, after each step and on failures, never inside a motion.

## What we agreed, and why

| Decision | Choice | Why |
|---|---|---|
| Planner | Claude through the Anthropic API (`ANTHROPIC_API_KEY` in the repo's `.env`). Default model `claude-opus-5-5`, configurable. | It takes images in tool results, and it's the API we already pay for. The connector is one file (`llm/claude.py`), so another provider is a swap. |
| Tool granularity | Few, semantic tools (`find`, `pick`, `place_on`, `place_at`, `look`, `open_gripper`, `home`, `done`), one call per turn | Published agent harnesses find semantic tools beat low-level ones (88.6 vs 73.3 for a frontier model, 22.9 vs 4.8 for a 4B one), and one call per turn beats batching several. |
| Which object | Numbered mask outlines on the image; the LLM answers by number | Much more reliable than asking a VLM for pixel coordinates. |
| Depth | None. Ray ∩ known plane (table top + object height), then a wrist-camera refine | The rig's ELP cameras are monocular. The table is a known plane and the objects have known sizes, so this is geometry, not an estimate. The wrist refine cancels most calibration error. |
| Footprint | The mask's outline projected onto the top-face plane and onto the base plane; their overlap, fitted with a minimum-area rectangle | Seen at an angle, an object's side faces enlarge its mask; on the two planes they push the outline in opposite directions, so the overlap is the true footprint (centre error from 13 mm to under 0.1 mm in the wrist view). A stacked object's level is the one whose expected size the measurement matches, unless the agent built that stack: then it remembers the height (`World.stacks`), because a real segmenter's mask of a tower covers the sides of every block under the top one and looks a level or more taller. |
| Grasp orientation | Rectangle yaw, folded by the object's symmetry toward the bearing from the arm's base (a radial approach), at the sim collector's 55° pitch | Radial halves the unreachable poses compared with folding toward the forward axis. GraspGen later, for odd shapes: a drop-in, since the step's output is only "a grasp pose". |
| Clearance | Before a pick or place, each equivalent yaw (four for a cube) is checked for reach (IK) and for the open gripper's boxes (`grasp.gripper_boxes`) against every other known object; the most radial one that clears is used, else the tool fails and names what is in the way | The descent is straight down, so the final pose's boxes cover it. Seen in the sim: an unchecked pick next to a tower knocked it over. Refusing and naming the obstacle hands the decision (move it, or work elsewhere) to the LLM. |
| Look pose | Home is high and near the base (0.20, 0, 0.30 m) | From over the table, the arm hid part of a block from the overhead camera in 7 of 36 placements; from here, in none. |
| Motion | IK through safe waypoints (the sim's damped least-squares IK; on the rig the same solver on a MuJoCo copy of the arm kept in sync with the real joints) | Enough for an uncluttered table. MoveIt 2 later, for clutter. Same interface: go to this pose. |
| Grasp | Close the gripper; check the width (and current on the rig). No learned policy. | Picking a block is easy. A learned policy is for when the tests show the scripted grasp failing. |
| Place | Descend to just above seated (`place.gap`), release | May be too coarse for DUPLO studs (about 1–2 mm and a press). If tests show that, the first ACT skill goes there (see "Later"). |
| Calibration error | Each wrist refine of an overhead estimate measures the overhead camera's error at that spot; the median of the last five is added to overhead estimates | A small tilt error in the overhead camera shifts everything it measures by nearly the same amount (1° from 0.85 m is about 15 mm). Places need this, because the held block hides the target from the wrist camera. |
| Segmentation | Pluggable, three sources: MuJoCo ground truth (`sim_truth`); the GPU-VM perception server's Grounding DINO + SAM 2.1 (`dino_sam2`, port 8100); SAM 3 as a second backend of the same server (`sam3`, port 8101). No depth needed | Reuses proven infrastructure: one server, one API, two backends side by side, so the agent can switch per run. Measured in sim (Status): both real segmenters do as well as ground truth once (a) DINO is prompted with the other colours as distractors (alone it returns every cube for "red cube"), (b) the stack level is chosen with a margin and settled by the wrist camera (SAM 3's masks run about 4 % large) |
| Cameras | Rig: ELP USB modules (two wrists, one centre), calibrated. Assumed wide-angle, not fisheye, with the centre camera looking roughly straight down, until we look. Insta360s: recording only. | ELPs are plain UVC webcams with a fixed image model. Action cameras stabilise electronically, which warps every frame differently and breaks calibration. |
| Arm choice (two arms) | A rule: the nearest arm whose IK reaches. Not the LLM. | A geometric fact, not a judgement. |
| Logging | Every LLM call, image, tool call and result per run in `runs/` | For debugging, and as training data for a smaller model later. |
| Any object | `dino_local`: frozen DINOv3 ViT-S/16 (the codex/dino-ik branch's backbone, same pinned revision) on this machine; a seen-objects list on disk; few-shot matching by cosine similarity | The codex/dino-ik branch's idea: cosine similarity to a few marked examples. No server; about 40 ms a frame on the Mac. At tabletop scale DINO tells object from table very well but one small object from another poorly (cosine about 0.9 between a red cube and a teal puck), so identity comes from the object's colours (up to three, shadow-tolerant). |
| Learning a new object | The code proposes numbered points (DINO features against the empty table's photo); the LLM chooses; two views measure the height | Typed pixel coordinates score 7-16 % in pointing benchmarks; choosing numbered candidates is the Set-of-Mark / MOKA pattern. Without depth, the overhead and wrist footprints coincide at the true height only. |
| Grasp choice | From the learned outline: across faces, widths that fit the jaws, fingers clear of the object itself, most radial first; parts by name; "top" for inserting | The cube's rectangle rule, generalised. GraspGen stays the upgrade for shapes a top view does not describe. |
| Checks by sight | In-hand after the lift (looking where the object should be; not seeing it is not seeing it dropped), and the place measured afterwards from the wrist. Lining up over the target from above before lowering exists but is off | Measured in the sim: lining up from above made places worse (the gripper hides the held object); the wrist's place check agrees with the truth within about 1 mm. |
| Contact skills | Geometry to just above contact, then a local policy on the wrist camera and the pose relative to the handoff | ManipGen's and SPHINX's interface: a relative frame and a wrist view carry across positions on the table. |
| Learning on the job | A supervisor (sim: privileged expert; rig: a person, a watchdog, the agent's own resets), DAgger, RLPD-style SAC with interventions, a success classifier; Claude labels episodes but is never the RL reward; a nightly retrain on the VM, promoted only if better on held-out layouts | HIL-SERL's recipe at small scale. VLM judges used as rewards get exploited (research notes). See "Learning" below. |
| Two arms | The nearest arm whose IK reaches (`arms:` in rig.yaml, `select_arm`) | A geometric fact. The sim has one arm until panthera_sim's bimanual scene (codex/dino-ik) is merged. |
| Motion profile | Quintic on the rig, smoothstep in the sim (`motion.profile`) | The quintic (codex/dino-ik) is gentler on the motors; the sim keeps the profile its results were measured with. |

**Sim vs. DUPLO.** The sim has 45 mm cubes (red, green, blue), not DUPLO. The geometry only needs each object's
height and footprint (`config/sim.yaml`, `objects:`), so DUPLO bricks are new entries there plus a scene with bricks.
DUPLO dimensions: 19.2 mm per brick height (body, without studs), studs 16 mm apart, 2×2 footprint 31.8 mm square.

## Layout

```
README.md            this plan
requirements.txt     Python 3.10+: mujoco, numpy, opencv-contrib (ChArUco), pyyaml, anthropic, pytest, torch, timm
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
  build.py           config, .env, segmenter and skills construction for the tools; per-scene config
  types.py           Result, Held
  grasp.py           grasps from an outline and parts (widths, faces, finger clearance, pinching flat things)
  learn.py           learning's pure parts: object pixels, DINO examples, height from two views, shape class
  skill_learn.py     propose_points, learn (survey poses predicted from the wrist camera's view, clear of obstacles)
  skill_checks.py    the in-hand check, lining up over a target, the place check
  skill_contact.py   insert (handoff + local policy or scripted search), pull_with_tool, pour_into
perception/
  base.py            Segmenter interface: (image, query) -> instances (mask, label, score)
  dino.py            frozen DINOv3 ViT-S/16 patch features on this machine (MPS, CUDA or CPU)
  candidates.py      numbered candidate points: regions that differ from the empty table, parts inside, dense detail
  memory.py          the seen-objects list on disk (examples, colours, shape, parts, grasp outcomes)
  dino_local.py      `dino_local`: learned objects by similarity and colour, pixel-sharpened masks
  sim_truth.py       ground-truth masks from MuJoCo's segmentation render (S1)
  server_client.py   the GPU VM's perception servers over an SSH tunnel: dino_sam2 (Grounding DINO + SAM 2.1) and sam3
                     (S2, rig)
  overlay.py         numbered mask outlines, and the table grid in cm, on images for the LLM
robots/
  base.py            Robot interface: observe, ee_pose, move_linear, set_gripper, gripper_width, home
  sim_robot.py       the Panthera MuJoCo sim (../panthera_sim), with S3's camera-pose error option
  real_robot.py      the real arms through the vendor SDK (one, or two with `arms:`), OpenCV cameras, calibration,
                     safety box, quintic moves (not yet run on hardware)
scenes/
  catalog.py         the agent's scenes: odd shapes, a peg and its hole (and a tight one for training), a hook tool
                     and a puck out of reach, a cup of beads and a bowl, a cloth; seeded layouts
  build.py           writes them next to panthera_sim's (../panthera_sim/sim/panthera/agent_*.xml)
learning/            the local skills' learning loop (see "Learning")
  envs.py            InsertEnv (sim: privileged set-up and expert), SkillEnv (any robot, through the skills)
  episodes.py        episodes on disk (data/episodes/<skill>/) with an index: source, outcome, labels
  policy.py          LocalPolicy (ACT-style action chunks), the SAC actor and critics, checkpoints (models/<skill>/)
  train.py           imitation, the success classifier
  practice.py        demos, imitation, DAgger, RL with interventions, evaluation
  supervisor.py      who demonstrates, corrects, judges and resets: the sim's expert, or the agent and a person
  label.py           Claude's labels of episodes (structured output) and their agreement with the recorded outcome
  teleop.py          a person's corrections from the keyboard on the rig
  nightly.py, .sh    the overnight job on the VM; setup_vm.sh makes its environment
  export_lerobot.py  successful episodes as a LeRobot dataset, for ACT with LeRobot's own trainer
llm/
  claude.py          Anthropic API connector: model, effort, one tool call per turn, prompt caching, refusal fallback
  tools.py           the tool schemas (strict) and their dispatch to the skills; results as text + images
  prompts.py         the system prompt
  loop.py            the agent loop (append-only conversation), logging every turn and image; live events for the
                     viewer
  scripted.py        scripted stand-ins for the model (no API): red on green, and sim oracles for the agent's tasks
                     that learn objects too (the points chosen from the sim's segmentation)
tools/
  render_views.py    S0: what each sim camera sees
  run_scripted.py    S1-S3: scripted pick-and-place through the skills, per-stage error against ground truth
  compare_segmenters.py  S2: each segmenter's masks against the true ones (found, wrong, IoU, position error, time)
  tunnel.sh          both VM perception servers on this machine's 8100 and 8101 (VM_HOST from the repo's .env)
  run_agent.py       S4: one instruction through the LLM (sim, or --robot rig)
  viewer.py, .html   S4, watched: the sim and every step of the run in the browser (below)
  evaluate.py        S5: tasks x seeds through the LLM, success from ground truth
  calibrate_intrinsics.py   R0: camera intrinsics from ChArUco views
  calibrate_extrinsics.py   R0: wrist hand-eye, centre camera pose and table plane from one board on the table
  touch_test.py      R1: centre camera -> table point -> TCP above it; measure the miss
  capture_empty_table.py   R0: the overhead camera's photo of the empty table (the points' background)
  rig_check.py       before every session: config, calibration, cameras, arms, kinematics, safety box (moves nothing)
  practice_rig.py    R5: practise a local skill on the rig (SkillEnv, keyboard corrections, the agent's resets)
  learning_sync.sh   episodes and the seen-objects list to the VM, checkpoints and reports back
tests/               pytest (no GPU, API key or rig): geometry against MuJoCo's cameras, segmentation, the skills in
                     the sim (stacking, calibration error, unstacking), calibration math, the agent loop with a
                     scripted stand-in for the model, the API request; the agent's scenes, grasps from shapes,
                     learning objects with local DINO (downloads it once), the learning loop's pieces
runs/                run logs (not kept in the repo)
memory/, data/, models/   the seen-objects list, episodes and checkpoints: per robot, not kept in the repo
```

## Setup

```bash
cd manipulation/panthera_agent
uv venv --python 3.10 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q tests                      # no API key or GPU needed
```

The LLM needs `ANTHROPIC_API_KEY` in the repo's `.env` (already listed in `.env.example`). The tools read it from
there or from the environment. The real-image segmenters need their servers running on the VM
(`services/perception_server/run_server.sh` and `run_sam3_server.sh`, see that README) and the tunnel:
`tools/tunnel.sh` (`PERCEPTION_SERVER_URL` / `SAM3_SERVER_URL` override the addresses;
`.venv/bin/python -m perception.server_client` checks both). Any tool takes
`--segmenter sim_truth|dino_local|dino_sam2|sam3`; the viewer has a Masks menu. `dino_local` needs no server: the
first use downloads DINOv3 ViT-S/16 (about 90 MB) into the Hugging Face cache, and the objects it knows are the ones
learned (`memory/`, or PANTHERA_AGENT_MEMORY). The agent's scenes are generated files:
`.venv/bin/python -m scenes.build` rewrites them after a change to `scenes/catalog.py`. On the rig's computer, install
`requirements.txt` into panthera_teleop's environment (the adapter uses its SDK set-up), photograph the empty table
once (`tools/capture_empty_table.py`), and run `tools/rig_check.py` before each session.

## Watching a run: the viewer

```bash
.venv/bin/python tools/viewer.py                         # opens http://127.0.0.1:8765
```

The left side is the sim (drag to orbit, scroll to zoom; or the overhead and wrist cameras), the right side the
whole stack, step by step, as it happens:

- your instruction, with the system prompt, tool schemas and request options sent to Claude;
- each request's new content (the tool result and images the model is shown), and Claude's reply: summarized
  thinking, text, the tool call, latency, tokens and cost;
- the tool running: each camera image, the segmenter's masks, the measured positions, the approach plan (yaws tried,
  reach, clearance), every straight-line move with its tracking error, wrist-camera corrections, gripper widths;
- the result returned to Claude, the model's `done`, and for a test task the ground-truth check;
- learning: the numbered points, the learned object's pixels, its second view from the wrist, the measured shape; the
  similarity map behind each `dino_local` find; masks dropped for an impossible size;
- grasping and placing: the grasp options drawn on the object's outline, the in-hand image, the place check;
- contact skills: each step of the local controller, the tip angle of a pour.

Under the controls: the **seen objects** (picture, measured shape, parts, grasps tried; forget one there), and
**Practice** for the insertion skill on the training scene: evaluate (the trained policy, the scripted search or the
expert), demonstrations, imitation, DAgger, RL with interventions, with each episode as a square (green: success,
outlined: the supervisor took over) and the success rate as it goes. While practising, hold W/S, A/D (move), R/F
(up, down), Q/E (turn) to take over, as a person does in HIL-SERL. "insert: use the trained policy" makes the agent's
`insert` run models/insert/latest instead of the scripted search.

Pick a test task (or a scene and seed) and Reset; type any instruction and Run (Cmd/Ctrl-Enter). "Pause each turn"
waits before every request and every tool call, "Pause each motion" also before every move and gripper action;
Next steps on, Stop ends the run. The planner "Scripted" needs no API key: red on green, or for the agent's own tasks
(learn_and_stack, mug_left, peg_insert, pour_beads) a sim oracle that learns the objects too, choosing the numbered
points from the simulator's segmentation where the model would look. Runs are logged to `runs/viewer_<time>/`, like
`run_agent.py`'s. The page binds to 127.0.0.1 only.

## Steps

### Sim first

| Step | What | Done when | Run |
|---|---|---|---|
| **S0** | Cameras as on the rig (placement, field of view, resolution); segmentation rendering | Sim and rig images look comparable | `tools/render_views.py` |
| **S1** | The skills with ground-truth masks: ray ∩ plane position, rectangle yaw, move above, refine, descend, grasp, check, place | Scripted pick-and-place works with perfect perception; position error per stage about 0 | `tools/run_scripted.py` |
| **S2** | Real segmentation on rendered images (the perception server) | Detection rate and position error per stage measured | `tools/compare_segmenters.py`, `tools/run_scripted.py --segmenter dino_sam2` (or `sam3`) |
| **S3** | Calibration noise: the overhead camera's estimated pose off by ±5 mm / ±1° | The wrist refine brings the error back under about 2 mm | `tools/run_scripted.py --camera-error 0.005 1.0` |
| **S4** | The LLM loop with the tools | Multi-step tasks run: stack, unstack, sort, make a row | `tools/run_agent.py "stack the red cube on the green cube"` |
| **S5** | A fixed test set: N seeds × tasks, success from the simulator's state, every call logged | A success rate to track; the logs are future training data | `tools/evaluate.py` |
| **S6** | Any object: learned on first sight (numbered points, two views), found with `dino_local`, grasped from its shape, checked by sight | Odd shapes picked and placed from their measured shape, no server | `tools/evaluate.py --planner oracle --fresh-memory --segmenter dino_local --tasks learn_and_stack mug_left` |
| **S7** | Contact and tools: insert, pull_with_tool, pour_into | Each succeeds in its scene | `tools/evaluate.py ... --tasks peg_insert pour_beads hook_puck` |
| **S8** | Learning the insertion's last centimetres: demos, imitation, DAgger, RL with interventions, a success classifier, Claude's labels | The trained policy beats the scripted search on held-out layouts | `python -m learning.practice ...`, the viewer's Practice |
| **S9** | The nightly loop on the VM | A report each morning; `latest` only ever improves | `learning/nightly.sh` (job `panthera_agent_nightly`) |

The sim is one arm. Two-arm selection is built (`select_arm`, nearest arm that reaches) and waits for the bimanual
scene and the rig.

### Then the real Panthera

| Step | What |
|---|---|
| **R0** | Mount the cameras rigidly and pin which wrist camera is on which arm (`../panthera_teleop` still lists this as unverified). Intrinsics with a ChArUco board (`tools/calibrate_intrinsics.py`). Then one board lying on the table gives the rest (`tools/calibrate_extrinsics.py`): the arm visits poses over it for the wrist camera's hand-eye (OpenCV `calibrateHandEye`), the board's pose in the base frame follows and the centre camera's from it, and the board's plane is the table's. Also: the gripper's motor range and jaw width, and how the centre image is oriented (`config/rig.yaml`). |
| **R1** | Touch test (`tools/touch_test.py`): the centre camera finds a printed marker, the arm puts its TCP 2 cm above the predicted point, measure the miss. Target under 3–5 mm before the wrist refine. |
| **R2** | The skills on the real arms with safety limits: workspace box, z floor, speed caps |
| **R3** | The same perception code on real images, one pick-and-place, then the agent loop |
| **R4** | The test set on the rig; the failures decide where the first ACT skill goes |
| **R5** | Objects learned on the rig (`tools/capture_empty_table.py` first), then the local skill practised there: `tools/practice_rig.py` (the insert skill's own handoff, keyboard corrections, the agent's resets), Claude's labels (`python -m learning.label`), `tools/learning_sync.sh push`, the nightly job, `pull` |

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

S4, first run (`stack_red_on_green`, seed 5, claude-opus-5-5 at medium effort, through the viewer): success by
ground truth, red 0.5 mm off green's centre, 6 turns (find, pick, place_on, home, find, done), $0.04 (the system
prompt and tools are cached after turn 1; each turn adds one image).

S2, the segmenters on rendered images (`tools/compare_segmenters.py --seeds 5`: overhead view, and the wrist view
over the red cube; found / wrong detections, top-face position error from the mask against the true mask's, time per
request on the VM):

| Segmenter | Colour query, overhead | "cube", overhead | Wrist view | Position error | Time on the VM |
|---|---|---|---|---|---|
| DINO + SAM 2.1 | 15/15 found, 0 wrong (with the other colours as distractors; about 30 wrong without) | 15/15, 0 wrong | 9/9 colour queries found, 2 wrong | median 0.1-0.2 mm, max 4.8 mm (wrist, green) | ~0.9 s for a colour query, 0.16 s otherwise |
| SAM 3 | 15/15 found, 0 wrong | 15/15, 0 wrong | 7/9 found, 0 wrong (misses: cubes cut by the image edge) | median 0.1-0.4 mm, max 1.4 mm | 0.06 s |

Then pick-and-place through the skills (`run_scripted.py`, seeds 0-9) with each:

| Segmenter | Red on green | Three-block tower |
|---|---|---|
| MuJoCo ground truth | 10/10 | 9/10 |
| DINO + SAM 2.1 | 10/10 | 8/10 |
| SAM 3 | 10/10 | 9/10 |

Every miss is the gripper-clearance refusal described above. Getting there took three fixes found on the way: the
sim's cameras drew the TCP marker (a red ball) that both models took for a red object; SAM 3 put cubes on the table
"2 high" (its masks are a little large, and from the overhead camera one level is only a 6 % size change), now a
higher level must fit 3 % better and the wrist camera re-decides it; and `home` failed silently far out over the
table (the lift to the start height was out of reach), leaving the arm in the overhead camera's view.

S4, the agent on the three-block tower (seed 2), the three segmenters in parallel: all three succeed by ground truth.
Ground truth: 10 turns, $0.05, spread 0.7 mm. DINO + SAM 2.1: 20 turns, $0.10, 5.8 mm. SAM 3: 17 turns, $0.09,
19 mm (inside the task's tolerance). With real masks the third cube stood too close to the first pair; pick refused
and named it, and Claude rearranged (DINO run: moved a cube aside and built elsewhere; SAM 3 run: stacked onto the
stranded cube instead).

Stack heights with real masks. In a viewer run (Claude, three-block tower, SAM 3 and then DINO) `find` reported a
2-high stack as "3 high" and kept saying 3 after the third block went on. DINO + SAM 2.1 and SAM 3 return a tower as
one mask (the top face and the visible sides of every block below), which looks taller; MuJoCo's per-body masks never
do that, so S1-S3 missed it. Now the skills remember the stacks they built and use that height ("stacked 2 high (a
stack I built)"), measuring a tower mask down to the table (0.3 mm from the true centre instead of 2-5 mm); a top
that clearly measures lower than what was built is reported as fallen. Built towers over seeds 0-7: the height after
each place is exact with both real segmenters (SAM 3 once found no "cube" on a finished tower). The agent on seed 2:
DINO + SAM 2.1 succeeds (19 turns, 0.3 mm); SAM 3 fails, on a different problem: for the plain query "cube" it also
returns the whole table at score 0.38 (above the 0.35 threshold), which `find` keeps as a 538 mm object that blocks
every later pick and place as an obstacle.

Not yet run: the other S4 tasks, and S5 with Claude.

### Any object (S6, S7), measured

All with `dino_local` (no server), each run starting with nothing learned, the numbered points chosen by the sim
oracle (the simulator's segmentation, standing in for Claude's choice; with Claude it costs API calls and was not
run at scale here).

- **Learning on first sight**, `agent_objects` and `agent_peg`, measured against the true shapes: cube 45 x 45 x 45 mm
  measured 45 x 45 x 45; can (44 mm, 75 mm tall) 43 x 43 x 76, round; long box 75 x 34 x 30 measured 75 x 34 x 31,
  long; mug (60 mm, 65 mm tall, handle) 60 x 60 x 66; puck 56 x 25 measured 56 x 55 x 26; peg 24 x 24 x 60 exact;
  holed fixture 75 x 75 x 35 measured 75 x 75 x 35. Heights are the median of up to three wrist views. Finding the
  learned objects again in new layouts: 25/25 found, mask IoU 0.75-1.0, centre 0-6 mm off.
- **Pick and place of the odd shapes** (S6, from the measured shape, six layouts): cube, can, long box and puck 24/24,
  placed within 0.2-6 mm; the mug with its handle 3/6.
- **The agent's tasks** (`tools/evaluate.py --planner oracle --fresh-memory --segmenter dino_local`, seeds 300-305,
  the final code): learn_and_stack (two unseen objects, puck on cube) 5/6, within 0.5-0.8 mm; peg_insert 5/6;
  pour_beads 5/6 (all 12 beads in the bowl when it works); mug_left 2/6. 17/24 in all. Earlier code on seeds 100-105
  and 200-205: 17/24 and 14/24.
- **Insertion with 3 mm clearance** (the scripted search after the handoff): 8/8 when the peg is in hand.
- **Not working yet**: the hook drags past the puck without catching it (a long tool gripped at its end turns in
  the jaws; the planning from its outline and hook part is in place); the cloth is learned (200 x 199 mm, its corner
  marked) but pinching it from the table does not catch it in the sim.

### Learning the insertion (S8, S9), measured

`agent_peg_tight` (1 mm clearance a side), the handoff deliberately off by up to 6 mm and 8 deg, success from the
simulator. Training on layouts 0-39, testing on 100-119, never seen:

| | Success |
|---|---|
| scripted search (press, spiral) | 0-4 % |
| privileged expert (the sim's supervisor) | 80-85 % |
| imitation on 40 expert demos | 0 % held out |
| + DAgger, 3 rounds (115 episodes) | 25 % on training layouts, 10 % held out |
| + DAgger on 40 layouts (340 episodes, 7 min of training on the Mac's GPU) | 37.5 % held out |
| + RL with interventions (RLPD-style SAC, 50 episodes) | worse (30 % then 20 %); stopped. Kept the imitation policy |

The success classifier (single wrist frames, 11,361 frames, 306 positive): 98.5 % held-out accuracy, 1.6 % false
positives. The nightly job ran end to end locally (inventory, classifier, candidate, held-out evaluation, promotion,
report). Claude's labels: the request works (structured output), but from 64-pixel wrist frames it could not see an
insertion and said so (low confidence on every one); episodes now keep full-resolution final images for labelling,
not yet measured.

## Learning

The local skills improve by practice, in two loops (research notes, "learning on the job"):

1. **During practice (minutes to hours)**: `learning/practice.py` on the sim (`InsertEnv`, privileged supervisor) or the
   rig (`tools/practice_rig.py`, `SkillEnv`): demonstrations, imitation, DAgger (the supervisor labels what should have
   happened), and RL with interventions (half of each batch from demonstrations and interventions, half the policy's
   own; the critic warms up first and the actor keeps an imitation term). The supervisor (`learning/supervisor.py`)
   fills HIL-SERL's human's four jobs: demonstrate, correct, judge, reset. In the sim a privileged expert does all
   four; on the rig a person's keyboard corrections (`learning/teleop.py`), a stall watchdog, the geometric success
   check and the agent's own resets (an instruction to Claude). Every attempt is an episode with full-resolution final
   images.
2. **Overnight (the GPU VM)**: `tools/learning_sync.sh push`, then the `panthera_agent_nightly` job
   (`learning/nightly.sh`, set up once with `learning/setup_vm.sh`): the success classifier retrained, a candidate
   policy trained on everything, both candidate and current evaluated in the sim on held-out layouts, the candidate
   promoted only if better; a report. `tools/learning_sync.sh pull` brings `models/` and the report back. With
   `--export-lerobot`, the successful episodes are also exported for an ACT fine-tune with LeRobot's own trainer
   (profile `../profiles/panthera_local.yaml`; `../finetune.sh local panthera_local` trains the small policy).

Claude labels episodes and filters training data (`python -m learning.label`); it is never the RL reward. The reward is
the simulator, the geometric check or the trained classifier.

## Later (not built yet)

- **GraspGen** (NVIDIA, on a point cloud) in place of the rectangle fit, for shapes a top-down rectangle does not
  describe. Its released models cover the Franka and Robotiq parallel grippers. Check its licence (reported to be
  non-commercial research only) and the Panthera jaw width.
- **MoveIt 2** (CPU, ROS 2 Jazzy; runs on the rig laptop now, the humanoid's computer later) in place of IK
  waypoints, for clutter.
- **Depth**: a wrist depth camera, or triangulation from two wrist poses.
- **A smaller planner**: the `runs/` logs from successful tasks are supervised fine-tuning data for a small local
  model, then RL with rewards the simulator can check. Worth it for speed, cost and offline use, after the big model
  works.
- **More local skills**: the insert skill's pattern (handoff, local policy, practice) for place-on-studs, the hook's
  drag and the cloth pinch, where the measurements above show the geometry failing.
- **Point tracking** between frames (CoTracker-style) for symmetric parts the per-frame match confuses.

### Training an ACT contact skill on the rig (notes kept from the plan)

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
- Known weaknesses of learning on first sight, all measured in the sim: objects whose colour is close to the
  table's (a white cup on the white table was measured 11 mm tall instead of 70) or grey ones whose shadows look like
  them; the hollow fixture's wrist view when its sides are lit differently; parts marked with the coarse points
  (use `detail`).
- On the rig, `insert` judges depth from the arm's pose: an object that slides up in the jaws reads as inserted. The
  success classifier and Claude's labels are the cross-check; a force or current check would be better.
- The RL stage made the insertion policy worse in the runs here (an untrained critic, interventions alternating with
  the policy); the imitation and DAgger stages are what improved it.
