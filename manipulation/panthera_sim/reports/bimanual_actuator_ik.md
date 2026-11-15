# Bimanual kept-session-inspired assembly IK

Reviewed on September 29, 2026 (America/Chicago).

The new simulation has two physical Panthera arms and three cameras. The
right wrist holds the carrier at a 45° lean, while the left arm partially
inserts `gear_3`, pushes it into place, inserts `pin_2`, and presses it down.
The controller has completed supported assembly from a direct tabletop start.
Pin engagement still varies with the randomized start; this is
not yet a robust autonomous assembly policy.

## Evidence from the kept recordings

Source: `/home/zeb/Desktop/Panthera-Teleop/data/kept`.
All 30 kept sessions' metadata, state logs and frame indices were inspected.
Raw duration median was 17.08 s, with a 13.38–44.51 s range. Visual sequences
from both wrists and overhead were reviewed for:

- `20260930T015415-0dadf220`
- `20260930T020311-ee8fe583`
- `20260930T021650-683688b1`

The observed sequence was carrier support in one gripper, gear pickup and
partial placement with the other, repositioning/pushing, then pin insertion
and pressing. The source processing configuration deliberately crops tails
before terminal separation/fall. Accordingly, simulation success is checked
at a final supported hold, with the carrier gripper still closed.

Real camera intrinsics/extrinsics and wrist-to-arm assignments are marked
uncalibrated/unverified in the recording metadata. The synthetic base layout,
45° lean and camera poses are approximations, not fitted calibration. Real
recording names remain `overhead`, `wrist_port2`, and `wrist_port3`; simulated
streams are `overhead`, `left_wrist`, and `right_wrist`.

## Implementation

- `sim/bimanual_actuator_env.py` builds a separate, portable scene containing
  12 arm joints, two gripper actuators, and exactly three cameras. Each wrist
  camera is attached to its corresponding link6. Solid arm visuals preserve
  camera occlusion. Both arms share one contact simulation, while IK scratch
  states and control addresses are independent.
- `tools/bimanual_actuator_ik.py` uses quintic Cartesian transitions, quaternion
  interpolation, warm-started damped IK, and bounds on the actual joint
  commands: 0.6 rad/s velocity, 2 rad/s² acceleration. Ordinary IK waypoint
  tolerance is 0.15 mm; small commanded placement offsets and contact stages
  distinguish placement from seating.
- Gear placement stops 4 mm short, retaining a shallow rim pinch for its
  first physical push. After release, closed fingers push the exposed rim;
  a gentle downward rim press precedes pin placement. A premature release
  4 mm short caused the gear to tip into the CAD's axial clearance and jam.
- Pushes slow axial advancement with measured contact load, perform small
  lateral searches, and reject loads above 35 N. The soft load limit is 20 N.
  Gear search radius is 0.3 mm; settling/pin press radius is 0.04 mm. Pushes
  do not continually retarget from measured object positions.
- Carrier, gear, and pin starts vary by ±2 mm and ±2°. Initial gear placement
  additionally samples a lateral offset within ±0.35 mm. Privileged object
  poses are used for pickup/in-hand transforms and the existing guided pin
  placement helper. The final presses use physical contacts.
- Only the small carrier is fixtured. Manipulated parts move through dynamics;
  there are no grasp welds, object teleportations during rollout, or seating
  snaps. The final check requires gear seating, full pin depth and engagement
  through both bores, and carrier pose retention.

The bimanual scene removes all gear pickup stands. Loose gears spawn directly
on the table at its surface height; the rim pickup uses the existing gripper
without extra support hardware. Blue rods are the loose pins. The legacy
single-arm scene retains its historical stands for recording compatibility.

## Validation

All 35 tests passed with:

```bash
OPENBLAS_NUM_THREADS=1 .venv-actuator-act/bin/python -m pytest \
  tests/test_bimanual_actuator.py \
  tests/test_actuator_assembly.py -q
```

Tests cover independent IK without live-state mutation, residual accuracy,
deterministic dual-arm resets, both arms' command interpolation, the 14-value
state/action schema, camera attachment/count, rack-free tabletop starts,
joint speed/acceleration bounds,
blocked pushes, contact overloads, and shallow pin-entry checks, plus existing
actuator regressions. A pin tip within 0.25 mm of the upper bore mouth can
proceed to the physical press without spanning the full carrier thickness
first. The final full-depth/both-bore check remains unchanged.

The final stand-free rollout measurements are saved under
`outputs/bimanual_ik/stand_free_0` and
`outputs/bimanual_ik/stand_free_press_1`. The seed-0 run completed the full
assembly: final gear XY error was 0.247 mm, pin depth error was 0.219 mm, and
both bores remained engaged. Its first push reduced the 4.023 mm partial
placement error to 0.166 mm after release. Carrier bore deviation was 0.079 mm
and gear bore deviation was 0.096 mm.

| Stand-free seed | Final gear XY error | Pin depth error | Both bores engaged | Full success |
| --- | ---: | ---: | --- | --- |
| 0 | 0.247 mm | 0.219 mm | yes | yes |
| 1 | 0.292 mm | 14.750 mm | no | no |

The seed-1 pin reached the upper bore mouth and proceeded to the physical
press, but remained short of full seating after two attempts. The failed run
is preserved. Joint command speed stayed below 0.227 rad/s and acceleration
below 1.157 rad/s² across these two trials.

These are selected seed trials, not a statistical success-rate estimate.
The larger-offset experiments demonstrate that gear seating can succeed
without reliable pin seating. A nominal-target seating variant and finer
free-motion IK tolerances did not improve this and were reverted. Failed
trials remain recorded and return a nonzero exit status. The final supported
assembly definition is unchanged; a near pin placement is never counted as
full insertion.

## Run and inspect

```bash
OPENBLAS_NUM_THREADS=1 .venv-assembly/bin/python tools/bimanual_actuator_ik.py \
  --seed 0 --output outputs/bimanual_ik/seed_0
OPENBLAS_NUM_THREADS=1 .venv-assembly/bin/python tools/replay_bimanual_actuator.py \
  outputs/bimanual_ik/seed_0 --stride 4
```

`--holder-angle-deg` selects the wrist lean (0–45°; default 45°).
`--xy-jitter`, `--yaw-jitter-deg`, and `--placement-error-mm` control variation.
`--video` records three synchronized camera views during the run. The replay
tool renders them from saved physics without resimulating the rollout.

Output is `result.json`, `stages.json`, and `rollout.npz`. The archive contains
20 Hz measured state and commanded action arrays in left-joints/gripper,
right-joints/gripper order, full qpos/ctrl, contact loads, tracking residuals,
stage IDs, and fixture transforms. Simulated grippers use per-finger opening
in metres; real recordings use motor radians and measured next-pose actions.
Existing single-arm assets, exports and checkpoints retain their original
contracts. This code does not control the physical robots.

Local reviewed artifacts:

- `outputs/bimanual_ik/kept_session_audit.json`
- `outputs/bimanual_ik/stand_free_0/bimanual.mp4` (4× playback, three camera panels)
- `outputs/bimanual_ik/holder_angle.png`
- `outputs/bimanual_ik/stand_free_0/result.json`
- `outputs/bimanual_ik/stand_free_press_1/result.json`
