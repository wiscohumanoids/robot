# Actuator assembly experiment — 2026-09-28

The contact-only IK controller picks up, inserts, releases, and seats the
large carrier and **all four planet gears**. Three complete randomized gear-only trials
(seeds 0, 1, 2) passed, with final gear XY errors below 0.16 mm. This is simulated
placement without pins. The optional pin workflow below extends the assembly;
neither workflow validates an operating transmission.

## Pin assembly extension

`--with-pins` selects a separate `scene_pins.xml` and places the carrier, four
gears, and four pins. It preserves the gear-only scene and its existing replays.
The four original CAD pins are free 8 mm diameter × 17.88 mm cylinders, initially
upright at X=0.24 m and Y=-0.14/-0.06/+0.06/+0.14 m. They receive the same ±2 mm /
±2° reset variation as the other parts. Source pin numbering is not gear
numbering: pin 3 fits gear 1, pin 4 fits gear 2, pin 2 fits gear 3, and pin 1 fits
gear 4.

Insertion uses the actual seated carrier pose. Its approximately 0.3 mm seating
offset is larger than the hole clearance, so the nominal fixture target alone
is insufficient. After engaging the upper carrier, the controller aligns the
held pin with the released gear's measured bore and lowers it further. A pin
that reaches the tilted gear face is still guided by the upper carrier. The
gripper releases and presses the pin head with closed fingertips, offset 5 mm
outward from its measured head center. This clears the central hub while
covering the full head; a larger offset could lever the pin out of the bore.
The controller permits one additional press attempt and rechecks the result
after retreat and settling.

The exported pins have no heads, clips, or interference fit to set their axial
position. Early tests showed that a released pin could slide below its CAD
height. The pin scene therefore adds four physical 3 mm radius backing posts
to the stationary fixture, with their tops at the CAD pin-bottom height. They
collide with the actual part pieces, not the convex support hulls that would
close holes. No pin welds, capture constraints, pose snapping, modified CAD
parts, or increased actuator limits are used. This is an assembly supported by
the fixture; retention after removing that fixture is not established.

Final pin checks require the shaft to span the entire gear bore and both faces
of the upper carrier plate. Centerline error is evaluated at both ends of each
bore: at most 0.15 mm for the gear and 0.17 mm for the carrier, including contact
compliance. Axial error must be below 0.75 mm and tilt below 3°. All gears, pins,
and the carrier are checked again after the final insertion. A pin resting on
a face or partially entering a bore is not counted as installed.

The final 5 mm press controller passed two complete randomized trials from
reset, including carrier placement, all four gears, and all four pins. Each
pin seated on its first press; no retry was required in these trials.

| Seed | Pins installed | Worst pin axial error | Worst pin tilt | Final carrier position error |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4/4 | 0.227 mm | 0.595° | 0.287 mm |
| 1 | 4/4 | 0.225 mm | 0.523° | 0.209 mm |

Across these final states, the maximum carrier-bore centerline error is
0.118 mm and the maximum gear-bore error is 0.098 mm. Every gear remains within
the original placement criteria; worst gear XY errors are 0.294 mm and 0.144 mm
respectively. Seed 0 runs for 318.95 simulated seconds, with the largest
per-frame IK position residual below 0.001 mm. These are two smoke tests, not
a statistical reliability estimate. An additional ten-second hold on an earlier
successful development trial also preserved all pin and gear criteria.

- [Final controller, seed 0](../outputs/actuator/pins_verified_0/result.json)
- [Final controller, seed 1](../outputs/actuator/pins_verified_1/result.json)
- [Recorded-state verification](../outputs/actuator/pins_verified_0/final_verification.json)
- [Pin-insertion replay](../outputs/actuator/pins_verified_0/pin_insertion.mp4)
- [Assembled pins image](../outputs/actuator/pins_verified_0/assembled_pins.png)

Validation: **28 tests pass** (17 assembly/IK tests and 11 existing gripper/control
tests). New checks cover optional-scene reset compatibility, CAD pin identity,
physical depth-stop contact, clear insertion and pressing paths, actual carrier
hole transforms, and rejection of shallow or laterally misaligned pins. Python
compilation and diff whitespace checks also pass.

```bash
OPENBLAS_NUM_THREADS=1 .venv-assembly/bin/python tools/try_actuator_assembly.py \
  --trial-only --with-pins --seed 0 --output outputs/actuator/pinned_assembly
.venv-assembly/bin/python tools/replay_actuator.py outputs/actuator/pinned_assembly \
  --start-stage pin_3_approach --video outputs/actuator/pin_insertion.mp4
```

## Working IK controller

Two separate problems caused the original failure. The 10° side approach
reached the wrist joint limit, and uncorrected position servos sagged roughly
0.6–0.9 mm under load. Although the plate cavity is wider, the internal gear
features leave only about 0.1 mm of vertical clearance near the final radial
insertion. Thus an accurate kinematic endpoint alone did not produce an
accurate physical path.

The controller now uses a vertical tool with an 18 mm outer-rim grasp, choosing
the symmetric jaw roll that fits the wrist limits. It measures the actual
in-hand transform after lifting and inserts radially over 30 mm. A bounded
±0.015 rad integral correction on the arm position-servo setpoints removes
static gravity/friction error. The original arm, fingers, joint limits, servo
gains, force limits, and contact geometry are unchanged. There are no grasp
welds, object pose commands, or placement snaps during a trial.

The fixture moved from X=0.40 to X=0.36 m. Gear pickup stations are now
(0.31, 0.09), (0.31, 0.18), (0.36, 0.12), and (0.36, -0.14) m so lifts and
transfers are reachable as well as insertion endpoints. The original ±2 mm
XY / ±2° yaw randomization remains enabled.

IK residuals now always describe the returned joint solution, including the
last solver iteration and rate limiting. Physical trials log those residuals
and actual tool tracking errors, reject unresolved stage endpoints, and exit
nonzero on failure. Overall success rechecks all placed gears and the carrier
after the final release and retreat.

| Seed | Gears seated | Worst final gear XY error | Worst gear tilt | Final carrier position error |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4/4 | 0.146 mm | 1.822° | 0.339 mm |
| 1 | 4/4 | 0.147 mm | 1.823° | 0.335 mm |
| 2 | 4/4 | 0.151 mm | 1.821° | 0.329 mm |

Seed 0 takes 152.05 simulated seconds. Its largest per-frame IK position
residual is 0.0010 mm; the largest commanded joint step is 0.0562 rad. Physical
tracking has transient error during motion; pre-release gear position errors
are 0.033–0.047 mm. After release the unpinned gears settle about 0.27 mm and
tilt about 1.8°, within the unchanged placement thresholds. Three trials are
a smoke test, not a statistical reliability estimate.

```bash
OPENBLAS_NUM_THREADS=1 .venv-assembly/bin/python tools/try_actuator_assembly.py \
  --trial-only --all-gears --seed 0 --output outputs/actuator/assembly
.venv-assembly/bin/python tools/replay_actuator.py outputs/actuator/assembly
```

- [Complete assembly replay](../outputs/actuator/ik_layout_0/assembly.mp4)
- [Final assembly image](../outputs/actuator/ik_layout_0/assembled.png)
- [Seed 0 result](../outputs/actuator/ik_layout_0/result.json) and
  [independent final-state verification](../outputs/actuator/ik_layout_0/final_verification.json)
- [Seed 1 result](../outputs/actuator/ik_full_1/result.json)
- [Seed 2 result](../outputs/actuator/ik_full_2/result.json)
- [Updated CAD and grasp audit](../outputs/actuator/ik_audit/audit.json)

Validation: 11 assembly tests and 11 existing gripper/control tests pass.
The new regressions verify returned IK residuals without live-state mutation
and collision-free, reachable radial paths for all four gears across three
fixture seeds. The updated audit also finds all four 18 mm / 90° grasp endpoints
reachable and clear. The earlier experiment below records the superseded
baseline and explains the remaining model limitations.

## Source and scene

- Source: [Actuator / Wisco_Actuator_Gear_Motion in Onshape](https://cad.onshape.com/documents/6ebda61ccb55656f56193289/w/a3ae93aac555a13ff6d266af/e/b429d4c9da58a01075feb849).
- Download: `sim/actuator/source/Wisco_Actuator_Gear_Motion.zip`, exported directly
  through Onshape, Fine OBJ, metres, Z up. Part identities and assembled
  transforms are preserved in `sim/actuator/manifest.json`.
- Scene: `sim/actuator/scene.xml`; API: `sim/actuator_env.py`.
- Small carrier: held by a stationary fixture. Large carrier and four gears:
  free bodies, with independent ±2 mm XY / ±2° yaw reset variation.
- Gear racks lift the loose gears 15 mm above the table for finger clearance.
- Stator and rotor are excluded. Pin meshes are available, but pins are not
  installed or manipulated in this initial task.
- Original Panthera joint limits, gripper geometry, servo gains, and force
  limits are retained. Trials use frictional contacts, without grasp welds,
  object teleportation, or final-pose snapping.

MuJoCo normally collides against a mesh's convex hull, which would fill the
assembly openings. The conversion therefore uses separate convex section
pieces for finger/part and part/part contacts. This follows the collision model
described in the [MuJoCo documentation](https://mujoco.readthedocs.io/en/latest/computation/).
Separate support hulls are used only for the table and racks, with collision
masks preventing them from closing the assembly openings. Collision-piece
volumes differ from exported visual-mesh volumes by less than 0.2%.

Masses assume steel; inertias use bounding-box approximations. These properties,
friction, mesh simplification, manufacturing clearances, and the rigid small
carrier unit need calibration before using the environment for hardware transfer.

## Earlier baseline physical results (superseded)

Carrier-only placement, assessed **after release, retreat, and two seconds of
settling**:

| Seed | Lift | Final position error | Final XY error | Result |
| --- | ---: | ---: | ---: | --- |
| 0 | 54.66 mm | 0.331 mm | 0.264 mm | Seated |
| 1 | 54.66 mm | 0.325 mm | 0.257 mm | Seated |
| 2 | 54.66 mm | 0.312 mm | 0.239 mm | Seated |

The acceptance threshold is 1.5 mm position error and 4° orientation error.
This is a small three-seed smoke test, not a statistically established success
rate. Seed 0 is recorded in `outputs/actuator/full_trial/result.json`; seeds 1
and 2 are under `outputs/actuator/seed_1/` and `seed_2/`.

For the near-side gear (`gear_3`), the controller pinches its outer rim at a
9 mm offset with a 10° downward tool pitch. It successfully lifts the gear
about 50 mm and approaches the side slot. Measuring and correcting the
in-hand tilt improves placement, but release remains unsuccessful:

| Corrected-grasp trial | Before-release position error | Released XY error | Released tilt | Result |
| --- | ---: | ---: | ---: | --- |
| Seed 0 | 2.561 mm | 4.090 mm | 8.945° | Not seated |
| Seed 1 | 2.627 mm | 3.074 mm | 10.440° | Not seated |

The gear settles approximately 1.4–1.5 mm below its CAD target. Its CAD pose has
2.24 mm of empty space below it, so some axial settling is expected without
pins; the lateral and angular errors still fail the placement criteria. The
insertion also displaces the large carrier: its final position error becomes
1.884 mm / 1.669 mm respectively. Thus the earlier carrier success does not
mean that the partial assembly remains correctly seated after the gear attempt.

The earlier uncorrected seed-0 gear attempt ended with 6.732 mm XY error and
9.860° tilt. Logs for the revised controller are in
`outputs/actuator/corrected_grasp_0/` and `corrected_grasp_1/`.

## Earlier geometry and IK findings

- The carrier has a clear straight vertical insertion path in the nominal CAD
  alignment, sampled every 0.5 mm over 45 mm of travel.
- All four gears have clear radial paths between the carrier plates. Vertical
  drop-in paths collide; the plate separation is 11.33 mm and gear thickness
  is 8.2 mm.
- A centered horizontal gripper grasp collides with the carrier. Shallow outer
  rim grasps provide useful tool clearance.
- Searching pitches 0°, 10°, 20°, 35°, 55°, 75°, and 90°, offsets 0/9/12 mm,
  symmetric gripper rolls, and multiple IK starts found reachable tool-clear
  endpoints for `gear_3` and one candidate for `gear_2`. It did not find such
  endpoints for `gear_1` or `gear_4` in that search. This is not a proof that
  those gears are unreachable.
- These candidate checks do not validate a stable grasp or a full collision-free
  robot trajectory. The physical gear trial demonstrates the remaining gap.

The next useful experiments are a carrier restraint during gear insertion,
contact-aware alignment/force limiting, and pin insertion to retain placed gears.
Reorienting the fixture between slots could give the arm the same near-side
approach for each gear. These changes have not been applied to the current scene.

## Run and inspect

Setup and full commands are in [the environment README](../sim/actuator/README.md).

```bash
# Open the environment.
.venv-assembly/bin/python -m sim.actuator_env --seed 0

# Reproduce the revised contact-only carrier + near-side gear attempt.
OPENBLAS_NUM_THREADS=1 .venv-assembly/bin/python tools/try_actuator_assembly.py \
  --trial-only --try-gear --seed 0 --output outputs/actuator/reproduction

# Re-run the CAD-path, IK, and gripper-clearance audit.
OPENBLAS_NUM_THREADS=1 .venv-assembly/bin/python tools/try_actuator_assembly.py \
  --audit-only --output outputs/actuator/final_audit

# Render a paired overview and assembly close-up.
OPENBLAS_NUM_THREADS=1 .venv-assembly/bin/python tools/replay_actuator.py \
  outputs/actuator/corrected_grasp_0 --video outputs/actuator/assembly_attempt.mp4
```

Artifacts:

- [Contact-only attempt video](../outputs/actuator/assembly_attempt.mp4)
- [Final audit JSON](../outputs/actuator/final_audit/audit.json)
- [Revised physical trial, seed 0](../outputs/actuator/corrected_grasp_0/result.json)
- [Revised physical trial, seed 1](../outputs/actuator/corrected_grasp_1/result.json)

Validation: five focused tests cover deterministic bounded resets, correct CAD
dimensions, nominal assembled clearance, blocked vertical versus clear radial
gear insertion, velocity reset, and collision-mask separation. The new Python
modules compile successfully. Existing unrelated workspace edits were preserved.
