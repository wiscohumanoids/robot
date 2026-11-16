# Actuator carrier / planet assembly

Real meshes exported from [Actuator / Wisco_Actuator_Gear_Motion](https://cad.onshape.com/documents/6ebda61ccb55656f56193289/w/a3ae93aac555a13ff6d266af/e/b429d4c9da58a01075feb849)
on 2026-09-28, OBJ, Fine, Z-up, metres. The original download, OBJ, and material
file are under `source/`; `manifest.json` records the hash, names, assembled
origins, bounds, and mesh volume. The stator and rotor are excluded from the scene.

The small carrier is held by a fixture. The large carrier and four gears have
free joints. Every reset places them at repeatable stations with independent
uniform XY variation of ±2 mm and yaw variation of ±2 degrees. The fixture has
the same small pose variation, and assembly targets follow its actual pose.
Gears start on 15 mm racks to give the finger tips clearance above the table.
The default scene contains the carriers and gears. `--with-pins` selects
`scene_pins.xml`, adding the four exported pins as free bodies and physical
insertion-depth stops in the fixture. Pin pickup stations are at X=0.24 m,
Y=-0.14/-0.06/+0.06/+0.14 m, with the same reset variation.
The fixture is at X=0.36 m. Loose gears occupy reachable pickup stations at
(0.31, 0.09), (0.31, 0.18), (0.36, 0.12), and (0.36, -0.14) m. This layout
leaves room for the vertical rim grasp and its lift within the wrist limits.

## Run

From the repository root:

```bash
uv venv .venv-assembly --python 3.10 --system-site-packages
uv pip install --python .venv-assembly/bin/python -r requirements-actuator.txt

# Open the dynamic scene in the MuJoCo viewer.
.venv-assembly/bin/python -m sim.actuator_env --seed 0

# Carrier pickup, insertion, release, and all four gears.
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 .venv-assembly/bin/python \
  tools/try_actuator_assembly.py --trial-only --all-gears --video \
  --output outputs/actuator/full_trial

# Complete carrier + four gears + four pins.
OPENBLAS_NUM_THREADS=1 .venv-assembly/bin/python \
  tools/try_actuator_assembly.py --trial-only --with-pins --seed 0 \
  --output outputs/actuator/pinned_assembly
.venv-assembly/bin/python tools/replay_actuator.py outputs/actuator/pinned_assembly

# CAD insertion paths, multistart arm IK, and ideal gripper clearance.
OPENBLAS_NUM_THREADS=1 .venv-assembly/bin/python \
  tools/try_actuator_assembly.py --audit-only

.venv-assembly/bin/python -m pytest tests/test_actuator_assembly.py -q

# Replay the recorded state, or make a paired overview/close-up video.
.venv-assembly/bin/python tools/replay_actuator.py outputs/actuator/full_trial
.venv-assembly/bin/python tools/replay_actuator.py outputs/actuator/full_trial \
  --video outputs/actuator/assembly_attempt.mp4
```

The viewer runs physics with the arm holding its initial posture; it is not a
teleoperation UI. Use `ActuatorSim.set_arm_ctrl`, `set_gripper`, and `step` for
control, or `ik` for Cartesian targets. `reset(rng=...)` accepts an external
generator; `reset(randomize=False)` restores exact nominal poses. Scene rebuilding
is explicit and does not overwrite files each time an environment starts.

Rebuild assets and scene:

```bash
.venv-assembly/bin/python tools/prepare_actuator.py
.venv-assembly/bin/python -m sim.actuator_env --build
.venv-assembly/bin/python -m sim.actuator_env --build --with-pins
```

## Collision geometry and limitations

Visual meshes preserve the Onshape export. A single convex hull would close
the holes and slots needed for insertion. Assembly and finger collisions instead
use convex prisms made from horizontal CAD cross-sections, with 15 micrometre
outline simplification. This approximates curved axial features by slabs;
it is not an exact CAD contact solver. Convex-piece volume differs from the
visual mesh volume by less than 0.2% for the selected parts. Small features and
real manufacturing tolerances still need calibration before hardware transfer.

The table and racks use a separate convex support hull per part to avoid hundreds
of redundant coplanar contacts. Collision masks prevent those hulls from
interacting with other parts or with the gripper. Exact section pieces continue
to govern part-to-part and gripper-to-part contact. Collision groups are hidden
in the viewer/rendering; group 2 shows the original CAD surfaces.

Mass uses CAD volume and assumed steel density (7,800 kg/m³), with box-approximate
inertia, rather than measured material properties. The small-carrier internal
gears are one rigid imported unit. The original Panthera arm, joint limits,
servos, and gripper force limits are retained. No grasp welds, magnetic captures,
part snapping, or object-pose commands are used during physical trials.

The gear cavity is approximately 11.33 mm high; each large gear is 8.2 mm thick.
Gears enter radially between the two plates. In the CAD pose there is 2.24 mm
clearance below a gear, so an unpinned released gear can settle downward. The
gear trial reports XY error, axial error, and tilt separately and measures
placement only, not a retained or functioning transmission.

`audit` is kinematic analysis: it deliberately sets candidate poses to measure
reachability and collisions. Its clear paths are not physical assembly successes.
The physical rollout logs and JSON results are separate. `--try-gear` currently
places only `gear_3`; `--all-gears` places the carrier followed by gears 1–4.
Each gear is pinched at an 18 mm outer-rim offset with the tool pointing down,
aligned using its measured in-hand pose, and inserted radially over 30 mm.
The controller adds a bounded ±0.015 rad integral bias to the arm servo
setpoints to remove gravity/friction tracking error. Original joint limits,
servo gains, force limits, and contact geometry are retained.

Stage completion checks reject unresolved IK targets. `stages.json` records
target poses and final IK/tracking errors; `carrier_rollout.npz` also records
per-frame residuals as `[IK position m, IK rotation rad, actual position m,
actual rotation rad]`. A failed physical trial exits nonzero. Overall success
requires every requested gear and the carrier to remain seated after all
releases and retreats.

## Pin insertion

`--with-pins` assembles all gears, then installs their pins. The pins are plain
8 mm diameter, 17.88 mm long CAD cylinders; their export order differs from the
gear numbering: pin 3 → gear 1, pin 4 → gear 2, pin 2 → gear 3, pin 1 → gear 4.

The controller grips each upright pin near its top, measures the in-hand pose,
and guides it through the actual seated carrier hole. It then aligns with the
released gear's measured bore before lowering farther. After releasing the
partially inserted pin, it presses the head with the edge of the closed fingers,
offset away from the central hub, and retreats. All motion uses the original
arm and gripper actuators and contact geometry.

Four 3 mm radius backing posts are added to the stationary fixture in the pin
scene. Their tops match the CAD pin-bottom heights. They contact the actual
part collision pieces, while support hulls remain excluded so holes stay open.
The plain exported pins otherwise have no positive axial stop and can fall
below the intended depth. This is a fixture-supported assembly, not a claim of
retention after removing the fixture or of a tested operating transmission.

Success requires each pin shaft to span both the upper carrier plate and the
entire gear bore. The centerline is checked at both ends of each bore, with
0.17 mm carrier and 0.15 mm gear limits (including compliant contact tolerance),
less than 0.75 mm axial depth error and 3° tilt. Final checks also re-evaluate
all gears, pins, and the carrier after the last insertion. The replay tool reads
the recorded scene variant automatically; existing gear-only recordings remain
compatible with `scene.xml`.

See [the experiment report](../../reports/actuator_assembly.md) for measured
trial results and artifacts.
