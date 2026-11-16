# Scripted stacking dataset — 2026-09-23

The scripted collector generated **1,000 accepted full three-block demonstrations**
from 1,402 randomized attempts (71.3% acceptance). Collection took 370 seconds
with six simulator workers. Rejected attempts are logged and are excluded from
training data. No simulator physics, object dynamics, or grasp-assist rules were
changed for this collection.

## Contents

- Raw native episodes: `data/scripted_stack/episode_0000` through `episode_0999`.
- 864,449 raw control frames, 28,814.956 simulated seconds (about eight hours).
- Episodes span 21.9–41.0 seconds.
- Native `data.npz` and `meta.json` include measured finger positions/velocities,
  physics tick counts, simulation timestamps, joint states/targets, end-effector
  poses, cube poses, gripper commands, IK residuals, stage boundaries, random
  seeds, generator/scene provenance, and acceptance metrics.
- Separate source data and output paths preserve the existing teleop datasets.

## Behavior and acceptance

The task order is consistent: red onto green, then blue onto red. The planner
uses smooth Cartesian interpolation and IK joint commands with an 0.08-radian
maximum command change per control tick. It approaches above a block, descends,
closes the jaws, verifies a bilateral contact grasp, lifts, transfers above the
support, lowers, opens, and retreats. Placement includes a small release gap
so the fingers do not collide with the supporting cube.

Cube positions and yaws use the simulator's normal randomized reset. Speeds
vary from 0.10–0.16 m/s; approach clearances vary from 6–8 cm. Initial arm poses
are randomized in a reachable elevated region. This region is narrower than
teleop's full 10–40 cm start-height distribution, and rejected unreachable
scenes introduce selection bias. Evaluation should report both matched starts
and the broader original deployment distribution.

Acceptance requires both blocks to be physically grasped and lifted, then a
released stack stable for a full second with at most 12 mm adjacent horizontal
misalignment. There are no object teleports during a recording. The only arm
pose initialization occurs before recording, as with teleop.

## Validation

Every raw episode passed finite-value, timestamp, control-bound, unique-seed,
and final stable-stack checks (`data/scripted_stack/audit.json`).

**All 1,000/1,000 production demonstrations also passed a fresh physics replay**
of the uniformly resampled 30 Hz controls, with a released stack held for at
least the final second (`data/scripted_stack/replay_audit.json`). Replaying uses
only the initial recorded state and subsequent joint/gripper commands; recorded
object trajectories are not imposed on the simulator.

The original ten-episode pilot reproduced all stacks under ACT's 30 Hz control
clock, both with and without the historical optional joint-step cap. The native
controls are resampled and shifted exactly as in training; successful replay is
therefore more informative than simply inspecting the recorded object poses.

All 23 repository tests passed, including new physical replay and compact-export
integration tests. The compact export test compares the optimized path against
upstream LeRobot serialization and checks identical embedded JPEG bytes,
state/action rows, and normalization statistics. A 10-episode export additionally
verified all 8,150 state/action rows against the existing conversion contract.
A two-optimizer-step ACT smoke run successfully loaded the camera dataset,
trained, and evaluated; this checks pipeline compatibility, not learned success.

The final export audit checked **every one of the 862,766 state/action rows**
against its rendered source trajectory, sequential frame/episode indices,
timestamps, metadata counts, and all provenance hashes. It also decoded 2,004
embedded images sampled across all 251 Parquet files. All checks passed; see
`outputs/lerobot/panthera_scripted_stack_30hz/verification.json`.

## Tools

```bash
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/collect_scripted.py --episodes 1000 --workers 6
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/validate_scripted_dataset.py
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/export_scripted_dataset.py --workers 6
```

The collector is resumable for unchanged code/scene settings. The exporter uses
fresh output paths and bounds its render queue. It embeds original rendered
JPEGs directly into LeRobot Parquet and removes disposable rendered images after
embedding them. Compact trajectories and source signatures remain available for
validation and replay. The usual renderer can regenerate images from raw data.
Both scripts check free disk space and stop below 5 GiB.

The dataset's final paths are `outputs/lerobot/panthera_scripted_stack_30hz` and
`outputs/scripted_stack_rendered_30hz`. The exporter writes `COMPLETE` only after
finalizing all episodes and provenance. Do not build the optional uncompressed
image cache without budgeting its substantially larger disk requirement.

The completed LeRobot export contains **1,000 episodes and 862,766 training
frames**, with 30 Hz observations and 256×256 RGB shoulder/wrist images. Export
took 1,097 seconds using six workers. The native recordings occupy 285 MiB,
the LeRobot dataset approximately 22 GiB, and retained rendered trajectories and
provenance 86 MiB. About 10.5 GiB remains free. No existing checkpoints were
deleted; disposable export trials created during this task were cleaned up.

The first full training run will additionally materialize an Arrow cache roughly
the size of the LeRobot dataset, plus training checkpoints. The remaining space
does not cover that additional cache; reclaim more space before starting a full
run. The optional decoded-image cache would be much larger still and is not
created by these tools.

Train a fresh model on this dataset before attributing prior ACT failures to
teleop quality. Success on scripted demonstrations would support that hypothesis;
collection and replay alone do not establish it.
