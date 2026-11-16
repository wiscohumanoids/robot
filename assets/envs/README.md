# Environments

Every environment is `<source>/<name>/` with an `env.yaml` (what to build) and a `build/` folder
produced by the builder (gitignored, rebuild any time):

    .venv/bin/python tools/env_build.py assets/envs/test/table_room

| build/ file | Used by |
|---|---|
| `world.mjb` | the ROS `sim` node, the Mac viewer, the standalone player (G1 training world + room, one self-contained file) |
| `map.pgm` + `map.yaml` | Nav2 `map_server` → `/map` |
| `objects.yaml` | named objects with position and size (planner now, ground-truth perception later) |
| `preview.png`, `preview_persp.png` | quick visual check |

Use one:

    # container
    ros2 launch hsim_bringup sim_stack.launch.py env:=test/table_room
    # Mac (must be the same env)
    tools/viewer.sh --env test/table_room
    tools/play.sh --env test/table_room          # standalone, no ROS

## RoboCasa kitchens

One-time setup (done on this Mac, outside the repo): `~/hsim_third_party/` holds robosuite + RoboCasa,
their Python env `.venv-robocasa` (Python 3.12, MuJoCo 3.3.1), and ~25 GB of assets.

    # export (run from the repo, not from ~/hsim_third_party) then build
    ~/hsim_third_party/.venv-robocasa/bin/python tools/robocasa_export.py --layout 3 --style 3
    .venv/bin/python tools/env_build.py assets/envs/robocasa/kitchen_l3_s3

The exporter builds a real RoboCasa task scene, removes RoboCasa's robot, and keeps the kitchen +
the task's objects. Our training physics always win over RoboCasa's settings. The G1 starts where
RoboCasa put its robot, turned to face the open room, moved out to ≥ 0.5 m clearance if needed.
A built kitchen `world.mjb` is ~450 MB (RoboCasa's textures are high resolution).

Rules: floors are flat (the all-joints walking policy only knows flat ground); the room's floor is the
training world's plane at z = 0; body names get the prefix `room/`.

| Env | Source | Status |
|---|---|---|
| `test/table_room` | hand-built: 8 × 6 m room, table with red/green/blue blocks, couch, shelf, ottoman | ✅ |
| `robocasa/kitchen_l<N>_s<N>` | RoboCasa kitchens (layout N, style N): counters, cabinets with working doors and drawers, appliances, 3 task objects | ✅ |

Built kitchens (RoboCasa's 10 "target" scenes), all walk-tested 2026-11-08: `robocasa/kitchen_l1_s1` … `robocasa/kitchen_l10_s10`, 44–121 fixtures each. When a room brings its own floor, the training floor stays for physics but is hidden, so the room's floor shows.
| `scan/*` | phone scan | ⬜ extension (`scan_pipeline/`) |
| generated | room from a sentence | ⬜ extension |
