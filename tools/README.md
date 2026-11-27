# tools

Developer scripts, grouped by intent. The two top-level entry points stay at the root so the common commands are short.

| Path | What |
|---|---|
| `ros.sh up\|shell\|build\|test\|autotest\|stop\|down` | the ROS 2 container (`infra/docker/`): start it, open a shell, build, smoke tests, autonomy checks |
| `ask.sh "<request>"` | send a sentence to the planner |
| `dev/vm.sh sync\|pull-policy\|pull-results\|jobs` | the GPU machine: copy code, fetch policies and results, drive the job queue |
| `dev/gen_docs.py [--check]` | regenerate the tables in `docs/interface_contract.md` from the contract YAML |
| `dev/requirements-mac.txt` | laptop-side Python dependencies for the tools below |
| `build/env_build.py <env>` | build an environment for a robot profile (`--robot wisco` default): `world.mjb`, Nav2 map, object list, previews |
| `build/build_robot_world.py wisco` | build `assets/robots/wisco_v1/mjlab_1.6.0/` (the simulation's robot world) from the training task |
| `build/robocasa_export.py` | export a RoboCasa kitchen as an environment (runs in RoboCasa's own Python) |
| `sim/viewer.sh` (`ros_viewer.py`, `viewer_camera.py`, `perception_view.py`) | MuJoCo window for the running ROS simulation: arrow-key driving, typed tasks, the robot's own view of the world |
| `sim/play.sh` (`sim2sim_play.py`) | run a **deployed** walking policy (ONNX + manifest, as the ROS node does) in plain MuJoCo, no ROS; defaults to the G1 all-joints policy. Wisco V1: `tools/sim/play.sh --policy models/locomotion/wisco_v1_velocity_arms_free --scene assets/robots/wisco_v1/mjlab_1.6.0/wisco_velocity_flat.xml` |
| `sim/play_mjlab.py` | run an exported policy (walking or motion tracking, any robot) inside its **training** environment in mjlab; viewer or video |
| `test/stack_smoke_test.sh`, `test/autonomy_test.sh` | what `ros.sh test` and `ros.sh autotest` run inside the container |
| `eval/` | graders for the autonomy checks and measurement tools; the only code allowed to read simulator ground truth |

Laptop-side Python tools use the repo `.venv` (MuJoCo 3.11, `dev/requirements-mac.txt`; created by `sim/viewer.sh` /
`sim/play.sh` on first run). `sim/play_mjlab.py` and `build/build_robot_world.py` need the mjlab environment
(`.venv-mjlab` on the laptop, `training/locomotion/.venv` on the GPU machine).
