# tools

Developer scripts. Mac-side Python tools use the repo `.venv` (MuJoCo 3.11, `tools/requirements-mac.txt`; created by
`viewer.sh` / `play.sh` on first run).

| Script | What |
|---|---|
| `ros.sh up\|shell\|build\|test\|autotest\|stop\|down` | the ROS 2 Jazzy container (`infra/docker/`): start it, open a shell, colcon build, smoke tests, autonomy checks |
| `stack_smoke_test.sh`, `autonomy_test.sh` | what `ros.sh test` / `ros.sh autotest` run inside the container |
| `viewer.sh` (`ros_viewer.py`, `viewer_camera.py`, `perception_view.py`) | native MuJoCo window for the running ROS sim: arrow-key driving, typed tasks, the robot's own view |
| `ask.sh "<request>"` | send a sentence to the planner |
| `play.sh` (`sim2sim_play.py`), `play_walk.sh`, `play_mocap.sh`, `showcase_play.py` | run a policy in plain MuJoCo on the Mac, no ROS (sim2sim). Defaults to the G1 all-joints policy; Wisco V1: `tools/play.sh --policy models/locomotion/wisco_v1_velocity_arms_free --scene assets/robots/wisco_v1/mjlab_1.6.0/wisco_velocity_flat.xml` |
| `env_build.py <env>` | build an environment for a robot profile (`--robot wisco` default): `world.mjb`, Nav2 map, objects, previews |
| `robocasa_export.py` | export a RoboCasa kitchen as an environment (runs in RoboCasa's own Python) |
| `build_robot_world.py wisco` | build `assets/robots/wisco_v1/mjlab_1.6.0/` (the sim's robot world) from the mjlab task |
| `gen_docs.py [--check]` | regenerate the tables in `docs/interface_contract.md` from the contract YAML |
| `sync_to_vm.sh`, `pull_policy.sh`, `vm_jobs.sh`, `vm_pull_results.sh` | GPU VM: copy code, fetch exported policies, overnight job control, fetch small results |
| `eval/` | graders and debug tools; the only code allowed to read privileged ground truth |
