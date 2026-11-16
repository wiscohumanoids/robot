# tests

Static checks, no ROS needed (CI job `static`): `python -m pytest tests ros2_ws/src/*/test` from the repo root.

| File | Checks |
|---|---|
| `test_contract.py` | the contract YAML is well formed; every owner is a real node; every topic the stack's code publishes is in the contract; interface files exist and are registered in CMake |
| `test_docs.py` | every relative link and anchor in the repo's markdown resolves; every ROS package has a README; `docs/interface_contract.md` tables are current |
| `test_repo_consistency.py` | robot profiles point at existing worlds and policies, manifest joints exist in the robot's world, manifests are self-consistent, `/hsim/...` paths exist, Python parses, shell scripts pass `bash -n` |

Package tests run in the same command: `ros2_ws/src/*/test/` (policy contract, VLA and perception clients, planner
prompt, and the hardware interface's Python tests: joint table, kinematics, IK, calibration stages in simulation).
The hardware interface's C++ tests (controller laws, mock EtherCAT, gravity parity) run under colcon in the CI job `ros`:
`colcon test --packages-select wisco_ethercat_ros2_hw`.

Checks against a running stack (Docker): `tools/ros.sh test [stub|sim] [env] [robot]` (contract check + drive test) and
`tools/ros.sh autotest [name] [robot]` (navigation, behavior tree, reach for `wisco`, `g1` or `g1_arms_free`).
