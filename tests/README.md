# tests

Static checks, no ROS needed (CI job `static`): `python -m pytest tests ros2_ws/src/*/test` from the repo root.

| File | Checks |
|---|---|
| `test_contract.py` | the contract YAML is well formed; every owner is a real node; every topic the stack's code publishes is in the contract; interface files exist and are registered in CMake |
| `test_docs.py` | every relative link and anchor in the repo's markdown resolves; every ROS package has a README; `docs/interface_contract.md` tables are current |
| `test_repo_consistency.py` | robot profiles point at existing worlds and policies, manifest joints exist in the robot's world, manifests are self-consistent, `/hsim/...` paths exist, Python parses, shell scripts pass `bash -n` |

Checks against a running stack: `tools/ros.sh test` (contract check + drive test) and `tools/ros.sh autotest`
(navigation, behavior tree, reach), see `CONTRIBUTING.md`.
