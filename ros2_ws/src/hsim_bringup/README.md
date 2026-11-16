# hsim_bringup

Launch files, robot profiles, parameters, and the machine-readable interface contract.

| File | What |
|---|---|
| `launch/autonomy.launch.py` | the full robot: `sim_stack` + state estimation, localization/SLAM, Nav2, perception, world model, safety, behavior tree, planner. Defaults: `robot:=wisco`, `mapping:=rtabmap`, `perception:=server` |
| `launch/sim_stack.launch.py` | MuJoCo sim + locomotion policy + twist_mux + sensor mounts (+ viewer bridges, foxglove) for one robot profile and optional `env:=` |
| `launch/stub_stack.launch.py` | kinematic sim stub + locomotion stub + twist_mux: the plumbing without physics (contract tests) |
| `config/robots/<name>.yaml` | robot profiles (`robot:=<name>`): `wisco` (Wisco V1, default), `g1`, `g1_arms_free`. World, policy, sensors, mounts, limits |
| `config/interface_contract.yaml` | topics, types, owners, rates, TF edges, actions and services, by phase (`docs/interface_contract.md` is generated from it) |
| `config/twist_mux.yaml` | `/cmd_vel_teleop` (100) > `/cmd_vel_approach` (20, behavior tree's final step) > `/cmd_vel_nav` (10) → `/cmd_vel`; `/safety/estop` lock (255) |
| `config/g1_joints.yaml` | G1 joint order + default pose (from mjlab 1.6.0), used by the stub stack |

Tools:
- `ros2 run hsim_bringup contract_check --phase N` checks the live graph against the contract
- `ros2 run hsim_bringup drive_check` sends a 2 s forward command and checks the robot moves, then stops
- `tools/ros.sh test [stub|sim] [env] [robot]` (from the Mac) launches a stack and runs both
