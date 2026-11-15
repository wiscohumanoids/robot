# hsim_bringup (Phase 1)

Launch files, parameters, and the machine-readable interface contract.

| File | What |
|---|---|
| launch/stub_stack.launch.py | sim + locomotion + twist_mux + state estimation + localization (+ foxglove_bridge) |
| config/interface_contract.yaml | topics, types, owners, rates, TF edges, by phase |
| config/g1_joints.yaml | G1 joint order + default pose (from mjlab 1.6.0) |
| config/twist_mux.yaml | /cmd_vel_teleop (priority 100) and /cmd_vel_nav (10) → /cmd_vel |

Tools:
- `ros2 run hsim_bringup contract_check --phase 1` checks the live graph against the contract
- `ros2 run hsim_bringup drive_check` sends a 2 s forward command and checks the robot moves, then stops
- `tools/ros.sh test` (from the Mac) runs both against a freshly launched stub stack
