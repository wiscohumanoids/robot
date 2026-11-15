# humanoid-sim-stack

Scan a room with a phone → a Unitree G1 appears in the reconstructed MuJoCo world → command it in natural language → watch every layer of the stack run live.

**Stack:** ROS 2 Jazzy (Ubuntu 24.04) · MuJoCo / mjlab · Nav2 · BehaviorTree.CPP · Grounded-SAM + FoundationPose · LeRobot VLA

**Machines:** AWS CPU VM (sim, ROS, control, ONNX inference) · AWS GPU VM, on demand (training + remote VLA/perception inference) · Mac laptop (editing, viewing)

| Folder | What it holds |
|---|---|
| docs/ | architecture, interface contract, infra, decisions |
| infra/ | VM + container setup |
| training/ | GPU-only training jobs (locomotion, manipulation) |
| models/ | exported policies + manifests |
| assets/ | G1 model, scenes |
| scan_pipeline/ | phone scan → environment (extension, later) |
| services/ | GPU inference servers (VLA, perception) |
| ros2_ws/ | ROS 2 packages, one per layer |
| tools/, tests/ | dev scripts, contract + smoke tests |

Status: Phases 0–2 done; Phase 3 autonomy built and tested (realistic sensors, leg odometry + EKF, AMCL/SLAM/RTAB-Map, Nav2, perception stub + object memory, behavior tree, safety, Claude planner). Next: Phase 4 real perception. Put ANTHROPIC_API_KEY in .env (see .env.example).
