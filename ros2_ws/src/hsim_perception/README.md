# hsim_perception — what the robot sees and remembers

| Executable | Node | Status | What |
|---|---|---|---|
| `perception_client` | `perception` | ✅ Phase 4 (`perception:=server`) | Real open-vocabulary detection: ZED X RGB-D -> perception server on the GPU VM (Grounding DINO base+tiny ensemble, SAM 2.1, depth -> 3D) -> `/perception/detections` (same message and semantics as the stub). Prompts: short default kitchen list + the environment's object labels + whatever the planner's plan names. Falls back to the stub if the server is unreachable, switches back when it returns. `/perception/status` (JSON, 1 Hz). Needs `sensors:=stereo` and `ssh -N -L 8100:localhost:8100 hsim-gpu` on the Mac |
| `perception_stub` | `perception` | 🟡 default (`perception:=stub`) and fallback | Honest about *what* is visible (head camera FOV, range, min angular size, occlusion by rays), privileged about *where* (reads true object positions). Noisy camera-frame detections, labels like "red block", no IDs. `/perception/detections` 5 Hz |
| `world_model` | `world_model` | ✅ (multi-view confirmation for real detectors: `confirm_views`, set to 2 by `perception:=server`) | Places detections on the map with the robot's own pose, tracks and merges sightings, confidence, last seen, "on" relations. `/world/objects`, `/world/summary` (JSON for the planner) |

Measured in the test room: table, blocks, couch remembered within ~2–7 cm after a walk; RoboCasa kitchen fixtures
within ~10 cm after one look-around.

Server, benchmark and accuracy numbers: services/perception_server/README.md, docs/runs/2026-11-10_worklog_E.md.
Tests: `test/test_server_link.py` (pytest, no ROS), `test/run_ros_smoke.sh` (throwaway ROS container).
