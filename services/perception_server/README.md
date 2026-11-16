# perception server (S1)

Open-vocabulary object detection for the robot's ZED X head camera, on the GPU VM:
**Grounding DINO** base + tiny (finds named objects: text -> boxes) -> **SAM 2.1** (box -> exact pixels) -> **depth
under the mask** -> 3D centre and size in the camera frame and, given the camera pose, the map frame -> clean-up:
**mask NMS** (one object found twice), **3D size/placement priors** (a "fridge" that ends at 0.86 m is a dishwasher;
`priors.py`), **SigLIP 2 crop check** for object/appliance classes (the crop must look like its label among the
kitchen vocabulary). Precision 0.50 -> 0.78 on our kitchen benchmark with recall kept (docs/runs/2026-11-10_worklog_E.md).
No SAM 3 (gated) and no FoundationPose (6-DoF pose not needed: the VLA grasps from images).
Decision context: docs/decisions/0003-perception-on-gpu.md.

| | |
|---|---|
| Where | GPU VM, `127.0.0.1:8100` (never exposed; the Mac uses an SSH tunnel) |
| Env | `~/hsim_envs/perception` (Python 3.12, torch cu128, transformers >= 4.56) via `setup_env_vm.sh` |
| Weights | `~/hsim_data/hf-cache` (Grounding DINO base/tiny, SAM 2.1 hiera base+/small) |
| Run | `run_server.sh` (tmux `hsim-svc-perception`, log `~/hsim_runs/S1_perception/server.log`) |
| API | `GET /health`, `POST /detect` (format: `perception_server/protocol.py`) |
| ROS side | `hsim_perception/perception_client` (launch `perception:=server`), falls back to the line-of-sight stub |

## Start (VM)
```
~/humanoid-sim-stack/services/perception_server/setup_env_vm.sh --prefetch   # once
~/humanoid-sim-stack/services/perception_server/run_server.sh                # tmux hsim-svc-perception
curl -s localhost:8100/health
```
Restart after a code change: `tmux kill-session -t hsim-svc-perception; bash ~/humanoid-sim-stack/services/perception_server/run_server.sh`.
Options: `--verifier none` (no crop check), `--no-priors`, `--no-mask-nms`; per request `"postprocess": {...}` and
`"return_rejected": true` (rejected detections with the reason, for debugging).
Fake backend (no GPU, returns ground truth sent in the request; for plumbing tests):
`python -m perception_server.server --backend fake`.

## Use from the sim stack (Mac)
```
ssh -N -L 8100:localhost:8100 hsim-gpu &          # tunnel; the container reaches it as host.docker.internal:8100
ros2 launch hsim_bringup autonomy.launch.py env:=robocasa/kitchen_l1_s1 sensors:=stereo perception:=server
ros2 topic echo /perception/status                 # mode, latency, detections, failures
```

## How the 3D estimate is made (perception_server/geometry.py)
Depth under the mask core (eroded) -> nearest depth cluster with >= 25 % of the points (a mask leaking onto
the wall behind an object adds a farther cluster) -> all mask pixels within a MAD band of that depth ->
2-98th percentile box. The camera only sees the front surface, so compact objects (< 0.4 m) get their box
completed along the view ray (assume about as deep as wide); fixtures stay as seen.

## Benchmark (bench/)
`render_frames.py` (Mac) renders RGB-D frames from our kitchens exactly like the ZED X sim node (640x320,
110 deg, 1.28 m, 25 deg down, stereo-like depth noise) with segmentation ground truth and a per-pixel object map;
`audit.py` (VM) scores a running server by what each detection's mask sits on (scorer rules in its docstring; caches
responses so filters can be re-scored offline); `memory_sim.py` simulates the world model's memory; `eval_server.py`
(old box-overlap scorer), `sweep.py`, `compare_configs.sh`, `verify_crops.py`. Results: docs/runs/2026-11-10_worklog_E.md.

## Tests
`tests/test_server_units.py` (RLE, lifting, phrase mapping, fake backend, HTTP) and
`ros2_ws/src/hsim_perception/test/test_server_link.py` (client logic + HTTP round trip): numpy, opencv-python-headless,
fastapi, httpx, uvicorn, pytest.
