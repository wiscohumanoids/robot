# perception server

Open-vocabulary object detection for the robot's ZED X head camera, on the GPU VM:
**Grounding DINO** base + tiny (finds named objects: text -> boxes) -> **SAM 2.1** (box -> exact pixels) -> **depth
under the mask** -> 3D centre and size in the camera frame and, given the camera pose, the map frame -> clean-up:
**mask NMS** (one object found twice), **3D size/placement priors** (a "fridge" that ends at 0.86 m is a dishwasher;
`priors.py`), **SigLIP 2 crop check** for object/appliance classes (the crop must look like its label among the
kitchen vocabulary). On a kitchen benchmark the clean-up raised precision from 0.50 to 0.78 with recall kept.
No SAM 3 (gated) and no FoundationPose (6-DoF pose not needed: the VLA grasps from images). It runs on the GPU
machine, not the robot, while the models are being chosen; the client falls back to a stub, so the stack runs without it.

| | |
|---|---|
| Where | GPU VM, `127.0.0.1:8100` (never exposed; the Mac uses an SSH tunnel) |
| Env | `~/hsim_envs/perception` (Python 3.12, torch cu128, transformers >= 4.56) via `setup_env_vm.sh` |
| Weights | `~/hsim_data/hf-cache` (Grounding DINO base/tiny, SAM 2.1 hiera base+/small) |
| Run | `run_server.sh` (tmux `hsim-svc-perception`, log `~/hsim_runs/perception/server.log`) |
| API | `GET /health`, `POST /detect` (format: `perception_server/protocol.py`) |
| ROS side | `hsim_perception/perception_client` (launch `perception:=server`), falls back to the line-of-sight stub |

## Start (VM)
```
~/robot/services/perception_server/setup_env_vm.sh --prefetch   # once
~/robot/services/perception_server/run_server.sh                # tmux hsim-svc-perception
curl -s localhost:8100/health
```
Restart after a code change: `tmux kill-session -t hsim-svc-perception; bash ~/robot/services/perception_server/run_server.sh`.
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

## Tests
`tests/test_server_units.py` (RLE, lifting, phrase mapping, fake backend, HTTP) and
`ros2_ws/src/hsim_perception/test/test_server_link.py` (client logic + HTTP round trip): numpy, opencv-python-headless,
fastapi, httpx, uvicorn, pytest.
