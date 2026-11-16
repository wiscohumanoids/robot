# vla_server (Phase 5)

GR00T policy behind a small HTTP API, on the GPU VM. Used by `ros2_ws/src/hsim_manipulation` (/skills/manipulate)
and by `training/manipulation/groot/eval_closed_loop.py`.

```
services/vla_server/run_vla_server.sh [checkpoint | --mock]      # VM: tmux hsim-svc-vla, 127.0.0.1:8200
ssh -N -L 8200:localhost:8200 hsim-gpu                            # Mac: tunnel (ROS container uses host.docker.internal:8200)
python client_test.py --url http://127.0.0.1:8200                 # check it
```

API: `GET /health`; `POST /act {instruction, images: {ego_view: base64 JPEG/PNG}, state: {left_arm[7], right_arm[7],
left_gripper[1], right_gripper[1], waist[3]}}` -> `{actions: {left_arm: [[7]x16], ...}, horizon, execution_horizon, latency_ms}`.
Actions are absolute joint targets; grippers 0 = closed, 1 = open; execute the first 8 at 20 Hz, then ask again.
Measured (N1.6, RTX PRO 6000): first call ~1.3 s (warm-up), then 175-240 ms; ~7 GB GPU.
`--mock` returns "hold current state" chunks without a model (for wiring tests).
