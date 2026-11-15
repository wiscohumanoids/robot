# hsim_sim — the simulated world and the robot's sensors

| Executable | Node | What |
|---|---|---|
| `mujoco_sim` | `sim` | MuJoCo 3.11 G1 world at 200 Hz (training physics). Sensors with realistic models (`sensors.py`): IMU (noise, bias, onboard attitude filter), head lidar → `/scan` (10 Hz), battery → `/battery`. Services `/sim/reset`, test-only `/sim/set_body_pose`. Privileged `/sim/ground_truth/*` for graders only. |
| `stereo_camera` | `stereo_camera` | ZED X-style color + depth (`sensors:=stereo`), stereo depth-error model, own process, ~2.5–5 Hz on CPU |
| `sim_stub` | `sim` | kinematic stand-in (no physics) for testing upper layers |
