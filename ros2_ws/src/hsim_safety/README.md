# hsim_safety — can the robot move?

Node `safety`: e-stop on operator request, fall (> 35° tilt), battery < 5%, or silent encoders/odometry;
warnings for battery < 20% and a silent lidar. Publishes `/safety/estop` (a twist_mux lock at priority 255: no
velocity command gets through) and `/safety/status` (read by the behavior tree every tick).

Operator e-stop: `ros2 run hsim_safety estop on` / `off`. Tested (`tools/eval/estop_check.py`): 0.41 m/s → 0.02 m/s
while engaged with commands still arriving → 0.44 m/s after release.
