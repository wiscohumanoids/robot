# wisco_hipnuc_imu — the robot's IMU on ROS 2

Driver for Wisco V1's IMU, a HiPNUC **EVAL_HI05** board on the back of the pelvis, plugged into the **RK3588
(Rockchip)** by USB. The board runs HiPNUC's own firmware: it does the sensor fusion itself and streams binary packets.
Nothing here is flashed onto it. This package reads those packets and publishes the `/imu` topic the stack already
uses in simulation, so the walking policy, the state estimator and the safety monitor work unchanged on the robot.

```
EVAL_HI05 ──USB serial──► imu node (this package, RK3588) ──/imu──► walking policy, EKF, safety monitor
 (fusion in its firmware)   HI91 packets → imu_link (pelvis axes)   (same topic and conventions as the simulator)
```

| File | What |
|---|---|
| `wisco_hipnuc_imu/protocol.py` | frame sync, CRC-16/XMODEM, the HI91 packet (no ROS) |
| `wisco_hipnuc_imu/imu_math.py` | board axes → `imu_link` (pelvis axes), heading zeroed at start like the simulated IMU (no ROS) |
| `wisco_hipnuc_imu/stamping.py` | message stamps from the board's clock, carried onto the host clock (no ROS) |
| `wisco_hipnuc_imu/imu_node.py` | node `imu`: serial reader, `/imu`, `/diagnostics`, reconnects on its own |
| `wisco_hipnuc_imu/imu_check.py` | bench check on the serial port: rate, CRC errors, status, noise, and the mount (`--mounted`) |
| `config/hipnuc.yaml` | port, baud rate, mount rotation, covariances |
| `config/99-hipnuc-imu.rules` | udev rule: a stable `/dev/hipnuc_imu` |
| `launch/imu.launch.py` | starts the node with the config |
| `test/` | decoder, mount, timestamps and bench-check tests (no hardware, run in CI) |

## What it publishes

`/imu` (`sensor_msgs/Imu`, reliable, frame `imu_link`):

| Field | Source | Convention |
|---|---|---|
| orientation | the board's fused quaternion, rotated into `imu_link` | z up; heading 0 at start (`zero_yaw`), as the simulated IMU starts; tilt is what the policy and the EKF use |
| angular_velocity | gyro, deg/s → rad/s, rotated | `imu_link` axes (x forward, y left, z up) |
| linear_acceleration | accelerometer, G → m/s², rotated | specific force: +9.81 on z with the robot upright and still |
| header.stamp | the board's millisecond clock, mapped onto the host clock along the quickest frames | keeps the board's sample spacing instead of the USB adapter's jitter (`use_device_time`) |
| covariances | `config/hipnuc.yaml` | the simulated IMU's values until `imu_check` measures the real ones |

`/diagnostics`: rate, CRC errors, dropped bytes, the board's coordinate setting, temperature, and its warnings (gyro
bias or attitude not yet converged, saturation, magnetic disturbance).

**Mount.** The board stands upright at the back of the pelvis, chip facing backward: board x → robot right, y →
down, z → forward (`robots/wisco_v1/model/robot_numbers.yaml`, `sensors.imu`, from the legs CAD). `mount_wxyz` in the
config is that rotation, and a test keeps the two equal. It assumes the board's factory output axes (`COORD 0`);
diagnostics warn if the board reports another setting.

## Set up on the RK3588

1. **Port:** `sudo cp config/99-hipnuc-imu.rules /etc/udev/rules.d/`, put the board's USB-serial ids in it (the file
   says how), `sudo udevadm control --reload && sudo udevadm trigger`, and check `ls -l /dev/hipnuc_imu`.
2. **Board setup (once; it keeps the settings after power-off):** the factory output is 100 Hz at 115200 baud; the
   policy and the EKF want 200 Hz, which needs 230400 baud. With HiPNUC's `hihost` tool (`pip install` from their
   repo's `python/` folder), driver stopped:
   ```bash
   hihost baudrate 230400 -p /dev/hipnuc_imu --save
   hihost command "LOG HI91 ONTIME 0.005" -p /dev/hipnuc_imu -b 230400 --save
   ```
   Leave the output coordinates at the factory setting (`COORD 0`), which the mount assumes.
3. **Bench check, board in the hand or on a table, still:** `ros2 run wisco_hipnuc_imu imu_check --seconds 10`:
   about 200 Hz, no CRC errors, |gravity| ≈ 9.81.
4. **Mount check, board on the robot, robot hung level or standing:** `ros2 run wisco_hipnuc_imu imu_check --mounted`:
   gravity must rotate onto +z in `imu_link` (within 5°). A failure prints which board axis points up; it means the
   printed board axes differ from the CAD part, so fix `mount_wxyz` here and `sensors.imu` in `robot_numbers.yaml`
   together (the hardware observation "IMU orientation" closes with this check). Put the measured noise into the
   covariances (variance = std²).
5. **Run:** `taskset -c 0-3 ros2 launch wisco_hipnuc_imu imu.launch.py`, off the cores isolated for the control loop
   (`infra/rk3588/`), then `ros2 topic hz /imu` and `ros2 topic echo /diagnostics`.

## State and limits

Decoder, mount rotation, timestamps and the bench check are tested without hardware (`test/`), and the node has been
run against a simulated board on a virtual serial port. Not yet done on the real board: the axis check (step 4), the
real noise figures, and the rate after the board setup. Only HI91 packets are decoded (the board's IMU packet);
others are counted in diagnostics and skipped.

## Sources

- HiPNUC, `github.com/hipnuc/products` (Apache-2.0): the serial decoder `c/hipnuc/hipnuc_decoder.c` and
  `hipnuc_data.h` (frame, CRC, HI91 layout and units, status bits), and the ROS 2 driver's README `ros/README.md`
  (output axes per `COORD` setting, output rate and baud rate, `hihost` commands, board-clock timestamps). This
  package is written for this robot rather than copied: it adds the mount rotation, heading convention and checks.
- `sensor_msgs/Imu` conventions: REP 145 (IMU sensor driver conventions) and REP 103 (units, axes).
