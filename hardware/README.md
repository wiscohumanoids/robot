# hardware

What runs on, or talks directly to, the physical robot, beyond the ROS hardware interface. The robots' models and
numbers live in [`robots/`](../robots/README.md); the `ros2_control` hardware interface and joint controllers live in
[`ros2_ws/src/wisco_ethercat_ros2_hw/`](../ros2_ws/src/wisco_ethercat_ros2_hw/README.md).

## Computers and buses

```
 Jetson Thor or Jetson Orin NX (central computer)       RK3588 (Rockchip, real-time computer)
   walking policy, perception, SLAM, planner,    ROS 2    EtherCAT master (IgH) + ros2_control:
   behavior tree, Nav2, manipulation policy     ───────►  ethercat_driver_ros2, joint controllers,
   targets at 50–100 Hz                          (DDS)     e-stop receiver, IMU driver (HiPNUC, USB) → /imu
                                                            │ EtherCAT ring (1 kHz with PREEMPT_RT)
                                                            ▼
                                                  MyActuator RMD-X drives (CiA 402)
 handheld ESP32-S3 ── Wi-Fi heartbeat ──► RK3588 GPIO ──► MOSFET ──► 48 V motor-power contactor
```

The real-time loop (controllers ↔ drives) never crosses the DDS link: the central computer sends targets, and the
controllers keep the joints safe if those targets stop arriving.

## What is here

| Path | What | State |
|---|---|---|
| [`estop/`](estop/README.md) | wireless e-stop: ESP32-S3 transmitter firmware (ESP-IDF) and the RK3588 receiver (Python, systemd). Motor power stays on only while heartbeats arrive; STOP, or 150 ms of silence, cuts it and latches until reset | tested end to end with a laptop as the receiver; the RK3588 GPIO path is still to be tested |

## Elsewhere in the repo

| Topic | Where |
|---|---|
| Drive configuration, controllers, mock EtherCAT, commissioning, current sensing, motion capture | `ros2_ws/src/wisco_ethercat_ros2_hw/README.md` |
| IMU board (HiPNUC EVAL_HI05) on the RK3588: driver, board setup, axis and mount check | `ros2_ws/src/wisco_hipnuc_imu/README.md` |
| RK3588 real-time setup: PREEMPT_RT kernel, core isolation, cyclictest, the `update_rate` ramp | `infra/rk3588/README.md` |
| Sim-to-real for walking: leg actuator identification (PACE), the steps to first steps | `training/sim2real/README.md` |
| arm01 calibration and identification (PACE) | `robots/arm01/README.md` |
| Wisco V1 hardware observations from simulation (masses, ankle motor sizing, calibration, IMU, drive gains and delay, supply voltage) | `robots/wisco_v1/hardware_observations.md` |
| Simulated current draw of Wisco V1 walking | `robots/wisco_v1/reports/current_draw/REPORT.md` |
| How ros2_control, EtherCAT, CiA 402 and real-time Linux work | `docs/knowledge/hardware_interface.md` |
