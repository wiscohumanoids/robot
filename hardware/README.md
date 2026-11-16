# Hardware

Everything that runs on, or talks directly to, the physical Wisco V1. The robot model and every number about it
(motors, masses, gains, ranges) live in [`robots/wisco_v1/`](../robots/wisco_v1/README.md).

## Compute and buses

```
 Jetson Thor (full ROS 2)                         RK3588 (real-time side)
   locomotion policy, perception, SLAM,    DDS      ros2_control: EtherCAT master (IgH) + ethercat_driver_ros2,
   planner, behavior tree, Nav2       ───────────►  arm controller (CST torque), e-stop receiver
   targets at 50–100 Hz                (non-RT)      │ EtherCAT daisy chain (1 kHz planned)
                                                     ▼
                                             MyActuator X-series drives (CiA 402)
 handheld ESP32-S3 ── Wi-Fi heartbeat ──► RK3588 GPIO ──► MOSFET ──► 48 V contactor
```

## What is here

| Where | What | Status |
|---|---|---|
| [`estop/`](estop/README.md) | wireless e-stop: ESP32-S3 transmitter firmware (ESP-IDF) and the RK3588 receiver (Python, systemd). Power stays on only while heartbeats arrive; STOP or 150 ms of silence cuts and latches | tested end to end with a laptop as the receiver; RK3588 GPIO path untested |
| [`../ros2_ws/src/wisco_ethercat_ros2_hw/`](../ros2_ws/src/wisco_ethercat_ros2_hw/README.md) | EtherCAT hardware configuration: MyActuator X4-10 slave/PDO map, `ros2_control` test-bench URDF, controllers, launch, example torque node | one motor spinning on a Linux laptop, CST mode, 100 Hz (no PREEMPT_RT) |
| [`../robots/wisco_v1/HARDWARE_ISSUES.md`](../robots/wisco_v1/HARDWARE_ISSUES.md) | hardware findings from simulation for the mechanical / electrical team (H1–H13: masses, ankle linkage and motor sizing, calibration, IMU, drive mode, supply voltage) | living list |
| [`../robots/wisco_v1/reports/current_draw/`](../robots/wisco_v1/reports/current_draw/REPORT.md) | per-motor and bus current estimates from the walking policy | 2026-11-12 |
| [`../docs/hardware/research_notes_ethercat_rt.md`](../docs/hardware/research_notes_ethercat_rt.md) | research notes: `ethercat_driver_ros2`, ros2_control plugins, CiA 402 state machine, PREEMPT_RT, multi-rate design | from the pre-unification scaffold |

## Decided

- Drive modes (2026-11-10): legs, waist and head in MyActuator mode 5 (position + kp + kd per cycle, PD in the
  driver, the same kp/kd as training); arms in mode 10 (CST torque from the arm controller). Details:
  `robots/wisco_v1/CHANGES.md`.
- Compute split: policy and perception on the Jetson Thor, EtherCAT and torque control on the RK3588.

## Next (see `STATUS.md` and `docs/migration/porting_backlog.md`)

Mode 5 PDO mapping and the full slave chain, 1 kHz on a PREEMPT_RT kernel, the ros2_control backend in simulation,
the IMU driver (EVAL_HI05 board; it must apply the board-to-pelvis rotation, HARDWARE_ISSUES H7), calibration (encoder zeros, sign checks, ankle crank offsets), the arm controller,
and a safety layer (soft limits, torque limits, missed-frame watchdog, fall → damping).
