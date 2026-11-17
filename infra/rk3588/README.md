# RK3588 real-time setup

The RK3588 (Rockchip) runs `ros2_control_node` and the IgH EtherCAT master: the loop that reads every drive and sends
it a command each cycle (`ros2_ws/src/wisco_ethercat_ros2_hw/`). For that loop to keep its period, the computer needs
a real-time kernel, cores kept free of other work, and the network card's interrupts on the loop's core. The central
computer (Jetson Thor or Jetson Orin NX) is not in this loop and needs none of this.

`setup_rt.sh` does the setup and the checks, on the RK3588 itself:

```bash
sudo infra/rk3588/setup_rt.sh check                # what is missing
sudo infra/rk3588/setup_rt.sh bootargs --apply     # isolate cores 6-7, then reboot
sudo infra/rk3588/setup_rt.sh tune                 # limits, governor, irqbalance off, NIC IRQs on cpu7 (+ boot unit)
sudo infra/rk3588/setup_rt.sh latency 30           # 30 min cyclictest under load → reports/
sudo infra/rk3588/setup_rt.sh rate-check 250 120   # while ros2_control runs at update_rate:=250
```

## Why each step

| Step | What it does | Without it |
|---|---|---|
| PREEMPT_RT kernel | almost all kernel code becomes preemptible, interrupts run as threads with priorities | a long kernel section can delay the control thread by milliseconds |
| `isolcpus` / `nohz_full` / `rcu_nocbs` on cores 6-7 | the scheduler, the timer tick and RCU callbacks stay off the second Cortex-A76 cluster; `irqaffinity=0-3` keeps other interrupts on the A55 cores | other processes and kernel housekeeping share the loop's core |
| `rtprio 98`, `memlock unlimited` | `ros2_control_node` may use SCHED_FIFO and lock its memory (it falls back to normal scheduling and says "Could not enable FIFO RT scheduling" otherwise) | page faults and normal-priority scheduling in the loop |
| `performance` governor | the cores stay at full clock | frequency changes add latency |
| irqbalance off, NIC IRQs on cpu7 | EtherCAT frames are handled on the loop's core | the interrupt moves between cores and its latency varies |

The RK3588 has four Cortex-A55 cores (cpu0-3) and four Cortex-A76 cores in two clusters (cpu4-5, cpu6-7). Run
`ros2_control_node` on the isolated cores, e.g. `taskset -c 6-7 ros2 launch ...` (the controller manager raises its
own update thread to FIFO priority 50 once the limits allow it).

## Kernel

There is no packaged PREEMPT_RT kernel for RK3588 boards. Two routes others have used:

- **Rockchip 6.1 vendor kernel + RT patch.** Forlinx applies the 6.1.99-rt36 patch to its 6.1.118 RK3588 kernel, plus
  RT config fragments for arm64 and Rockchip, and publishes cyclictest results for it. Armbian's `vendor` branch for
  RK3588 is the same 6.1 Rockchip kernel line, so that patch is the starting point for an Armbian build
  (`KERNEL_CONFIGURE=yes`, then "Fully Preemptible Kernel (Real-Time)"); expect some conflicts to resolve.
- **Mainline 6.12 or newer.** PREEMPT_RT is merged in mainline from 6.12: no patch, only `CONFIG_PREEMPT_RT=y`.
  Mainline RK3588 support covers the Ethernet, USB and storage needed here; check the board's other peripherals.

Either way, rebuild the IgH master's kernel modules against the new kernel.

## EtherCAT master

The IgH master either takes over the network card with a native driver (lowest latency) or uses its generic driver
on top of the normal Linux driver (works with every card, slower). `check` prints the card's driver. Which cards have
a native driver depends on the IgH and kernel versions (IgH's "Device Drivers" page). Use the native driver if there
is one for the card and kernel; otherwise use the generic driver and measure.

## Measured latency

`latency` runs cyclictest at a 1 ms period (a 1 kHz control loop) on the isolated cores, with stress-ng loading the
other cores, and writes `reports/cyclictest_<kernel>_<date>.txt` (min / avg / max per core). For a control loop at
`update_rate` *f*, the worst-case latency should stay well under the period 1/*f*: below about 100 µs before trying
1 kHz. Record the result in `ros2_ws/src/wisco_ethercat_ros2_hw/README.md` ("Real-time setup").

## Raising update_rate

The launch uses `update_rate` for both the controller manager and the EtherCAT cycle. Raise it one step at a time:
**100 → 250 → 500 → 1000 Hz**. At each step:

1. Launch with `update_rate:=<hz>`, with the robot held or the joints free to move, and let it run for at least two
   minutes with the usual load (perception, logging) on the other cores.
2. `sudo infra/rk3588/setup_rt.sh rate-check <hz> 120`: EtherCAT lost frames and working-counter changes must both
   be zero. A working-counter change means a drive missed a cycle's data.
3. The controller manager's log must show no overrun warnings ("Overrun detected"), and `/joint_states` must arrive
   at the configured `joint_state_rate`.
4. Pass: try the next rate. Fail: keep the previous one and look at the latency report first.

Keep the highest rate that passes, with margin. The walking policy itself runs at 50 Hz on the central computer.
The drives run their PD (mode 5) at their own internal rate, so a faster loop mainly helps the arms (`ArmController`)
and the watchdogs.

## Sources

- Forlinx, PREEMPT_RT on the OK3588 (6.1.118 kernel, 6.1.99-rt36 patch, cyclictest results):
  https://www.forlinx.net/industrial-news/rk3588-development-board-preempt-rt-patch-testing-810.html
- Forlinx, PREEMPT_RT + IgH EtherCAT master on RK3576 (same 6.1 Rockchip kernel line):
  https://www.forlinx.net/industrial-news/igh-ethercat-preempt-rt-forlinx-ok3576-776.html
- Armbian forum, real-time kernel for the Orange Pi 5 (RK3588S): https://forum.armbian.com/topic/28559-realtime-kernel-for-orange-pi-5
- linux-rt-users, Armbian 6.12+ needs only the PREEMPT_RT config: https://spinics.net/lists/linux-rt-users/msg28416.html
- OSADL, real-time Linux checklist and latency practice: https://www.osadl.org/?id=930
- acontis, running EtherCAT in real time on Linux (core isolation, IRQ pinning): https://www.acontis.com/en/running-ethercat-in-real-time-on-linux.html
- IgH EtherCAT master 1.6 documentation (native and generic drivers): https://docs.etherlab.org/ethercat/1.6/doxygen/index.html
- ros2_control controller manager, real-time scheduling setup: https://control.ros.org/jazzy/doc/ros2_control/controller_manager/doc/userdoc.html
