# Robot network: Jetson ↔ RK3588

The robot's two computers sit on one Ethernet cable: the **Jetson Thor or Jetson Orin NX** (central computer: policies,
perception, planning) and the **RK3588 (Rockchip)** (real-time computer: EtherCAT, ros2_control). The walking
policy's joint targets go Jetson → RK3588 over ROS 2 every 20 ms, and joint states and the IMU come back. Two things
have to be set up for that:

- **ROS 2 on the cable, and only there.** Both computers use the same ROS domain, look for each other directly
  (`ROS_STATIC_PEERS`), and Fast DDS is limited to the cable's address (and shared memory inside each computer), so
  robot traffic never goes out over Wi-Fi and a laptop on the Wi-Fi cannot see or command the robot.
- **One clock.** Message timestamps (IMU, joints, camera) are compared across the computers, so the RK3588's clock
  follows the Jetson's (chrony). The Jetson follows the internet when it has it and keeps serving time on its own when
  it does not.

```bash
# addresses in the repo .env: ROBOT_JETSON_IP, ROBOT_RK3588_IP (+ ROBOT_LINK_PREFIX, ROBOT_ROS_DOMAIN_ID)
infra/robot_network/robot_network.sh plan jetson --iface eth1          # print the files, change nothing
sudo infra/robot_network/robot_network.sh apply jetson --iface eth1    # on the Jetson
sudo infra/robot_network/robot_network.sh apply rk3588 --iface eth0    # on the RK3588
infra/robot_network/robot_network.sh check rk3588                      # ping, clock offset < 1 ms, ROS nodes visible
```

| File written | What |
|---|---|
| `/etc/netplan/60-wisco-robot-link.yaml` | static address on the cable, no gateway |
| `/etc/chrony/conf.d/wisco-robot.conf` | Jetson: serves time to the RK3588 (`local stratum 8 orphan`); RK3588: follows the Jetson, polling every 1–4 s, stepping the clock at any offset so it is right from boot |
| `/etc/profile.d/wisco-ros.sh` | `ROS_DOMAIN_ID`, `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`, `ROS_STATIC_PEERS` (the other computer), Fast DDS, its profile |
| `/etc/wisco/fastdds_robot.xml` | Fast DDS profile: UDP only on the cable's address (`interfaceWhiteList`) plus shared memory |

Chrony over a direct cable keeps the clocks within tens of microseconds, plenty for a 50 Hz policy and 200 Hz
sensors. PTP (`linuxptp`) reaches sub-microsecond if both network ports timestamp in hardware; it is not needed here.

**Tested** as two containers ("Jetson", "RK3588") on a "cable" network and a "Wi-Fi" network, with a "laptop" on the
Wi-Fi only: the RK3588 locked to the Jetson's clock (9 µs offset), a topic published on the RK3588 reached the
Jetson, and the laptop in the same ROS domain saw nothing. With the Fast DDS profile switched off, the laptop did see
it: the profile is what keeps the robot's traffic on its cable. Not yet run on the robot's own computers.

## Sources

- ROS 2 Jazzy, discovery settings (`ROS_AUTOMATIC_DISCOVERY_RANGE`, `ROS_STATIC_PEERS`):
  https://docs.ros.org/en/jazzy/Tutorials/Advanced/Improved-Dynamic-Discovery.html
- Fast DDS, interface allowlist and transports: https://fast-dds.docs.eprosima.com/en/2.14.x/fastdds/transport/whitelist.html
- Unitree's ROS 2 setup pins DDS to the robot's network interface the same way (CycloneDDS `NetworkInterface`):
  https://github.com/unitreerobotics/unitree_ros2
- chrony configuration (`allow`, `local ... orphan`, `server ... iburst minpoll maxpoll`, `makestep`):
  https://chrony-project.org/doc/4.5/chrony.conf.html
