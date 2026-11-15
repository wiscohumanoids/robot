## WiscoHumanoids Hardware Interface

Demo: https://drive.google.com/file/d/1zwNxs48vFlNf9jD_jDFN_XyyGplYSpis/view?usp=sharing

This repository contains the physical hardware configuration, PDO mapping, demo script, and ROS 2 launch files to bridge the RK3588's real-time EtherCAT master to the Jetson Thor's high-level AI policy. Its contents serve as a test demo that proves you can run a single servo with ROS 2 Jazzy and ethercat_driver_ros2 on a Linux laptop (replace with RK3588 in final implementation). Note that it is configured to run on 100Hz instead of the intended 1kHz since it was made on a device that did not have PREEMPT_RT patched.

RK3588: The software is split into hardware definitions (this repo) and mathematical control plugins (loaded dynamically).

```text
wisco_ws/src/
├── ethercat_driver_ros2/          # (add separately, you need to install this for it to work github.com/ICube-Robotics/ethercat_driver_ros2)
│
├── wisco_ethercat_ros2_hw/        # <----- (THIS REPO: Hardware configuration)
│   ├── config/rmd_x4_10.yaml      #      Hexadecimal PDO maps & gear ratios
│   ├── config/controllers.yaml    #      Dictates which controllers to load
│   ├── launch/test_bench.launch.py 
│   ├── urdf/test_bench.urdf.xacro #      Physical robot description & slave IDs
│   └── README.md                  # (you are here)
│
└── wisco_controllers/             # (C++ controls stack to be added separately later along with the rest of the humanoid)
    ├── include/wisco_controllers/
    │   └── wbc_controller.hpp     # Class headers, Eigen matrix declarations
    └── src/
        └── wbc_controller.cpp     # The 1kHz QP solver, Pinocchio rigid body dynamics, etc.

ros2_control (controller_manager process, same process as hw_interface)
├── ethercat_driver_ros2 (SystemInterface) ── 14x RMD-X4-10 CiA 402
├── wbc_controller plugin
└── State/Command Interfaces (hardware interface exposed by ethercat_driver_ros2, consumed/written by wbc_controller)

ROS 2 (minimal) on RK3588
└── Command surface exposed by wbc_controller (action/topic for
    Cartesian/task targets) — receives targets from Jetson Thor
```

Jetson Thor (not included here; for context):
```text
jetson_ws/src/
└── wisco_policy/
    ├── wisco_policy/
    │   └── policy_node.py              # RL locomotion policy / MPC
    └── launch/
        └── policy.launch.py

ROS 2 (full) on Jetson Thor
├── MoveIt 2 / AI pipeline
├── RL locomotion policy / MPC
├── Vision / SLAM
└── Sends high-level targets (50–100Hz) over DDS ──┐
                                                     │
                                                     ▼
                               ROS 2 DDS Network (non-RT link)
                                                     │
                                                     ▼
                     RK3588: wbc_controller's ROS 2 command surface
```

### Quick Start Workflow

**1. Build**
Run this from the workspace root (`[DIRECTORY]/wisco_ws`) whenever you make changes
```bash
colcon build --symlink-install
```

**2. Source the Environment**
Run this in any new terminal before executing ROS 2 commands.
```bash
source /opt/ros/jazzy/setup.bash
source [DIRECTORY]/wisco_ws/install/setup.bash
```

**3. Launch**
Requires `sudo` to allow the EtherCAT driver to access the raw `/dev/EtherCAT0` network device.
```bash
# 1. Start the kernel module (make sure the slave is connected)
sudo systemctl start ethercat

# 2. Launch ros2_control and the hardware interface
sudo -E bash -c 'source /opt/ros/jazzy/setup.bash && source ~/Code/wisco_ws/install/setup.bash && ros2 launch wisco_ethercat_ros2_hw test_bench.launch.py'
```

Note that the test bench by default runs an example python script that sends torque commands to the motor to spin every couple seconds.

rviz2 is recommended to visualize the servo as it runs.

### Useful Debugging Command Examples

Open a second (sourced) terminal while the launch file is running to inspect the real-time loop:

* **List active ROS2 topics:** `ros2 topic list`
* **Stream live motor telemetry (pos/vel/effort):** `ros2 topic echo /joint_states`
* **Verify hardware loaded:** `ros2 control list_hardware_interfaces`
* **Verify controllers loaded:** `ros2 control list_controllers`
* **Command torque manually:** `ros2 topic pub /effort_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.5]}"`


