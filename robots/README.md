# robots

One folder per physical robot we build: its model (URDF / MuJoCo, from CAD), every number about it with its source,
its calibration, and what simulation has found about its hardware.

| Robot | What | Used by |
|---|---|---|
| [`wisco_v1/`](wisco_v1/README.md) | Wisco V1, the full humanoid (31 joints) | the simulation, the walking and tracking policies, the ROS stack (`robot:=wisco`) |
| [`arm01/`](arm01/README.md) | arm01, a 7-DOF test arm on a stand (also the `arm01_description` ROS package) | the arm controller and calibration toolkit in `ros2_ws/src/wisco_ethercat_ros2_hw/` |

The Unitree G1, a reference robot in simulation, has no folder here: its model comes from mjlab (`assets/robots/g1/`).

**Adding a robot:** a folder with its URDF and MuJoCo model, a `model/robot_numbers.yaml` (or a joint config for an arm)
with sources for every value, and a README. For the ROS stack, add a robot profile in
`ros2_ws/src/hsim_bringup/config/robots/`; for the hardware interface, a joint table (see the hardware package README).
