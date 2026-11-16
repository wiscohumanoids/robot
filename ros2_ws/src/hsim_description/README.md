# hsim_description (planned)

Robot description for `robot_state_publisher` and RViz: publishes every robot link's TF from the URDF and
`/joint_states`. Not built yet; today only the sensor-mount frames are published, as static transforms placed from
the robot profile's `mounts` (`hsim_bringup/launch/sim_stack.launch.py`).

URDFs it will use: Wisco V1 `robots/wisco_v1/robot.urdf` (generated, see `robots/wisco_v1/CHANGES.md`); G1 from the
same model as `assets/robots/g1/`. On the real robot the same URDF also carries the `<ros2_control>` tags for the
EtherCAT hardware plugin (`ros2_ws/src/wisco_ethercat_ros2_hw/`).
