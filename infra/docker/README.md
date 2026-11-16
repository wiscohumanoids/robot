# docker

The ROS 2 Jazzy development image (`Dockerfile`) and its compose file. One image for every laptop (macOS has no native
Jazzy) and for CI: navigation, SLAM, behavior trees, MuJoCo 3.11 (the training version), ONNX Runtime, `ros2_control`
with Pinocchio for the controllers, and software OpenGL so the simulator renders camera images without a GPU.

`tools/ros.sh up` builds the image and starts the container `hsim-ros` with the repo mounted at `/hsim` and the repo's
`.env` loaded (API keys). The EtherCAT master and `ethercat_driver_ros2` are hardware-only and are not in the image.
