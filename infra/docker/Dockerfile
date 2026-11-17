# ROS 2 Jazzy development image for the whole stack: simulation, autonomy, and the ros2_control controllers.
# Runs on a Mac (arm64, Docker Desktop) and on any Linux x86_64 host. The repo is mounted at /hsim (compose.yaml).
# The EtherCAT master and ethercat_driver_ros2 are hardware-only and are not installed here.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Navigation, behavior trees, teleop, state estimation, visualization bridge
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
      ros-jazzy-twist-mux \
      ros-jazzy-joy ros-jazzy-teleop-twist-joy ros-jazzy-teleop-twist-keyboard \
      ros-jazzy-robot-state-publisher ros-jazzy-xacro \
      ros-jazzy-tf2-ros ros-jazzy-tf2-tools \
      ros-jazzy-robot-localization \
      ros-jazzy-behaviortree-cpp \
      ros-jazzy-vision-msgs \
      ros-jazzy-foxglove-bridge \
      python3-pip python3-pytest python3-yaml \
      git less nano tmux \
    && rm -rf /var/lib/apt/lists/*

# MuJoCo matches the training version (mjlab 1.6.0). numpy stays < 2 so ROS's own compiled Python packages
# (built against Ubuntu's numpy 1.26) keep working.
RUN pip3 install --no-cache-dir --break-system-packages \
      "mujoco==3.11.0" "onnxruntime>=1.20" "numpy<2"

# Software OpenGL so the simulator renders camera images without a GPU or display (MUJOCO_GL=osmesa)
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Stereo camera path, SLAM, C++ build tools, Claude planner client
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-rtabmap-ros ros-jazzy-pointcloud-to-laserscan \
      ros-jazzy-spatio-temporal-voxel-layer ros-jazzy-slam-toolbox \
      build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages "anthropic>=2.6,<3" "pydantic>=2.7,<3" "numpy<2"

# ros2_control and the arm controllers: controller manager and controllers, Pinocchio dynamics, yaml-cpp,
# SciPy for the calibration toolkit, the mock hardware components used without drives, and the upstream MuJoCo
# hardware plugin (mujoco_ros2_control) for the sim stack's backend:=ros2_control, and the VRPN motion-capture
# client (vrpn_mocap) for the arm calibrations.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-mujoco-ros2-control ros-jazzy-vrpn-mocap \
      ros-jazzy-pinocchio ros-jazzy-realtime-tools \
      libyaml-cpp-dev python3-scipy \
    && rm -rf /var/lib/apt/lists/*

# Source ROS and the workspace (once built) in every shell
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc && \
    echo '[ -f /hsim/ros2_ws/install/setup.bash ] && source /hsim/ros2_ws/install/setup.bash' >> /root/.bashrc

WORKDIR /hsim/ros2_ws
CMD ["sleep", "infinity"]
