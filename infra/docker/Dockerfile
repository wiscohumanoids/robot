# ROS 2 Jazzy dev environment for humanoid-sim-stack.
# Runs on the Mac (arm64, via Docker Desktop) and on any Linux x86_64 host (a future CPU VM).
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
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

# Same MuJoCo as training (mjlab 1.6.0 on the GPU VM). numpy stays < 2 so ROS's
# own compiled Python packages (built against Ubuntu's numpy 1.26) keep working.
RUN pip3 install --no-cache-dir --break-system-packages \
      "mujoco==3.11.0" "onnxruntime>=1.20" "numpy<2"

# Software OpenGL so the sim node can render camera images without a GPU or display
# (Docker on macOS). Selected with MUJOCO_GL=osmesa in compose.yaml.
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Phase 3: stereo (ZED X style) sensor path, BT build deps, Claude planner.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-rtabmap-ros ros-jazzy-pointcloud-to-laserscan \
      ros-jazzy-spatio-temporal-voxel-layer ros-jazzy-slam-toolbox \
      build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages "anthropic>=2.6,<3" "pydantic>=2.7,<3" "numpy<2"

# Source ROS and the workspace (once built) in every shell.
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc && \
    echo '[ -f /hsim/ros2_ws/install/setup.bash ] && source /hsim/ros2_ws/install/setup.bash' >> /root/.bashrc

WORKDIR /hsim/ros2_ws
CMD ["sleep", "infinity"]
