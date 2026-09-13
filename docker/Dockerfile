# ROS2 Humble + MuJoCo dev container for the `robot` workspace.
#
# Pattern (MuJoCo prebuilt-binary download + MUJOCO_DIR, apt deps, rosdep,
# colcon build) adapted directly from the vendored mujoco_ros2_control
# package's own docker/Dockerfile (bipedal_nav/src/mujoco_ros2_control/docker/Dockerfile,
# MIT licensed) -- reused rather than reinvented, extended with this repo's
# additional dependencies (Eigen3, yaml-cpp, control_toolbox, effort
# controllers, PyYAML, onnxruntime for locomotion_runner's optional ONNX slot).
#
# This is the OPTIONAL containerized path -- the sim path also runs directly
# on a bare Ubuntu 22.04 + ROS2 Humble desktop install; see the top-level
# README.md "Prerequisites" for the native-install equivalent of everything
# below.

ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}

ARG MUJOCO_VERSION=3.2.7
ARG CPU_ARCH=x86_64

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    python3-pip wget \
    libglfw3-dev libeigen3-dev libyaml-cpp-dev \
    ros-${ROS_DISTRO}-control-toolbox \
    ros-${ROS_DISTRO}-effort-controllers \
    ros-${ROS_DISTRO}-joint-state-broadcaster \
    ros-${ROS_DISTRO}-robot-state-publisher \
    ros-${ROS_DISTRO}-xacro \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install xacro onnxruntime

# --- MuJoCo (prebuilt binary release, not built from source) ---
ENV MUJOCO_VERSION=${MUJOCO_VERSION}
ENV MUJOCO_DIR="/opt/mujoco/mujoco-${MUJOCO_VERSION}"
RUN mkdir -p ${MUJOCO_DIR} && \
    wget -q https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-${CPU_ARCH}.tar.gz && \
    tar -xzf "mujoco-${MUJOCO_VERSION}-linux-${CPU_ARCH}.tar.gz" -C $(dirname "${MUJOCO_DIR}") && \
    rm "mujoco-${MUJOCO_VERSION}-linux-${CPU_ARCH}.tar.gz"

# --- Workspace ---
ENV ROS_WS="/home/ros2_ws/"
RUN mkdir -p ${ROS_WS}/src
WORKDIR ${ROS_WS}
COPY src/ src/

RUN . /opt/ros/${ROS_DISTRO}/setup.bash && \
    rosdep update && \
    rosdep install --from-paths src/ --ignore-src -y

RUN . /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo

RUN echo "source ${ROS_WS}/install/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
