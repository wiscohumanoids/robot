# ROS2 Humble + MuJoCo dev container for the `robot` workspace.
#
# Works on x86_64 AND arm64 hosts (Apple-silicon Macs, ARM Linux): the MuJoCo
# release tarball is chosen from the build architecture below. MuJoCo is built
# with OSMesa support so the sim can run headless (`headless:=true`) with no
# display server -- the only option on macOS, in CI, and on remote machines.
#
# Pattern (MuJoCo prebuilt-binary download + MUJOCO_DIR, apt deps, rosdep,
# colcon build) adapted from the vendored mujoco_ros2_control package's own
# docker/Dockerfile (MIT licensed), extended with this repo's dependencies.
#
# The workspace is built with --symlink-install so that, when ./docker/run.sh
# bind-mounts your working copy over /home/ros2_ws/src, Python edits take effect
# without rebuilding (C++/message changes still need `colcon build`).
#
# This is the OPTIONAL containerized path -- everything also runs on a bare
# Ubuntu 22.04 + ROS2 Humble install; see README.md "Prerequisites".

ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}

ARG MUJOCO_VERSION=3.2.7

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    python3-pip python3-pytest wget \
    libglfw3-dev libosmesa6-dev libeigen3-dev libyaml-cpp-dev \
    ros-${ROS_DISTRO}-control-toolbox \
    ros-${ROS_DISTRO}-effort-controllers \
    ros-${ROS_DISTRO}-joint-state-broadcaster \
    ros-${ROS_DISTRO}-robot-state-publisher \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-nav2-msgs \
    ros-${ROS_DISTRO}-tf2-ros-py \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install xacro onnxruntime

# --- MuJoCo (prebuilt binary release, arch picked from the build platform) ---
ENV MUJOCO_VERSION=${MUJOCO_VERSION}
ENV MUJOCO_DIR="/opt/mujoco/mujoco-${MUJOCO_VERSION}"
RUN case "$(dpkg --print-architecture)" in \
      amd64) MJ_ARCH=x86_64 ;; \
      arm64) MJ_ARCH=aarch64 ;; \
      *) echo "unsupported architecture: $(dpkg --print-architecture)" >&2; exit 1 ;; \
    esac && \
    mkdir -p ${MUJOCO_DIR} && \
    wget -q https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-${MJ_ARCH}.tar.gz && \
    tar -xzf "mujoco-${MUJOCO_VERSION}-linux-${MJ_ARCH}.tar.gz" -C $(dirname "${MUJOCO_DIR}") && \
    rm "mujoco-${MUJOCO_VERSION}-linux-${MJ_ARCH}.tar.gz"

# --- Workspace ---
ENV ROS_WS="/home/ros2_ws/"
RUN mkdir -p ${ROS_WS}/src
WORKDIR ${ROS_WS}
COPY src/ src/

RUN . /opt/ros/${ROS_DISTRO}/setup.bash && \
    rosdep update && \
    rosdep install --from-paths src/ --ignore-src -y

RUN . /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMUJOCO_HEADLESS_OSMESA=ON

# Interactive shells you open later (`docker exec`) also get the workspace.
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source ${ROS_WS}/install/setup.bash" >> ~/.bashrc

COPY docker/robot_entrypoint.sh /robot_entrypoint.sh
ENTRYPOINT ["/robot_entrypoint.sh"]
CMD ["/bin/bash"]
