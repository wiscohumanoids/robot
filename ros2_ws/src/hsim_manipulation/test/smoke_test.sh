#!/usr/bin/env bash
# Build + run hsim_manipulation in a throwaway container (no user stack touched):
#   docker run --rm -v $REPO:/repo:ro hsim/ros-jazzy:dev bash /repo/ros2_ws/src/hsim_manipulation/test/smoke_test.sh
# Builds hsim_interfaces (+ the proposed Manipulate.action patch) and hsim_manipulation under /tmp/ws, starts
# the VLA server in --mock mode, fakes a camera + joint states, sends one goal, checks that arm targets come out.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
mkdir -p /tmp/ws/src && cp -r /repo/ros2_ws/src/hsim_interfaces /repo/ros2_ws/src/hsim_manipulation /tmp/ws/src/
patch -d /tmp/ws/src -p3 --forward < /repo/docs/runs/patches/F2_hsim_interfaces_manipulate.patch >/dev/null || true
cd /tmp/ws
grep -q Manipulate /tmp/ws/src/hsim_interfaces/CMakeLists.txt || { echo "patch did not apply"; exit 1; }
colcon build --base-paths /tmp/ws/src --build-base /tmp/ws/build --install-base /tmp/ws/install --event-handlers console_cohesion- 2>&1 | tail -3
source /tmp/ws/install/setup.bash
python3 -m pytest -q /tmp/ws/src/hsim_manipulation/test/test_vla_client.py 2>&1 | tail -2
python3 /repo/services/vla_server/server.py --mock --port 8200 > /tmp/vla.log 2>&1 &
ros2 run hsim_manipulation manipulation_node --ros-args -p vla_url:=http://127.0.0.1:8200 -p use_sim_time:=false -p max_duration_s:=${MAXD:-3.0} > /tmp/node.log 2>&1 &
python3 - <<'PY' &
import rclpy, numpy as np, time
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
rclpy.init(); n = Node('fake_robot')
pi, pj = n.create_publisher(Image, '/camera/head/color/image_raw', 1), n.create_publisher(JointState, '/joint_states', 10)
names = [f'{s}_{j}_joint' for s in ('left', 'right') for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
end = time.time() + 30
k = 0
img_every = max(1, int(20 / float(__import__('os').environ.get('FAKE_IMG_HZ', 20))))
while time.time() < end:
    k += 1
    im = Image(); H, W = int(__import__("os").environ.get("FAKE_H", 480)), int(__import__("os").environ.get("FAKE_W", 640)); im.height, im.width, im.encoding, im.step = H, W, "rgb8", W * 3; im.data = np.full((H, W, 3), 120, np.uint8).tobytes()
    (k % img_every == 0) and pi.publish(im); js = JointState(); js.name = names; js.position = [0.1] * 14; pj.publish(js); time.sleep(0.05)
PY
sleep 3
timeout 20 ros2 topic echo --once /arm/joint_targets > /tmp/targets.txt 2>&1 & ECHO=$!
ros2 action send_goal /skills/manipulate hsim_interfaces/action/Manipulate "{instruction: 'pick up the cup and put it on the plate', max_duration_s: ${MAXD:-3.0}}" 2>&1 | tail -8
wait $ECHO 2>/dev/null || true
echo "--- one /arm/joint_targets message:"; head -12 /tmp/targets.txt
grep -q left_shoulder_pitch_joint /tmp/targets.txt && echo "SMOKE PASS" || { echo "SMOKE FAIL"; cat /tmp/node.log; exit 1; }
