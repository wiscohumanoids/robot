#!/usr/bin/env bash
# Usage (in the container): ROBOT=wisco|g1|g1_arms_free [SENSORS=lidar|stereo] [ENV=robocasa/kitchen_l1_s1] bash /hsim/tools/eval/e2e_realistic.sh
# Needs the perception tunnel on the Mac: ssh -f -N -L 8100:localhost:8100 hsim-gpu
# end-to-end: SLAM (rtabmap) + real perception (VM) + BT + locomotion, no prebuilt map, no privileged perception
set +u
source /opt/ros/jazzy/setup.bash; source /hsim/ros2_ws/install/setup.bash
R=${ROBOT:-wisco}; S=${SENSORS:-}; ENV=${ENV:-robocasa/kitchen_l1_s1}
export HSIM_BUILD_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('/hsim/ros2_ws/src/hsim_bringup/config/robots/$R.yaml'))['env_build_dir'])")
LOG=/tmp/e2e_${R}_${S:-default}.log
setsid ros2 launch hsim_bringup autonomy.launch.py viewer:=false planner:=false robot:=$R ${S:+sensors:=$S} env:=$ENV > $LOG 2>&1 &
L=$!; sleep 35
timeout 20 python3 - <<'PY'
import rclpy, time, json
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
rclpy.init(); n=Node("e2e_probe"); got={}
lat=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
n.create_subscription(OccupancyGrid,"/map",lambda m: got.__setitem__("map",m),lat)
n.create_subscription(String,"/perception/status",lambda m: got.__setitem__("pst",m.data),5)
n.create_subscription(Detection3DArray,"/world/objects",lambda m: got.__setitem__("obj",m),5)
t0=time.time()
while time.time()-t0<12: rclpy.spin_once(n,timeout_sec=0.3)
m=got.get("map"); print("MAP:", f"{m.info.width}x{m.info.height} @ {m.info.resolution:.2f} ({sum(1 for v in m.data if v>=0)} known cells)" if m else "none")
print("PERCEPTION:", (got.get("pst") or "no status")[:160])
o=got.get("obj"); print("OBJECTS:", sorted({d.results[0].hypothesis.class_id for d in o.detections}) if o else "none")
PY
TARGET=${TARGET:-sink}; OBJ=${OBJ:-sink_main_group}
# track the TRUE path while the task runs: did the robot ever leave the room's floor (env geometry map)?
timeout 200 python3 - "$ENV" > /tmp/e2e_track.txt 2>&1 <<'PY' &
import sys, time, os, yaml, numpy as np, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
env = sys.argv[1]; b = f"/hsim/assets/envs/{env}/{os.environ.get('HSIM_BUILD_DIR', 'build')}"
meta = yaml.safe_load(open(f"{b}/map.yaml")); raw = open(f"{b}/map.pgm", "rb").read()
parts = raw.split(b"\n", 3); w, h = map(int, parts[1].split()); img = np.frombuffer(parts[3][:w * h], np.uint8).reshape(h, w)
res, ox, oy = meta["resolution"], meta["origin"][0], meta["origin"][1]
rclpy.init(); n = Node("e2e_track"); out = {"n": 0, "outside": 0, "in_wall": 0, "path": 0.0, "last": None}
def cb(m):
    p = m.pose.pose.position; out["n"] += 1
    if out["last"] is not None: out["path"] += float(np.hypot(p.x - out["last"][0], p.y - out["last"][1]))
    out["last"] = (p.x, p.y)
    c, r = int((p.x - ox) / res), h - 1 - int((p.y - oy) / res)
    if not (0 <= r < h and 0 <= c < w): out["outside"] += 1
    elif img[r, c] < 50: out["in_wall"] += 1
n.create_subscription(Odometry, "/sim/ground_truth/odom", cb, 10)
t0 = time.time()
while time.time() - t0 < 190 and not os.path.exists("/tmp/e2e_stop"): rclpy.spin_once(n, timeout_sec=0.2)
print(f"TRACK: walked {out['path']:.1f} m, samples off the room's map {out['outside']}, inside walls/furniture {out['in_wall']}")
PY
TR=$!; rm -f /tmp/e2e_stop
echo "PLAN: $(timeout 180 python3 /hsim/tools/eval/run_plan.py look_around navigate_to:$TARGET 2>&1 | tail -1)"
touch /tmp/e2e_stop; wait $TR; cat /tmp/e2e_track.txt | grep TRACK
echo "GRADE: $(timeout 20 python3 /hsim/tools/eval/reach_check.py $ENV $OBJ 2>&1 | tail -1)"
echo "NAV: goals $(grep -c 'Begin navigating\|Received goal' $LOG), path replans $(grep -c 'Received goal preemption' $LOG), staging moves $(grep -c 'staging spot moved' $LOG), plan failures $(grep -c 'failed to plan' $LOG)"
echo "FALLS: $(grep -c 'robot fell' $LOG)  TF map->odom from: $(grep -oE '\[(localization|rtabmap)[^]]*\]' $LOG | head -1)"
kill -TERM -- -$L; sleep 3; kill -KILL -- -$L 2>/dev/null; rm -f /dev/shm/fastrtps_* 2>/dev/null
