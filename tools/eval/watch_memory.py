"""Debug: every 2 s print the table entry, the robot's own pose estimate, and its true pose."""
import json, math, time, sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
rclpy.init(); n = Node("mem_watch"); st = {}
n.create_subscription(String, "/world/summary", lambda m: st.__setitem__("s", json.loads(m.data)), 5)
n.create_subscription(Odometry, "/sim/ground_truth/odom", lambda m: st.__setitem__("t", m.pose.pose), 10)
t0 = time.monotonic()
while time.monotonic() - t0 < float(sys.argv[1]):
    e = time.monotonic() + 2
    while time.monotonic() < e: rclpy.spin_once(n, timeout_sec=0.05)
    s, tp = st.get("s"), st.get("t")
    if not s or not tp: continue
    tabs = [(o["id"], o["position"][:2], o["times_seen"]) for o in s["objects"] if o["label"] == "table"]
    q = tp.orientation; ty = math.degrees(math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))
    print(f"t={time.monotonic()-t0:4.0f}s robot est {s['robot']} | true ({tp.position.x:.2f},{tp.position.y:.2f}, {ty:.0f} deg) | tables {tabs}", flush=True)
