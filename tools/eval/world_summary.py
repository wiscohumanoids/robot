"""Print the robot's object memory (/world/summary) once."""
import json, time, rclpy
from rclpy.node import Node
from std_msgs.msg import String
rclpy.init(); n = Node("summary_probe"); got = []
n.create_subscription(String, "/world/summary", lambda m: got.append(m.data), 5)
t = time.time()
while not got and time.time() - t < 5: rclpy.spin_once(n, timeout_sec=0.1)
d = json.loads(got[-1]); print("robot (its own estimate):", d["robot"])
for o in d["objects"]:
    print(f"  {o['id']:18s} at {o['position']}  conf {o['confidence']:.2f}  seen {o['times_seen']}x, {o['seen_ago_s']}s ago" + (f"  ON {o['on']}" if o['on'] else ""))
if not d["objects"]: print("  (nothing seen yet)")
