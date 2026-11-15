"""Debug: for every dynamic TF link, how far its newest stamp trails the sim clock (seconds)."""
import time, rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
rclpy.init(); n = Node("tf_lag"); st = {"clock": 0.0, "lag": {}}
n.create_subscription(Clock, "/clock", lambda m: st.__setitem__("clock", m.clock.sec + m.clock.nanosec * 1e-9), 50)
def on_tf(m):
    for t in m.transforms:
        k = f"{t.header.frame_id} -> {t.child_frame_id}"
        st["lag"].setdefault(k, []).append(st["clock"] - (t.header.stamp.sec + t.header.stamp.nanosec * 1e-9))
n.create_subscription(TFMessage, "/tf", on_tf, 200)
end = time.monotonic() + 6
while time.monotonic() < end: rclpy.spin_once(n, timeout_sec=0.01)
for k, v in sorted(st["lag"].items()):
    v = v[len(v) // 3:]
    print(f"{k:34s} {len(v)/4:6.1f} msgs/s   lag behind sim clock: mean {sum(v)/len(v):+.3f} s, max {max(v):+.3f} s")
