"""Save one color + depth frame from the head camera, and summarize the depth-derived scan."""
import time, numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
rclpy.init(); n = Node("cam_probe"); got = {}
n.create_subscription(Image, "/camera/head/depth/image_raw", lambda m: got.__setitem__("d", m), 2)
n.create_subscription(Image, "/camera/head/color/image_raw", lambda m: got.__setitem__("c", m), 2)
n.create_subscription(LaserScan, "/scan", lambda m: got.__setitem__("s", m), 2)
t = time.time()
while len(got) < 3 and time.time() - t < 8: rclpy.spin_once(n, timeout_sec=0.1)
d = np.frombuffer(got["d"].data, np.float32).reshape(got["d"].height, got["d"].width)
c = np.frombuffer(got["c"].data, np.uint8).reshape(got["c"].height, got["c"].width, 3)
s = got["s"]; r = np.array(s.ranges); fin = np.isfinite(r)
print(f"depth {d.shape[1]}x{d.shape[0]}: {np.isfinite(d).mean():.0%} pixels have depth, median {np.nanmedian(d):.2f} m")
print(f"scan from depth: {len(r)} beams over {np.degrees(s.angle_max - s.angle_min):.0f} deg, {fin.mean():.0%} hit, nearest {r[fin].min() if fin.any() else float('nan'):.2f} m")
np.save("/hsim/.cam_rgb.npy", c); np.save("/hsim/.cam_depth.npy", d)
