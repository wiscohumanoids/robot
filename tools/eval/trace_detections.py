"""Debug: for the 'table' detection, show camera-frame measurement, where world_model would put it,
and the estimated vs true robot pose, while the robot walks forward."""
import math, os, time
import numpy as np, rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection3DArray
from tf2_ros import Buffer, TransformListener
def qm(x, y, z, w):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
rclpy.init(); n = Node("teleop"); buf = Buffer(); TransformListener(buf, n, spin_thread=True)
st = {}
n.create_subscription(Odometry, "/sim/ground_truth/odom", lambda m: st.__setitem__("truth", m.pose.pose), 10)
n.create_subscription(Detection3DArray, "/perception/detections", lambda m: st.__setitem__("det", m), 5)
pub = n.create_publisher(TwistStamped, "/cmd_vel_teleop", 10)
t0 = time.monotonic()
while not buf.can_transform("map", "head_camera", Time()) and time.monotonic() - t0 < 20:
    rclpy.spin_once(n, timeout_sec=0.1)
t0 = time.monotonic(); last = 0
while time.monotonic() - t0 < 9:
    rclpy.spin_once(n, timeout_sec=0.02)
    if time.monotonic() - t0 > 2:
        m = TwistStamped(); m.header.stamp = n.get_clock().now().to_msg(); m.twist.linear.x = 0.5; pub.publish(m)
    if time.monotonic() - last < 1.0 or "det" not in st: continue
    last = time.monotonic(); d = st["det"]
    tab = [x for x in d.detections if x.results[0].hypothesis.class_id == "table"]
    if not tab: continue
    c = tab[0].bbox.center.position; rel = np.array([c.x, c.y, c.z])
    try:
        tf = buf.lookup_transform("map", "head_camera", Time.from_msg(d.header.stamp)); how = "exact"
    except Exception:
        tf = buf.lookup_transform("map", "head_camera", Time()); how = "latest"
    tr, q = tf.transform.translation, tf.transform.rotation
    pos = np.array([tr.x, tr.y, tr.z]) + qm(q.x, q.y, q.z, q.w) @ rel
    tb = st["truth"].position
    det_t = d.header.stamp.sec + d.header.stamp.nanosec * 1e-9; tf_t = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
    print(f"t={time.monotonic()-t0:4.1f} rel=({rel[0]:.2f},{rel[1]:.2f},{rel[2]:.2f}) cam_in_map=({tr.x:.2f},{tr.y:.2f},{tr.z:.2f}) [{how}, det stamp {det_t:.2f} tf stamp {tf_t:.2f}] "
          f"-> table at ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) | true pelvis ({tb.x:.2f},{tb.y:.2f},{tb.z:.2f})")
import sys; sys.stdout.flush(); os._exit(0)
