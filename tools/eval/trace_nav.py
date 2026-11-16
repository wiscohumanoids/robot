"""Debug trace: per 1 s, what Nav2 commands vs what the robot does (reads truth for grading)."""
import math, time, sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, Path
rclpy.init(); n = Node("nav_tracer"); st = {"cmd": [], "path": None}
n.create_subscription(TwistStamped, "/cmd_vel", lambda m: st["cmd"].append((m.twist.linear.x, m.twist.linear.y, m.twist.angular.z)), 50)
n.create_subscription(Odometry, "/sim/ground_truth/odom", lambda m: st.__setitem__("p", m.pose.pose), 10)
n.create_subscription(Path, "/plan", lambda m: st.__setitem__("path", m), 5)
end = time.monotonic() + float(sys.argv[1]); t0 = time.monotonic()
while time.monotonic() < end:
    s = time.monotonic() + 1.0
    while time.monotonic() < s: rclpy.spin_once(n, timeout_sec=0.02)
    c = st["cmd"]; st["cmd"] = []
    p = st.get("p"); path = st["path"]
    pts = [(q.pose.position.x, q.pose.position.y) for q in path.poses] if path else []
    near = min(pts, key=lambda q: abs(q[0] + 0.1)) if pts else None   # path point near x = -0.1 (box area)
    if p:
        print(f"t={time.monotonic()-t0:4.0f}s pos=({p.position.x:5.2f},{p.position.y:5.2f}) cmds={len(c):2d} "
              f"mean vx={sum(a for a,_,_ in c)/max(len(c),1):+.2f} wz={sum(w for *_,w in c)/max(len(c),1):+.2f} "
              f"| plan {len(pts)} pts, y at box x: {near[1] if near else float('nan'):+.2f}")
