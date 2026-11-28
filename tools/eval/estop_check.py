"""GRADER: the e-stop must stop the robot even while commands keep coming, and release must work.

Keeps sending 'walk forward 0.5 m/s' as teleop the whole time; engages the operator e-stop
after 5 s, releases it after 9 s. Measures true speed (privileged) in each window.
"""
import sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
rclpy.init()
tele, op = Node("teleop"), Node("operator")
pub = tele.create_publisher(TwistStamped, "/cmd_vel_teleop", 10)
req = op.create_publisher(Bool, "/safety/estop_request", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
st = {}
tele.create_subscription(Odometry, "/sim/ground_truth/odom", lambda m: st.__setitem__("v", m.twist.twist.linear.x), 10)
tele.create_subscription(Bool, "/safety/estop", lambda m: st.__setitem__("estop", m.data), 10)
speeds = {"before": [], "estop": [], "released": []}
t0 = time.monotonic(); nxt = 0; engaged = released = False
while time.monotonic() - t0 < 15:
    t = time.monotonic() - t0
    if t >= nxt:
        m = TwistStamped(); m.header.stamp = tele.get_clock().now().to_msg(); m.twist.linear.x = 0.5; pub.publish(m); nxt = t + 0.02
    if t > 5 and not engaged: req.publish(Bool(data=True)); engaged = True
    if t > 9 and not released: req.publish(Bool(data=False)); released = True
    rclpy.spin_once(tele, timeout_sec=0.005); rclpy.spin_once(op, timeout_sec=0.0)
    if "v" in st:
        if 3 < t < 5: speeds["before"].append(st["v"])
        elif 6.5 < t < 9: speeds["estop"].append(st["v"])
        elif 11.5 < t < 15: speeds["released"].append(st["v"])
avg = {k: sum(v) / max(len(v), 1) for k, v in speeds.items()}
ok = avg["before"] > 0.3 and abs(avg["estop"]) < 0.08 and avg["released"] > 0.3
print(f"true forward speed: before {avg['before']:.2f} m/s | e-stopped {avg['estop']:.2f} m/s | released {avg['released']:.2f} m/s "
      f"(commands kept coming the whole time) -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
