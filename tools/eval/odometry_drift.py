"""GRADER (reads privileged ground truth): how far does the robot's own pose estimate drift?

Drives the robot through ROS as the `teleop` node (launch the stack with viewer:=false),
then compares /odometry/filtered (the robot's estimate, odom frame) with the simulator's
true pose (expressed relative to where the robot started).

    ros2 run ... / python3 tools/eval/odometry_drift.py
"""
import math, time, sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

def yaw(q):
    """Yaw of a ROS quaternion."""
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    """Angle wrapped to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


rclpy.init(); n = Node('teleop')
pub = n.create_publisher(TwistStamped, '/cmd_vel_teleop', 10)
st = {}
n.create_subscription(Odometry, '/odometry/filtered', lambda m: st.__setitem__('est', m), 10)
n.create_subscription(Odometry, '/sim/ground_truth/odom', lambda m: st.__setitem__('true', m), 10)

def spin(sec, cmd=None):
    """Spin for `sec`, publishing `cmd` at 50 Hz when given."""
    end, nxt = time.monotonic() + sec, 0
    while time.monotonic() < end:
        if cmd and time.monotonic() >= nxt:
            m = TwistStamped(); m.header.stamp = n.get_clock().now().to_msg(); m.header.frame_id = 'base_link'
            m.twist.linear.x, m.twist.linear.y, m.twist.angular.z = cmd; pub.publish(m); nxt = time.monotonic() + 0.02
        rclpy.spin_once(n, timeout_sec=0.005)

spin(3.0)
if 'est' not in st or 'true' not in st:
    print('FAIL: no odometry'); sys.exit(1)
t0 = st['true'].pose.pose; y0 = yaw(t0.orientation)
script = [(5.0, (0.5, 0.0, 0.0)), (3.0, (0.0, 0.0, 1.0)), (4.0, (0.5, 0.0, 0.3)), (2.0, None)]
dist = 0.0; prev = (t0.position.x, t0.position.y)
for sec, cmd in script:
    end = time.monotonic() + sec
    while time.monotonic() < end:
        spin(0.1, cmd)
        p = st['true'].pose.pose.position; dist += math.hypot(p.x - prev[0], p.y - prev[1]); prev = (p.x, p.y)
tp, ep = st['true'].pose.pose, st['est'].pose.pose
dx, dy = tp.position.x - t0.position.x, tp.position.y - t0.position.y
true_x = math.cos(-y0) * dx - math.sin(-y0) * dy; true_y = math.sin(-y0) * dx + math.cos(-y0) * dy
true_yaw = wrap(yaw(tp.orientation) - y0)
err = math.hypot(ep.position.x - true_x, ep.position.y - true_y)
yerr = math.degrees(wrap(yaw(ep.orientation) - true_yaw))
print(f"traveled {dist:.2f} m, turned {math.degrees(true_yaw):.0f} deg")
print(f"true (start frame) x={true_x:.2f} y={true_y:.2f} | estimate x={ep.position.x:.2f} y={ep.position.y:.2f}")
print(f"position drift {err:.2f} m ({100 * err / max(dist, 1e-6):.1f}% of distance), heading drift {yerr:+.1f} deg")
