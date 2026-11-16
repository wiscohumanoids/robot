"""GRADER: send Nav2 a goal (as the behavior tree will) and grade where the robot really ended up.

    python3 tools/eval/nav_goal.py X Y YAW_DEG [--frame-offset SPAWN_X SPAWN_Y SPAWN_YAW_DEG] [--timeout 90]

Goal is in the map frame. With SLAM the map frame starts at the robot's spawn pose, so pass
--frame-offset (the spawn pose) to grade against world coordinates.
"""
import argparse, math, time, sys
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry

ap = argparse.ArgumentParser()
ap.add_argument("x", type=float); ap.add_argument("y", type=float); ap.add_argument("yaw_deg", type=float)
ap.add_argument("--frame-offset", type=float, nargs=3, default=[0, 0, 0])
ap.add_argument("--timeout", type=float, default=90.0)
a = ap.parse_args()

rclpy.init(); n = Node("nav_grader")
truth = {}
n.create_subscription(Odometry, "/sim/ground_truth/odom", lambda m: truth.__setitem__("p", m.pose.pose), 10)
cli = ActionClient(n, NavigateToPose, "/navigate_to_pose")
if not cli.wait_for_server(timeout_sec=30):
    print("FAIL: Nav2 not available"); sys.exit(1)
g = NavigateToPose.Goal(); g.pose.header.frame_id = "map"; g.pose.header.stamp = n.get_clock().now().to_msg()
g.pose.pose.position.x, g.pose.pose.position.y = a.x, a.y
g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(math.radians(a.yaw_deg) / 2), math.cos(math.radians(a.yaw_deg) / 2)
t0 = time.monotonic()
fut = cli.send_goal_async(g); rclpy.spin_until_future_complete(n, fut, timeout_sec=10)
gh = fut.result()
if not gh or not gh.accepted:
    print("FAIL: goal rejected"); sys.exit(1)
res = gh.get_result_async(); rclpy.spin_until_future_complete(n, res, timeout_sec=a.timeout)
elapsed = time.monotonic() - t0
status = res.result().status if res.done() else None
for _ in range(20): rclpy.spin_once(n, timeout_sec=0.05)
# goal in world coordinates
ox, oy, oyaw = a.frame_offset[0], a.frame_offset[1], math.radians(a.frame_offset[2])
gx = ox + math.cos(oyaw) * a.x - math.sin(oyaw) * a.y; gy = oy + math.sin(oyaw) * a.x + math.cos(oyaw) * a.y
p = truth["p"]; err = math.hypot(p.position.x - gx, p.position.y - gy)
q = p.orientation; yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
yerr = math.degrees((yaw - (oyaw + math.radians(a.yaw_deg)) + math.pi) % (2 * math.pi) - math.pi)
ok = status == 4 and err < 0.5
print(f"Nav2 status {status} ({'SUCCEEDED' if status == 4 else 'not succeeded'}) in {elapsed:.0f} s; "
      f"true final position off by {err:.2f} m, heading off by {yerr:+.0f} deg -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
