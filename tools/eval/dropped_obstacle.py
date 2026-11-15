"""GRADER: the "someone drops a box in the way" scenario.

Sends a Nav2 goal, waits until the robot is walking, then drops a box (via the test-only
/sim/set_body_pose) on the straight line between the robot and the goal, ~1.3 m ahead.
The robot must notice it with its own lidar, go around, reach the goal, and never touch it.

    python3 tools/eval/dropped_obstacle.py GOAL_X GOAL_Y   (map frame = world frame, mapping:=map)
"""
import math, sys, time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from hsim_interfaces.srv import SetBodyPose

gx, gy = float(sys.argv[1]), float(sys.argv[2])
BOX, BOX_HALF = "room/obstacle_box", 0.2
rclpy.init(); n = Node("obstacle_grader")
truth = {}
n.create_subscription(Odometry, "/sim/ground_truth/odom", lambda m: truth.__setitem__("p", m.pose.pose.position), 10)
nav = ActionClient(n, NavigateToPose, "/navigate_to_pose"); setp = n.create_client(SetBodyPose, "/sim/set_body_pose")
nav.wait_for_server(timeout_sec=30); setp.wait_for_service(timeout_sec=10)
g = NavigateToPose.Goal(); g.pose.header.frame_id = "map"; g.pose.pose.position.x, g.pose.pose.position.y = gx, gy; g.pose.pose.orientation.w = 1.0
f = nav.send_goal_async(g); rclpy.spin_until_future_complete(n, f, timeout_sec=10); gh = f.result(); res = gh.get_result_async()
t0 = time.monotonic()
while time.monotonic() - t0 < 4.0: rclpy.spin_once(n, timeout_sec=0.05)       # let it start walking
p = truth["p"]; d = math.hypot(gx - p.x, gy - p.y); ux, uy = (gx - p.x) / d, (gy - p.y) / d
bx, by = p.x + 1.3 * ux, p.y + 1.3 * uy
req = SetBodyPose.Request(); req.body = BOX; req.pose.position.x, req.pose.position.y, req.pose.position.z = bx, by, 0.2; req.pose.orientation.w = 1.0
rf = setp.call_async(req); rclpy.spin_until_future_complete(n, rf, timeout_sec=5)
print(f"dropped a 0.4 m box at ({bx:.2f}, {by:.2f}), {1.3} m ahead of the walking robot")
closest = 9.0
while not res.done() and time.monotonic() - t0 < 120:
    rclpy.spin_once(n, timeout_sec=0.05)
    q = truth["p"]; closest = min(closest, max(abs(q.x - bx), abs(q.y - by)) - BOX_HALF)
status = res.result().status if res.done() else None
q = truth["p"]; err = math.hypot(q.x - gx, q.y - gy)
# park the box again for the next test
req.pose.position.x, req.pose.position.y = 20.0, 20.0; rclpy.spin_until_future_complete(n, setp.call_async(req), timeout_sec=5)
ok = status == 4 and err < 0.5 and closest > 0.15
print(f"Nav2 {'SUCCEEDED' if status == 4 else f'status {status}'} after {time.monotonic() - t0:.0f} s; final error {err:.2f} m; "
      f"closest approach to the box (pelvis to box edge) {closest:.2f} m -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
