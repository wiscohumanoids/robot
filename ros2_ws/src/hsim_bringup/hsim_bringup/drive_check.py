"""End-to-end drive test through the real command path.

Acts as the teleop node (node name `teleop`): sends 0.5 m/s forward on
/cmd_vel_teleop for 2 s, then stops. Passes if the robot moved forward
(via twist_mux -> /cmd_vel -> sim -> state estimation -> /odometry/filtered)
and came to a stop once commands ended (twist_mux timeout).

    ros2 run hsim_bringup drive_check
"""

import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

SPEED = 0.5
DRIVE_S = 2.0


def main():
    """Drive the robot forward through /cmd_vel_teleop, then check it moved and that it stops once commands stop."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--stop-tolerance', type=float, default=0.02,
                    help='max drift (m) in the 1 s after commands stop; a walking robot needs more than the stub')
    ap.add_argument('--settle', type=float, default=3.0, help='seconds to wait before driving')
    args = ap.parse_args()
    rclpy.init()
    node = Node('teleop')
    pub = node.create_publisher(TwistStamped, '/cmd_vel_teleop', 10)
    state = {'x': None, 'y': None, 'yaw': None}

    def on_odom(m):
        q = m.pose.pose.orientation
        state['x'], state['y'] = m.pose.pose.position.x, m.pose.pose.position.y
        state['yaw'] = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    node.create_subscription(Odometry, '/odometry/filtered', on_odom, 10)

    def spin(seconds, publish=False):
        end = time.monotonic() + seconds
        next_pub = 0.0
        while time.monotonic() < end:
            if publish and time.monotonic() >= next_pub:
                msg = TwistStamped()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.header.frame_id = 'base_link'
                msg.twist.linear.x = SPEED
                pub.publish(msg)
                next_pub = time.monotonic() + 0.02
            rclpy.spin_once(node, timeout_sec=0.005)

    end = time.monotonic() + 10.0            # wait for the estimator to come up (slower under load)
    while state['x'] is None and time.monotonic() < end:
        spin(0.1)
    spin(args.settle)
    if state['x'] is None:
        print('FAIL: no /odometry/filtered received')
        return 1
    # "Forward" = the robot's starting heading (an environment may spawn it facing any way)
    x0, y0, yaw0 = state['x'], state['y'], state['yaw']
    spin(DRIVE_S, publish=True)
    spin(1.0)                      # let twist_mux time out
    x1, y1 = state['x'], state['y']
    spin(1.0)
    x2, y2 = state['x'], state['y']

    moved = (x1 - x0) * math.cos(yaw0) + (y1 - y0) * math.sin(yaw0)
    drift = math.hypot(x2 - x1, y2 - y1)
    ok_move = moved > 0.5 * SPEED * DRIVE_S
    ok_stop = drift < args.stop_tolerance
    print(f'moved {moved:.2f} m forward in {DRIVE_S:.0f}s at {SPEED} m/s commanded -> {"ok" if ok_move else "FAIL"}')
    print(f'after commands stopped: drifted {drift:.3f} m in 1s -> {"ok" if ok_stop else "FAIL"}')
    node.destroy_node()
    rclpy.try_shutdown()
    print('PASS' if ok_move and ok_stop else 'FAIL')
    return 0 if ok_move and ok_stop else 1


if __name__ == '__main__':
    sys.exit(main())
