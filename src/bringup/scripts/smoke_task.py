#!/usr/bin/env python3
"""End-to-end scenario against a RUNNING stack: type an intent, expect the whole
task layer to carry it out.

    ros2 launch bringup full_stack.launch.py sim:=false &
    ros2 run bringup smoke_task.py [--intent "pick up the cube"] [--timeout 90]

Flow exercised (all stubs today; must still pass when a stub is replaced by the
real node, that is the point):
  /user_intent -> task_planner -> /skill_sequence -> behavior_tree
    -> navigate_to_pose (nav -> /cmd_vel_nav -> cmd_vel_mux -> /cmd_vel
       -> state_estimation -> /robot_pose)
    -> execute_manipulation (manipulation -> /manipulation/joint_targets -> wbc)
  and back out as /task_status == "SUCCEEDED".

Exit 0 on SUCCEEDED, 1 on FAILED or timeout. Acts as the (temporary) publisher
of /user_intent, so don't run it while another node is publishing there.
"""
import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Smoke(Node):

    def __init__(self):
        super().__init__('smoke_task')
        self.statuses = []
        self._pub = self.create_publisher(String, '/user_intent', 10)
        self.create_subscription(String, '/task_status', self._on_status, 10)

    def _on_status(self, msg):
        if not self.statuses or self.statuses[-1] != msg.data:
            self.statuses.append(msg.data)
            print(f'  /task_status: {msg.data}', flush=True)

    def wait_for_planner(self, timeout_s):
        end = time.monotonic() + timeout_s
        while self._pub.get_subscription_count() == 0:
            if time.monotonic() > end:
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return True

    def say(self, text):
        self._pub.publish(String(data=text))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--intent', default='pick up the cube')
    ap.add_argument('--timeout', type=float, default=90.0)
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = Smoke()
    code = 1
    try:
        if not node.wait_for_planner(20.0):
            print('FAIL: nobody subscribes to /user_intent -- is the stack (task_planner) running?')
            return 1
        # perception must have reported an object before the planner can plan; the
        # planner logs and drops an intent that arrives too early, so retry gently.
        deadline = time.monotonic() + args.timeout
        next_send = 0.0
        print(f'intent: "{args.intent}"', flush=True)
        while time.monotonic() < deadline:
            if not any(s.startswith('RUNNING') or s.startswith('SUCCEEDED') for s in node.statuses):
                if time.monotonic() >= next_send:
                    node.say(args.intent)
                    next_send = time.monotonic() + 3.0
            rclpy.spin_once(node, timeout_sec=0.1)
            if any(s.startswith('SUCCEEDED') for s in node.statuses):
                print('PASS: task completed')
                code = 0
                break
            if any(s.startswith('FAILED') for s in node.statuses):
                print('FAIL: behavior tree reported failure')
                break
        else:
            print(f'FAIL: timed out after {args.timeout:.0f}s; statuses seen: {node.statuses}')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
