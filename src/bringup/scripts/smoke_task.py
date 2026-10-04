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

The intent is sent ONCE (the planner queues it and plans it when the tree is idle).
Only statuses that arrive after it was sent count, and a PASS needs a RUNNING first,
so a SUCCEEDED left over from an earlier run is never mistaken for this one.
Exit 0 on SUCCEEDED, 1 if the planner says NO_PLAN / REJECTED / GAVE_UP, or on
timeout. Acts as the (temporary) publisher of /user_intent, so don't run it while
another node is publishing there.

The hand must be empty for the default "pick up the cube": the manipulation stub keeps
holding the cube after a pick, so run it on a fresh stack (or --intent "put the cube down").
"""
import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

PLANNER_FAILURES = ('NO_PLAN', 'REJECTED', 'GAVE_UP')


class Smoke(Node):

    def __init__(self):
        super().__init__('smoke_task')
        self.statuses = []          # /task_status changes seen after the intent was sent
        self.planner_lines = []     # /planner_status lines seen after the intent was sent
        self.held = None            # /manipulation/held_object ('' = empty), None = not heard
        self.sent = False
        self._pub = self.create_publisher(String, '/user_intent', 10)
        self.create_subscription(String, '/task_status', self._on_status, 10)
        self.create_subscription(String, '/planner_status', self._on_planner, 50)
        self.create_subscription(String, '/manipulation/held_object', self._on_held, 10)

    def _on_status(self, msg):
        if self.sent and (not self.statuses or self.statuses[-1] != msg.data):
            self.statuses.append(msg.data)
            print(f'  /task_status: {msg.data}', flush=True)

    def _on_planner(self, msg):
        if self.sent:
            self.planner_lines.append(msg.data)
            print(f'  /planner_status: {msg.data}', flush=True)

    def _on_held(self, msg):
        self.held = msg.data

    def wait_for_planner(self, timeout_s):
        end = time.monotonic() + timeout_s
        while self._pub.get_subscription_count() == 0:
            if time.monotonic() > end:
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return True

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def say(self, text):
        self.sent = True
        self._pub.publish(String(data=text))

    def verdict(self):
        """('PASS'|'FAIL'|None, text): None = keep waiting."""
        for line in self.planner_lines:
            if line.startswith(PLANNER_FAILURES):
                return 'FAIL', f'planner: {line}'
        started = False
        for status in self.statuses:
            if status.startswith('RUNNING'):
                started = True
            elif started and status.startswith('SUCCEEDED'):
                return 'PASS', 'task completed'
            elif started and status.startswith('FAILED') and not self.planner_lines:
                return 'FAIL', f'behavior tree reported failure ({status}) and no planner is answering'
        return None, ''


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
            print('FAIL: nobody subscribes to /user_intent, is the stack (task_planner) running?')
            return 1
        node.spin_for(2.0)      # hear /manipulation/held_object and the tree's heartbeat
        if node.held:
            print(f'note: the hand already holds {node.held}; a "pick" will be refused '
                  f'(restart the stack, or run --intent "put the {node.held} down" first)', flush=True)
        print(f'intent: "{args.intent}"', flush=True)
        node.say(args.intent)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            result, text = node.verdict()
            if result:
                print(f'{result}: {text}')
                code = 0 if result == 'PASS' else 1
                break
        else:
            print(f'FAIL: timed out after {args.timeout:.0f}s; statuses seen: {node.statuses}')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
