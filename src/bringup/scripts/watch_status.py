#!/usr/bin/env python3
"""Stand-in for the user-facing feedback: prints what the task layer is doing.

    ros2 run bringup watch_status.py

Prints every change of /planner_status (what happened to each typed intent) and of
/task_status (what the behavior tree is executing), one line each, in order. Only
prints; publishes nothing. Replace with the real UI / speech output later.
"""
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Watch(Node):

    def __init__(self):
        super().__init__('watch_status')
        self._start = time.monotonic()
        self._last = {}
        self.create_subscription(String, '/planner_status', lambda m: self._show('planner', m.data), 50)
        self.create_subscription(String, '/task_status', lambda m: self._show('tree   ', m.data), 50)
        print('watching /planner_status and /task_status (Ctrl-C to stop)', flush=True)

    def _show(self, who, text):
        # /task_status repeats itself at 1 Hz; only print when it changes.
        if who.startswith('tree') and self._last.get(who) == text:
            return
        self._last[who] = text
        print(f'[{time.monotonic() - self._start:7.1f}s] {who}: {text}', flush=True)


def main():
    rclpy.init()
    node = Watch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
