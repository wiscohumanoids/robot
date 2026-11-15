"""Talk to the robot (node name: `ask`):  ros2 run hsim_planner ask "walk to the table"

Publishes the request on /planner/command, then prints what the robot says and does
until the planner reports done/failed (or 5 minutes pass). `ask stop` cancels the task.
"""
import json
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main():
    text = ' '.join(sys.argv[1:]).strip()
    if not text:
        print('usage: ros2 run hsim_planner ask "<request>"   (or: ask stop)')
        return 1
    rclpy.init()
    n = Node('ask')
    pub = n.create_publisher(String, '/planner/command', 10)
    state = {'done': False}

    def on_reply(m):
        print(f'robot: {m.data}', flush=True)

    def on_status(m):
        s = json.loads(m.data)
        if s['state'] == 'executing':
            steps = ' -> '.join(f"{k['name']}({k['target'] or k['text']})" for k in s.get('plan', []))
            print(f'  [plan, attempt {s.get("attempt")}] {steps}', flush=True)
        elif s['state'] == 'planning':
            print(f'  [thinking, attempt {s.get("attempt")}]', flush=True)
        elif s['state'] in ('done', 'failed', 'idle', 'error'):
            state['done'] = True

    n.create_subscription(String, '/planner/reply', on_reply, 10)
    n.create_subscription(String, '/planner/status', on_status, 10)
    n.create_subscription(String, '/robot/say', lambda m: print(f'robot says: {m.data}', flush=True), 10)
    t0 = time.monotonic()
    while pub.get_subscription_count() == 0 and time.monotonic() - t0 < 5:
        rclpy.spin_once(n, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print('no planner is running. Start the full stack first (inside tools/ros.sh shell):\n'
              '  ros2 launch hsim_bringup autonomy.launch.py env:=test/table_room\n'
              '(sim_stack.launch.py is the robot only, without the planner)')
        n.destroy_node()
        rclpy.try_shutdown()
        return 1
    pub.publish(String(data=text))
    end = time.monotonic() + 300
    try:
        while not state['done'] and time.monotonic() < end:
            rclpy.spin_once(n, timeout_sec=0.1)
    except KeyboardInterrupt:
        print('stopped watching (the robot keeps going; tools/ask.sh stop cancels the task)')
        return 0
    for _ in range(5):
        rclpy.spin_once(n, timeout_sec=0.1)
    n.destroy_node()
    rclpy.try_shutdown()
    return 0
