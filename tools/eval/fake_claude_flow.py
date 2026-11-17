"""End-to-end planner test WITHOUT the API: a scripted stand-in for Claude drives the real planner
node logic, behavior tree, Nav2 and the robot. Launch the stack with planner:=false first."""
import json, sys, time, types
import rclpy
from hsim_planner.planner_node import Planner

SCRIPT = [
    # 1) a bad plan: go to an object the robot has never seen -> the planner's check must reject it
    {'understood': True, 'reasoning': 'go straight to the block', 'reply': '', 'skills': [
        {'name': 'navigate_to', 'target': 'red block', 'hints': [], 'text': ''}]},
    # 2) a good plan: go to the table, find the block there, approach it, try to pick it
    {'understood': True, 'reasoning': 'the block should be on the table', 'reply': 'Going to the table to look for the red block.', 'skills': [
        {'name': 'navigate_to', 'target': 'table', 'hints': [], 'text': ''},
        {'name': 'find_object', 'target': 'red block', 'hints': [], 'text': ''},
        {'name': 'navigate_to', 'target': 'red block', 'hints': [], 'text': ''},
        {'name': 'pick', 'target': 'red block', 'hints': [], 'text': ''}]},
    # 3) after pick fails: explain instead of retrying
    {'understood': False, 'reasoning': 'manipulation unavailable', 'reply': "I'm next to the red block, but I can't pick things up yet.", 'skills': []},
]
calls = []

class FakeMessages:
    """Stand-in for the Claude messages API that returns scripted plans."""
    def create(self, **kw):
        """Record the request and return the next scripted plan."""
        calls.append(kw['messages'][0]['content'])
        plan = SCRIPT[min(len(calls) - 1, len(SCRIPT) - 1)]
        return types.SimpleNamespace(stop_reason='end_turn', content=[types.SimpleNamespace(type='text', text=json.dumps(plan))])

client = types.SimpleNamespace(beta=types.SimpleNamespace(messages=FakeMessages()))
rclpy.init()
node = Planner(client=client)
replies = []
orig = node.reply
node.reply = lambda t: (replies.append(t), orig(t))
t0 = time.monotonic()
while time.monotonic() - t0 < 4: rclpy.spin_once(node, timeout_sec=0.1)       # get memory first
from std_msgs.msg import String
node.on_command(String(data='pick up the red block'))
while node.task is not None and time.monotonic() - t0 < 200: rclpy.spin_once(node, timeout_sec=0.05)
print(f"Claude was called {len(calls)} times")
for i, c in enumerate(calls):
    fail = [l for l in c.splitlines() if l.startswith('It failed')]
    print(f"  call {i+1}: told about {len(fail)} earlier failure(s)" + (f": {fail[-1][:110]}" if fail else ''))
print("replies to the person:", replies)
ok = len(calls) == 3 and "can't pick" in replies[-1]
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
