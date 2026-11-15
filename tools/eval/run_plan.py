"""Send a hand-written skill plan to the behavior tree (no LLM) and print progress + result.

    python3 tools/eval/run_plan.py "navigate_to:table" "find_object:red block" "pick:red block"
    skills: navigate_to:X  find_object:X  search_for:X[@hint1;hint2]  look_around  pick:X  place:X  say:text
"""
import sys, time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from hsim_interfaces.action import ExecutePlan
from hsim_interfaces.msg import Skill
rclpy.init(); n = Node("plan_tester")
cli = ActionClient(n, ExecutePlan, "/behavior/execute_plan")
if not cli.wait_for_server(timeout_sec=20): print("behavior tree not running"); sys.exit(1)
g = ExecutePlan.Goal(); g.plan.task = " then ".join(sys.argv[1:])
for a in sys.argv[1:]:
    name, _, rest = a.partition(":"); s = Skill(); s.name = name
    if name == "say": s.text = rest
    elif "@" in rest: s.target, hints = rest.split("@", 1); s.hints = hints.split(";")
    else: s.target = rest
    g.plan.skills.append(s)
last = [None]
def fb(m):
    f = m.feedback
    if (f.current_skill_index, f.status) != last[0]:
        last[0] = (f.current_skill_index, f.status); print(f"  [{time.monotonic()-t0:5.1f}s] step {f.current_skill_index}: {f.status}")
t0 = time.monotonic()
fut = cli.send_goal_async(g, feedback_callback=fb); rclpy.spin_until_future_complete(n, fut, timeout_sec=10)
res = fut.result().get_result_async(); rclpy.spin_until_future_complete(n, res, timeout_sec=240)
r = res.result().result
print(f"RESULT after {time.monotonic()-t0:.0f}s: {'SUCCESS' if r.success else 'FAILED at step ' + str(r.failed_skill_index)}: {r.message}")
