"""behavior_tree / bt_node

The ROS wrapper. All decisions live in bt_logic.py (flow) and skills.py (the
skills); this file only moves messages and action calls in and out.

INPUT:
  humanoid_interfaces/SkillSequence on /skill_sequence (from task_planner).
  Plans are queued (FIFO) and run one after another.

ACTION CLIENTS (it calls, it does not serve):
  - navigate_to_pose     nav2_msgs/NavigateToPose                  skill "navigate_to"
  - execute_manipulation humanoid_interfaces/ExecuteManipulation   pick, place, handover

OUTPUT (this node is the only publisher):
  std_msgs/String on /task_status, e.g. "IDLE", "RUNNING 2/3 pick",
  "RUNNING 1/2 navigate_to: attempt 2/3", "SUCCEEDED",
  "FAILED 1/2 navigate_to: timed out after 60 s (attempt 3/3)".
  Published on every change and re-sent at 1 Hz as a heartbeat.
"""
import threading

import rclpy
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from humanoid_interfaces.action import ExecuteManipulation
from humanoid_interfaces.msg import SkillSequence

from behavior_tree.action_call import call_action
from behavior_tree.bt_logic import BehaviorRunner
from behavior_tree.skills import Skill


class RosSkillContext:
    """The real SkillContext: the two action clients (see skills.SkillContext)."""

    def __init__(self, node):
        self._nav = ActionClient(node, NavigateToPose, 'navigate_to_pose')
        self._manip = ActionClient(node, ExecuteManipulation, 'execute_manipulation')

    def navigate(self, pose, timeout_s):
        goal = NavigateToPose.Goal()
        goal.pose = pose
        return call_action(self._nav, goal, timeout_s)

    def manipulate(self, task, object_id, pose, timeout_s):
        goal = ExecuteManipulation.Goal()
        goal.task = task
        goal.object_id = object_id
        goal.target_pose = pose
        return call_action(self._manip, goal, timeout_s, manipulation=True)


class BehaviorTreeNode(Node):

    def __init__(self):
        super().__init__('behavior_tree')
        self._status_pub = self.create_publisher(String, '/task_status', 10)
        self._runner = BehaviorRunner(
            RosSkillContext(self),
            publish_status=lambda text: self._status_pub.publish(String(data=text)),
            log=lambda msg: self.get_logger().info(msg))
        self.create_subscription(SkillSequence, '/skill_sequence', self._on_sequence, 10)
        self.create_timer(1.0, lambda: self._status_pub.publish(String(data=self._runner.status)))

        # The flow blocks on action results, so it runs on its own thread and
        # leaves the executor free to service the action clients' callbacks.
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._runner.run_forever, args=(self._stop,), daemon=True)
        self._worker.start()
        self.get_logger().info('behavior_tree up: queued plans, 3 attempts x 60 s per skill')

    def _on_sequence(self, msg: SkillSequence):
        plan = [Skill(s.name, s.object_id, s.target_pose) for s in msg.skills]
        if self._runner.submit(plan, msg.plan_id):
            self.get_logger().info(f'plan queued ({len(plan)} skills, {len(self._runner.queue)} waiting)')

    def shutdown(self):
        self._stop.set()
        self._worker.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorTreeNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
