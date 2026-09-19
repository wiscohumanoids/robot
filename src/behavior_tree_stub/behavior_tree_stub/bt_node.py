"""behavior_tree_stub / bt_node

STATUS: STUB. The interface is real; the internals are not a behavior tree. It
runs a SkillSequence's skills one after another and stops at the first failure.
The real replacement is BehaviorTree.CPP with your own condition and action
nodes; it must keep the same inputs, outputs and action clients below.

INPUT:
  humanoid_interfaces/SkillSequence on /skill_sequence (from task_planner).
  A plan that arrives while one is running is ignored (logged), not queued.

ACTION CLIENTS (it calls, it does not serve):
  - navigate_to_pose    nav2_msgs/NavigateToPose            for skill "navigate_to"
  - execute_manipulation humanoid_interfaces/ExecuteManipulation  for every other skill

OUTPUT (this node is the only publisher):
  std_msgs/String on /task_status, e.g. "IDLE", "RUNNING 2/3 pick",
  "SUCCEEDED", "FAILED 1/2 navigate_to: goal rejected". Published on every
  change and re-sent at 1 Hz as a heartbeat.
"""
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from humanoid_interfaces.action import ExecuteManipulation
from humanoid_interfaces.msg import SkillSequence

from behavior_tree_stub.bt_logic import action_kind, format_status

SERVER_WAIT_S = 5.0
SKILL_TIMEOUT_S = 120.0
POLL_S = 0.05


class BehaviorTreeStub(Node):

    def __init__(self):
        super().__init__('behavior_tree')
        self._nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._manip = ActionClient(self, ExecuteManipulation, 'execute_manipulation')
        self._status = format_status('IDLE')
        self._busy = False
        self._lock = threading.Lock()

        self.create_subscription(SkillSequence, '/skill_sequence', self._on_sequence, 10)
        self._status_pub = self.create_publisher(String, '/task_status', 10)
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info('behavior_tree STUB up: sequential SkillSequence executor')

    # status ------------------------------------------------------------
    def _set_status(self, text):
        self._status = text
        self.get_logger().info(f'task_status: {text}')
        self._publish_status()

    def _publish_status(self):
        self._status_pub.publish(String(data=self._status))

    # plan intake ---------------------------------------------------------
    def _on_sequence(self, msg: SkillSequence):
        with self._lock:
            if self._busy:
                self.get_logger().warn('plan received while busy, ignored')
                return
            if not msg.skills:
                self.get_logger().warn('empty plan received, ignored')
                return
            self._busy = True
        # Run on a worker thread so this callback returns immediately and the
        # executor stays free to service the action clients' callbacks.
        threading.Thread(target=self._run_plan, args=(msg,), daemon=True).start()

    def _run_plan(self, msg: SkillSequence):
        total = len(msg.skills)
        try:
            for i, skill in enumerate(msg.skills, start=1):
                self._set_status(format_status('RUNNING', i, total, skill.name))
                ok, detail = self._run_skill(skill)
                if not ok:
                    self._set_status(format_status('FAILED', i, total, skill.name, detail))
                    return
            self._set_status(format_status('SUCCEEDED'))
        except Exception as exc:  # never leave the BT stuck in RUNNING
            self._set_status(format_status('FAILED', detail=f'unexpected error: {exc}'))
        finally:
            with self._lock:
                self._busy = False

    # skill execution -----------------------------------------------------
    def _run_skill(self, skill):
        if action_kind(skill.name) == 'nav':
            client = self._nav
            goal = NavigateToPose.Goal()
            goal.pose = skill.target_pose
        else:
            client = self._manip
            goal = ExecuteManipulation.Goal()
            goal.task = skill.name
            goal.object_id = skill.object_id
            goal.target_pose = skill.target_pose

        if not client.wait_for_server(timeout_sec=SERVER_WAIT_S):
            return False, 'action server not available'

        send_future = client.send_goal_async(goal)
        if not self._wait(send_future):
            return False, 'timed out sending goal'
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            return False, 'goal rejected'

        result_future = goal_handle.get_result_async()
        if not self._wait(result_future):
            goal_handle.cancel_goal_async()
            return False, 'timed out waiting for result'
        wrapped = result_future.result()
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            return False, f'action ended with status {wrapped.status}'
        if action_kind(skill.name) == 'manip' and not wrapped.result.success:
            return False, f'manipulation reported failure: {wrapped.result.message}'
        return True, ''

    @staticmethod
    def _wait(future, timeout_s=SKILL_TIMEOUT_S):
        deadline = time.monotonic() + timeout_s
        while not future.done():
            if time.monotonic() > deadline:
                return False
            time.sleep(POLL_S)
        return True


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorTreeStub()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
