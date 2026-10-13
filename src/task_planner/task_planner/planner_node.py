"""task_planner / planner_node

With OPENAI_API_KEY set (in .env)
an LLM turns each intent into a validated skill list (llm_planner.py, skills from
skill_catalog.py). Without a key it falls back to a canned two-skill plan (navigate
to a known object, then pick or place it; planner_logic.py: that fallback is the
only STUB part). If the LLM is
configured but fails or refuses, nothing is published (the reason is logged): a
wrong canned plan would be worse than none.

The queue holds raw text (dispatcher.py). An intent is planned only when the
behavior tree is idle, using the robot pose and objects at that moment. If the
tree reports FAILED, the LLM gets ONE replan call (original request, old plan, where
it failed, current world); a second failure is final. No replanning without a key.

INPUTS:
  - std_msgs/String on /user_intent (typed by hand today:
      ros2 topic pub --once /user_intent std_msgs/msg/String "{data: 'pick up the cube'}"
    later, the speech front-end)
  - humanoid_interfaces/ObjectPoseArray on /object_poses (from perception)
  - geometry_msgs/PoseStamped on /robot_pose (from state_estimation): where the
    robot is when an intent is planned
  - std_msgs/String on /planner_command: `clear`, `remove <n>` (1 = next to run) or
    `list`; edits the waiting queue (not the running plan). Answers on /planner_status.
  - std_msgs/String on /manipulation/held_object (from manipulation): the object in
    the hand ("" = empty); unknown until the first message
  - std_msgs/String on /task_status (from behavior_tree): tells the queue when the
    tree is done

OUTPUTS (this node is the only publisher of both):
  humanoid_interfaces/SkillSequence on /skill_sequence (event-driven: one
  message per planned intent).
  std_msgs/String on /planner_status (event-driven): what happened to each intent,
  e.g. 'QUEUED "pick up the cup": 1 waiting', 'PLANNING "..."', 'SENT "...": 2 steps',
  'NO_PLAN "...": <reason>', 'REJECTED "...": queue full (N)' (only with a cap), and the outcome:
  'DONE "..."' (only after the tree said SUCCEEDED), 'RETRYING "...": <tree failure>' +
  'REPLANNING "..."', 'GAVE_UP "...": <failure, or the tree stopped / went silent>'.
"""
import threading

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String

from humanoid_interfaces.msg import ObjectPoseArray, Skill, SkillSequence

from task_planner.dispatcher import Dispatcher, yaw_from_quaternion
from task_planner.llm_config import config_from_env, load_env
from task_planner.llm_planner import PlanResult, plan_with_llm, replan_with_llm
from task_planner.planner_logic import plan_checked

# ROS keeps only this many unread messages per topic. The default (10) silently dropped
# intents when more than ten arrived faster than the planner read them (found by the stress
# test); the queue cap, not DDS, must be what turns extra intents away.
BACKLOG = 200


class PlannerNode(Node):

    def __init__(self):
        super().__init__('task_planner')
        self.declare_parameter('standoff_m', 0.6)
        self._objects = []  # list of dicts, latest /object_poses
        self._robot = None  # {'x', 'y', 'yaw'}, latest /robot_pose
        self._held = None   # latest /manipulation/held_object: id, '' = empty hand, None = unknown
        load_env()
        self._llm = config_from_env()
        self.create_subscription(ObjectPoseArray, '/object_poses', self._on_objects, 10)
        self.create_subscription(PoseStamped, '/robot_pose', self._on_robot_pose, 10)
        self.create_subscription(String, '/manipulation/held_object', self._on_held, 10)
        self.create_subscription(String, '/task_status', self._on_status, 10)
        self.create_subscription(String, '/user_intent', self._on_intent, BACKLOG)
        self.create_subscription(String, '/planner_command', self._on_command, BACKLOG)
        self._pub = self.create_publisher(SkillSequence, '/skill_sequence', 10)
        self._report_pub = self.create_publisher(String, '/planner_status', BACKLOG)

        self._dispatcher = Dispatcher(
            plan_fn=self._plan, publish=self._publish_plan,
            get_world=lambda: {'robot': self._robot, 'objects': list(self._objects),
                               'held': self._held},
            log=lambda msg: self.get_logger().info(msg),
            report=lambda text: self._report_pub.publish(String(data=text)),
            replan_fn=self._replan if self._llm.configured else None)
        # Planning blocks (the LLM call takes seconds): its own thread keeps the
        # executor free for /object_poses and /task_status.
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._dispatcher.run_forever, args=(self._stop,), daemon=True)
        self._worker.start()

        if self._llm.configured:
            self.get_logger().info(f'task_planner up: queued /user_intent -> LLM ({self._llm.model}) -> /skill_sequence')
        else:
            self.get_logger().warn('task_planner up: OPENAI_API_KEY not set, using the canned fallback planner')

    def _on_objects(self, msg: ObjectPoseArray):
        self._objects = [
            {'object_id': o.object_id, 'label': o.label,
             'x': o.pose.position.x, 'y': o.pose.position.y, 'z': o.pose.position.z}
            for o in msg.objects]

    def _on_robot_pose(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        self._robot = {'x': p.x, 'y': p.y, 'yaw': yaw_from_quaternion(q.x, q.y, q.z, q.w)}

    def _on_held(self, msg: String):
        self._held = msg.data

    def _on_status(self, msg: String):
        self._dispatcher.on_status(msg.data)

    def _on_command(self, msg: String):
        self._dispatcher.command(msg.data)

    def _on_intent(self, msg: String):
        self._dispatcher.submit(msg.data)

    def _plan(self, intent, world):
        standoff = float(self.get_parameter('standoff_m').value)
        if self._llm.configured:
            return plan_with_llm(intent, world['objects'], self._llm, standoff, world['robot'], world['held'])
        plan, error = plan_checked(intent, world['objects'], standoff, world['held'])
        return PlanResult(plan=plan, error=error)

    def _replan(self, intent, old_plan, failure, world):
        standoff = float(self.get_parameter('standoff_m').value)
        return replan_with_llm(intent, old_plan, failure, world['objects'], self._llm, standoff,
                               world['robot'], world['held'])

    def _publish_plan(self, intent, plan, plan_id):
        seq = SkillSequence()
        seq.header.stamp = self.get_clock().now().to_msg()
        seq.header.frame_id = 'map'
        seq.intent = intent
        seq.plan_id = plan_id
        for step in plan:
            skill = Skill()
            skill.name = step['name']
            skill.object_id = step['object_id']
            skill.target_pose.header.frame_id = 'map'
            skill.target_pose.pose.position.x = float(step['x'])
            skill.target_pose.pose.position.y = float(step['y'])
            skill.target_pose.pose.position.z = float(step['z'])
            skill.target_pose.pose.orientation.w = 1.0
            seq.skills.append(skill)
        self._pub.publish(seq)
        self.get_logger().info(
            f'intent "{intent}" -> plan {plan_id}: {[s.name for s in seq.skills]}')

    def shutdown(self):
        self._stop.set()
        self._worker.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
