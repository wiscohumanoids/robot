"""task_planner_stub / planner_node

STATUS: STUB. The interface is real; there is no LLM. Any /user_intent string
becomes a canned two-skill plan (navigate to a known object, then pick or place
it; see planner_logic.py). The real replacement is the LLM + skill-library
planner publishing the same SkillSequence.

INPUTS:
  - std_msgs/String on /user_intent (typed by hand today:
      ros2 topic pub --once /user_intent std_msgs/msg/String "{data: 'pick up the cube'}"
    later, the speech front-end)
  - humanoid_interfaces/ObjectPoseArray on /object_poses (from perception);
    a plan is only possible once perception has reported at least one object.

OUTPUT (this node is the only publisher):
  humanoid_interfaces/SkillSequence on /skill_sequence (event-driven: one
  message per accepted intent).
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from humanoid_interfaces.msg import ObjectPoseArray, Skill, SkillSequence

from task_planner_stub.planner_logic import plan_for_intent


class PlannerStub(Node):

    def __init__(self):
        super().__init__('task_planner')
        self.declare_parameter('standoff_m', 0.6)
        self._objects = []  # list of dicts, latest /object_poses
        self.create_subscription(ObjectPoseArray, '/object_poses', self._on_objects, 10)
        self.create_subscription(String, '/user_intent', self._on_intent, 10)
        self._pub = self.create_publisher(SkillSequence, '/skill_sequence', 10)
        self.get_logger().info('task_planner STUB up: any /user_intent -> canned /skill_sequence')

    def _on_objects(self, msg: ObjectPoseArray):
        self._objects = [
            {'object_id': o.object_id, 'label': o.label,
             'x': o.pose.position.x, 'y': o.pose.position.y, 'z': o.pose.position.z}
            for o in msg.objects]

    def _on_intent(self, msg: String):
        if not msg.data.strip():
            self.get_logger().warn('ignoring an empty /user_intent')
            return
        plan = plan_for_intent(
            msg.data, self._objects, float(self.get_parameter('standoff_m').value))
        if not plan:
            self.get_logger().warn(
                f'no plan for intent "{msg.data}": perception has reported no objects yet')
            return

        seq = SkillSequence()
        seq.header.stamp = self.get_clock().now().to_msg()
        seq.header.frame_id = 'map'
        seq.intent = msg.data
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
            f'intent "{msg.data}" -> plan: {[s.name for s in seq.skills]}')


def main(args=None):
    rclpy.init(args=args)
    node = PlannerStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
