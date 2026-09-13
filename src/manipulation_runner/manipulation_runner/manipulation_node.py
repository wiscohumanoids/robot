"""manipulation_runner / manipulation_node

STATUS: STUB task execution, REAL action-server interface. Accepts any
ExecuteManipulation goal and runs a fixed scripted arm-pose sequence
(approach -> grasp -> retract) over a few seconds, then reports success. No
perception, no grasp planning, no policy inference happens here -- see
INTEGRATION_POINTS.md "Manipulation" for what a real implementation
(wrapping lerobot_alohamini's evaluate_bi.py-style inference loop) needs to
replace this with.

FREQUENCY: 10 Hz internal control-loop rate while a goal is executing
(CONTROL_RATE_HZ). The action server itself is event-driven (goal/cancel
callbacks), not rate-limited.

OUTPUT: humanoid_interfaces/JointTargets on /manipulation/joint_targets,
source="manipulation", published ONLY while a goal is active. wbc_stub reads
this as an override for the arm joint indices (13-22) only -- see
wbc_stub/README.md for the exact arbitration rule. The leg/waist indices in
this message are filled with the same nominal pose as locomotion_runner's
but are NOT meant to be trusted downstream; they exist only because
JointTargets is a fixed 23-length array and this node has no legitimate
opinion about leg/waist targets.
"""
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from humanoid_interfaces.msg import JointTargets
from humanoid_interfaces.action import ExecuteManipulation

from manipulation_runner.joint_order import load_canonical_joint_order, NUM_JOINTS

CONTROL_RATE_HZ = 10.0
PHASE_DURATION_S = 1.0  # each of approach/grasp/retract takes this long

NOMINAL_POSE = [
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    0.0,
    0.3, 0.15, 0.0, 0.5, 0.0,
    0.3, -0.15, 0.0, 0.5, 0.0,
]
assert len(NOMINAL_POSE) == NUM_JOINTS

LEFT_ARM = slice(13, 18)
RIGHT_ARM = slice(18, 23)

# A scripted "reach forward" arm pose, purely illustrative -- not derived from
# any real grasp/IK computation. shoulder_pitch more negative = arm forward,
# elbow more flexed = forearm raised.
REACH_POSE_LEFT_ARM = [0.9, 0.15, 0.0, 1.1, 0.0]
REACH_POSE_RIGHT_ARM = [0.9, -0.15, 0.0, 1.1, 0.0]


class ManipulationNode(Node):

    def __init__(self):
        super().__init__('manipulation_runner')
        self.joint_names = load_canonical_joint_order()
        self._targets_pub = self.create_publisher(
            JointTargets, '/manipulation/joint_targets', 10)

        self._action_server = ActionServer(
            self,
            ExecuteManipulation,
            'execute_manipulation',
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(
            f'manipulation_runner up, ExecuteManipulation action server ready '
            f'at {CONTROL_RATE_HZ} Hz internal loop (STUB task execution)')

    def _goal_callback(self, goal_request):
        self.get_logger().info(
            f'accepted manipulation goal: task={goal_request.task} object_id={goal_request.object_id}')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def _publish_pose(self, left_arm, right_arm):
        pose = list(NOMINAL_POSE)
        pose[LEFT_ARM] = left_arm
        pose[RIGHT_ARM] = right_arm
        msg = JointTargets()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.positions = pose
        msg.velocities = [0.0] * NUM_JOINTS
        msg.source = 'manipulation'
        self._targets_pub.publish(msg)

    def _execute_callback(self, goal_handle):
        """
        === REAL MANIPULATION INFERENCE GOES HERE ===
        Replace this scripted three-phase sequence with a call into a ROS2
        wrapper around lerobot_alohamini's evaluate_bi.py-style
        observation->action loop, publishing real JointTargets per tick
        instead of interpolating between two hand-picked poses. See
        INTEGRATION_POINTS.md "Manipulation" for the transport/joint-mapping
        gaps that must be resolved first.
        """
        feedback = ExecuteManipulation.Feedback()
        result = ExecuteManipulation.Result()
        dt = 1.0 / CONTROL_RATE_HZ
        ticks_per_phase = int(PHASE_DURATION_S * CONTROL_RATE_HZ)

        phases = [
            ('approaching', REACH_POSE_LEFT_ARM, REACH_POSE_RIGHT_ARM),
            ('grasping', REACH_POSE_LEFT_ARM, REACH_POSE_RIGHT_ARM),
            ('retracting', NOMINAL_POSE[LEFT_ARM], NOMINAL_POSE[RIGHT_ARM]),
        ]

        for phase_idx, (phase_name, left_pose, right_pose) in enumerate(phases):
            for tick in range(ticks_per_phase):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = 'canceled'
                    return result

                self._publish_pose(left_pose, right_pose)
                feedback.phase = phase_name
                feedback.progress = (phase_idx * ticks_per_phase + tick + 1) / (
                    len(phases) * ticks_per_phase)
                goal_handle.publish_feedback(feedback)
                time.sleep(dt)

        goal_handle.succeed()
        result.success = True
        result.message = f'stub execution of task={goal_handle.request.task} complete'
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ManipulationNode()
    executor = rclpy.executors.MultiThreadedExecutor()
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
