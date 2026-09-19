"""nav_stub / nav_node

STATUS: STUB standing in for Nav2. The interface is real (the standard
nav2_msgs/action/NavigateToPose action); the internals are a small
proportional go-to-point controller. There are no costmaps, no planning and no
obstacle avoidance. It drives in a straight line to the goal position and
ignores the goal heading.

INPUTS:
  - geometry_msgs/PoseStamped on /robot_pose (from state_estimation)
  - action goals on `navigate_to_pose` (from behavior_tree)

OUTPUTS:
  - Action server `navigate_to_pose` (nav2_msgs/NavigateToPose). This is the
    name and type Nav2's bt_navigator serves, so the behavior tree's client does
    not change when Nav2 replaces this node.
  - geometry_msgs/Twist on /cmd_vel_nav at CONTROL_RATE_HZ, only while a goal is
    active. This node is the only publisher; cmd_vel_mux forwards it to /cmd_vel.
    When Nav2 replaces this node, remap Nav2's velocity output to /cmd_vel_nav.

Only one goal runs at a time; a goal that arrives while another is active is
rejected. A goal is aborted if /robot_pose goes stale, and a zero command is
always sent when a goal ends, whatever the reason.
"""
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from nav_stub.nav_logic import (
    distance_to_goal, go_to_goal_command, reached, yaw_from_quaternion)

CONTROL_RATE_HZ = 20.0
GOAL_TIMEOUT_S = 60.0
POSE_WAIT_S = 5.0    # how long to wait for the first /robot_pose
POSE_STALE_S = 1.0   # abort if /robot_pose is older than this mid-goal


class NavStub(Node):

    def __init__(self):
        super().__init__('nav')
        self._pose = None        # latest PoseStamped
        self._pose_time = 0.0    # time.monotonic() when it arrived
        self._busy = False
        self._lock = threading.Lock()

        group = ReentrantCallbackGroup()
        self.create_subscription(
            PoseStamped, '/robot_pose', self._on_pose, 10, callback_group=group)
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self._server = ActionServer(
            self, NavigateToPose, 'navigate_to_pose',
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda goal_handle: CancelResponse.ACCEPT,
            callback_group=group)
        self.get_logger().info(
            'nav stub up: navigate_to_pose action server (go-to-point, no obstacle avoidance)')

    def _on_pose(self, msg: PoseStamped):
        self._pose = msg
        self._pose_time = time.monotonic()

    def _on_goal(self, _goal_request):
        with self._lock:
            if self._busy:
                self.get_logger().warn('rejecting navigate_to_pose goal: another goal is active')
                return GoalResponse.REJECT
            self._busy = True
        return GoalResponse.ACCEPT

    def _publish_cmd(self, vx, vy, wz):
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = vx, vy, wz
        self._cmd_pub.publish(cmd)

    def _execute(self, goal_handle):
        try:
            return self._drive(goal_handle)
        finally:
            # Runs on every exit path: stop the robot and free the node for the next goal.
            self._publish_cmd(0.0, 0.0, 0.0)
            with self._lock:
                self._busy = False

    def _drive(self, goal_handle):
        goal = goal_handle.request.pose.pose.position
        self.get_logger().info(f'navigate_to_pose goal: ({goal.x:.2f}, {goal.y:.2f})')

        result = NavigateToPose.Result()
        feedback = NavigateToPose.Feedback()
        period = 1.0 / CONTROL_RATE_HZ
        started = time.monotonic()

        while True:
            now = time.monotonic()
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return result
            if now - started > GOAL_TIMEOUT_S:
                self.get_logger().error('goal timed out, aborting')
                goal_handle.abort()
                return result

            pose_age = now - self._pose_time if self._pose is not None else float('inf')
            if pose_age > POSE_STALE_S:
                if self._pose is None and now - started < POSE_WAIT_S:
                    time.sleep(period)   # still waiting for the first pose
                    continue
                self.get_logger().error('/robot_pose is missing or stale, aborting')
                goal_handle.abort()
                return result

            pose = self._pose.pose
            x, y = pose.position.x, pose.position.y
            if reached(x, y, goal.x, goal.y):
                goal_handle.succeed()
                return result

            yaw = yaw_from_quaternion(
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            self._publish_cmd(*go_to_goal_command(x, y, yaw, goal.x, goal.y))

            feedback.current_pose = self._pose
            feedback.distance_remaining = float(distance_to_goal(x, y, goal.x, goal.y))
            goal_handle.publish_feedback(feedback)
            time.sleep(period)


def main(args=None):
    rclpy.init(args=args)
    node = NavStub()
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
