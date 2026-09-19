"""nav_stub / nav_node

STATUS: STUB standing in for Nav2. Real interface (the standard
nav2_msgs/action/NavigateToPose action), fake internals: a ~30-line
proportional go-to-point controller. No costmaps, no planning, no obstacle
avoidance -- it drives in a straight line to the goal position and ignores the
goal heading.

INPUTS:
  - geometry_msgs/PoseStamped on /robot_pose (from state_estimation)
  - action goal on `navigate_to_pose` (from behavior_tree)

OUTPUTS:
  - action server `navigate_to_pose` (nav2_msgs/NavigateToPose), the same
    action name Nav2's bt_navigator serves -- so the behavior tree's client
    does not change when real Nav2 replaces this node.
  - geometry_msgs/Twist on /cmd_vel_nav at CONTROL_RATE_HZ, only while a goal
    is active (this node is its ONLY publisher; cmd_vel_mux forwards it to
    /cmd_vel). When Nav2 replaces this, remap Nav2's velocity output to
    /cmd_vel_nav.
"""
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
POSE_WAIT_S = 5.0


class NavStub(Node):

    def __init__(self):
        super().__init__('nav')
        self._pose = None  # latest PoseStamped
        self._cb_group = ReentrantCallbackGroup()
        self.create_subscription(
            PoseStamped, '/robot_pose', self._on_pose, 10, callback_group=self._cb_group)
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self._server = ActionServer(
            self, NavigateToPose, 'navigate_to_pose',
            execute_callback=self._execute,
            goal_callback=lambda goal: GoalResponse.ACCEPT,
            cancel_callback=lambda goal_handle: CancelResponse.ACCEPT,
            callback_group=self._cb_group)
        self.get_logger().info(
            'nav STUB up: navigate_to_pose action server (go-to-point, no obstacle avoidance)')

    def _on_pose(self, msg: PoseStamped):
        self._pose = msg

    def _publish_cmd(self, vx, vy, wz):
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = vx, vy, wz
        self._cmd_pub.publish(cmd)

    def _execute(self, goal_handle):
        gx = goal_handle.request.pose.pose.position.x
        gy = goal_handle.request.pose.pose.position.y
        self.get_logger().info(f'navigate_to_pose goal: ({gx:.2f}, {gy:.2f})')

        dt = 1.0 / CONTROL_RATE_HZ
        started = time.monotonic()
        result = NavigateToPose.Result()

        while self._pose is None:
            if time.monotonic() - started > POSE_WAIT_S:
                self.get_logger().error('no /robot_pose received -- aborting goal')
                goal_handle.abort()
                return result
            time.sleep(dt)

        feedback = NavigateToPose.Feedback()
        while True:
            if goal_handle.is_cancel_requested:
                self._publish_cmd(0.0, 0.0, 0.0)
                goal_handle.canceled()
                return result
            if time.monotonic() - started > GOAL_TIMEOUT_S:
                self.get_logger().error('goal timed out -- aborting')
                self._publish_cmd(0.0, 0.0, 0.0)
                goal_handle.abort()
                return result

            p = self._pose.pose
            x, y = p.position.x, p.position.y
            if reached(x, y, gx, gy):
                self._publish_cmd(0.0, 0.0, 0.0)
                goal_handle.succeed()
                return result

            yaw = yaw_from_quaternion(
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
            self._publish_cmd(*go_to_goal_command(x, y, yaw, gx, gy))

            feedback.current_pose = self._pose
            feedback.distance_remaining = float(distance_to_goal(x, y, gx, gy))
            goal_handle.publish_feedback(feedback)
            time.sleep(dt)


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
