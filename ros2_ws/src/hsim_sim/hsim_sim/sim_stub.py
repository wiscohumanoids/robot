"""STUB simulation node (node name: `sim`).

Stands in for the MuJoCo node until Phase 1 replaces it. No physics: the robot
"walks" by integrating the commanded velocity, and joints sit wherever the
locomotion targets put them. It publishes exactly what the real sim will:
/clock, /joint_states, /imu, /sim/ground_truth/odom.

Stub-only shortcut: it reads /cmd_vel to fake the motion. The real sim never
reads /cmd_vel; it only receives joint targets.
"""

import math

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from builtin_interfaces.msg import Time

CMD_TIMEOUT_S = 0.5
GRAVITY = 9.81


def to_time(t: float) -> Time:
    # whole nanoseconds first, then split: t = 5.9999999999 must become 6.000000000, not 5.0
    sec, nsec = divmod(int(round(t * 1e9)), 1_000_000_000)
    return Time(sec=sec, nanosec=nsec)


class SimStub(Node):
    def __init__(self):
        super().__init__('sim')
        self.declare_parameter('joint_names', Parameter.Type.STRING_ARRAY)
        self.declare_parameter('default_joint_pos', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('base_height', 0.76)
        self.declare_parameter('physics_dt', 0.005)

        self.joint_names = list(self.get_parameter('joint_names').value)
        self.q = dict(zip(self.joint_names, self.get_parameter('default_joint_pos').value))
        if len(self.q) != len(self.joint_names):
            raise ValueError('joint_names and default_joint_pos lengths differ')
        self.z = self.get_parameter('base_height').value
        self.dt = self.get_parameter('physics_dt').value

        self.t = 0.0                        # sim time, seconds
        self.x = self.y = self.yaw = 0.0
        self.cmd = (0.0, 0.0, 0.0)          # vx, vy, wz in the body frame
        self.cmd_t = -1.0

        self.pub_clock = self.create_publisher(Clock, '/clock', 10)
        self.pub_js = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_imu = self.create_publisher(Imu, '/imu', 10)
        self.pub_gt = self.create_publisher(Odometry, '/sim/ground_truth/odom', 10)
        self.create_subscription(JointState, '/locomotion/joint_targets', self.on_targets, 10)
        self.create_subscription(TwistStamped, '/cmd_vel', self.on_cmd, 10)

        self.create_timer(self.dt, self.step)
        self.get_logger().info(
            f'STUB sim: {len(self.joint_names)} joints, dt={self.dt}s (kinematic, no physics)')

    def on_targets(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in self.q:
                self.q[name] = pos

    def on_cmd(self, msg: TwistStamped):
        tw = msg.twist
        self.cmd = (tw.linear.x, tw.linear.y, tw.angular.z)
        self.cmd_t = self.t

    def step(self):
        self.t += self.dt
        vx, vy, wz = self.cmd if self.t - self.cmd_t < CMD_TIMEOUT_S else (0.0, 0.0, 0.0)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * vx - s * vy) * self.dt
        self.y += (s * vx + c * vy) * self.dt
        self.yaw += wz * self.dt

        stamp = to_time(self.t)
        self.pub_clock.publish(Clock(clock=stamp))

        js = JointState()
        js.header.stamp = stamp
        js.name = self.joint_names
        js.position = [self.q[n] for n in self.joint_names]
        js.velocity = [0.0] * len(self.joint_names)
        js.effort = [0.0] * len(self.joint_names)
        self.pub_js.publish(js)

        qz, qw = math.sin(self.yaw / 2), math.cos(self.yaw / 2)

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = 'imu_link'
        imu.orientation.z, imu.orientation.w = qz, qw
        imu.angular_velocity.z = wz
        imu.linear_acceleration.z = GRAVITY
        self.pub_imu.publish(imu)

        gt = Odometry()
        gt.header.stamp = stamp
        gt.header.frame_id = 'odom'
        gt.child_frame_id = 'base_link'
        gt.pose.pose.position.x, gt.pose.pose.position.y, gt.pose.pose.position.z = self.x, self.y, self.z
        gt.pose.pose.orientation.z, gt.pose.pose.orientation.w = qz, qw
        gt.twist.twist.linear.x, gt.twist.twist.linear.y, gt.twist.twist.angular.z = vx, vy, wz
        self.pub_gt.publish(gt)


def main():
    rclpy.init()
    node = SimStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
