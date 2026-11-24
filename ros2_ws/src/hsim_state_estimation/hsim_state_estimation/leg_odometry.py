"""Leg odometry (node name: `leg_odometry`): body velocity from joint encoders + IMU.

Uses only what a real robot has: its own kinematic model (the robot profile's model file,
parameter `robot_model`, e.g. Wisco's or the G1's: link lengths and joint layout, not the
simulator's state), joint angles and speeds from /joint_states, and the IMU's gyro and tilt
from /imu.

The foot on the ground (stance foot) is not moving, so the body's velocity in its own
frame is the negative of how the stance foot appears to move relative to the body:

    v_body = -( omega x r_foot + J(q) qdot )

where r_foot is the foot position in the body frame (forward kinematics) and J(q) qdot is
its velocity from joint motion. Stance = the lower foot in the gravity-aligned frame,
blended smoothly in double support. Published as /leg_odometry (twist in base_link);
the EKF (`state_estimation`) fuses it with the IMU.

It also publishes the body's own pose transforms (what robot_state_publisher does with a URDF):
  base_footprint -> base_link   pelvis height above the stance foot sole + roll/pitch from the IMU
  base_link -> torso_link       the three waist joints (the head camera sits on the torso)
"""

import math

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from tf2_ros import TransformBroadcaster

PREFIX = 'robot/'
SOLE_OFFSET = 0.0015          # foot sites sit this far above the sole (measured at the standing pose)


def tilt_only_rotation(qx, qy, qz, qw):
    """Rotation matrix with the IMU's roll and pitch but zero yaw (body -> gravity-aligned)."""
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    return np.array([[cp, sp * sr, sp * cr], [0, cr, -sr], [-sp, cp * sr, cp * cr]])


class LegOdometry(Node):
    """Node `leg_odometry`: base velocity from stance-foot kinematics, and the base/torso TF the rest of the stack
    uses.
    """
    def __init__(self):
        """Load the robot model for kinematics, declare the profile parameters, subscribe to joints and IMU, create
        publishers.
        """
        super().__init__('leg_odometry')
        self.declare_parameter('robot_model', '/hsim/assets/robots/g1/mjlab_1.6.0/g1_velocity_flat.xml')
        self.declare_parameter('blend_width', 0.003)    # m: how sharply stance switches between feet (validated: <2% speed error)
        self.declare_parameter('torso_body', 'torso_link')   # robot profile: MuJoCo body published as TF torso_link
        self.declare_parameter('sole_offset', SOLE_OFFSET)  # robot profile: foot sites above the sole, m
        self.declare_parameter('rate_divider', 4)       # every 4th /joint_states: 200 -> 50 Hz (TF listeners in Python can't take more)
        self.m = mujoco.MjModel.from_xml_path(self.get_parameter('robot_model').value)
        self.d = mujoco.MjData(self.m)
        self.blend = self.get_parameter('blend_width').value
        self.div = self.get_parameter('rate_divider').value

        self.site = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, PREFIX + n) for n in ('left_foot', 'right_foot')]
        self.jac = np.zeros((3, self.m.nv))
        self.joint_slot = None                         # /joint_states index -> (qpos adr, dof adr)
        self.imu = None
        self.count = 0
        self.torso = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + self.get_parameter('torso_body').value)
        self.sole_offset = self.get_parameter('sole_offset').value
        self.tf = TransformBroadcaster(self)
        self.pub = self.create_publisher(Odometry, '/leg_odometry', 10)
        self.create_subscription(Imu, '/imu', lambda m: setattr(self, 'imu', m), 10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.get_logger().info('leg odometry: stance-foot kinematics + IMU gyro')

    def on_joints(self, msg: JointState):
        """Every few joint messages: each foot's velocity relative to the body gives a body-velocity estimate; blend
        the two by which foot is lower (stance), publish it with the body height.
        """
        self.count += 1
        if self.imu is None or self.count % self.div:
            return
        if self.joint_slot is None:
            slots = []
            for name in msg.name:
                j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + name)
                slots.append((self.m.jnt_qposadr[j], self.m.jnt_dofadr[j]) if j >= 0 else None)
            self.joint_slot = slots

        # Body frame = world frame here: base at the origin, no rotation.
        self.d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        self.d.qvel[:6] = 0.0
        for slot, q, dq in zip(self.joint_slot, msg.position, msg.velocity):
            if slot:
                self.d.qpos[slot[0]], self.d.qvel[slot[1]] = q, dq
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

        w = self.imu.angular_velocity
        omega = np.array([w.x, w.y, w.z])
        o = self.imu.orientation
        R_tilt = tilt_only_rotation(o.x, o.y, o.z, o.w)

        v_est, heights = [], []
        for s in self.site:
            r = self.d.site_xpos[s].copy()
            mujoco.mj_jacSite(self.m, self.d, self.jac, None, s)
            rdot = self.jac[:, 6:] @ self.d.qvel[6:]
            v_est.append(-(np.cross(omega, r) + rdot))
            heights.append((R_tilt @ r)[2])
        # weight each foot by how much lower it is (smooth switch, both count in double support)
        dz = (heights[1] - heights[0]) / self.blend
        w_left = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, dz))))
        v = w_left * v_est[0] + (1.0 - w_left) * v_est[1]
        height = -(w_left * heights[0] + (1.0 - w_left) * heights[1]) + self.sole_offset
        self.publish_body_tf(msg.header.stamp, height, o)

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'odom'
        out.child_frame_id = 'base_link'
        out.twist.twist.linear.x, out.twist.twist.linear.y, out.twist.twist.linear.z = map(float, v)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.02 ** 2 * 4      # vx, vy
        cov[14] = 0.05 ** 2                  # vz
        cov[21] = cov[28] = cov[35] = 1e3    # angular: not measured here
        out.twist.covariance = cov
        self.pub.publish(out)

    def publish_body_tf(self, stamp, height, imu_q):
        """Publish base_footprint -> base_link (height, IMU tilt) and base_link -> torso_link."""
        # tilt only (roll/pitch); heading lives in odom -> base_footprint
        R = tilt_only_rotation(imu_q.x, imu_q.y, imu_q.z, imu_q.w)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.ravel())            # parent (level) <- child (pelvis) rotation
        a = TransformStamped()
        a.header.stamp = stamp
        a.header.frame_id, a.child_frame_id = 'base_footprint', 'base_link'
        a.transform.translation.z = float(height)
        a.transform.rotation.w, a.transform.rotation.x, a.transform.rotation.y, a.transform.rotation.z = map(float, q)
        t = self.d.xpos[self.torso]
        tq = self.d.xquat[self.torso]
        b = TransformStamped()
        b.header.stamp = stamp
        b.header.frame_id, b.child_frame_id = 'base_link', 'torso_link'
        b.transform.translation.x, b.transform.translation.y, b.transform.translation.z = map(float, t)
        b.transform.rotation.w, b.transform.rotation.x, b.transform.rotation.y, b.transform.rotation.z = map(float, tq)
        self.tf.sendTransform([a, b])


def main():
    """Entry point: spin the node until Ctrl-C."""
    rclpy.init()
    node = LegOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
