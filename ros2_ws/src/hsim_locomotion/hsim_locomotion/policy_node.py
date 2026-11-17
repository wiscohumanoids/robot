"""Locomotion policy node (node name: `locomotion`). Replaces locomotion_stub.

Runs an mjlab-trained ONNX policy on CPU at the manifest's rate (50 Hz, sim time).
Builds the same observation the policy saw in training (99 values for the all-joints (29-joint)
policy, 85 for a legs+waist (arms-free) policy), from ROS inputs:

  base_lin_vel       /odometry/filtered twist.linear (body frame; the state estimator's job)
  base_ang_vel       /imu angular_velocity
  projected_gravity  /imu orientation applied to (0, 0, -1)
  joint_pos          /joint_states position - default pose   (by joint name)
  joint_vel          /joint_states velocity                  (by joint name)
  actions            this node's previous raw output (one per policy-driven joint)
  command            /cmd_vel (vx, vy, wz); zero if older than command_timeout

Publishes /locomotion/joint_targets and /locomotion/status (1 Hz).
  Manifest v1 (policy drives all joints): targets = default_pos + action_scale * action.
  Manifest v2 (policy drives a subset, e.g. legs + waist): the same for the driven joints, plus
  the held joints (arms): default pose until the first message on held_joints.source_topic
  (default /arm/joint_targets, sensor_msgs/JointState with names + positions), then the latest
  commanded target per joint. Names that are not held joints are ignored (warned once).
  Manifest v3 (motion tracking): the policy drives every joint and plays the clip carried by its ONNX file.
  On the first tick the clip is aligned to the robot (yaw and xy of the anchor body, TF world_frame -> anchor_frame),
  then it advances one clip step per policy step; at the end it holds the last frame (or restarts with loop:=true).
  /cmd_vel is ignored. The observation adds the clip's reference joint state and its anchor pose relative to the
  robot's anchor body.
  The manifest/observation logic lives in policy_contract.py (shared with tools/sim2sim_play.py).
"""

from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState

from hsim_locomotion.policy_contract import HeldTargets, MotionAlignment, load_contract, quat_to_mat


def quat_rotate_inverse(x, y, z, w, v):
    """Rotate world vector v into the frame of unit quaternion (x, y, z, w)."""
    q = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v * (2 * w * w - 1) - np.cross(q, v) * 2 * w + q * 2 * np.dot(q, v)


class PolicyNode(Node):
    """Node `locomotion`: runs the walking policy at its manifest rate and publishes targets for every joint."""
    def __init__(self):
        """Load the policy folder and its contract, create the ONNX session, subscriptions, publishers and the
        policy timer.
        """
        super().__init__('locomotion')
        self.declare_parameter('policy_dir', '/hsim/models/locomotion/g1_velocity_all_joints')
        self.declare_parameter('command_timeout', 0.5)
        self.declare_parameter('world_frame', 'odom')        # tracking: frame the robot's anchor pose is read in
        self.declare_parameter('anchor_frame', 'torso_link')  # tracking: TF frame of the clip's anchor body
        self.declare_parameter('loop', False)                # tracking: restart the clip when it ends
        folder = Path(self.get_parameter('policy_dir').value)
        self.cmd_timeout = self.get_parameter('command_timeout').value

        self.contract = c = load_contract(folder)
        self.policy_id = c.policy_id
        self.joint_names = c.obs_joint_names          # every joint the observation reads
        policy_hz = c.policy_hz
        self.session = ort.InferenceSession(str(folder / 'policy.onnx'), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        n_in = self.session.get_inputs()[0].shape[-1]
        if isinstance(n_in, int) and n_in != c.obs_dim:
            raise ValueError(f'policy expects {n_in} observations, manifest implies {c.obs_dim}')

        self.last_action = np.zeros(len(c.act_joint_names))
        self.held = HeldTargets(c)
        self.unknown_arm_warned = set()
        self.js = self.imu = self.odom = None
        self.js_index = None
        self.cmd = np.zeros(3)
        self.cmd_time = None
        self.infer_ms = []
        self.tilted = False
        self.step = 0                       # tracking: clip step of the next policy step
        self.alignment = None               # tracking: clip-to-robot frame, fixed on the first tick
        if c.kind == 'tracking':
            from tf2_ros import Buffer, TransformListener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.world_frame = self.get_parameter('world_frame').value
            self.anchor_frame = self.get_parameter('anchor_frame').value
            self.loop = bool(self.get_parameter('loop').value)

        self.pub_targets = self.create_publisher(JointState, '/locomotion/joint_targets', 10)
        self.pub_status = self.create_publisher(DiagnosticArray, '/locomotion/status', 10)
        self.create_subscription(JointState, '/joint_states', self.on_js, 10)
        self.create_subscription(Imu, '/imu', lambda m: setattr(self, 'imu', m), 10)
        self.create_subscription(Odometry, '/odometry/filtered', lambda m: setattr(self, 'odom', m), 10)
        self.create_subscription(TwistStamped, '/cmd_vel', self.on_cmd, 10)
        if c.held_joint_names:
            topic = c.held_topic or '/arm/joint_targets'
            self.create_subscription(JointState, topic, self.on_arm, 10)
            self.get_logger().info(f'{len(c.held_joint_names)} held joints: default pose until {topic} arrives')
        self.create_timer(1.0 / policy_hz, self.tick)
        self.create_timer(1.0, self.publish_status)
        self.get_logger().info(f'policy {self.policy_id} (manifest v{c.version}): drives '
                               f'{len(c.act_joint_names)}/{len(self.joint_names)} joints at {policy_hz:.0f} Hz')

    def on_js(self, msg: JointState):
        """Keep the latest /joint_states; on the first message, map policy joints to message slots (and fail loudly
        if any is missing).
        """
        if self.js_index is None:
            pos = {n: i for i, n in enumerate(msg.name)}
            missing = [n for n in self.joint_names if n not in pos]
            if missing:
                self.get_logger().error(f'/joint_states is missing policy joints {missing}')
                return
            self.js_index = np.array([pos[n] for n in self.joint_names])
        self.js = msg

    def on_arm(self, msg: JointState):
        """Update held-joint targets from /arm/joint_targets, warning once per name that is not a held joint."""
        for n in self.held.update(msg.name, msg.position):
            if n not in self.unknown_arm_warned:
                self.get_logger().warn(f'arm target for "{n}" ignored (not a held joint of this policy)')
                self.unknown_arm_warned.add(n)

    def on_cmd(self, msg: TwistStamped):
        """Store the latest velocity command and when it arrived."""
        t = msg.twist
        self.cmd = np.array([t.linear.x, t.linear.y, t.angular.z])
        self.cmd_time = self.get_clock().now()

    def command(self):
        """The current command (vx, vy, wz), or zeros once it is older than `command_timeout`."""
        if self.cmd_time is None:
            return np.zeros(3)
        age = (self.get_clock().now() - self.cmd_time).nanoseconds * 1e-9
        return self.cmd if age < self.cmd_timeout else np.zeros(3)

    def anchor_pose(self):
        """The robot's anchor body pose in world_frame as (position, rotation matrix), or None without TF."""
        try:
            tf = self.tf_buffer.lookup_transform(self.world_frame, self.anchor_frame, rclpy.time.Time())
        except Exception:  # noqa: BLE001 - not localized yet, or TF not published yet
            return None
        t, r = tf.transform.translation, tf.transform.rotation
        return np.array([t.x, t.y, t.z]), quat_to_mat([r.w, r.x, r.y, r.z])

    def reference(self, step):
        """The clip at `step`: reference joint positions and velocities, anchor position and (w, x, y, z) quat."""
        c = self.contract
        feeds = {self.input_name: np.zeros((1, c.obs_dim), np.float32), 'time_step': np.array([[step]], np.float32)}
        out = dict(zip(self.output_names, self.session.run(None, feeds)))
        return (out['joint_pos'][0], out['joint_vel'][0], out['body_pos_w'][0][c.anchor_index],
                out['body_quat_w'][0][c.anchor_index])

    def tracking_observation(self, w, v, q, dq):
        """Tracking observation for the current clip step (aligning the clip on the first call); None without TF."""
        robot = self.anchor_pose()
        if robot is None:
            return None
        ref_q, ref_dq, m_pos, m_quat = self.reference(self.step)
        if self.alignment is None:
            r_pos, r_R = robot
            yaw = float(np.arctan2(r_R[1, 0], r_R[0, 0]))
            self.alignment = MotionAlignment(m_pos, m_quat, r_pos, [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            self.get_logger().info(f'tracking: clip aligned to the robot, {self.contract.motion_steps or "?"} steps')
        a_pos, a_R = self.alignment.apply(m_pos, m_quat)
        return self.contract.tracking_observation(ref_q, ref_dq, a_pos, a_R, robot[0], robot[1],
                                                  [v.x, v.y, v.z], [w.x, w.y, w.z], q, dq, self.last_action)

    def tick(self):
        """One policy step: build the observation exactly as in training, run the ONNX model, publish the joint
        targets.
        """
        if self.js is None or self.imu is None or self.odom is None:
            return  # sim holds the standing keyframe until every input exists
        o, w, v = self.imu.orientation, self.imu.angular_velocity, self.odom.twist.twist.linear
        gravity_b = quat_rotate_inverse(o.x, o.y, o.z, o.w, (0.0, 0.0, -1.0))
        q = np.asarray(self.js.position)[self.js_index]
        dq = np.asarray(self.js.velocity)[self.js_index]
        tracking = self.contract.kind == 'tracking'
        if tracking:
            obs = self.tracking_observation(w, v, q, dq)
            if obs is None:
                return  # waiting for the anchor body's TF
            feeds = {self.input_name: obs[None], 'time_step': np.array([[self.step]], np.float32)}
        else:
            obs = self.contract.observation([v.x, v.y, v.z], [w.x, w.y, w.z], gravity_b,
                                            q, dq, self.last_action, self.command())
            feeds = {self.input_name: obs[None]}

        t0 = time.perf_counter()
        action = self.session.run(None, feeds)[0][0]
        self.infer_ms.append((time.perf_counter() - t0) * 1e3)
        self.last_action = action.astype(np.float64)
        self.tilted = gravity_b[2] > -0.5   # torso tilted > 60 degrees
        if tracking:
            self.step += 1
            steps = self.contract.motion_steps
            if steps and self.step >= steps and self.loop:
                self.step, self.alignment = 0, None

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        c = self.contract
        msg.name = list(c.act_joint_names) + list(c.held_joint_names)
        msg.position = c.action_targets(self.last_action).tolist() + self.held.targets.tolist()
        self.pub_targets.publish(msg)

    def publish_status(self):
        """Publish /locomotion/status: waiting for inputs, tilted, or running (policy id, inference time, command).
        """
        running = self.js is not None and self.imu is not None and self.odom is not None
        c = self.command()
        st = DiagnosticStatus(name='locomotion', hardware_id='robot')
        if not running:
            st.level, st.message = DiagnosticStatus.WARN, 'waiting for /joint_states, /imu, /odometry/filtered'
        elif self.tilted:
            st.level, st.message = DiagnosticStatus.ERROR, 'robot tilted over'
        elif self.contract.kind == 'tracking':
            done = self.contract.motion_steps and self.step >= self.contract.motion_steps
            st.level, st.message = DiagnosticStatus.OK, 'holding the last frame' if done else 'tracking'
        else:
            st.level, st.message = DiagnosticStatus.OK, 'walking'
        ms = self.infer_ms[-50:]
        st.values = [
            KeyValue(key='mode', value='policy'),
            KeyValue(key='policy_id', value=self.policy_id),
            KeyValue(key='kind', value=self.contract.kind),
            KeyValue(key='clip_step', value=f'{self.step}/{self.contract.motion_steps}')
            if self.contract.kind == 'tracking' else KeyValue(key='clip_step', value='n/a'),
            KeyValue(key='cmd', value=f'vx={c[0]:+.2f} vy={c[1]:+.2f} wz={c[2]:+.2f}'),
            KeyValue(key='inference_ms_mean', value=f'{np.mean(ms):.3f}' if ms else 'n/a'),
            KeyValue(key='tilted', value=str(self.tilted)),
            KeyValue(key='driven_joints', value=str(len(self.contract.act_joint_names))),
            KeyValue(key='held_joints_source', value=('commanded' if self.held.received else 'default pose')
                     if self.contract.held_joint_names else 'n/a'),
        ]
        self.infer_ms = ms
        arr = DiagnosticArray(status=[st])
        arr.header.stamp = self.get_clock().now().to_msg()
        self.pub_status.publish(arr)


def main():
    """Entry point: spin the node until Ctrl-C."""
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
