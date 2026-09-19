"""locomotion_runner / locomotion_node

STATUS: STUB gait, real interface. The gait (a sinusoidal leg-pitch correction
on top of a nominal crouch pose) is a placeholder written for this repo, not
berkeley_humanoid's trained policy. The messages it reads and writes, and its
50 Hz timing, are what a dropped-in ONNX policy would use too. See the
"REAL POLICY GOES HERE" block below and INTEGRATION_POINTS.md, "Locomotion".

FREQUENCY: 50 Hz (PUBLISH_RATE_HZ).

INPUTS:
  - geometry_msgs/Twist on /cmd_vel (from cmd_vel_mux, 50 Hz; only linear.x,
    linear.y, angular.z are used; this node reads whatever was last
    published, no blocking wait; see RESEARCH_NOTES.md "Multi-rate architecture")
  - humanoid_interfaces/RobotState on /robot_state (from lowlevel_control)

OUTPUTS:
  - humanoid_interfaces/PolicyObservation on /locomotion/observation (for
    logging/replay/debugging: see PolicyObservation.msg's own docstring for
    why this is published even though nothing currently subscribes to it)
  - humanoid_interfaces/JointTargets on /locomotion/joint_targets, source="locomotion"
"""
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

from geometry_msgs.msg import Twist
from humanoid_interfaces.msg import RobotState, PolicyObservation, JointTargets

NUM_JOINTS = 23  # CANONICAL_JOINT_ORDER, see humanoid_interfaces/config/canonical_joint_order.yaml

PUBLISH_RATE_HZ = 50.0

# Nominal standing ("crouch") pose, CANONICAL_JOINT_ORDER, radians. Matches
# the <state_interface name="position"><param name="initial_value">...</param>
# values in g1_description's <ros2_control> block, so the stub gait's resting
# pose is the same pose the hardware/sim interface itself boots into.
NOMINAL_POSE = [
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,      # left leg (0-5)
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,      # right leg (6-11)
    0.0,                                  # waist_yaw (12)
    0.3, 0.15, 0.0, 0.5, 0.0,             # left arm (13-17)
    0.3, -0.15, 0.0, 0.5, 0.0,            # right arm (18-22)
]
assert len(NOMINAL_POSE) == NUM_JOINTS

# Canonical (23) index -> hip/knee-pitch joints used by the stub walking gait.
LEFT_HIP_PITCH, LEFT_KNEE = 0, 3
RIGHT_HIP_PITCH, RIGHT_KNEE = 6, 9

GAIT_FREQUENCY_HZ = 1.0        # one full stride cycle per second
GAIT_HIP_AMPLITUDE = 0.25      # rad
GAIT_KNEE_AMPLITUDE = 0.20     # rad
WALK_CMD_THRESHOLD = 0.05      # m/s or rad/s; below this the command means "stand"

# berkeley_humanoid G1Env is trained on the 29-DOF G1 (adds waist_roll,
# waist_pitch, and wrist_pitch/wrist_yaw per arm: 6 extra DOF vs. our
# canonical 23). This is the index map from canonical (23) into
# berkeley_humanoid's 29-vector, derived directly from g1_env.py's own
# qpos-layout docstring (environment/g1_env.py lines ~14-65 at the time this
# was written: RE-VERIFY against the actual checkpoint before trusting
# this for a real policy, per INTEGRATION_POINTS.md).
CANONICAL_TO_BERKELEY29 = [
    0, 1, 2, 3, 4, 5,            # left leg
    6, 7, 8, 9, 10, 11,          # right leg
    12,                          # waist_yaw (waist_roll=13, waist_pitch=14 are dropped)
    15, 16, 17, 18, 19,          # left arm: shoulder p/r/y, elbow, wrist_roll
                                 # (left_wrist_pitch=20, left_wrist_yaw=21 dropped)
    22, 23, 24, 25, 26,          # right arm: shoulder p/r/y, elbow, wrist_roll
                                 # (right_wrist_pitch=27, right_wrist_yaw=28 dropped)
]
BERKELEY29_ZERO_PAD_INDICES = [13, 14, 20, 21, 27, 28]


class LocomotionNode(Node):

    def __init__(self):
        super().__init__('locomotion_runner')

        self.declare_parameter('policy_onnx_path', '')
        self._onnx_session = None
        self._onnx_input_name = None
        self._np = None   # numpy, imported only when a policy is loaded
        self._maybe_load_policy(self.get_parameter('policy_onnx_path').value)

        self._latest_cmd = Twist()
        self._latest_state = None  # RobotState, None until first message arrives
        self._previous_action = [0.0] * NUM_JOINTS
        self._t = 0.0  # gait phase clock, seconds

        latest_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, latest_qos)
        self.create_subscription(RobotState, '/robot_state', self._on_state, latest_qos)
        self._obs_pub = self.create_publisher(PolicyObservation, '/locomotion/observation', 10)
        self._targets_pub = self.create_publisher(JointTargets, '/locomotion/joint_targets', 10)

        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._on_timer)
        self.get_logger().info(
            f'locomotion_runner up at {PUBLISH_RATE_HZ} Hz: '
            f'{"ONNX policy loaded" if self._onnx_session else "STUB gait (no policy_onnx_path given)"}')

    def _maybe_load_policy(self, path: str):
        if not path:
            return
        if not os.path.isfile(path):
            self.get_logger().warn(f'policy_onnx_path={path} does not exist, falling back to stub gait')
            return
        try:
            import numpy  # optional dependencies, only needed on this path
            import onnxruntime
        except ImportError:
            self.get_logger().error(
                'policy_onnx_path was given but onnxruntime is not installed '
                '(pip install onnxruntime), falling back to stub gait')
            return
        try:
            session = onnxruntime.InferenceSession(path)
        except Exception as exc:  # corrupt or unsupported model file
            self.get_logger().error(f'could not load {path} ({exc}), falling back to stub gait')
            return
        self._np = numpy
        self._onnx_input_name = session.get_inputs()[0].name
        self._onnx_session = session
        self.get_logger().info(f'loaded locomotion policy from {path}')

    def _on_cmd(self, msg: Twist):
        self._latest_cmd = msg

    def _on_state(self, msg: RobotState):
        self._latest_state = msg

    def _build_observation(self) -> PolicyObservation:
        obs = PolicyObservation()
        obs.header.stamp = self.get_clock().now().to_msg()
        obs.cmd_vx = self._latest_cmd.linear.x
        obs.cmd_vy = self._latest_cmd.linear.y
        obs.cmd_vyaw = self._latest_cmd.angular.z

        if self._latest_state is not None:
            obs.joint_positions = list(self._latest_state.joint_positions)
            obs.joint_velocities = list(self._latest_state.joint_velocities)
            obs.base_orientation = self._latest_state.imu_orientation
            obs.base_angular_velocity = self._latest_state.imu_angular_velocity
            obs.projected_gravity = self._latest_state.gravity_vector
            # base_linear_velocity has no real-hardware source in this repo yet
            # (no state estimator): see ARCHITECTURE.md gap list. Zero until one exists.
        else:
            obs.joint_positions = list(NOMINAL_POSE)
            obs.joint_velocities = [0.0] * NUM_JOINTS

        obs.previous_action = list(self._previous_action)
        return obs

    def _compute_action_stub(self, cmd: Twist) -> list:
        """Nominal pose + sinusoidal leg-pitch correction while a nonzero
        forward/yaw command is present, otherwise hold the nominal pose.
        This is OUR stub, not berkeley_humanoid's trained gait."""
        is_walking = (abs(cmd.linear.x) > WALK_CMD_THRESHOLD
                      or abs(cmd.angular.z) > WALK_CMD_THRESHOLD)
        action = list(NOMINAL_POSE)
        if is_walking:
            phase = 2.0 * math.pi * GAIT_FREQUENCY_HZ * self._t
            hip_swing = GAIT_HIP_AMPLITUDE * math.sin(phase)
            knee_lift = GAIT_KNEE_AMPLITUDE * max(0.0, math.sin(phase))
            action[LEFT_HIP_PITCH] += hip_swing
            action[LEFT_KNEE] += knee_lift
            action[RIGHT_HIP_PITCH] -= hip_swing
            action[RIGHT_KNEE] += GAIT_KNEE_AMPLITUDE * max(0.0, math.sin(phase + math.pi))
        return action

    def _compute_action_onnx(self, obs: PolicyObservation) -> list:
        """
        === REAL POLICY GOES HERE (load ONNX) ===
        This branch runs only when --ros-args -p policy_onnx_path:=/path/to/policy.onnx
        pointed at a real berkeley_humanoid export AND onnxruntime is installed.
        It performs the 23->29 pad / 29->23 truncate documented at
        CANONICAL_TO_BERKELEY29 above and in INTEGRATION_POINTS.md.

        UNVERIFIED: no real policy.onnx has been run through this path yet,
        it is written to the exact contract berkeley_humanoid's G1Env
        describes in its own docstring, but has not been validated against an
        actual exported checkpoint. Whoever drops in the first real
        policy.onnx should treat this function as the first thing to
        sanity-check, not as proven-correct code.
        """
        obs29 = [0.0] * 29
        for canon_idx, berkeley_idx in enumerate(CANONICAL_TO_BERKELEY29):
            obs29[berkeley_idx] = obs.joint_positions[canon_idx]
        # NOTE: this only forwards joint positions into the padded slot; a real
        # adapter must also assemble the rest of G1Env's 69-dim observation
        # (behaviour flag, pelvis quat, velocities) in the exact order given
        # in berkeley_humanoid/environment/g1_env.py's constructor docstring
        # before calling session.run(). That full assembly is intentionally
        # left as the integrating team's job: see INTEGRATION_POINTS.md.
        result = self._onnx_session.run(
            None, {self._onnx_input_name: self._np.array([obs29], dtype=self._np.float32)})
        action29 = result[0][0].tolist()

        action23 = [action29[b] for b in CANONICAL_TO_BERKELEY29]
        return action23

    def _on_timer(self):
        dt = 1.0 / PUBLISH_RATE_HZ
        self._t += dt

        obs = self._build_observation()
        self._obs_pub.publish(obs)

        action = None
        if self._onnx_session is not None:
            try:
                action = self._compute_action_onnx(obs)
            except Exception as exc:
                # A policy whose input shape or output does not match the adapter would
                # otherwise raise on every 20 ms tick and leave nothing publishing.
                # Fall back to the stub gait once, loudly, and stay there.
                self.get_logger().error(
                    f'ONNX policy failed ({exc}); falling back to the stub gait permanently')
                self._onnx_session = None
        if action is None:
            action = self._compute_action_stub(self._latest_cmd)

        self._previous_action = action

        targets = JointTargets()
        targets.header.stamp = self.get_clock().now().to_msg()
        targets.positions = action
        targets.velocities = [0.0] * NUM_JOINTS
        targets.source = 'locomotion'
        self._targets_pub.publish(targets)


def main(args=None):
    rclpy.init(args=args)
    node = LocomotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
