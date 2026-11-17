"""Joint target bridge (node name: `joint_target_bridge`) for the ros2_control backend.

The walking policy publishes /locomotion/joint_targets (sensor_msgs/JointState, targets by name, possibly a subset of
the joints: arms-free policies leave the arms out). A ros2_control forward controller takes a Float64MultiArray in
its own joint order. This node keeps one target per controller joint, starting at the standing pose, overwrites the
named ones on every message and publishes the full vector to /joint_targets_controller/commands.

Startup, like lowering the robot from a gantry: the simulation runs from the moment the plugin loads, but nothing
commands the actuators until the spawners finish, and a position hold alone does not keep a humanoid up (it topples in
about two seconds). Once the forward controller is subscribed (the simulation cannot be paused earlier: the controller
manager cycles on /clock and the spawners would time out), this node pauses the simulation, publishes the standing
targets, resets the world to the standing keyframe and unpauses. Targets from the policy before that moment are
ignored. The standing targets are republished at `hold_hz` (steady clock) until the policy's next message.

Service: /sim/reset (std_srvs/Trigger), as the MuJoCo sim node offers it: back to the standing keyframe and the
standing hold.

It also publishes the backend's description (`robot_description_file`) on /robot_description, latched, which is where
controller_manager reads it from. The description holds only the `<ros2_control>` block, so no robot_state_publisher
runs and nothing but the stack's own nodes publishes TF.

Parameters: `joints_file` (joints.yaml written by ros2_control_backend: joint order and standing targets), `hold_hz`,
`keyframe`, `plugin_node` (the plugin's node name, whose reset_world and set_pause services are used),
`robot_description_file` (robot.urdf written by ros2_control_backend).
"""

import rclpy
import yaml
from mujoco_ros2_control_msgs.srv import ResetWorld, SetPause
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger


class JointTargetBridge(Node):
    """Named joint targets → the forward position controller's command array."""

    def __init__(self):
        """Read the joint order and standing pose, then subscribe, offer /sim/reset and start the startup timer."""
        super().__init__('joint_target_bridge')
        self.declare_parameter('joints_file', '')
        self.declare_parameter('hold_hz', 20.0)
        self.declare_parameter('keyframe', 'init_state')
        self.declare_parameter('plugin_node', '/mujoco_ros2_control_node')
        self.declare_parameter('robot_description_file', '')
        spec = yaml.safe_load(open(self.get_parameter('joints_file').value))
        self.index = {n: i for i, n in enumerate(spec['joints'])}
        self.standing = [float(v) for v in spec['standing']]
        self.targets = list(self.standing)
        self.holding = True        # republishing the standing targets until the policy's first message
        self.running = False       # the startup reset has been sent
        self.unknown_warned = set()
        self.pub = self.create_publisher(Float64MultiArray, '/joint_targets_controller/commands', 10)
        description = self.get_parameter('robot_description_file').value
        if description:
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            self.pub_description = self.create_publisher(String, '/robot_description', latched)
            self.pub_description.publish(String(data=open(description).read()))
        self.create_subscription(JointState, '/locomotion/joint_targets', self.on_targets, 10)
        plugin = self.get_parameter('plugin_node').value
        self.reset_client = self.create_client(ResetWorld, plugin + '/reset_world')
        self.pause_client = self.create_client(SetPause, plugin + '/set_pause')
        self.create_service(Trigger, '/sim/reset', self.on_reset)
        # steady clock: /clock stops while the simulation is paused
        self.create_timer(1.0 / self.get_parameter('hold_hz').value, self.tick,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))

    def tick(self):
        """Startup (once the controller listens: pause, stand, reset, unpause) and the standing hold."""
        if self.pub.get_subscription_count() == 0:
            return
        if not self.running:
            if not (self.reset_client.service_is_ready() and self.pause_client.service_is_ready()):
                return
            self.running = True
            self.release()
            self.get_logger().info('controller active: robot reset to the standing keyframe')
        if self.holding:
            self.publish()

    def release(self):
        """Pause, command the standing pose, reset to the keyframe, unpause (each call after the previous answers)."""
        self.targets, self.holding = list(self.standing), True
        unpause = lambda _f: self.pause_client.call_async(SetPause.Request(paused=False))   # noqa: E731
        self.pause_client.call_async(SetPause.Request(paused=True)).add_done_callback(
            lambda _f: (self.publish(), self.reset_world().add_done_callback(unpause)))

    def reset_world(self):
        """Ask the plugin to restore the standing keyframe; returns the future."""
        return self.reset_client.call_async(ResetWorld.Request(keyframe=self.get_parameter('keyframe').value))

    def on_reset(self, _req, res):
        """/sim/reset: standing keyframe and standing targets, held until the policy publishes again."""
        if not (self.reset_client.service_is_ready() and self.pause_client.service_is_ready()):
            res.success, res.message = False, 'the plugin\'s reset_world/set_pause services are not available'
            return res
        self.release()
        res.success, res.message = True, 'reset to the standing keyframe'
        return res

    def on_targets(self, msg):
        """Overwrite the named targets and publish; ends the standing hold. Ignored before the startup reset."""
        if not self.running:
            return
        for name, pos in zip(msg.name, msg.position):
            i = self.index.get(name)
            if i is None:
                if name not in self.unknown_warned:
                    self.get_logger().warn(f'target for unknown joint "{name}" ignored')
                    self.unknown_warned.add(name)
                continue
            self.targets[i] = float(pos)
        self.holding = False
        self.publish()

    def publish(self):
        """Send the full target vector."""
        self.pub.publish(Float64MultiArray(data=self.targets))


def main():
    """Run the node."""
    rclpy.init()
    rclpy.spin(JointTargetBridge())


if __name__ == '__main__':
    main()
