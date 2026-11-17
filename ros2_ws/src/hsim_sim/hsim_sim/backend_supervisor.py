"""Backend supervisor (node name: `backend_supervisor`) for the sim stack's controller-manager backends
(backend:=ros2_control and ethercat_sim).

On these backends the robot is run by its own controllers (JointTargetController instances `leg_targets`,
`arm_targets` from wisco_ethercat_ros2_hw, fed by the walking policy's /locomotion/joint_targets) and MuJoCo runs inside
the controller manager (mujoco_ros2_control, or hsim_mujoco_plugins' VirtualRobotSystem under simulated EtherCAT
drives). This node does what the hsim_sim MuJoCo node did around them:

- publishes the backend's description (`robot_description_file`) on /robot_description, latched, which is where the
  controller manager reads it from (the description holds only the `<ros2_control>` block, so no
  robot_state_publisher runs and nothing but the stack's own nodes publishes TF);
- starts the robot standing, like lowering it from a gantry: the simulation runs from the moment the plugin loads,
  but nothing commands the actuators until the spawners finish, and a position hold alone does not keep a humanoid
  up (it topples in about two seconds). Once every controller in `controllers` is active, it pauses the simulation,
  resets the world to the standing keyframe, asks each controller in `hold_controllers` to hold the pose it now
  measures (`~/hold`), and unpauses. /clock stops while paused, so the policy (on sim time) starts with the robot
  standing, and the controllers ramp in from there;
- offers /sim/reset (std_srvs/Trigger) as the MuJoCo node does: the same pause, reset, hold, unpause;
- watches for a second stack: two simulations (two /clock publishers) or two walking policies (two
  /locomotion/joint_targets publishers) fight over the robot and knock it over; it logs an error naming the fix
  (tools/ros.sh stop, then one launch) every few seconds while that lasts.

Parameters: `robot_description_file`, `controllers` (wait until active), `hold_controllers`, `keyframe`,
`plugin_node` (the plugin's node, whose reset_world and set_pause services are used: /mujoco_ros2_control_node or
/virtual_robot).
"""

import rclpy
from controller_manager_msgs.srv import ListControllers
from mujoco_ros2_control_msgs.srv import ResetWorld, SetPause
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


class BackendSupervisor(Node):
    """Description publisher, standing start and world reset for the controller-manager backends."""

    def __init__(self):
        """Publish the description, create the service clients and /sim/reset, start the startup poll."""
        super().__init__('backend_supervisor')
        self.declare_parameter('robot_description_file', '')
        self.declare_parameter('controllers', ['leg_targets'])
        self.declare_parameter('hold_controllers', ['leg_targets'])
        self.declare_parameter('keyframe', 'init_state')
        self.declare_parameter('plugin_node', '/mujoco_ros2_control_node')
        description = self.get_parameter('robot_description_file').value
        if description:
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            self.pub_description = self.create_publisher(String, '/robot_description', latched)
            self.pub_description.publish(String(data=open(description).read()))
        plugin = self.get_parameter('plugin_node').value
        self.wanted = list(self.get_parameter('controllers').value)
        self.reset_client = self.create_client(ResetWorld, plugin + '/reset_world')
        self.pause_client = self.create_client(SetPause, plugin + '/set_pause')
        self.list_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        self.hold_clients = [self.create_client(Trigger, f'/{c}/hold')
                             for c in self.get_parameter('hold_controllers').value]
        self.create_service(Trigger, '/sim/reset', self.on_reset)
        self.started = False
        self.asking = False
        # steady clock: /clock stops while the simulation is paused
        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.poll = self.create_timer(0.2, self.wait_for_controllers, clock=steady)
        self.create_timer(3.0, self.check_single_stack, clock=steady)

    def check_single_stack(self):
        """Log an error while another simulation or a second walking policy is running."""
        clocks = self.count_publishers('/clock')
        policies = self.count_publishers('/locomotion/joint_targets')
        if clocks > 1 or policies > 1:
            self.get_logger().error(
                f'a second stack is running ({clocks} simulations, {policies} walking policies): they fight over the '
                'robot and knock it over. Stop everything (tools/ros.sh stop), then start one launch.')

    def wait_for_controllers(self):
        """Startup: once every wanted controller is active, run the standing start (once)."""
        if self.started or self.asking or not self.list_client.service_is_ready():
            return
        self.asking = True
        self.list_client.call_async(ListControllers.Request()).add_done_callback(self.on_list)

    def on_list(self, future):
        """Start when every wanted controller reports active."""
        self.asking = False
        active = {c.name for c in future.result().controller if c.state == 'active'}
        missing = [c for c in self.wanted if c not in active]
        if missing or self.started or not self.services_ready():
            return
        self.started = True
        self.poll.cancel()
        self.release()
        self.get_logger().info(f'controllers active ({", ".join(self.wanted)}): robot reset to the standing keyframe')

    def services_ready(self):
        """The plugin's reset/pause services and every controller's ~/hold are up."""
        return self.reset_client.service_is_ready() and self.pause_client.service_is_ready() and \
            all(c.service_is_ready() for c in self.hold_clients)

    def release(self):
        """Pause, reset to the keyframe, have the controllers hold the new pose, unpause (each after the previous)."""
        def unpause(_f=None):
            self.pause_client.call_async(SetPause.Request(paused=False))

        def hold(_f):
            pending = [c.call_async(Trigger.Request()) for c in self.hold_clients]
            if not pending:
                unpause()
                return
            left = {'n': len(pending)}

            def one_done(_g):
                left['n'] -= 1
                if left['n'] == 0:
                    unpause()
            for p in pending:
                p.add_done_callback(one_done)

        def reset(_f):
            self.reset_client.call_async(
                ResetWorld.Request(keyframe=self.get_parameter('keyframe').value)).add_done_callback(hold)
        self.pause_client.call_async(SetPause.Request(paused=True)).add_done_callback(reset)

    def on_reset(self, _req, res):
        """/sim/reset: back to the standing keyframe, controllers holding it."""
        if not self.services_ready():
            res.success, res.message = False, 'the plugin\'s reset_world/set_pause or a controller\'s ~/hold is not up'
            return res
        self.release()
        res.success, res.message = True, 'reset to the standing keyframe'
        return res


def main():
    """Run the node."""
    rclpy.init()
    rclpy.spin(BackendSupervisor())


if __name__ == '__main__':
    main()
