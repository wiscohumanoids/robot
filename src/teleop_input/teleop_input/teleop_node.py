"""teleop_input / teleop_node

STATUS: real.

FREQUENCY: 10 Hz (enforced by a single rclpy timer, see PUBLISH_RATE_HZ).

INPUTS:
  - Keyboard, read non-blockingly from stdin via termios/tty (same technique
    as the upstream teleop_twist_keyboard package). w/s = +/-vx, a/d = +/-vy,
    q/e = +/-vyaw, space = zero all, x = quit.
  - sensor_msgs/Joy on /joy, if a joystick driver (e.g. joy_node) is running.
    Left stick vertical -> vx, left stick horizontal -> vy, right stick
    horizontal -> vyaw. Whichever input source produced a nonzero command
    most recently wins (both write into the same self._vx/_vy/_vyaw state).

OUTPUT:
  geometry_msgs/Twist on /cmd_vel_teleop (not /cmd_vel, which belongs solely
  to cmd_vel_mux; it arbitrates teleop against navigation, see
  INTERFACE_CONTRACT.md). Published at 10 Hz, holding the last commanded value
  between key/joystick events (the command latches until changed, so the
  operator does not need to hold a key down), but only while the command is
  nonzero, plus one final zero when it returns to zero.
  Going quiet is what lets navigation take over again: cmd_vel_mux falls back
  to /cmd_vel_nav once teleop has been silent for its timeout.

Only linear.x, linear.y and angular.z are used (REP-103 base_link frame:
x-forward, y-left, z-up). The other three Twist fields are always zero.

KEYBOARD NOTE: keyboard input needs a real TTY, and `ros2 launch` does not give
its nodes one. Run this node in its own terminal (`ros2 run teleop_input
teleop_node`); bringup's full_stack.launch.py starts it only with teleop:=true,
where only the joystick path works.
"""
import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

PUBLISH_RATE_HZ = 10.0
LINEAR_STEP = 0.1     # m/s per keypress
ANGULAR_STEP = 0.2    # rad/s per keypress
JOY_LINEAR_SCALE = 0.5
JOY_ANGULAR_SCALE = 1.0

KEY_BINDINGS = {
    'w': (LINEAR_STEP, 0.0, 0.0),
    's': (-LINEAR_STEP, 0.0, 0.0),
    'a': (0.0, LINEAR_STEP, 0.0),
    'd': (0.0, -LINEAR_STEP, 0.0),
    'q': (0.0, 0.0, ANGULAR_STEP),
    'e': (0.0, 0.0, -ANGULAR_STEP),
}


def _get_key(settings, timeout_s=0.0):
    """Non-blocking single-character stdin read. Returns '' if nothing
    was typed within timeout_s. Standard termios raw-mode technique."""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout_s)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class TeleopNode(Node):

    def __init__(self):
        super().__init__('teleop_input')
        self._vx = 0.0
        self._vy = 0.0
        self._vyaw = 0.0

        self._was_active = False
        self._pub = self.create_publisher(Twist, '/cmd_vel_teleop', 10)
        self._joy_sub = self.create_subscription(Joy, '/joy', self._on_joy, 10)
        self._timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self._on_timer)

        self._term_settings = None
        if sys.stdin.isatty():
            self._term_settings = termios.tcgetattr(sys.stdin)
            self._key_timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self._on_key_poll)
        else:
            self.get_logger().warn(
                'stdin is not a TTY, keyboard input disabled, joystick-only mode.')

        self.get_logger().info(
            f'teleop_input up: w/s=vx a/d=vy q/e=vyaw space=zero x=quit, '
            f'publishing at {PUBLISH_RATE_HZ} Hz on /cmd_vel_teleop while active')

    def _on_key_poll(self):
        key = _get_key(self._term_settings, timeout_s=0.0)
        if key == '':
            return
        if key == ' ':
            self._vx = self._vy = self._vyaw = 0.0
        elif key == 'x' or key == '\x03':  # 'x' or Ctrl-C
            self.get_logger().info('quit key received, shutting down teleop_input')
            rclpy.shutdown()
        elif key in KEY_BINDINGS:
            dvx, dvy, dvyaw = KEY_BINDINGS[key]
            self._vx += dvx
            self._vy += dvy
            self._vyaw += dvyaw

    def _on_joy(self, msg: Joy):
        if len(msg.axes) < 4:
            return
        # Common convention: axes[1]=left-stick-vertical, axes[0]=left-stick-horizontal,
        # axes[3]=right-stick-horizontal. Verify against your specific joystick's
        # joy_node output before trusting these indices on real hardware.
        self._vx = msg.axes[1] * JOY_LINEAR_SCALE
        self._vy = msg.axes[0] * JOY_LINEAR_SCALE
        self._vyaw = msg.axes[3] * JOY_ANGULAR_SCALE

    def _on_timer(self):
        active = self._vx != 0.0 or self._vy != 0.0 or self._vyaw != 0.0
        if not active and not self._was_active:
            return  # stay silent so cmd_vel_mux can hand control to navigation
        self._was_active = active  # the tick that returns to zero still publishes it
        msg = Twist()
        msg.linear.x = self._vx
        msg.linear.y = self._vy
        msg.angular.z = self._vyaw
        self._pub.publish(msg)

    def destroy_node(self):
        if self._term_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._term_settings)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
