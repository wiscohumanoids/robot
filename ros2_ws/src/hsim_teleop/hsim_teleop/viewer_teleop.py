"""Teleop bridge (node name: `teleop`): Mac MuJoCo viewer keys -> /cmd_vel_teleop, typed tasks -> planner.

The viewer (tools/viewer.sh) sends its current command over UDP ~30 times a second:
    b"HCMD", float32 vx, vy, wz, uint8 flags   (flags bit 0 = reset requested)
This node republishes the latest command as geometry_msgs/TwistStamped on
/cmd_vel_teleop at 50 Hz while packets keep arriving AND the command is not zero (plus 0.5 s of zeros
after you stop, so the robot halts). An idle viewer sends zeros all the time; publishing those would
let the viewer (priority 100 in twist_mux) silently block navigation. If the viewer stops
(closed, crashed), publishing stops and twist_mux times out to zero, so the
robot stands still. A reset request calls the sim's /sim/reset service.

Tasks typed in the viewer's terminal arrive as b"HTXT" + UTF-8 text and are published on
/planner/command, exactly like `tools/ask.sh "..."`.

Docker Desktop doesn't always forward UDP from the Mac INTO the container, but replies to traffic the
container started do get through. So this node also says b"HELO" to the viewer (viewer_host:hello_port,
5 times a second) from its listening socket, and the viewer sends its commands back to where the hello
came from.
"""

import socket
import struct
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

PACKET = struct.Struct('<4s3fB')


class ViewerTeleop(Node):
    def __init__(self):
        super().__init__('teleop')
        self.declare_parameter('port', 9871)
        self.declare_parameter('timeout', 0.3)
        self.declare_parameter('viewer_host', 'host.docker.internal')
        self.declare_parameter('hello_port', 9873)
        self.timeout = self.get_parameter('timeout').value
        port = self.get_parameter('port').value

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(('0.0.0.0', port))
        except OSError as e:
            raise SystemExit(f'udp/{port} is taken ({e}): is another stack running? tools/ros.sh stop')
        self.sock.setblocking(False)
        try:
            self.hello_addr = (socket.gethostbyname(self.get_parameter('viewer_host').value),
                               self.get_parameter('hello_port').value)
        except OSError:
            self.hello_addr = None
        self.create_timer(0.2, self.hello)

        self.pub = self.create_publisher(TwistStamped, '/cmd_vel_teleop', 10)
        self.reset_client = self.create_client(Trigger, '/sim/reset')
        self.pub_task = self.create_publisher(String, '/planner/command', 10)
        self.cmd, self.last_rx, self.connected = (0.0, 0.0, 0.0), 0.0, False
        self.last_active = -1e9                  # last time the command was not zero
        self.create_timer(0.02, self.tick)   # 50 Hz, wall time: keys are a wall-clock thing
        self.get_logger().info(f'listening for viewer commands on udp/{port}')

    def hello(self):
        if self.hello_addr:
            try:
                self.sock.sendto(b'HELO', self.hello_addr)
            except OSError:
                pass                                   # viewer not running

    def tick(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            if data[:4] == b'HTXT':
                text = data[4:].decode('utf-8', 'replace').strip()
                if text:
                    self.get_logger().info(f'task from the viewer: "{text}"')
                    self.pub_task.publish(String(data=text))
                continue
            if len(data) != PACKET.size:
                continue
            magic, vx, vy, wz, flags = PACKET.unpack(data)
            if magic != b'HCMD':
                continue
            self.cmd, self.last_rx = (vx, vy, wz), time.monotonic()
            if flags & 1 and self.reset_client.service_is_ready():
                self.reset_client.call_async(Trigger.Request())

        fresh = time.monotonic() - self.last_rx < self.timeout
        if fresh != self.connected:
            self.connected = fresh
            self.get_logger().info('viewer connected' if fresh else 'viewer gone: stopped sending commands')
        if not fresh:
            return
        now = time.monotonic()
        if any(abs(v) > 1e-6 for v in self.cmd):
            self.last_active = now
        elif now - self.last_active > 0.5:
            return                               # keys idle: leave the robot to navigation
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z = self.cmd
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ViewerTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
