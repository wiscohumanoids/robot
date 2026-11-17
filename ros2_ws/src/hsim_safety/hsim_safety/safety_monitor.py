"""Safety monitor (node name: `safety`).

Watches the robot's health and decides whether it may move. Two outputs:
  /safety/estop   std_msgs/Bool, 10 Hz. A twist_mux LOCK at top priority: while true, no
                  velocity command reaches the walking policy, whoever sent it.
  /safety/status  diagnostic_msgs/DiagnosticArray, 2 Hz: OK / WARN / ERROR with reasons.
                  The behavior tree's IsSafe check reads this every tick.

E-stop (ERROR) when any of:
  - the operator requested it (/safety/estop_request, latched until released)
  - the torso is tilted more than `max_tilt_deg` (falling / fallen)
  - battery below `battery_critical`
  - joint encoders or odometry have gone silent (> `stale_s`)
Warning (WARN, robot may still move) when: battery below `battery_low`, lidar silent.
Automatic e-stops clear themselves `clear_hold_s` after the condition is gone.
"""

import math
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import BatteryState, Imu, JointState, LaserScan
from std_msgs.msg import Bool


class SafetyMonitor(Node):
    """Node `safety`: owns /safety/estop; engages it on an operator request, a fall, a critical battery or stale
    sensors.
    """
    def __init__(self):
        """Declare the thresholds, subscribe to the monitored topics, create the publishers and timers."""
        super().__init__('safety')
        for name, default in (('max_tilt_deg', 35.0), ('battery_low', 0.20), ('battery_critical', 0.05),
                              ('stale_s', 0.5), ('scan_stale_s', 1.0), ('clear_hold_s', 2.0)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value
        self.max_tilt, self.batt_low, self.batt_crit = p('max_tilt_deg'), p('battery_low'), p('battery_critical')
        self.stale, self.scan_stale, self.hold = p('stale_s'), p('scan_stale_s'), p('clear_hold_s')

        self.last = {}                    # topic -> wall time of last message
        self.tilt_deg = 0.0
        self.battery = None
        self.operator_estop = False
        self.estop_since_clear = None     # wall time when automatic conditions last cleared
        self.estop = False

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_estop = self.create_publisher(Bool, '/safety/estop', 10)
        self.pub_status = self.create_publisher(DiagnosticArray, '/safety/status', 10)
        self.create_subscription(Imu, '/imu', self.on_imu, 10)
        self.create_subscription(JointState, '/joint_states', lambda m: self.seen('joint_states'), 10)
        self.create_subscription(Odometry, '/odometry/filtered', lambda m: self.seen('odometry'), 10)
        self.create_subscription(LaserScan, '/scan', lambda m: self.seen('scan'), 5)
        self.create_subscription(BatteryState, '/battery', self.on_battery, 5)
        self.create_subscription(Bool, '/safety/estop_request', self.on_request, latched)
        self.create_timer(0.1, self.tick)
        self.create_timer(0.5, self.publish_status)
        self.reasons, self.warnings = [], []
        self.get_logger().info('safety monitor up: e-stop on operator request, fall, critical battery, stale sensors')

    def seen(self, key):
        """Mark a source as alive."""
        self.last[key] = time.monotonic()

    def on_imu(self, m: Imu):
        """Mark the IMU as alive and compute the body tilt from vertical."""
        self.seen('imu')
        q = m.orientation
        # angle between the body's up axis and gravity
        up_z = 1 - 2 * (q.x * q.x + q.y * q.y)
        self.tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))

    def on_battery(self, m: BatteryState):
        """Store the battery level."""
        self.battery = m.percentage

    def on_request(self, m: Bool):
        """Apply an operator e-stop request."""
        if m.data != self.operator_estop:
            self.get_logger().warn(f'operator e-stop {"ENGAGED" if m.data else "released"}')
        self.operator_estop = m.data

    def age(self, key):
        """Seconds since a source was last seen."""
        return time.monotonic() - self.last[key] if key in self.last else math.inf

    def tick(self):
        """Collect stop reasons and warnings; publish the e-stop, which only clears after `clear_hold_s` without
        reasons.
        """
        reasons, warnings = [], []
        if self.operator_estop:
            reasons.append('operator e-stop')
        if self.tilt_deg > self.max_tilt:
            reasons.append(f'tilted {self.tilt_deg:.0f} deg (falling)')
        if self.battery is not None and self.battery < self.batt_crit:
            reasons.append(f'battery critical {100 * self.battery:.0f}%')
        for key in ('joint_states', 'odometry'):
            if self.age(key) > self.stale and key in self.last:
                reasons.append(f'{key} silent for {self.age(key):.1f} s')
        if self.battery is not None and self.batt_crit <= self.battery < self.batt_low:
            warnings.append(f'battery low {100 * self.battery:.0f}%')
        if 'scan' in self.last and self.age('scan') > self.scan_stale:
            warnings.append(f'lidar silent for {self.age("scan"):.1f} s')

        now = time.monotonic()
        if reasons:
            if not self.estop:
                self.get_logger().error('E-STOP: ' + '; '.join(reasons))
            self.estop, self.estop_since_clear = True, None
        elif self.estop:
            # operator e-stop clears immediately on release; automatic ones after a hold time
            if self.estop_since_clear is None:
                self.estop_since_clear = now
            if now - self.estop_since_clear >= self.hold:
                self.estop = False
                self.get_logger().info('e-stop cleared')
        self.reasons, self.warnings = reasons, warnings
        self.pub_estop.publish(Bool(data=self.estop))

    def publish_status(self):
        """Publish /safety/status: e-stop with its reasons, warnings, or OK."""
        st = DiagnosticStatus(name='safety', hardware_id='robot')
        if self.estop:
            st.level, st.message = DiagnosticStatus.ERROR, 'E-STOP: ' + ('; '.join(self.reasons) or 'clearing')
        elif self.warnings:
            st.level, st.message = DiagnosticStatus.WARN, '; '.join(self.warnings)
        else:
            st.level, st.message = DiagnosticStatus.OK, 'ok'
        st.values = [
            KeyValue(key='estop', value=str(self.estop)),
            KeyValue(key='battery', value='n/a' if self.battery is None else f'{self.battery:.3f}'),
            KeyValue(key='tilt_deg', value=f'{self.tilt_deg:.1f}'),
            KeyValue(key='operator_estop', value=str(self.operator_estop)),
        ]
        arr = DiagnosticArray(status=[st])
        arr.header.stamp = self.get_clock().now().to_msg()
        self.pub_status.publish(arr)


def main():
    """Entry point: spin the node until Ctrl-C."""
    rclpy.init()
    node = SafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
