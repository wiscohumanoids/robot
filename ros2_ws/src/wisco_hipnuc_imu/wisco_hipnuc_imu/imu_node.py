"""HiPNUC IMU driver (node name: `imu`): the robot's IMU board over USB serial → /imu.

Runs on the RK3588 (Rockchip), where the board is plugged in. Reads HI91 packets (protocol.py), rotates them from
the board's axes into `imu_link` (pelvis axes, imu_math.py) and publishes:

  /imu           sensor_msgs/Imu, frame imu_link, stamped with the board's clock carried onto the host clock
                 (stamping.py): orientation (heading zeroed at start, like the simulated IMU),
                 angular velocity (rad/s), linear acceleration (m/s^2, specific force: +9.81 up at rest)
  /diagnostics   rate, CRC errors, dropped bytes, the board's status warnings (bias / attitude not converged,
                 saturation) and its coordinate setting, once a second

The stack's consumers (walking policy, state estimator, safety monitor) read /imu exactly as from the simulator.
If the port disappears (cable, power) the node keeps retrying and reports the gap; on reconnect it zeroes the heading
again only if `rezero_on_reconnect` is set.

Parameters: config/hipnuc.yaml (port, baudrate, mount_wxyz, expected_coord, expected_rate_hz, covariances).
"""

import threading
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from sensor_msgs.msg import Imu

from wisco_hipnuc_imu.imu_math import MountConverter
from wisco_hipnuc_imu.protocol import Decoder
from wisco_hipnuc_imu.stamping import DeviceClock


class ImuNode(Node):
    """Serial reader thread + publisher."""

    def __init__(self):
        """Declare parameters, build the converter, start the reader thread and the diagnostics timer."""
        super().__init__('imu')
        p = {
            'port': '/dev/hipnuc_imu', 'baudrate': 115200, 'frame_id': 'imu_link',
            'mount_wxyz': [1.0, 0.0, 0.0, 0.0], 'zero_yaw': True, 'rezero_on_reconnect': False,
            'expected_coord': 0, 'expected_rate_hz': 200.0, 'use_device_time': True,
            'orientation_covariance': [1e-4, 1e-4, 1e-2], 'angular_velocity_covariance': [1.2e-5] * 3,
            'linear_acceleration_covariance': [9e-4] * 3,
        }
        for k, v in p.items():
            self.declare_parameter(k, v)
        g = lambda k: self.get_parameter(k).value                        # noqa: E731
        self.port, self.baud, self.frame = g('port'), int(g('baudrate')), g('frame_id')
        self.expected_coord, self.expected_rate = int(g('expected_coord')), float(g('expected_rate_hz'))
        self.rezero = bool(g('rezero_on_reconnect'))
        self.conv = MountConverter(g('mount_wxyz'), zero_yaw=bool(g('zero_yaw')))
        self.cov = {k: [float(x) for x in g(k)] for k in
                    ('orientation_covariance', 'angular_velocity_covariance', 'linear_acceleration_covariance')}
        self.decoder = Decoder()
        self.clock = DeviceClock() if g('use_device_time') else None
        self.pub = self.create_publisher(Imu, '/imu', 10)     # reliable, as the simulator's /imu: its subscribers expect that
        self.pub_diag = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.lock = threading.Lock()
        self.count, self.last_sample, self.connected, self.last_status = 0, None, False, None
        self.window = (time.monotonic(), 0)
        self.running = True
        self.reader = threading.Thread(target=self.read_loop, daemon=True)
        self.reader.start()
        self.create_timer(1.0, self.publish_diagnostics)

    def read_loop(self):
        """Open the port (retrying every second), feed bytes to the decoder, publish every sample."""
        import serial
        first = True
        while self.running and rclpy.ok():
            try:
                with serial.Serial(self.port, self.baud, timeout=0.05) as s:
                    if not first and self.rezero:
                        self.conv.reset_heading()
                    first = False
                    self.connected = True
                    self.get_logger().info(f'reading {self.port} at {self.baud} baud')
                    while self.running and rclpy.ok():
                        data = s.read(s.in_waiting or 1)
                        if data:
                            for sample in self.decoder.feed(data):
                                self.publish(sample)
            except Exception as e:          # serial.SerialException, OSError: unplugged, no permission, wrong path
                if self.connected or first:
                    self.get_logger().error(f'{self.port}: {e}; retrying every second')
                self.connected, first = False, False
                time.sleep(1.0)

    def publish(self, s):
        """One HI91 sample as sensor_msgs/Imu in imu_link."""
        m = Imu()
        now = self.get_clock().now()
        if self.clock is None:
            m.header.stamp = now.to_msg()
        else:
            ns = int(round(self.clock.stamp(s.system_time_ms, now.nanoseconds * 1e-9) * 1e9))
            m.header.stamp.sec, m.header.stamp.nanosec = divmod(ns, 1_000_000_000)
        m.header.frame_id = self.frame
        q = self.conv.orientation(s.quat_wxyz)
        if q is None:
            m.orientation_covariance[0] = -1.0          # sensor_msgs/Imu: orientation not available
        else:
            m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z = map(float, q)
            m.orientation_covariance = _diag(self.cov['orientation_covariance'])
        gx, gy, gz = self.conv.vector(s.gyr)
        ax, ay, az = self.conv.vector(s.acc)
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = float(gx), float(gy), float(gz)
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = float(ax), float(ay), float(az)
        m.angular_velocity_covariance = _diag(self.cov['angular_velocity_covariance'])
        m.linear_acceleration_covariance = _diag(self.cov['linear_acceleration_covariance'])
        self.pub.publish(m)
        with self.lock:
            self.count += 1
            self.last_sample, self.last_status = time.monotonic(), s

    def publish_diagnostics(self):
        """Rate, errors and the board's status, once a second."""
        now = time.monotonic()
        with self.lock:
            t0, n0 = self.window
            rate = (self.count - n0) / max(now - t0, 1e-6)
            self.window = (now, self.count)
            s, last = self.last_status, self.last_sample
        st = DiagnosticStatus(name='imu: HiPNUC', hardware_id=self.port)
        problems = []
        if not self.connected:
            problems.append('serial port not open')
        elif last is None or now - last > 0.5:
            problems.append('no samples (check baud rate and that the board outputs HI91)')
        elif rate < 0.9 * self.expected_rate:
            problems.append(f'{rate:.0f} Hz, expected {self.expected_rate:.0f} (README: output rate)')
        if s is not None:
            if s.coord != self.expected_coord:
                problems.append(f'board COORD {s.coord}, mount_wxyz assumes {self.expected_coord}')
            problems += s.warnings()
        st.level = DiagnosticStatus.OK if not problems else (
            DiagnosticStatus.ERROR if (not self.connected or last is None or now - last > 0.5) else DiagnosticStatus.WARN)
        st.message = '; '.join(problems) or 'ok'
        st.values = [KeyValue(key=k, value=str(v)) for k, v in (
            ('rate_hz', round(rate, 1)), ('frames', self.decoder.frames), ('crc_errors', self.decoder.crc_errors),
            ('skipped_bytes', self.decoder.skipped_bytes), ('other_packets', self.decoder.unknown),
            ('coord', s.coord if s else ''), ('temperature_c', s.temperature_c if s else ''),
            ('board_clock_resets', self.clock.resets if self.clock else ''))]
        arr = DiagnosticArray(status=[st])
        arr.header.stamp = self.get_clock().now().to_msg()
        self.pub_diag.publish(arr)

    def destroy_node(self):
        """Stop the reader thread."""
        self.running = False
        super().destroy_node()


def _diag(values):
    """Row-major 3x3 covariance with `values` on the diagonal."""
    return [float(values[i // 3]) if i % 4 == 0 else 0.0 for i in range(9)]


def main():
    """Run the node."""
    rclpy.init()
    node = ImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
