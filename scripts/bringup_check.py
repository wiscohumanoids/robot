#!/usr/bin/env python3
"""Check that the joint table matches the wiring, one drive at a time.

Run with the bench up in velocity mode and the arm clear to move:

  ros2 launch wisco_ethercat_ros2_hw test_bench.launch.py
  ros2 run wisco_ethercat_ros2_hw bringup_check.py

For each joint in bus order it jogs that joint a little, then checks:
  * the named joint moved, by about the commanded amount, in the commanded
    direction (so the table's `direction` matches the URDF axis), and
  * no other joint moved (so bus positions are not swapped),
and asks the operator to confirm the right physical joint moved the right way.
If the `ethercat` tool is installed it also compares the slave count on the
ring with the joint table. Exits non-zero if anything failed.
"""

import shutil
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from wisco_ethercat_ros2_hw.ros_tools import get_controller_limits


class BringupCheck(Node):

    def __init__(self):
        super().__init__('bringup_check')
        self.declare_parameter('controller', 'velocity_controller')
        self.declare_parameter('jog_rad', 0.15)
        self.declare_parameter('jog_velocity', 0.3)  # rad/s
        self.declare_parameter('settle_s', 0.5)
        self.declare_parameter('still_threshold_rad', 0.02)
        self.declare_parameter('interactive', True)
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.controller = p('controller')
        self.jog_rad = float(p('jog_rad'))
        self.jog_velocity = float(p('jog_velocity'))
        self.settle_s = float(p('settle_s'))
        self.still = float(p('still_threshold_rad'))
        self.interactive = bool(p('interactive'))

        limits = get_controller_limits(self, self.controller)
        if limits['interface_name'] != 'velocity':
            raise RuntimeError('bring-up check needs velocity mode; %s commands %s'
                               % (self.controller, limits['interface_name']))
        self.joints = limits['joints']
        self.lo = limits['position_min']
        self.hi = limits['position_max']
        self.max_command = limits['max_command']

        self.lock = threading.Lock()
        self.positions = {}
        self.pub = self.create_publisher(Float64MultiArray, '/%s/commands' % self.controller, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)

    def _on_joint_states(self, msg):
        with self.lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.positions[name] = msg.position[i]

    def snapshot(self):
        with self.lock:
            return [self.positions.get(j, float('nan')) for j in self.joints]

    def hold(self, command, seconds):
        """Publish `command` continuously (the controller times out otherwise)."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.pub.publish(Float64MultiArray(data=command))
            time.sleep(0.01)

    def jog(self, k, start):
        """Jog joint k away from its nearer limit; return the sign that was commanded."""
        sign = 1.0 if (self.hi[k] - start[k]) >= (start[k] - self.lo[k]) else -1.0
        speed = min(self.jog_velocity, self.max_command[k])
        command = [0.0] * len(self.joints)
        command[k] = sign * speed
        self.hold(command, self.jog_rad / speed)
        self.hold([0.0] * len(self.joints), self.settle_s)
        return sign


def ask(question):
    while True:
        answer = input('%s [y/n/q] ' % question).strip().lower()
        if answer in ('y', 'n', 'q'):
            return answer


def count_slaves():
    if shutil.which('ethercat') is None:
        return None
    try:
        out = subprocess.run(['ethercat', 'slaves'], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return len([line for line in out.stdout.splitlines() if line.strip()])


def main(args=None):
    rclpy.init(args=args)
    node = BringupCheck()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    failures = []
    try:
        slaves = count_slaves()
        if slaves is None:
            print('ethercat tool not found: skipping the slave count check')
        elif slaves != len(node.joints):
            failures.append('ring has %d slave(s) but the joint table lists %d'
                            % (slaves, len(node.joints)))
        else:
            print('ring has %d slaves, matching the joint table' % slaves)

        deadline = time.monotonic() + 10.0
        while any(p != p for p in node.snapshot()):  # NaN: no state yet
            if time.monotonic() > deadline:
                raise RuntimeError('no joint_states for all of %s' % ', '.join(node.joints))
            time.sleep(0.1)

        for k, joint in enumerate(node.joints):
            print('\n[%d/%d] bus position %d: %s' % (k + 1, len(node.joints), k, joint))
            if node.interactive and ask('Jog %s by %.2f rad now?' % (joint, node.jog_rad)) != 'y':
                failures.append('%s: skipped' % joint)
                continue
            before = node.snapshot()
            sign = node.jog(k, before)
            after = node.snapshot()
            delta = [a - b for a, b in zip(after, before)]

            moved = delta[k]
            ok = moved * sign > 0.5 * node.jog_rad
            print('  commanded %+.2f rad, measured %+.3f rad' % (sign * node.jog_rad, moved))
            if not ok:
                failures.append('%s: commanded %+.2f rad, measured %+.3f rad (wrong bus position, '
                                'wrong direction, or not enabled)'
                                % (joint, sign * node.jog_rad, moved))
            others = [(node.joints[i], d) for i, d in enumerate(delta)
                      if i != k and abs(d) > node.still]
            for name, d in others:
                failures.append('%s moved %+.3f rad while jogging %s (bus positions swapped?)'
                                % (name, d, joint))
            if node.interactive:
                answer = ask('Did the physical %s move %s along its URDF axis?'
                             % (joint, 'positively' if sign > 0 else 'negatively'))
                if answer == 'q':
                    break
                if answer == 'n':
                    failures.append('%s: operator saw the wrong joint or direction' % joint)
    except KeyboardInterrupt:
        failures.append('interrupted')
    finally:
        try:
            node.hold([0.0] * len(node.joints), 0.2)
        except Exception:  # noqa: BLE001 - context may already be shut down
            pass
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print()
    if failures:
        print('FAILED:')
        for f in failures:
            print('  - ' + f)
        sys.exit(1)
    print('PASSED: every drive is where the joint table says, moving the right way')


if __name__ == '__main__':
    main()
