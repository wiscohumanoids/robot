"""ArmClient on the real arm: talks to the ArmController (src/arm_controller.cpp).

  /arm_controller/command  Float64MultiArray, published per send():
      [mode (0 P, 1 T), flags (1 gravity_ff, 2 ignore_soft_limits), friction_ff_gain,
       q[n], qd[n], kp[n], kd[n], tau_ff[n], torque_clamp[n]]   NaN = not set
      optionally followed by [payload_kg, x, y, z] (point mass in tool0; sticky in the
      controller, kg <= 0 clears it); sent only when Command.payload is set
  /arm_controller/state    Float64MultiArray, every RT cycle:
      [t, mode, fallback, limit_hit, clamped, q[n], qd[n], tau[n], tau_out[n], q_target[n]]

The RT loop owns timing and the 50 ms watchdog; this client only streams
targets. Keep the layouts in sync with arm_controller.hpp.

What the state topic does not carry, and how this client fills it:
  raw_counts    re-derived from q with the configured units (a later unit fix
                cannot reprocess these; log 0x6064 directly once the drive
                manifest exposes it as a state interface)
  raw_permille  re-derived from tau the same way
  status        0x6041 is not exposed by the controller yet; reported as
                0x1237 (operation enabled) unless the watchdog fallback is on
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np

from ..config import ArmConfig
from ..units import Units
from .client import ArmClient, Command, JointState

HEADER = 3
STATE_HEADER = 5
PAYLOAD_BLOCK = 4


def encode_command(cmd: Command, n: int) -> list:
    def block(v):
        return [math.nan] * n if v is None else [float(x) for x in np.broadcast_to(np.asarray(v, float), (n,))]
    flags = (1 if cmd.gravity_ff else 0) | (2 if cmd.ignore_soft_limits else 0)
    out = [1.0 if cmd.mode == 'T' else 0.0, float(flags), float(cmd.friction_ff_gain)]
    for v in (cmd.q, cmd.qd, cmd.kp, cmd.kd, cmd.tau_ff, cmd.torque_clamp):
        out += block(v)
    payload = getattr(cmd, 'payload', None)
    if payload is not None:
        # optional trailing block; the controller keeps it until the next one
        out += [float(x) for x in payload][:PAYLOAD_BLOCK]
    return out


def decode_state(data, n: int):
    d = np.asarray(data, float)
    if len(d) != STATE_HEADER + 5 * n:
        raise ValueError('state has %d values, expected %d' % (len(d), STATE_HEADER + 5 * n))
    b = lambda k: d[STATE_HEADER + k * n:STATE_HEADER + (k + 1) * n]  # noqa: E731
    return {'t': d[0], 'mode': 'T' if d[1] > 0.5 else 'P', 'fallback': d[2] > 0.5,
            'limit_hit': d[3] > 0.5, 'clamped': d[4] > 0.5, 'q': b(0), 'qd': b(1), 'tau': b(2),
            'tau_out': b(3), 'q_target': b(4)}


class RosArmClient(ArmClient):

    def __init__(self, arm: ArmConfig, calibration: dict, controller: str = 'arm_controller',
                 power_on_monotonic: Optional[float] = None, node_name: str = 'arm_stack'):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Float64MultiArray

        self.arm = arm
        self.names = arm.names
        self.n = arm.n
        self.dt = arm.dt
        self.cmd_dt = max(arm.dt, 0.002)      # Python streams at up to 500 Hz; the RT loop interpolates nothing
        self.units = Units(arm, calibration)
        self._msg_type = Float64MultiArray
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(node_name)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.node.create_publisher(Float64MultiArray, '/%s/command' % controller, 10)
        self._lock = threading.Lock()
        self._state = None
        self._cond = threading.Condition(self._lock)
        self.node.create_subscription(Float64MultiArray, '/%s/state' % controller, self._on_state, qos)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._spin = threading.Thread(target=self.executor.spin, daemon=True)
        self._spin.start()
        self._power_on = power_on_monotonic
        self._mocap = None
        self._last_cmd = None
        if not self._wait_state(5.0):
            raise RuntimeError('no /%s/state within 5 s: is the arm launched (arm01.launch.py)?' % controller)

    def _on_state(self, msg):
        try:
            st = decode_state(msg.data, self.n)
        except ValueError:
            return
        with self._cond:
            self._state = st
            self._cond.notify_all()

    def _wait_state(self, timeout):
        with self._cond:
            return self._cond.wait_for(lambda: self._state is not None, timeout)

    def state(self) -> JointState:
        with self._lock:
            st = dict(self._state)
        counts = self.units.rad_to_counts(st['q'])
        permille = self.units.nm_to_permille(st['tau'])
        status = np.full(self.n, 0x1237 if not st['fallback'] else 0x0237)
        return JointState(t=float(st['t']), q=st['q'].copy(), qd=st['qd'].copy(), tau=st['tau'].copy(),
                          raw_counts=counts, raw_permille=permille, status=status, mode=st['mode'],
                          fallback=bool(st['fallback']), tau_cmd=st['tau_out'].copy())

    def send(self, cmd: Command) -> None:
        msg = self._msg_type()
        msg.data = encode_command(cmd, self.n)
        self.pub.publish(msg)
        self._last_cmd = cmd

    def step(self, seconds: float = 0.0) -> JointState:
        # wall-clock pacing; the RT loop runs on its own clock
        time.sleep(max(seconds, self.cmd_dt))
        return self.state()

    def hand_guide(self, prompt: str, hint_q=None) -> JointState:
        if hint_q is not None:
            print('  suggested configuration (deg): %s' % np.round(np.degrees(hint_q), 1).tolist())
        done = threading.Event()
        zeros = np.zeros(self.n)

        def feed():
            while not done.is_set():
                self.send(Command(mode='T', q=None, kp=zeros, kd=zeros))
                time.sleep(0.01)
        t = threading.Thread(target=feed, daemon=True)
        t.start()
        try:
            input(prompt + ' [Enter] ')
            # settle, then average 0.5 s of readings
            time.sleep(0.3)
            reads = []
            for _ in range(50):
                reads.append(self.state())
                time.sleep(0.01)
        finally:
            done.set()
            t.join()
        s = reads[-1]
        s.q = np.mean([r.q for r in reads], axis=0)
        s.tau = np.mean([r.tau for r in reads], axis=0)
        self.hold(0.1)
        return s

    def drive_gains(self):
        # read from the drives over SDO and recorded in the joint config (drive_gains: {kp, kd})
        rows = {r['name']: r for r in self.arm.raw.get('joints') or []}
        g = [(rows.get(n) or {}).get('drive_gains') or {} for n in self.names]
        if not all(x.get('kp') and x.get('kd') for x in g):
            return None
        return np.array([x['kp'] for x in g], float), np.array([x['kd'] for x in g], float)

    def mocap(self):
        cfg = self.arm.raw.get('mocap') or {}
        if not cfg.get('topic'):
            return None
        if self._mocap is None:
            self._mocap = RosMocap(self.node, cfg['topic'], cfg.get('length_bar_error_m'))
        return self._mocap

    def time_since_power_on(self) -> float:
        return float('nan') if self._power_on is None else time.monotonic() - self._power_on

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()


class RosMocap:
    """Flange marker-cluster pose from a geometry_msgs/PoseStamped topic (mocap frame)."""

    def __init__(self, node, topic: str, length_bar_error_m: Optional[float]):
        import pinocchio as pin
        from geometry_msgs.msg import PoseStamped
        self._pin = pin
        self._lock = threading.Lock()
        self._pose = None
        self._bar = length_bar_error_m
        node.create_subscription(PoseStamped, topic, self._on_pose, 10)

    def _on_pose(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        q = self._pin.Quaternion(o.w, o.x, o.y, o.z)
        with self._lock:
            self._pose = self._pin.SE3(q.matrix(), np.array([p.x, p.y, p.z]))

    def marker_pose(self):
        deadline = time.monotonic() + 1.0
        while True:
            with self._lock:
                if self._pose is not None:
                    return self._pose.copy()
            if time.monotonic() > deadline:
                raise RuntimeError('no mocap pose received')
            time.sleep(0.01)

    def static_noise_3sigma_m(self, seconds: float = 60.0) -> float:
        P = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            P.append(self.marker_pose().translation)
            time.sleep(0.01)
        return float(3 * np.max(np.std(P, axis=0)))

    def length_bar_error_m(self):
        return self._bar
