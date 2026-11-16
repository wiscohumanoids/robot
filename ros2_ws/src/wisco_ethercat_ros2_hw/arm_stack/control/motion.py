"""Trajectories streamed to an ArmClient: moves, sweeps, chirps, holds.

Everything streams one setpoint per `client.cmd_dt` and records what was
commanded and measured, which is what the fits read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from ..hw.client import ArmClient, Command, JointState


@dataclass
class Trace:
    t: List[float] = field(default_factory=list)
    q_cmd: List[np.ndarray] = field(default_factory=list)
    qd_cmd: List[np.ndarray] = field(default_factory=list)
    q: List[np.ndarray] = field(default_factory=list)
    qd: List[np.ndarray] = field(default_factory=list)
    tau: List[np.ndarray] = field(default_factory=list)
    permille: List[np.ndarray] = field(default_factory=list)

    def add(self, s: JointState, q_cmd, qd_cmd):
        self.t.append(s.t)
        self.q_cmd.append(np.array(q_cmd, float))
        self.qd_cmd.append(np.array(qd_cmd, float))
        self.q.append(s.q.copy())
        self.qd.append(s.qd.copy())
        self.tau.append(s.tau.copy())
        self.permille.append(s.raw_permille.copy())

    def arrays(self):
        return {k: np.array(getattr(self, k)) for k in ('t', 'q_cmd', 'qd_cmd', 'q', 'qd', 'tau', 'permille')}


def min_jerk(q0, q1, duration, dt):
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    n = max(2, int(np.ceil(duration / dt)))
    s = np.linspace(0, 1, n)
    p = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    v = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / duration
    return q0 + np.outer(p, q1 - q0), np.outer(v, q1 - q0)


def stream(client: ArmClient, qs, qds, mode='P', trace: Optional[Trace] = None, kp=None, kd=None,
           tau_ff=None, torque_clamp=None, gravity_ff=True, ignore_soft_limits=False,
           friction_ff_gain=1.0,
           stop: Optional[Callable[[JointState], bool]] = None) -> JointState:
    s = client.state()
    for q, qd in zip(qs, qds):
        client.send(Command(mode=mode, q=q, qd=qd, kp=kp, kd=kd, tau_ff=tau_ff,
                            torque_clamp=torque_clamp, gravity_ff=gravity_ff,
                            ignore_soft_limits=ignore_soft_limits,
                            friction_ff_gain=friction_ff_gain))
        s = client.step(client.cmd_dt)
        if trace is not None:
            trace.add(s, q, qd)
        if stop is not None and stop(s):
            break
    return s


def move_to(client: ArmClient, q_goal, max_speed=0.5, min_time=0.5, settle=0.3, mode='P',
            trace=None, **kw) -> JointState:
    s = client.state()
    dist = np.max(np.abs(np.asarray(q_goal) - s.q))
    duration = max(min_time, 1.875 * dist / max_speed)   # min-jerk peak speed = 1.875 * avg
    qs, qds = min_jerk(s.q, q_goal, duration, client.cmd_dt)
    s = stream(client, qs, qds, mode=mode, trace=trace, **kw)
    if settle > 0:
        n = int(settle / client.cmd_dt)
        s = stream(client, [qs[-1]] * n, [np.zeros_like(qs[-1])] * n, mode=mode, trace=trace, **kw)
    return s


def hold_and_average(client: ArmClient, seconds: float, q_hold=None, mode='P', **kw):
    """Hold a pose and average the measured state."""
    s = client.state()
    q_hold = s.q.copy() if q_hold is None else np.asarray(q_hold, float)
    trace = Trace()
    n = max(1, int(seconds / client.cmd_dt))
    stream(client, [q_hold] * n, [np.zeros_like(q_hold)] * n, mode=mode, trace=trace, **kw)
    a = trace.arrays()
    return {k: a[k].mean(axis=0) for k in ('q', 'qd', 'tau', 'permille')}


def constant_velocity(client: ArmClient, joint: int, q_from, q_to, speed, trace=None, **kw):
    """Move one joint from q_from to q_to at constant speed (others held)."""
    s = client.state()
    q0 = s.q.copy()
    start = q0.copy()
    start[joint] = q_from
    move_to(client, start, settle=0.3, **kw)
    span = q_to - q_from
    duration = abs(span) / speed
    n = max(2, int(duration / client.cmd_dt))
    qs = np.repeat(start[None, :], n, axis=0)
    qs[:, joint] = np.linspace(q_from, q_to, n)
    qds = np.zeros_like(qs)
    qds[:, joint] = np.sign(span) * speed
    return stream(client, qs, qds, trace=trace, **kw)


def chirp_profile(amplitude, f0, f1, duration, dt, max_vel=None):
    """Linear chirp; with max_vel the amplitude tapers as 1/f so the peak speed
    stays at max_vel (the RT loop rate-limits faster targets, which would make
    the logged command differ from what the drive got)."""
    t = np.arange(0, duration, dt)
    k = (f1 - f0) / duration
    f = f0 + k * t
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
    ramp = np.minimum(1.0, t / 0.5) * np.minimum(1.0, (duration - t) / 0.5)
    amp = np.full_like(t, amplitude)
    if max_vel is not None:
        amp = np.minimum(amp, max_vel / (2 * np.pi * np.maximum(f, 1e-3)))
    x = ramp * amp * np.sin(phase)
    return t, x, np.gradient(x, dt)


def chirp(client: ArmClient, joint: int, center, amplitude, f0, f1, duration, trace=None,
          max_vel=None, **kw):
    s = client.state()
    q = s.q.copy()
    q[joint] = center
    move_to(client, q, settle=0.5, **kw)
    t, x, xd = chirp_profile(amplitude, f0, f1, duration, client.cmd_dt, max_vel)
    qs = np.repeat(q[None, :], len(t), axis=0)
    qds = np.zeros_like(qs)
    qs[:, joint] = center + x
    qds[:, joint] = xd
    return stream(client, qs, qds, trace=trace, **kw)
