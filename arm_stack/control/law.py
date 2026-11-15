"""The RT control law, in Python. The C++ ArmController implements the same
law (src/arm_controller.cpp); the sim backend runs this one so procedures see
the same behavior in --sim.

Per cycle:
  Mode P: q_target -> 0x607A, tau_offset = g(q) + f(qd_d) + tau_ff -> 0x60B2
  Mode T: tau = Kp(q_d - q) + Kd(qd_d - qd) + g(q) + f(qd) + tau_ff -> 0x6071
then clamp torque and velocity, enforce soft limits, and fall back to gravity
comp plus joint damping if no command arrived within the watchdog (50 ms).

Payload: a command may carry `payload` = (kg, x, y, z), a point mass in the
tool0 frame. The law keeps the last one it got (mass <= 0 clears it) and adds
its gravity to g(q) everywhere, the watchdog fallback included, as the C++
GravityModel does by adding it to the inertia of tool0's parent link.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..hw.client import Command

WATCHDOG_S = 0.050


@dataclass
class LawOutput:
    mode: str                 # 'P' or 'T' actually applied
    q_target: np.ndarray      # Mode P position target
    tau: np.ndarray           # Mode P: torque offset; Mode T: target torque
    clamped: bool
    limit_hit: bool
    fallback: bool


def payload_gravity(model, q, payload, frame: Optional[str] = None) -> np.ndarray:
    """Holding torque of a point mass (kg, x, y, z) fixed in `frame` (tool0):
    J_p(q)^T m g_up, J_p the world-aligned linear Jacobian of that point."""
    n = len(q)
    if payload is None or not payload[0] > 0:
        return np.zeros(n)
    frame = frame or getattr(model, 'tool_frame', 'tool0')
    M = model.fk(q, frame)
    r = M.rotation @ np.asarray(payload[1:4], float)
    J = model.jacobian(q, frame)
    skew = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
    Jp = J[:3] - skew @ J[3:]
    return Jp.T @ np.array([0.0, 0.0, 9.81 * float(payload[0])])


class ControlLaw:

    def __init__(self, model, max_torque, max_vel, soft_lower, soft_upper, kd_fallback,
                 soft_margin=0.05):
        self.model = model
        self.max_torque = np.asarray(max_torque, float)
        self.max_vel = np.asarray(max_vel, float)
        self.lower = np.asarray(soft_lower, float)
        self.upper = np.asarray(soft_upper, float)
        self.kd_fallback = np.asarray(kd_fallback, float)
        self.margin = soft_margin
        self.limit_stiffness = 4.0 * self.max_torque / max(soft_margin, 1e-3)
        self.payload = None       # (kg, x, y, z) in tool0, sticky across commands

    def gravity(self, q) -> np.ndarray:
        """g(q) of the arm plus the current payload."""
        g = self.model.gravity(q)
        if self.payload is not None:
            g = g + payload_gravity(self.model, q, self.payload)
        return g

    def compute(self, cmd: Optional[Command], age_s: float, q, qd, last_mode: str) -> LawOutput:
        n = len(q)
        q = np.asarray(q, float)
        qd = np.asarray(qd, float)
        p = getattr(cmd, 'payload', None) if cmd is not None else None
        if p is not None and np.isfinite(p[0]):
            self.payload = tuple(float(x) for x in p) if p[0] > 0 else None
        g = self.gravity(q)
        fallback = cmd is None or age_s > WATCHDOG_S or cmd.q is None and cmd.mode == 'P'
        if fallback:
            # gravity comp plus joint damping, in torque mode
            tau = g - self.kd_fallback * qd
            tau, clamped = self._clamp(tau, None)
            return LawOutput('T', q.copy(), tau, clamped, False, True)

        clamp = cmd.torque_clamp
        zeros = np.zeros(n)
        q_d = np.asarray(cmd.q if cmd.q is not None else q, float)
        qd_d = np.asarray(cmd.qd if cmd.qd is not None else zeros, float)
        tau_ff = np.asarray(cmd.tau_ff if cmd.tau_ff is not None else zeros, float)
        fgain = getattr(cmd, 'friction_ff_gain', 1.0)
        ff = (g + fgain * self.model.friction(qd_d if cmd.mode == 'P' else qd)) if cmd.gravity_ff else zeros

        # soft limits: never command a target outside them
        limit_hit = bool(np.any(q_d < self.lower) or np.any(q_d > self.upper))
        if not cmd.ignore_soft_limits:
            q_d = np.clip(q_d, self.lower, self.upper)
        # velocity clamp on the target rate
        qd_d = np.clip(qd_d, -self.max_vel, self.max_vel)

        if cmd.mode == 'P':
            tau, clamped = self._clamp(ff + tau_ff, clamp)
            return LawOutput('P', q_d, tau, clamped, limit_hit, False)

        kp = np.asarray(cmd.kp if cmd.kp is not None else zeros, float)
        kd = np.asarray(cmd.kd if cmd.kd is not None else zeros, float)
        tau = kp * (q_d - q) + kd * (qd_d - qd) + ff + tau_ff
        # outside a soft limit: spring back in, never push further out
        over = q - self.upper
        under = self.lower - q
        if np.any(over > 0) or np.any(under > 0):
            limit_hit = True
            tau = np.where(over > 0, np.minimum(tau, -self.limit_stiffness * over - self.kd_fallback * qd), tau)
            tau = np.where(under > 0, np.maximum(tau, self.limit_stiffness * under - self.kd_fallback * qd), tau)
        # velocity clamp: damp hard above the limit
        fast = np.abs(qd) > self.max_vel
        tau = np.where(fast, tau - 5.0 * self.kd_fallback * (qd - np.sign(qd) * self.max_vel), tau)
        tau, clamped = self._clamp(tau, clamp)
        return LawOutput('T', q.copy(), tau, clamped, limit_hit, False)

    def _clamp(self, tau, clamp):
        limit = self.max_torque if clamp is None else np.minimum(self.max_torque, clamp)
        out = np.clip(tau, -limit, limit)
        return out, bool(np.any(out != tau))
