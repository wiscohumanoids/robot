"""`arm limits`: find the hard stops, set soft limits 3 deg inside.

Each joint moves slowly toward each URDF limit and on past it in Mode P, with
gravity feedforward and 0x6072 clamped to the model gravity torque plus a
margin. Confirm contact from sustained low speed and measured torque in the
approach direction above gravity. Then sweep to the inset software limits.
Contact detection is adapted from Enactic's OpenArm zero-position calibration.
"""

from __future__ import annotations

import numpy as np

from ..control.motion import move_to, stream
from .common import DEG, StageContext, StageResult, safe_home

INSET = 3 * DEG
SEARCH_BEYOND = 10 * DEG
SPEED = 0.15            # rad/s
STALL_ERROR = 1.5 * DEG


def path_gravity(model, home, j, end, samples=25):
    """Largest |gravity torque| per joint while joint j sweeps from home to `end`."""
    q = home.copy()
    worst = np.zeros(model.n)
    for v in np.linspace(home[j], end, samples):
        q[j] = v
        worst = np.maximum(worst, np.abs(model.gravity(q)))
    return worst


def find_stop(ctx, j, direction, home, clamp, reference=None):
    """Drive joint `j` slowly toward a stop until it stalls; returns where it stopped."""
    client, model = ctx.client, ctx.model
    speed = SPEED * (2 if ctx.quick else 1)
    if reference is None:
        reference = model.upper[j] if direction > 0 else model.lower[j]
    target_end = reference + direction * SEARCH_BEYOND
    # approach to just inside the URDF limit at normal speed
    near = home.copy()
    near[j] = np.clip(reference - direction * 8 * DEG, model.lower[j], model.upper[j])
    move_to(client, near, max_speed=0.4, torque_clamp=clamp)
    s = client.state()
    q_cmd = s.q.copy()
    n = int(abs(target_end - q_cmd[j]) / speed / client.cmd_dt) + 1
    stall_q = None
    contact_since = None
    # Scale contact detection to the existing search torque margin for this joint.
    # Subtract modeled gravity: a stationary, loaded shoulder already draws current.
    contact_nm = 0.04 * ctx.arm.joints[j].peak_torque_nm
    for k in range(n):
        q_cmd[j] += direction * speed * client.cmd_dt
        qd = np.zeros(model.n)
        qd[j] = direction * speed
        s = stream(client, [q_cmd.copy()], [qd], torque_clamp=clamp, ignore_soft_limits=True)
        lag = direction * (q_cmd[j] - s.q[j])
        if lag > 2 * STALL_ERROR:
            print(f'{ctx.arm.names[j]}: search stopped at {s.q[j]/DEG:.2f} deg, '
                  f'lag {lag/DEG:.2f} deg, speed {s.qd[j]:.3f} rad/s, torque {s.tau[j]:.2f} Nm')
            break
        if lag > STALL_ERROR and abs(s.qd[j]) < 0.02:
            excess_torque = direction * (s.tau[j] - model.gravity(s.q)[j])
            if excess_torque >= contact_nm:
                if contact_since is None:
                    contact_since = s.t
                if s.t - contact_since >= 0.04:
                    stall_q = float(s.q[j])
                    break
                continue
        contact_since = None
    # back off
    move_to(client, near, max_speed=0.3, torque_clamp=clamp)
    return stall_q


def run(ctx: StageContext) -> StageResult:
    """`arm limits`: find each joint's stops and set soft limits 3 degrees inside them (URDF range without hard
    stops).
    """
    client, model, arm = ctx.client, ctx.model, ctx.arm
    home = safe_home(model)
    peak = np.array([j.peak_torque_nm for j in arm.joints])
    lines, flags = [], []
    soft = {}
    stops = {}
    ok = True
    for j, name in enumerate(arm.names):
        found = []
        for direction in (-1, +1):
            end = (model.upper[j] + SEARCH_BEYOND) if direction > 0 else (model.lower[j] - SEARCH_BEYOND)
            # model gravity anywhere on the way, plus a margin
            g = path_gravity(model, home, j, end)
            clamp = g + 0.08 * peak
            max_t = np.array([jt.max_torque_nm for jt in arm.joints])
            if np.any(g > 0.9 * max_t):
                flags.append('%s: gravity on the way to its %s limit (%.1f Nm on %s) is near the torque limit'
                             % (name, 'upper' if direction > 0 else 'lower', g.max(),
                                arm.names[int(np.argmax(g / max_t))]))
            stop = find_stop(ctx, j, direction, home, clamp)
            if stop is None:
                flags.append('%s: no confirmed stop on bounded search (maximum %.0f deg past the %s limit)'
                             % (name, SEARCH_BEYOND / DEG, 'upper' if direction > 0 else 'lower'))
                return StageResult('limits', False, {'hard_stops_rad': stops}, {}, lines, flags)
            found.append(stop)
        lo, hi = found
        soft[name] = [float(lo + INSET), float(hi - INSET)]
        stops[name] = [float(lo), float(hi)]
        lines.append('%-15s stops [%+7.2f, %+7.2f] deg (URDF [%+7.2f, %+7.2f]) -> soft [%+7.2f, %+7.2f]' % (
            name, lo / DEG, hi / DEG, model.lower[j] / DEG, model.upper[j] / DEG,
            soft[name][0] / DEG, soft[name][1] / DEG))
        if lo > model.lower[j] + 1 * DEG or hi < model.upper[j] - 1 * DEG:
            flags.append('%s: hard stop inside the URDF limit; update the URDF or the stop' % name)

    # verify: sweep to each soft limit, no stall on the way
    for j, name in enumerate(arm.names):
        for value in soft[name]:
            q = home.copy()
            q[j] = value
            clamp = path_gravity(model, home, j, value) + 0.15 * peak
            s = move_to(client, q, max_speed=0.6 if ctx.quick else 0.3, torque_clamp=clamp)
            if abs(s.q[j] - value) > STALL_ERROR:
                ok = False
                flags.append('%s: stalled at %.2f deg before its soft limit %.2f deg'
                             % (name, s.q[j] / DEG, value / DEG))
        move_to(client, home, max_speed=0.4)
    return StageResult('limits', ok, {'hard_stops_rad': stops}, {'soft_limits_rad': soft}, lines, flags)
