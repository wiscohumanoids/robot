"""Stage 3, `arm limits`: find the hard stops, set soft limits 3 deg inside.

Each joint moves slowly toward each URDF limit and on past it in Mode P, with
gravity feedforward and 0x6072 clamped to the model gravity torque plus a
margin, until it stalls (tracking error grows while the joint stops). The
stall position is the hard stop. Then each joint is swept to its new soft
limits to confirm no stop is reached first.
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


def find_stop(ctx, j, direction, home, clamp):
    client, model = ctx.client, ctx.model
    speed = SPEED * (2 if ctx.quick else 1)
    s = client.state()
    target_end = (model.upper[j] + SEARCH_BEYOND) if direction > 0 else (model.lower[j] - SEARCH_BEYOND)
    # approach to just inside the URDF limit at normal speed
    near = home.copy()
    near[j] = (model.upper[j] - 8 * DEG) if direction > 0 else (model.lower[j] + 8 * DEG)
    move_to(client, near, max_speed=0.4, torque_clamp=clamp)
    s = client.state()
    q_cmd = s.q.copy()
    n = int(abs(target_end - q_cmd[j]) / speed / client.cmd_dt) + 1
    stall_q = None
    for k in range(n):
        q_cmd[j] += direction * speed * client.cmd_dt
        qd = np.zeros(model.n)
        qd[j] = direction * speed
        s = stream(client, [q_cmd.copy()], [qd], torque_clamp=clamp, ignore_soft_limits=True)
        if abs(q_cmd[j] - s.q[j]) > STALL_ERROR:
            stall_q = s.q[j]
            break
    # back off
    move_to(client, near, max_speed=0.3, torque_clamp=clamp)
    return stall_q


def run(ctx: StageContext) -> StageResult:
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
            # model gravity anywhere on the way, plus a margin (spec: 0x6072 = gravity + margin)
            g = path_gravity(model, home, j, end)
            clamp = g + 0.08 * peak
            max_t = np.array([jt.max_torque_nm for jt in arm.joints])
            if np.any(g > 0.9 * max_t):
                flags.append('%s: gravity on the way to its %s limit (%.1f Nm on %s) is near the torque limit'
                             % (name, 'upper' if direction > 0 else 'lower', g.max(),
                                arm.names[int(np.argmax(g / max_t))]))
            stop = find_stop(ctx, j, direction, home, clamp)
            if stop is None:
                flags.append('%s: no stop found within %.0f deg past the URDF %s limit'
                             % (name, SEARCH_BEYOND / DEG, 'upper' if direction > 0 else 'lower'))
                stop = (model.upper[j] if direction > 0 else model.lower[j]) + direction * SEARCH_BEYOND
                ok = False
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
