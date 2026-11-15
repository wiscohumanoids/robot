"""Stage 4, `arm kt`: output-side torque constant per joint.

A known mass hangs on a known lever. The joint is raised and lowered slowly
through the test pose in Mode P, with and without the load, and the current
is averaged over both directions so Coulomb friction cancels. The load's
torque comes from the model (mass x lever x gravity through the Jacobian),
and the current difference it causes gives

    Kt = d(tau_load) / d(I)

Three loads check linearity. On the bench rig the same code runs with one
joint; on the arm the load hangs from the joint's child link.

Pass: within 15% of the datasheet value and linear across 3 loads
(the line through the unloaded point and the 3 loads misses no point by more
than 5% of the full-scale load torque).
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from ..control.motion import Trace, move_to, stream
from ..hw.client import Payload
from .common import DEG, StageContext, StageResult, mount_payload, safe_home

DATASHEET_TOL = 0.15
LINEARITY_TOL = 0.05
LEVER_M = 0.15
LOAD_FRACTIONS = (0.15, 0.30, 0.45)     # of the joint's max torque
SWEEP = 6 * DEG
SPEED = 0.05                            # rad/s


def rig_setup(arm, model, j):
    """(hook offset, test masses or None) from the joint config's `rig:` block when
    it describes joint j's child link (the bench lever), else (None, None)."""
    rig = arm.raw.get('rig') or {}
    if not rig or rig.get('lever_link') != model.link_names()[j]:
        return None, None
    offset = rig.get('hook_offset_m')
    masses = rig.get('test_masses_kg')
    if isinstance(masses, dict):
        masses = masses.get(arm.joints[j].model)
    return (None if offset is None else np.array(offset, float),
            None if not masses else [float(m) for m in masses])


def _lever_offset(model, j):
    """A lever arm in the child link frame, perpendicular to the joint axis."""
    axis = model.axis_in_child(j)
    ref = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    perp = np.cross(axis, ref)
    return LEVER_M * perp / np.linalg.norm(perp)


def load_torque(model, q, link, mass, offset):
    """Joint torques from a point mass at `offset` in `link`'s frame."""
    m = model.copy()
    links = m.link_names()
    k = links.index(link)
    j = m.jid[k]
    m.model.inertias[j] = m.model.inertias[j] + pin.Inertia(mass, np.array(offset, float), np.eye(3) * 1e-6)
    m.data = m.model.createData()
    return m.gravity(q) - model.gravity(q)


def best_pose(model, j, home, offset, rng, tries=300):
    link = model.link_names()[j]
    best, best_t = home.copy(), abs(load_torque(model, home, link, 1.0, offset)[j])
    for _ in range(tries):
        q = home.copy()
        for i in range(j + 1):
            lo, hi = model.lower[i] + SWEEP + 5 * DEG, model.upper[i] - SWEEP - 5 * DEG
            if hi > lo:
                q[i] = rng.uniform(lo, hi)
        t = abs(load_torque(model, q, link, 1.0, offset)[j])
        # prefer poses where the other joints carry little extra load
        if t > best_t:
            best, best_t = q, t
    return best, best_t


def swept_current(ctx, j, q0, clamp):
    """Mean current (A) at q0 over an up pass and a down pass of joint j."""
    client = ctx.client
    units = client.units if hasattr(client, 'units') else None
    rated = ctx.arm.joints[j].rated_current_a
    dirn = ctx.arm.joints[j].direction
    means = []
    for sgn in (+1, -1):
        start = q0.copy()
        start[j] -= sgn * SWEEP
        move_to(client, start, max_speed=0.2, settle=0.3, torque_clamp=clamp)
        n = int(2 * SWEEP / SPEED / client.cmd_dt)
        qs = np.repeat(start[None, :], n, axis=0)
        qs[:, j] = start[j] + sgn * np.linspace(0, 2 * SWEEP, n)
        qds = np.zeros_like(qs)
        qds[:, j] = sgn * SPEED
        tr = Trace()
        stream(client, qs, qds, trace=tr, torque_clamp=clamp)
        a = tr.arrays()
        mid = np.abs(a['q'][:, j] - q0[j]) < 0.5 * SWEEP
        if not np.any(mid):
            mid = slice(None)
        means.append(dirn * a['permille'][mid, j].mean() * rated / 1000.0)
    _ = units
    return 0.5 * (means[0] + means[1])


def run(ctx: StageContext) -> StageResult:
    client, model, arm = ctx.client, ctx.model, ctx.arm
    home = safe_home(model)
    lines, flags = [], []
    kt_out, metrics = {}, {}
    ok = True
    joints = range(arm.n)
    fractions = LOAD_FRACTIONS if not ctx.quick else LOAD_FRACTIONS
    for j in joints:
        spec = arm.joints[j]
        name = spec.name
        link = model.link_names()[j]
        rig_offset, rig_masses = rig_setup(arm, model, j)
        offset = rig_offset if rig_offset is not None else _lever_offset(model, j)
        q0, unit_t = best_pose(model, j, home, offset, ctx.rng)
        if unit_t < 0.05:
            flags.append('%s: no pose gives the lever a gravity moment; run it on the bench rig' % name)
            ok = False
            continue
        peak = np.array([s.peak_torque_nm for s in arm.joints])
        g = np.abs(model.gravity(q0))
        mount_payload(ctx, None)
        clamp = np.minimum(peak, g + 0.3 * peak)
        i0 = swept_current(ctx, j, q0, clamp)
        dtaus, dis = [0.0], [0.0]
        # loads sized from the headroom every joint has left after the arm's own
        # gravity, so no joint reaches its torque clamp (a clamped or sagging joint
        # bends the line)
        max_t = np.array([s.max_torque_nm for s in arm.joints])
        per_kg = np.abs(load_torque(model, q0, link, 1.0, offset))
        room = 0.8 * max_t - g
        with np.errstate(divide='ignore'):
            m_max = float(np.min(np.where(per_kg > 1e-6, room / per_kg, np.inf)))
        if not np.isfinite(m_max) or m_max * unit_t < 0.1 * spec.max_torque_nm:
            flags.append('%s: little torque headroom at the lever pose (%.2f Nm of load); run it on the '
                         'bench rig' % (name, m_max * unit_t if np.isfinite(m_max) else 0.0))
        if rig_masses and max(rig_masses) > m_max * 1.001:
            # the rig's real weights are used as they are; past the headroom they are flagged
            flags.append('%s: rig mass %.2f kg exceeds the %.2f kg the torque limit allows here'
                         % (name, max(rig_masses), m_max))
        m_max = min(m_max, max(fractions) * spec.max_torque_nm / unit_t)
        if rig_masses:
            masses = rig_masses
        else:
            masses = [(frac / max(fractions)) * max(m_max, 0.1 * spec.max_torque_nm / unit_t)
                      for frac in fractions]
        for mass in masses:
            mount_payload(ctx, Payload(link, mass, offset))
            tau_load = load_torque(model, q0, link, mass, offset)
            clamp = np.minimum(peak, g + np.abs(tau_load) + 0.3 * peak)
            i_load = swept_current(ctx, j, q0, clamp)
            dtaus.append(tau_load[j])
            dis.append(i_load - i0)
        mount_payload(ctx, None)
        dtaus, dis = np.array(dtaus), np.array(dis)
        # line through the unloaded point and the 3 loads, intercept free (the
        # unloaded current is one measurement; forcing it through the origin
        # charges its noise to every load). Nonlinearity = worst residual as a
        # share of the full-scale load torque.
        A = np.column_stack([dis, np.ones_like(dis)])
        (kt, c0), *_ = np.linalg.lstsq(A, dtaus, rcond=None)
        kt = float(kt)
        lin = float(np.max(np.abs(A @ np.array([kt, c0]) - dtaus)) / max(np.max(np.abs(dtaus)), 1e-9))
        each = dtaus[1:] / np.where(np.abs(dis[1:]) > 1e-9, dis[1:], np.nan)
        nominal = spec.kt_nominal_nm_per_a
        dev = abs(kt - nominal) / nominal
        good = dev < DATASHEET_TOL and lin < LINEARITY_TOL and kt > 0
        ok &= good
        kt_out[name] = kt
        metrics[name] = {'kt': kt, 'nominal': nominal, 'deviation': dev, 'linearity': lin,
                         'per_load': each.tolist(), 'masses_kg': [float(m) for m in masses],
                         'lever_offset_m': [float(x) for x in offset]}
        if spec.serial:
            metrics[name]['actuator_serial'] = spec.serial
        lines.append('%-15s Kt %.4f Nm/A (datasheet %.4f, %+.1f%%), linearity %.1f%%%s' % (
            name, kt, nominal, 100 * (kt / nominal - 1), 100 * lin, '' if good else '  <-- FAIL'))
        if kt <= 0:
            flags.append('%s: negative Kt; torque sign is reversed (check direction / 0x6077 sign)' % name)
    move_to(client, home, max_speed=0.3)
    return StageResult('kt', ok, metrics, {'kt_output_nm_per_a': kt_out}, lines, flags)
