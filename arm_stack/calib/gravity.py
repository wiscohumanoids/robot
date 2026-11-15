"""Stage 5, `arm gravity`: fit each link's center of mass.

Static joint torque is linear in each link's first moment h_k = m_k c_k:

    tau_g(q) = Y(q) h + Y0(q) m

with Y, Y0 from Pinocchio's joint torque regressor and m the masses weighed
at assembly (joint config `link_masses_kg`, URDF values where still null).
At each pose all joints sweep slowly up and then down through it in Mode P;
the up and down means are averaged so friction cancels. h is fit by
regularized least squares (prior = URDF), since some first-moment directions
are unobservable from joint torques and stay at their URDF values.

Pass: torque residual under 5% of each joint's peak gravity torque on 10
held-out poses. A center of mass that moves more than 10 mm from the URDF
is flagged: check that link's CAD and wiring before trusting the fit.
"""

from __future__ import annotations

import numpy as np

from ..control.motion import Trace, move_to, stream
from .common import DEG, StageContext, StageResult, random_poses, safe_home

RESIDUAL_FRACTION = 0.05
COM_FLAG_M = 0.010
SWEEP = 2 * DEG
SPEED = 0.04
PRIOR_WEIGHT = 1e-3


def measure_pose(ctx, q, clamp, sweep=SWEEP, speed=SPEED):
    """Measured joint torque at q, averaged over an up and a down pass."""
    client = ctx.client
    passes = []
    for sgn in (+1, -1):
        start = q - sgn * sweep
        move_to(client, start, max_speed=0.4, settle=0.3, torque_clamp=clamp)
        n = max(2, int(2 * sweep / speed / client.cmd_dt))
        s = np.linspace(0, 1, n)[:, None]
        qs = start[None, :] + sgn * 2 * sweep * s
        qds = np.repeat((sgn * speed * np.ones(len(q)))[None, :], n, axis=0)
        tr = Trace()
        stream(client, qs, qds, trace=tr, torque_clamp=clamp)
        a = tr.arrays()
        mid = slice(n // 4, 3 * n // 4)
        passes.append((a['tau'][mid].mean(axis=0), a['q'][mid].mean(axis=0)))
    tau = 0.5 * (passes[0][0] + passes[1][0])
    q_meas = 0.5 * (passes[0][1] + passes[1][1])
    return tau, q_meas


def fit_first_moments(model, Q, T, masses, prior_h, weight=PRIOR_WEIGHT, scale=None):
    rows, rhs = [], []
    scale = np.ones(model.n) if scale is None else scale
    for q, tau in zip(Q, T):
        Y, Y0 = model.gravity_first_moment_regressor(q)
        rows.append(Y / scale[:, None])
        rhs.append((tau - Y0 @ masses) / scale)
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    lam = weight * np.trace(A.T @ A) / A.shape[1]
    H = A.T @ A + lam * np.eye(A.shape[1])
    h = np.linalg.solve(H, A.T @ b + lam * prior_h)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    fit_first_moments.row_space = Vt[sv > sv[0] * 1e-6]   # identifiable directions of h
    return h, sv


def residuals(model, Q, T):
    return np.array([tau - model.gravity(q) for q, tau in zip(Q, T)])


def run(ctx: StageContext, n_fit: int = None, n_test: int = 10) -> StageResult:
    client, arm = ctx.client, ctx.arm
    model = ctx.model.copy()
    n_fit = n_fit or (min(16, 4 + 4 * model.n) if ctx.quick else 32)   # 3 params per link
    n_test = 6 if ctx.quick else n_test
    home = safe_home(model)
    peak_g = model.peak_gravity_torque()
    peak = np.array([j.peak_torque_nm for j in arm.joints])
    poses = random_poses(model, n_fit + n_test, ctx.rng, margin=12 * DEG, spread=0.8)
    data = []
    for q in poses:
        clamp = np.minimum(peak, np.abs(model.gravity(q)) * 1.5 + 0.25 * peak)
        tau, q_meas = measure_pose(ctx, q, clamp)
        data.append((q_meas, tau))
    move_to(client, home, max_speed=0.4)
    fit, test = data[:n_fit], data[n_fit:]
    Qf, Tf = [d[0] for d in fit], [d[1] for d in fit]
    Qt, Tt = [d[0] for d in test], [d[1] for d in test]

    masses = model.masses()
    prior = model.first_moments()
    before = np.abs(residuals(model, Qt, Tt)).max(axis=0)
    h, sv = fit_first_moments(model, Qf, Tf, masses, prior, scale=np.maximum(peak_g, 0.1))
    model.set_first_moments(h)
    after = residuals(model, Qt, Tt)
    worst = np.abs(after).max(axis=0)
    # a joint that carries almost no gravity cannot be judged below the drive's
    # torque resolution: the threshold is max(5% of peak gravity, one 0x6077 count)
    from ..units import Units
    count_nm = np.abs(Units(arm, ctx.calibration).nm_per_permille())
    allowed = np.maximum(RESIDUAL_FRACTION * peak_g, count_nm)
    frac = worst / np.maximum(peak_g, 1e-6)
    passed = bool(np.all(worst < allowed))

    lines, flags = [], []
    inertials = {}
    links = model.link_names()
    # only the identifiable part of the change says anything about the hardware;
    # the rest is the ridge prior at work
    V = fit_first_moments.row_space
    dh_id = V.T @ (V @ (h - prior))
    for k, link in enumerate(links):
        com = h[3 * k:3 * k + 3] / masses[k]
        shift = float(np.linalg.norm(dh_id[3 * k:3 * k + 3]) / masses[k])
        inertials[link] = {'mass_kg': float(masses[k]), 'com_m': [float(c) for c in com]}
        lines.append('%-15s com [%+.4f %+.4f %+.4f] m, identifiable shift %.1f mm from URDF' % (
            link, com[0], com[1], com[2], shift * 1e3))
        if shift > COM_FLAG_M:
            flags.append('%s: COM moved %.1f mm from the URDF; check its CAD and wiring' % (link, shift * 1e3))
    for i, name in enumerate(arm.names):
        lines.append('%-15s held-out residual %.3f Nm = %.1f%% of peak gravity %.2f Nm (before fit %.3f)%s' % (
            name, worst[i], 100 * frac[i], peak_g[i], before[i], '' if worst[i] < allowed[i] else '  <-- FAIL'))
    cond = float(sv[0] / sv[sv > sv[0] * 1e-9][-1]) if len(sv) else float('nan')
    rank = int(np.sum(sv > sv[0] * 1e-6))
    lines.append('regressor rank %d of %d first-moment parameters (rest held at URDF values)' % (rank, len(h)))
    return StageResult('gravity', passed,
                       {'residual_frac': frac.tolist(), 'residual_nm': worst.tolist(),
                        'before_nm': before.tolist(), 'rank': rank, 'cond': cond,
                        'fit_poses': [q.tolist() for q in Qf], 'test_poses': [q.tolist() for q in Qt]},
                       {'link_inertials': inertials}, lines, flags)
