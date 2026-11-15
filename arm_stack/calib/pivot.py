"""Stage 8, `arm pivot`: kinematics check with the probe tip in a table divot.

The arm is hand guided (Mode T, gravity comp, no stiffness) into 20 to 30
varied configurations with the probe tip in the divot; FK of each must land on
one point. Pass: tip spread under 3 mm.

The same data also fits joint zero offsets as a cross-check on the jig: find
the offsets d and divot point p minimizing sum |FK(q_k + d) - p|^2. A pivot
about a fixed point cannot see every offset (J1 rotates every tip about its own
axis, which only moves p; J7 barely moves a tip that sits near its axis), so
only offsets the poses make observable are reported. Pass: each observable
offset within 0.5 deg of the jig zero (the current calibration).

Joint compliance from the calibration (stage mocap-calib) is taken out of the
encoder angles before FK when it is known.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .common import DEG, StageContext, StageResult, safe_home

SPREAD_MM = 3.0
OFFSET_TOL = 0.5 * DEG
MIN_POSES = 20
OBSERVABLE = 0.9          # share of an offset that lies in the observable subspace


def link_angles(ctx: StageContext, q, tau) -> np.ndarray:
    """Encoder (motor side) angles minus the modeled joint deflection."""
    k = ctx.calibration.get('joint_stiffness_nm_per_rad') or {}
    c = np.array([1.0 / k[n] if k.get(n) else 0.0 for n in ctx.arm.names])
    return np.asarray(q, float) - c * np.asarray(tau, float)


def spread_mm(tips: np.ndarray) -> float:
    """Largest distance of a tip from the mean tip, mm."""
    return float(np.max(np.linalg.norm(tips - tips.mean(axis=0), axis=1)) * 1e3)


def _pivot_residual(model, Q, d, p):
    return np.concatenate([model.tool_position(q + d) - p for q in Q])


def fit_offsets(model, Q: np.ndarray, iters: int = 30, rank_tol: float = 1e-3,
                max_step: float = 2 * DEG):
    """Damped Gauss-Newton on (offsets d, divot p). Returns d, p, observability, rms.

    Directions the pivot cannot see (a rotation of the whole arm about J1 just
    moves the divot) have near-zero singular values; they are cut at rank_tol
    of the largest, each step is capped at max_step per offset, and a step is
    kept only if it lowers the residual, so a weak direction cannot throw the
    fit into another minimum.
    """
    n = model.n
    d = np.zeros(n)
    p = np.mean([model.tool_position(q) for q in Q], axis=0)
    r = _pivot_residual(model, Q, d, p)
    cost = float(r @ r)
    observability = np.zeros(n)
    lam = 1e-6
    for _ in range(iters):
        A = np.vstack([np.hstack([model.jacobian(q + d)[:3], -np.eye(3)]) for q in Q])
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        keep = S > S[0] * rank_tol
        V = Vt[keep].T
        observability = np.linalg.norm(V[:n], axis=1)
        Sk = S[keep]
        accepted = False
        for _ in range(8):
            step = -V @ ((U[:, keep].T @ r) * Sk / (Sk ** 2 + lam * Sk[0] ** 2))
            big = np.max(np.abs(step[:n]))
            if big > max_step:
                step *= max_step / big
            r_new = _pivot_residual(model, Q, d + step[:n], p + step[n:])
            c_new = float(r_new @ r_new)
            if c_new < cost:
                d, p, r, accepted = d + step[:n], p + step[n:], r_new, True
                rel = (cost - c_new) / max(cost, 1e-30)
                cost = c_new
                lam = max(lam / 3, 1e-9)
                break
            lam *= 10
        if not accepted or rel < 1e-12 or np.linalg.norm(step) < 1e-10:
            break
    rms = float(np.sqrt(np.mean(np.sum(np.reshape(r, (-1, 3)) ** 2, axis=1))))
    return d, p, observability, rms


def sim_divot_poses(ctx: StageContext, count: int, divot: Optional[np.ndarray] = None, accept=None):
    """Sim 'operator': varied true configurations with the true tip on the divot.
    accept(q) -> bool, if given, rejects configurations the setup cannot reach
    (the plate stage keeps the probe inside a cone about the plate normal)."""
    sim = ctx.client
    truth = sim.truth
    home = safe_home(truth)
    if divot is None:
        q_ref = home.copy()
        # a reachable point in front of and below the shoulder
        q_ref[0] = np.clip(0.6, truth.lower[0], truth.upper[0])
        q_ref[3] = np.clip(1.0, truth.lower[3], truth.upper[3])
        divot = truth.tool_position(q_ref)
    rng = ctx.rng
    poses = []
    lo = truth.lower + 8 * DEG
    hi = truth.upper - 8 * DEG
    for _ in range(count * 40):
        seed = home + rng.uniform(-1, 1, truth.n) * 0.6 * (hi - lo) / 2
        seed = np.clip(seed, lo, hi)
        q = truth.ik_position(divot, seed, iters=300)
        if np.linalg.norm(truth.tool_position(q) - divot) > 1e-5:
            continue
        if np.any(q < lo) or np.any(q > hi):
            continue
        if accept is not None and not accept(q):
            continue
        if poses and min(np.max(np.abs(q - p)) for p in poses) < 12 * DEG:
            continue          # want varied configurations, not repeats
        poses.append(q)
        if len(poses) == count:
            break
    return poses, divot


def collect(ctx: StageContext, count: int):
    """Hand guide into `count` configurations; returns (link angles, torques)."""
    Q, T = [], []
    if ctx.sim:
        targets, _ = sim_divot_poses(ctx, count)
    else:
        targets = [None] * count
    for k, q_true in enumerate(targets):
        s = ctx.client.hand_guide('pivot %d/%d: probe tip in the divot, vary the elbow and wrist, '
                                  'then confirm' % (k + 1, len(targets)), q_true)
        Q.append(link_angles(ctx, s.q, s.tau))
        T.append(s.tau.copy())
    return np.array(Q), np.array(T)


def evaluate(ctx: StageContext, Q: np.ndarray):
    model = ctx.model
    tips = np.array([model.tool_position(q) for q in Q])
    spread = spread_mm(tips)
    d, p, obs, rms = fit_offsets(model, Q)
    return tips, spread, d, p, obs, rms


def run(ctx: StageContext, count: int = None) -> StageResult:
    count = count or (MIN_POSES if ctx.quick else 25)
    Q, _ = collect(ctx, count)
    lines, flags = [], []
    if len(Q) < MIN_POSES:
        flags.append('only %d configurations (want %d+)' % (len(Q), MIN_POSES))
    tips, spread, d, p, obs, rms = evaluate(ctx, Q)
    lines.append('%d configurations, tip spread %.2f mm (pass < %.1f), divot at %s m'
                 % (len(Q), spread, SPREAD_MM, np.round(p, 4).tolist()))
    lines.append('offset fit: residual RMS %.3f mm' % (rms * 1e3))
    offsets_ok = True
    fitted = {}
    for i, n in enumerate(ctx.arm.names):
        if obs[i] < OBSERVABLE:
            lines.append('  %-15s not observable from a pivot (%.0f%%)' % (n, 100 * obs[i]))
            continue
        ok = abs(d[i]) < OFFSET_TOL
        offsets_ok &= ok
        fitted[n] = float(d[i])
        lines.append('  %-15s fitted offset %+.3f deg vs jig %s' % (n, d[i] / DEG, 'ok' if ok else 'OFF'))
    passed = bool(spread < SPREAD_MM and offsets_ok and len(Q) >= MIN_POSES)
    return StageResult('pivot', passed,
                       {'spread_mm': spread, 'poses': len(Q), 'fit_rms_mm': rms * 1e3,
                        'fitted_offsets_rad': fitted, 'observability': obs.tolist(),
                        'divot_m': p.tolist()},
                       {'validation': {'pivot_spread_mm': spread}}, lines, flags)
