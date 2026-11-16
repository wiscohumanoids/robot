"""`arm gravity-check`: how good gravity comp is in float, and how much friction
help float can use. Complements validate's hold test (which measures torque
residuals in Mode P and drift with gravity comp only).

1. Drift. At N poses the whole arm floats (Mode T, kp = 0) with no friction
   help and no damping for 3 s. A gravity error that beats the breakaway
   accelerates the arm:  M(q) qdd + f(qd) = g_model - g_true.  qdd per joint
   is the slope of a line fit to qd over the window, f the calibrated
   friction model, so the gravity the model is missing is

       dtau = g_true - g_model ~ -(M(q) a + mean f(qd))

   for each joint that moved. Flag: |dtau| > 5% of that joint's peak gravity
   torque. A joint that stayed stuck has |dtau| under its breakaway; it is
   reported as "< Fc" (it will not drift in float either). M includes the
   rotor armature from the PACE fit, or a nominal per-model value until then.

2. Friction gain. Per joint, the others locked: a short torque kick each way
   to ~V_KICK, then float with Coulomb help at gain g on that joint. Below the
   breakaway (g Fc_model < real friction) the joint coasts to a stop; above it
   the joint keeps going (creeps). Bisection finds the largest g that stops
   both ways; MARGIN x that is recommended as the joint config's
   `float_friction_gain` (applied with --write-config, as bringup does).

--payload KG@x,y,z (tool0 frame) puts the payload in the RT gravity for the
whole stage, so the drift test also checks the payload compensation.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..control.float_mode import FloatConfig, FloatController, exit_to_hold, run_float
from ..control.motion import move_to
from ..hw.client import Command, Payload
from .common import DEG, StageContext, StageResult, feasible_poses
from .gravity import RESIDUAL_FRACTION

DRIFT_S = 3.0
DRIFT_STOP_RAD = 8 * DEG       # end a drift window early past this
DRIFT_STOP_SPEED = 0.25        # rad/s
MOVED_RAD = 0.1 * DEG
MOVED_SPEED = 0.005
GAIN_MAX = 0.95
MARGIN = 0.9
V_KICK = 0.12                  # rad/s, above the Coulomb help's deadband + ramp
KICK_MAX_S = 0.5
OBSERVE_S = 2.0
STOP_SPEED = 0.01
STOP_HOLD_S = 0.1
CREEP_RAD = 12 * DEG
# kg m^2 at the output, until the PACE stage fits it (sim plant uses the same)
ARMATURE_NOMINAL = {'x6_60': 0.03, 'x4_36': 0.012, 'x4_10': 0.004}


def armature(ctx) -> np.ndarray:
    fit = ((ctx.calibration.get('pace') or {}).get('armature') or {})
    return np.array([float(fit[j.name]) if fit.get(j.name) is not None else ARMATURE_NOMINAL.get(j.model, 0.0)
                     for j in ctx.arm.joints])


def send_payload(client, payload, sim=False):
    """One Mode P hold command carrying the payload (the RT loop keeps it). On the
    sim the mass is also hung on the plant's tool0 (on hardware it is mounted)."""
    if payload is None:
        return
    if sim and hasattr(client, 'set_payload'):
        client.set_payload(Payload('tool0', float(payload[0]), tuple(payload[1:4])) if payload[0] > 0 else None)
    s = client.state()
    client.send(Command(mode='P', q=s.q.copy(), payload=payload))
    client.step(client.cmd_dt)


def estimate_residual(model, arm_inertia, trace, coulomb, viscous):
    """Gravity the model is missing, g_true - g_model = -(M a + mean f(qd)), from a
    float window (the unbalanced torque M a + f is what the model left over);
    returns (residual, moved mask)."""
    t, q, qd = trace['t'], trace['q'], trace['qd']
    w = t >= min(0.05, t[-1] / 2)          # skip the T blend
    t, q, qd = t[w], q[w], qd[w]
    tc = t - t.mean()
    a = (tc[:, None] * (qd - qd.mean(axis=0))).sum(axis=0) / max(float(np.dot(tc, tc)), 1e-12)
    M = model.mass_matrix(q[len(q) // 2]) + np.diag(arm_inertia)
    f = (coulomb[None] * np.tanh(qd / 0.002) + viscous[None] * qd).mean(axis=0)
    speed = np.max(np.abs(qd), axis=0)
    # a window cut short by a fast drift is brief: speed alone says it moved
    moved = (speed > MOVED_SPEED) & ((np.max(np.abs(q - q[0]), axis=0) > MOVED_RAD) | (speed > 10 * MOVED_SPEED))
    return -(M @ a + f), moved


def drift_test(ctx, poses, cfg: FloatConfig, drift_s: float):
    client, model = ctx.client, ctx.model
    inertia = armature(ctx)
    zero = dataclasses.replace(cfg, friction_gain=np.zeros(cfg.n), damping=np.zeros(cfg.n),
                               lock=np.zeros(cfg.n, bool))
    out = []
    for q in poses:
        move_to(client, q, max_speed=0.5, settle=0.4, torque_clamp=cfg.torque_clamp)
        stop = lambda s, t, q0=client.state().q.copy(): bool(  # noqa: E731
            np.max(np.abs(s.q - q0)) > DRIFT_STOP_RAD or np.max(np.abs(s.qd)) > DRIFT_STOP_SPEED)
        r = run_float(client, zero, duration=drift_s, stop=stop, log=ctx.log, hold_s=0.2)
        dtau, moved = estimate_residual(model, inertia, r.trace, model.coulomb, model.viscous)
        out.append({'q': np.asarray(q), 'dtau': dtau, 'moved': moved, 'seconds': r.seconds,
                    'travel': r.travel, 'stopped_early': r.reason == 'stop'})
    return out


def _creeps(ctx, cfg: FloatConfig, j: int, gain: float, sign: int, pose, observe_s: float) -> bool:
    """Kick joint j (others locked) toward sign * V_KICK, release into float at `gain`.
    True if it does not stop within observe_s."""
    client = ctx.client
    dt = client.cmd_dt
    move_to(client, pose, max_speed=0.5, settle=0.2, torque_clamp=cfg.torque_clamp)
    lock = np.ones(cfg.n, bool)
    lock[j] = False
    g = np.zeros(cfg.n)
    g[j] = gain
    c = dataclasses.replace(cfg, friction_gain=g, damping=np.zeros(cfg.n), lock=lock)
    s = client.state()
    ctrl = FloatController(c, s.q)
    q0 = s.q[j]
    fc = max(cfg.coulomb[j], 0.02 * cfg.torque_clamp[j])
    t = 0.0
    while t < KICK_MAX_S and sign * s.qd[j] < V_KICK:
        cmd = ctrl.command(s, dt)
        # torque ramp from 1x to 4x the model Coulomb: breaks away, then stops at V_KICK
        cmd.tau_ff[j] += sign * min(fc * (1.0 + 3.0 * t / KICK_MAX_S), 0.5 * cfg.torque_clamp[j])
        client.send(cmd)
        s = client.step(dt)
        t += dt
    creep, still, t = True, 0.0, 0.0
    while t < observe_s:
        client.send(ctrl.command(s, dt))
        s = client.step(dt)
        t += dt
        still = still + dt if abs(s.qd[j]) < STOP_SPEED else 0.0
        if still >= STOP_HOLD_S:
            creep = False
            break
        if abs(s.q[j] - q0) > CREEP_RAD or abs(s.qd[j]) > 2.0 * V_KICK:
            break
    exit_to_hold(client, c, ctrl, s, hold_s=0.1)
    return creep


def bisect_gain(ctx, cfg, j, pose, iters, observe_s):
    trials = []

    def ok(gain):
        res = [not _creeps(ctx, cfg, j, gain, sgn, pose, observe_s) for sgn in (+1, -1)]
        trials.append((float(gain), bool(all(res))))
        return all(res)

    if ok(GAIN_MAX):
        return GAIN_MAX, trials
    lo, hi = 0.0, GAIN_MAX
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    if lo == 0.0:
        ok(0.0)
    return lo, trials


def kick_pose(model, arm, rng):
    """Mid-range pose (room both ways for the kicks) if Mode P holds it easily, else a feasible one."""
    q = 0.5 * (model.lower + model.upper)
    clamp = np.array([j.max_torque_nm for j in arm.joints])
    if np.all(np.abs(model.gravity(q)) < 0.6 * clamp):
        return q
    return feasible_poses(model, 1, rng, clamp, fraction=0.5, margin=25 * DEG)[0]


def run(ctx: StageContext, payload: Optional[Sequence[float]] = None, n_poses: Optional[int] = None,
        drift_s: float = DRIFT_S, bisect: bool = True, joints: Optional[List[int]] = None,
        bisect_iters: Optional[int] = None, mount_sim_payload: bool = True) -> StageResult:
    """mount_sim_payload: on the sim, hang the --payload on the plant too (False:
    the plant keeps whatever the test mounted, e.g. a different mass)."""
    client, model, arm = ctx.client, ctx.model, ctx.arm
    n = arm.n
    lines, flags = [], []
    metrics: Dict = {}
    payload = (0.0, 0.0, 0.0, 0.0) if payload is None else tuple(payload)
    cfg = FloatConfig.from_arm(arm, model, payload=payload)
    clamp = cfg.torque_clamp
    send_payload(client, payload, sim=ctx.sim and mount_sim_payload)
    if payload[0] > 0:
        lines.append('payload %.3f kg at %s m (tool0) in the RT gravity' % (payload[0], list(payload[1:])))

    # 1. drift with gravity comp only
    n_poses = n_poses or (3 if ctx.quick else 6)
    rng = np.random.default_rng(7)
    poses = feasible_poses(model, n_poses, rng, clamp, fraction=0.5, margin=15 * DEG,
                           extra_mass=payload[0])
    peak_g = model.peak_gravity_torque()
    results = drift_test(ctx, poses, cfg, drift_s)
    worst = np.zeros(n)
    resolved = np.zeros(n, bool)
    for r in results:
        a = np.where(r['moved'], np.abs(r['dtau']), 0.0)
        worst = np.maximum(worst, a)
        resolved |= r['moved']
    frac = worst / np.maximum(peak_g, 1e-6)
    flagged = resolved & (frac > RESIDUAL_FRACTION)
    lines.append('drift: %d poses x %.0f s, gravity comp only (no friction help, no damping)' % (len(poses), drift_s))
    for i, j in enumerate(arm.joints):
        if resolved[i]:
            msg = 'residual %.3f Nm = %.1f%% of peak gravity %.2f Nm' % (worst[i], 100 * frac[i], peak_g[i])
        elif model.coulomb[i] > 0:
            msg = 'did not move: residual < breakaway (~Fc %.3f Nm = %.0f%% of peak gravity)' % (
                model.coulomb[i], 100 * model.coulomb[i] / max(peak_g[i], 1e-6))
        else:
            msg = 'did not move: residual < breakaway (no friction model to put a number on it)'
        lines.append('  %-15s %s%s' % (j.name, msg, '  <-- FLAG (> %.0f%%)' % (100 * RESIDUAL_FRACTION)
                                       if flagged[i] else ''))
    for i in np.flatnonzero(flagged):
        flags.append('%s: gravity residual %.0f%% of peak in float; rerun arm gravity (or check the payload)'
                     % (arm.names[i], 100 * frac[i]))
    if any(r['stopped_early'] for r in results):
        lines.append('  (windows cut short past %.0f deg or %.2f rad/s)' % (DRIFT_STOP_RAD / DEG, DRIFT_STOP_SPEED))
    metrics['drift'] = {'residual_nm': worst.tolist(), 'residual_frac': frac.tolist(),
                        'resolved': resolved.tolist(), 'flagged': flagged.tolist(),
                        'peak_gravity_nm': peak_g.tolist(),
                        'per_pose': [{'q': r['q'].tolist(), 'dtau': r['dtau'].tolist(), 'moved': r['moved'].tolist(),
                                      'travel_deg': (r['travel'] / DEG).tolist(), 'seconds': r['seconds']}
                                     for r in results]}

    # 2. largest friction help that does not creep, per joint
    updates = {}
    if bisect:
        iters = bisect_iters or (3 if ctx.quick else 4)
        observe = 1.5 if ctx.quick else OBSERVE_S
        pose = kick_pose(model, arm, rng)
        gains = {}
        lines.append('friction gain: kick +/- %.2f rad/s, others locked, must stop within %.1f s; '
                     'bisection %d steps, recommend %.0f%% of the largest that stops' % (
                         V_KICK, observe, iters, 100 * MARGIN))
        for j in (range(n) if joints is None else joints):
            name = arm.names[j]
            if not model.coulomb[j] > 0:
                lines.append('  %-15s no friction model (run arm friction): gain not tuned' % name)
                continue
            g, trials = bisect_gain(ctx, cfg, j, pose, iters, observe)
            rec = round(MARGIN * g, 2)
            gains[name] = {'max_no_creep': g, 'recommended': rec, 'trials': trials}
            updates[name] = rec
            drifts_at_zero = g == 0.0 and trials and not trials[-1][1]
            lines.append('  %-15s friction gain: no creep up to %.2f -> float_friction_gain %.2f%s' % (
                name, g, rec, '  (drifts even with no help: gravity residual above breakaway)'
                if drifts_at_zero else ''))
            if drifts_at_zero:
                flags.append('%s creeps with no friction help: fix gravity before floating' % name)
        metrics['friction_gain'] = gains
    move_to(client, client.state().q, max_speed=0.3, settle=0.2)
    passed = not bool(np.any(flagged))
    return StageResult('gravity-check', passed, metrics, {}, lines, flags,
                       joint_config_updates={'float_friction_gain': updates} if updates else {})
