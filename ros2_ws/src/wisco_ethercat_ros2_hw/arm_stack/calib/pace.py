"""Stage 9, `arm pace`: actuator system ID by replay (PACE).

Each joint runs a chirp in Mode P (others held). The logged commands are
replayed in a simulator with candidate actuator parameters, and CMA-ES
minimizes the position error. Per joint: armature, Coulomb friction, viscous
friction, encoder bias, delay, and the drive's CSP gains when the joint
config does not give them. Link inertials stay at their stage 5 values (the
current calibration's model).

With the drive gains known, the replay also simulates the held joints under
their CSP loops through the hold-pose mass matrix (pace/backend.py), so the
chirped joint's drag on them does not end up in its armature and viscous fit.
The joint stiffness from plate / mocap-calib (run before PACE) puts the
transmission compliance in the replay too.

Pass: every joint's fit converges (cost drop > 90%, RMS error under 10% of
the chirp amplitude), and the fitted delay (median over joints) is within
one sim step of the stage 7 latency.
"""

from __future__ import annotations

import numpy as np

from ..control.motion import Trace, chirp, move_to
from ..pace.backend import PARAMS, IsaacLabBackend, JointLog, JointReplayBackend, default_bounds
from ..pace.cmaes import CMAES
from .common import DEG, StageContext, StageResult, safe_home

LOG_SCALE = {'armature', 'drive_kp', 'drive_kd'}
LOG_FLOOR = {'armature': 1e-4}


class _Map:
    """[0, 1]^k <-> physical params, log scale where it helps, fixed dims dropped."""

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.free = np.where(self.hi > self.lo)[0]
        self.log = np.array([PARAMS[i] in LOG_SCALE for i in range(len(PARAMS))])

    def to_phys(self, X):
        X = np.atleast_2d(X)
        P = np.repeat(self.lo[None, :], len(X), axis=0)
        for c, i in enumerate(self.free):
            lo, hi = self.lo[i], self.hi[i]
            if self.log[i]:
                lo = max(lo, LOG_FLOOR.get(PARAMS[i], 1e-6))
                P[:, i] = np.exp(np.log(lo) + X[:, c] * (np.log(hi) - np.log(lo)))
            else:
                P[:, i] = lo + X[:, c] * (hi - lo)
        return P


F1 = 12.0


def record_chirps(ctx, amplitude_deg=8.0, duration=None, f0=0.2, f1=F1):
    client, model = ctx.client, ctx.model
    duration = duration or (3.0 if ctx.quick else 8.0)
    home = safe_home(model)
    logs = []
    gains = client.drive_gains()
    M = np.diag(model.mass_matrix(home))
    for j in range(model.n):
        # keep the sweep below the joint's servo resonance: above it the joints the
        # chirp shakes through the coupled inertia dominate, which the per-joint
        # replay cannot reproduce (the full-arm Isaac Lab replay can)
        fj = f1
        if gains is not None:
            f_servo = np.sqrt(gains[0][j] / max(M[j], 1e-6)) / (2 * np.pi)
            fj = float(np.clip(0.8 * f_servo, 3.0, f1))
        span = model.upper[j] - model.lower[j]
        amp = min(amplitude_deg * DEG, 0.2 * span)
        center = float(np.clip(home[j], model.lower[j] + amp + 3 * DEG, model.upper[j] - amp - 3 * DEG))
        tr = Trace()
        chirp(client, j, center, amp, f0, fj, duration, trace=tr,
              max_vel=0.7 * ctx.arm.joints[j].max_vel_rad_s)
        a = tr.arrays()
        hold = a['q'][0].copy()
        hold[j] = center
        actual = float(np.max(np.abs(a['q_cmd'][:, j] - center)))     # after the speed taper
        logs.append(JointLog(j, client.cmd_dt, a['q_cmd'][:, j], a['qd_cmd'][:, j], a['q'][:, j], hold,
                             (center, actual), (f0, fj)))
        if ctx.log is not None:
            ctx.log.event('pace_chirp', joint=ctx.arm.names[j], center=center, amplitude=amp, f0=f0, f1=fj,
                          duration=duration)
        move_to(client, home, max_speed=0.4)
    return logs


def decimate(log: JointLog, k: int) -> JointLog:
    if k <= 1:
        return log
    return JointLog(log.joint, log.dt * k, log.q_cmd[::k], log.qd_cmd[::k], log.q_meas[::k], log.q_hold,
                    log.sweep, log.freq)


def armature_prior(ctx) -> np.ndarray:
    """Rotor armature for the joints held during another joint's chirp: the last PACE
    fit, else the nominal per-model value. It enters only through the coupling."""
    from .gravity_check import armature
    return armature(ctx)


def stiffness_prior(ctx) -> np.ndarray:
    """Joint stiffness from plate / mocap-calib (inf where not calibrated)."""
    k = ctx.calibration.get('joint_stiffness_nm_per_rad') or {}
    return np.array([float(k[n]) if k.get(n) else np.inf for n in ctx.arm.names])


def fit(model, arm, logs, generations=60, popsize=14, seed=0, backend='joint', drive_gains=None,
        hold_armature=None, stiffness=None, fixed_delay=None, torque_limit=None):
    """fixed_delay: per log, None (fit the delay) or seconds to pin it to."""
    if backend == 'isaac':
        sim = IsaacLabBackend(model, logs)
    else:
        # with the drive gains known, the held joints are simulated too (coupled
        # replay), with their rotor armature from hold_armature
        sim = JointReplayBackend(model, logs, hold_gains=drive_gains, hold_armature=hold_armature,
                                 stiffness=stiffness, torque_limit=torque_limit)
    maps, opts = [], []
    for k, log in enumerate(logs):
        spec = arm.joints[log.joint]
        row = next((r for r in arm.raw.get('joints', []) if r.get('name') == spec.name), {})
        known = row.get('drive_gains')
        if drive_gains is not None:
            known = {'kp': float(drive_gains[0][log.joint]), 'kd': float(drive_gains[1][log.joint])}
        lo, hi = default_bounds(spec, known)
        if fixed_delay is not None and fixed_delay[k] is not None:
            i = PARAMS.index('delay_s')
            lo[i] = hi[i] = float(fixed_delay[k])
        m = _Map(lo, hi)
        maps.append(m)
        x0 = np.full(len(m.free), 0.5)
        opts.append(CMAES(x0, 0.25, np.zeros(len(m.free)), np.ones(len(m.free)), popsize=popsize, seed=seed + k))
    scale = [max(l.sweep[1], 1e-3) ** 2 for l in logs]
    first = [None] * len(logs)
    for gen in range(generations):
        live = 0
        for k, (m, o) in enumerate(zip(maps, opts)):
            if o.converged() or o.plateaued(0.01, 15):
                continue
            live += 1
            X = o.ask()
            P = m.to_phys(X)
            # unstable candidates (too little damping for their delay) diverge:
            # score them as a large finite cost so CMA-ES ranks them last
            with np.errstate(over='ignore', invalid='ignore'):
                q = sim.simulate(k, P)
                cost = np.mean((q - logs[k].q_meas[None, :]) ** 2, axis=1) / scale[k]
            cost = np.where(np.isfinite(cost), cost, 1e12)
            if first[k] is None:
                first[k] = float(np.median(cost))
            o.tell(X, cost)
        if not live:
            break
    out = []
    for k, (m, o) in enumerate(zip(maps, opts)):
        p = m.to_phys(o.best_x)[0]
        rms = float(np.sqrt(o.best_f * scale[k]))
        # converged: CMA-ES stopped itself, or the best cost found improved by
        # under 1% over the last 15 generations
        stable = o.plateaued(0.01, 15)
        out.append(dict(params=dict(zip(PARAMS, map(float, p))), cost=o.best_f, first=first[k],
                        rms_rad=rms, gens=o.gen, converged=bool(o.converged() or stable)))
    return out


def run(ctx: StageContext, backend: str = 'joint') -> StageResult:
    arm = ctx.arm
    logs = record_chirps(ctx)
    k = 2
    gains = ctx.client.drive_gains()
    stiff = stiffness_prior(ctx)
    # a chirp capped well below the bus rate cannot see a few ms of delay, which then
    # trades off against armature: pin it to the stage 7 latency on those joints
    latency = ctx.calibration.get('latency_ms')
    low_band = [l.freq[1] < 0.8 * F1 for l in logs]
    # (unless no chirp reaches the band: then the delay check needs every joint's fit)
    fixed = [latency * 1e-3 if (lb and latency is not None and not all(low_band)) else None
             for lb in low_band]
    fits = fit(ctx.model, arm, [decimate(l, k) for l in logs], generations=150 if ctx.quick else 300,
               backend=backend, drive_gains=gains, hold_armature=armature_prior(ctx), stiffness=stiff,
               fixed_delay=fixed, torque_limit=np.array([j.max_torque_nm for j in arm.joints], float))
    sim_dt = float((arm.raw.get('pace') or {}).get('sim_dt_s', 0.002))
    lines, flags = [], []
    if not np.all(np.isfinite(stiff)):
        flags.append('no joint stiffness for %s (run `arm plate` or `arm mocap-calib` first): the replay '
                     'treats them as rigid, so their armature and viscous absorb the compliance' %
                     ', '.join(n for n, v in zip(arm.names, stiff) if not np.isfinite(v)))
    if gains is None:
        flags.append('drive CSP gains unknown (no drive_gains in the joint config): kp/kd fitted too, '
                     'which trades off against the delay; read them over SDO and record them')
    ok = True
    writes = {n: {} for n in ('armature', 'viscous', 'friction', 'encoder_bias', 'drive_kp', 'drive_kd')}
    delays = []
    for idx, (log, f) in enumerate(zip(logs, fits)):
        name = arm.names[log.joint]
        p = f['params']
        amp = log.sweep[1]
        good = f['converged'] and f['rms_rad'] < 0.1 * amp
        ok &= good
        # a chirp capped well below the bus rate cannot see a few ms of delay
        if log.freq[1] >= 0.8 * F1:
            delays.append(p['delay_s'])
        writes['armature'][name] = p['armature']
        writes['viscous'][name] = p['viscous']
        writes['friction'][name] = p['coulomb']
        writes['encoder_bias'][name] = p['encoder_bias']
        writes['drive_kp'][name] = p['drive_kp']
        writes['drive_kd'][name] = p['drive_kd']
        lines.append('%-15s armature %.4f, Fc %.3f, Fv %.3f, bias %+.3f deg, delay %.2f ms%s, kp %.0f kd %.2f, '
                     'chirp to %.1f Hz, rms %.3f deg (%.1f%% of amp)%s' % (
                         name, p['armature'], p['coulomb'], p['viscous'], p['encoder_bias'] / DEG,
                         p['delay_s'] * 1e3, ' (stage 7)' if fixed[idx] is not None else '',
                         p['drive_kp'], p['drive_kd'], log.freq[1], f['rms_rad'] / DEG,
                         100 * f['rms_rad'] / amp,
                         '' if good else ('  <-- did not converge' if not f['converged'] else '  <-- FAIL (rms)')))
    if not delays:      # no chirp reached the bus-rate band: use every joint
        delays = [f['params']['delay_s'] for f in fits]
    delay = float(np.median(delays))
    if latency is None:
        flags.append('no stage 7 latency in the calibration; run `arm latency` first')
        ok = False
    else:
        diff = abs(delay - latency * 1e-3)
        within = diff <= sim_dt
        ok &= within
        lines.append('fitted delay %.2f ms vs stage 7 latency %.2f ms: %s one sim step (%.1f ms)' % (
            delay * 1e3, latency, 'within' if within else 'NOT within', sim_dt * 1e3))
    writes['delay_steps'] = int(round(delay / sim_dt))
    writes['delay_ms'] = delay * 1e3
    writes['sim_dt_s'] = sim_dt
    writes['backend'] = backend
    metrics = {'fits': fits, 'delay_ms': delay * 1e3}
    return StageResult('pace', ok, metrics, {'pace': writes}, lines, flags)
