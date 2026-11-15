"""`arm mocap-repeat`: repeatability, measured and pushed down (spec: Mocap path).

Calibration fixes accuracy, not repeatability. This stage:

1. measures ISO 9283 pose repeatability with the flange marker cluster
   (5 poses, cycles from a fixed home; RP = mean(l) + 3 S_l per pose, where l
   is each cycle's distance from the mean attained position),
2. separates the effects: backlash (each pose from opposite directions),
   compliance (with and without a hanging load), thermal drift (a reference
   pose revisited through the session) and control error (final error against
   settle time), and names the dominant one with its lever,
3. tunes Kp, Kd, friction feedforward gain, approach direction and settle
   time by Bayesian optimization, objective = mean RP across the 5 poses on a
   short run (5 poses, 10 cycles), no overshoot as a constraint, gains inside
   fixed bounds. Kp/Kd act on the drives' CSP loop and need the CSP gains
   settable over SDO; when the client cannot set them they are left out,
4. confirms the winner with one full ISO run, and also reports RP with a free
   (random) approach direction, since teleop does not fix it.

Pass: RP under 0.5 mm per pose on the confirmation run, no steady drift.
Writes validation.repeatability_rp_mm; the tuned values are reported for the
joint config (gains, friction_ff_gain, approach, settle_s).

The ISO measurement itself (iso_run, iso_measure) also takes the dial-indicator
cube (hw/indicator_cube.py) as its position source, one pose per fixture
setup: `arm repeat --indicators` and `arm validate` without mocap use it. The
tuning here needs the mocap (every pose, every cycle, no operator).
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from ..control.motion import Trace, move_to, stream, min_jerk
from ..hw.client import Command, Payload
from ..pace.bo import GPBO, Space
from .common import DEG, StageContext, StageResult, feasible_poses, mount_payload, safe_home

RP_MM = 0.5
DRIFT_MM = 0.25
OVERSHOOT = 0.05 * DEG
APPROACH = 3 * DEG             # pre-position offset for a fixed approach direction
POSES = 5


def iso_rp(positions: np.ndarray) -> Dict[str, np.ndarray]:
    """positions: (cycles, poses, 3) attained positions. ISO 9283 RP per pose."""
    mean = positions.mean(axis=0)
    l = np.linalg.norm(positions - mean[None], axis=2)          # (cycles, poses)
    lbar = l.mean(axis=0)
    S = np.sqrt(np.sum((l - lbar) ** 2, axis=0) / max(positions.shape[0] - 1, 1))
    return {'rp': lbar + 3 * S, 'lbar': lbar, 'S': S, 'mean': mean}


def drift(positions: np.ndarray) -> np.ndarray:
    """Steady drift per pose: the linear trend of position across cycles, end to end."""
    c = positions.shape[0]
    if c < 3:
        return np.zeros(positions.shape[1])
    t = np.arange(c) - (c - 1) / 2
    slope = np.einsum('c,cpk->pk', t, positions - positions.mean(axis=0)) / np.sum(t * t)
    return np.linalg.norm(slope, axis=1) * (c - 1)


def iso_poses(ctx: StageContext) -> np.ndarray:
    """5 poses spread through the workspace (ISO uses a cube's diagonal plane);
    from the joint config if listed, else feasible random poses."""
    listed = (ctx.arm.raw.get('validation') or {}).get('iso_poses_rad')
    if listed:
        return np.array(listed, float)
    rng = np.random.default_rng(42)
    return feasible_poses(ctx.model, POSES, rng, [j.max_torque_nm for j in ctx.arm.joints],
                          spread=0.6)


class Trial:
    """Settings of one repeatability run."""

    def __init__(self, kp_scale=1.0, kd_scale=1.0, friction_ff_gain=1.0, approach=1, settle_s=0.5):
        self.kp_scale = kp_scale
        self.kd_scale = kd_scale
        self.friction_ff_gain = friction_ff_gain
        self.approach = approach       # +1 / -1: fixed final direction; 0: free (random)
        self.settle_s = settle_s

    def as_dict(self):
        return dict(self.__dict__)


def reach(ctx, q_goal, trial: Trial, rng, trace=None, speed=0.8):
    """Move to q_goal, last leg from the trial's approach side; returns max overshoot (rad)."""
    client = ctx.client
    direction = trial.approach if trial.approach else rng.choice([-1, 1])
    pre = np.clip(q_goal - direction * APPROACH, ctx.model.lower, ctx.model.upper)
    kw = {'friction_ff_gain': trial.friction_ff_gain}
    move_to(client, pre, max_speed=speed, settle=0.0, **kw)
    s = client.state()
    qs, qds = min_jerk(s.q, q_goal, max(0.25, 1.875 * APPROACH / 0.3), client.cmd_dt)
    stream(client, qs, qds, **kw)
    n = max(1, int(trial.settle_s / client.cmd_dt))
    tr = trace or Trace()
    stream(client, [q_goal] * n, [np.zeros_like(q_goal)] * n, trace=tr, **kw)
    q = np.array(tr.q)
    past = direction * (q - q_goal)       # positive = beyond the target in the approach direction
    return float(np.max(past)) if len(q) else 0.0


def marker_position(mocap, client, q_hold=None, seconds=0.05):
    """Average the cluster position while holding q_hold (keeps the watchdog fed)."""
    n = max(1, int(seconds / client.cmd_dt))
    q_hold = client.state().q.copy() if q_hold is None else np.asarray(q_hold, float)
    P = []
    for _ in range(n):
        client.send(Command(mode='P', q=q_hold))
        client.step(client.cmd_dt)
        P.append(mocap.marker_pose().translation)
    return np.mean(P, axis=0)


def source_position(source, client, q_hold):
    """Attained position from either source: the mocap cluster (averaged while
    holding) or a fixture source such as the indicator cube (`position`)."""
    if hasattr(source, 'marker_pose'):
        return marker_position(source, client, q_hold)
    return source.position(client, q_hold)


def iso_run(ctx, poses, cycles, trial: Trial, mocap, rng, home=None, ref_pose=None):
    """cycles x poses from home. Returns positions (cycles, poses, 3), overshoot, ref positions.

    `mocap` is the position source. The mocap sees every pose, so each cycle
    visits all poses. A fixture source (`one_pose_per_setup`, the indicator
    cube) is set up at one pose at a time: all cycles there, then the next
    pose; each pose's positions are then in that setup's own zeroed frame,
    which RP and drift do not mind. The thermal reference pose needs the
    mocap and is skipped for a fixture."""
    client = ctx.client
    home = safe_home(ctx.model) if home is None else home
    _apply_gains(ctx, trial)
    P = np.zeros((cycles, len(poses), 3))
    over = 0.0
    refs = []
    if getattr(mocap, 'one_pose_per_setup', False):
        for k, q in enumerate(poses):
            move_to(client, home, max_speed=0.8, settle=0.05)
            reach(ctx, q, trial, rng)
            mocap.setup(ctx, k, q)
            for c in range(cycles):
                move_to(client, home, max_speed=0.8, settle=0.05)
                over = max(over, reach(ctx, q, trial, rng))
                P[c, k] = source_position(mocap, client, q)
        return P, over, refs
    for c in range(cycles):
        for k, q in enumerate(poses):
            move_to(client, home, max_speed=0.8, settle=0.05)
            over = max(over, reach(ctx, q, trial, rng))
            P[c, k] = marker_position(mocap, client, q)
        if ref_pose is not None:
            move_to(client, home, max_speed=0.8, settle=0.05)
            reach(ctx, ref_pose, Trial(approach=1, settle_s=0.5), rng)
            refs.append((client.time_since_power_on(), marker_position(mocap, client, ref_pose)))
    return P, over, refs


def warm_up(ctx, poses, warm_s: float = None):
    """Motion through the poses until warm_s has passed (20 min on hardware, 2 s in sim)."""
    client = ctx.client
    warm_s = warm_s if warm_s is not None else (2.0 if ctx.sim else 1200.0)
    home = safe_home(ctx.model)
    t_end = client.time_since_power_on() + warm_s
    while True:
        for q in poses:
            move_to(client, q, max_speed=0.8, settle=0.0)
        move_to(client, home, max_speed=0.8, settle=0.0)
        if math.isnan(t_end) or client.time_since_power_on() >= t_end:
            break


def iso_measure(ctx, source, cycles: int, rng, warm_s: float = None, poses=None) -> Dict:
    """ISO 9283 measurement with the tuned settings (joint config
    `repeatability_tuning`, else defaults): warm-up, cycles x poses, RP and
    drift per pose. Used by `arm validate` and `arm repeat`, with the mocap or
    the indicator cube."""
    poses = iso_poses(ctx) if poses is None else poses
    warm_up(ctx, poses, warm_s)
    noise = source.static_noise_3sigma_m()
    tuned = (ctx.arm.raw.get('repeatability_tuning') or {})
    trial = Trial(**{k: v for k, v in tuned.items() if k in Trial().as_dict()})
    P, _, _ = iso_run(ctx, poses, cycles, trial, source, rng)
    rp = iso_rp(P)['rp'] * 1e3
    dr = drift(P) * 1e3
    ok = bool(noise < 1e-4 and np.all(rp < RP_MM) and np.all(dr < DRIFT_MM))
    return {'rp_mm': rp, 'drift_mm': dr, 'noise_3sigma_m': noise, 'passed': ok, 'positions': P,
            'cycles': cycles, 'source': getattr(source, 'name', 'mocap')}


def _apply_gains(ctx, trial: Trial) -> bool:
    n = ctx.arm.n
    try:
        ctx.client.set_drive_gains(np.full(n, trial.kp_scale), np.full(n, trial.kd_scale))
        return True
    except NotImplementedError:
        return False


def effects(ctx, poses, mocap, rng, cycles) -> Dict[str, float]:
    """Size of each repeatability effect, mm."""
    client = ctx.client
    home = safe_home(ctx.model)
    out = {}
    # backlash: same pose from opposite directions
    diffs = []
    for q in poses:
        sides = []
        for direction in (1, -1):
            pts = []
            for _ in range(max(2, cycles // 2)):
                move_to(client, home, max_speed=0.8, settle=0.05)
                reach(ctx, q, Trial(approach=direction), rng)
                pts.append(marker_position(mocap, client, q))
            sides.append(np.mean(pts, axis=0))
        diffs.append(np.linalg.norm(sides[0] - sides[1]))
    out['backlash'] = float(np.max(diffs) * 1e3)
    # compliance: with and without a hanging load
    q = poses[0]
    move_to(client, home, max_speed=0.8, settle=0.05)
    reach(ctx, q, Trial(), rng)
    a = marker_position(mocap, client, q)
    mount_payload(ctx, Payload(link=ctx.model.link_names()[-1], mass_kg=1.0))
    client.hold(0.5)
    b = marker_position(mocap, client, q)
    mount_payload(ctx, None)
    client.hold(0.3)
    out['compliance'] = float(np.linalg.norm(a - b) * 1e3)
    # control error: final position against settle time, relative to a long settle
    ref = None
    errs = {}
    for settle in (2.0, 0.1, 0.3, 0.6, 1.0):
        move_to(client, home, max_speed=0.8, settle=0.05)
        reach(ctx, q, Trial(settle_s=settle), rng)
        p = marker_position(mocap, client, q)
        if ref is None:
            ref = p
        else:
            errs[settle] = float(np.linalg.norm(p - ref) * 1e3)
    out['control'] = max(errs.values())
    out['control_by_settle_mm'] = errs
    return out


LEVERS = {
    'backlash': 'same final approach direction every time; preload or an output-side encoder if large',
    'compliance': 'gravity feedforward; stiffer links',
    'thermal': 'warm-up routine; cap continuous current',
    'control': 'higher Kp or an integral term, friction feedforward, longer settle',
}


def run(ctx: StageContext, trials: int = None, warm_s: float = None) -> StageResult:
    mocap = ctx.client.mocap()
    if mocap is None:
        return StageResult('mocap-repeat', False, {}, {}, ['no mocap source on this client'])
    lines, flags = [], []
    rng = np.random.default_rng(7)
    poses = iso_poses(ctx)
    noise = mocap.static_noise_3sigma_m()
    lines.append('mocap still-cluster 3 sigma %.3f mm%s' % (noise * 1e3, '' if noise < 1e-4 else ' (> 0.1 mm!)'))
    short_cycles = 3 if ctx.quick else 10
    full_cycles = 5 if ctx.quick else 30
    trials = trials or (6 if ctx.quick else 24)

    # warm-up: 20 min of motion on hardware (shortened in sim)
    warm_s = warm_s if warm_s is not None else (2.0 if ctx.sim else 1200.0)
    home = safe_home(ctx.model)
    t_end = ctx.client.time_since_power_on() + warm_s if not math.isnan(ctx.client.time_since_power_on()) else None
    k = 0
    while True:
        move_to(ctx.client, poses[k % len(poses)], max_speed=0.8, settle=0.0)
        move_to(ctx.client, home, max_speed=0.8, settle=0.0)
        k += 1
        if t_end is None or ctx.client.time_since_power_on() >= t_end:
            break

    baseline = Trial()
    P0, _, _ = iso_run(ctx, poses, short_cycles, baseline, mocap, rng)
    rp0 = iso_rp(P0)['rp'] * 1e3
    lines.append('baseline (%d cycles): RP %s mm' % (short_cycles, np.round(rp0, 3).tolist()))

    eff = effects(ctx, poses, mocap, rng, short_cycles)

    gains_settable = _apply_gains(ctx, baseline)
    dims = {'friction_ff_gain': ('lin', 0.0, 1.5), 'approach': ('choice', [1, -1]),
            'settle_s': ('lin', 0.15, 1.2)}
    if gains_settable:
        dims = {'kp_scale': ('log', 0.5, 2.0), 'kd_scale': ('log', 0.5, 2.0), **dims}
    else:
        flags.append('CSP gains not settable over SDO here: Kp/Kd left out of the tuning')
    space = Space(dims)
    bo = GPBO(space, n_init=min(4, trials), seed=3)
    history = []
    ref_pose = poses[0]
    refs_all = []
    for i in range(trials):
        u = bo.ask()
        params = space.to_params(u)
        trial = Trial(**params)
        P, over, refs = iso_run(ctx, poses, short_cycles, trial, mocap, rng, ref_pose=ref_pose)
        refs_all += refs
        rp = iso_rp(P)['rp'] * 1e3
        # settle time costs cycle time: a small penalty so equal RP prefers shorter settles
        score = float(np.mean(rp)) + 0.01 * trial.settle_s
        if over > OVERSHOOT:
            score += 10.0 * (over - OVERSHOOT) / DEG + 1.0
        bo.tell(u, score)
        history.append({'trial': i, **trial.as_dict(), 'rp_mean_mm': float(np.mean(rp)),
                        'overshoot_deg': over / DEG, 'score': score})
    u_best, _ = bo.best()
    best = Trial(**space.to_params(u_best))
    lines.append('tuning: %d trials, best %s' % (trials, {k: (round(v, 3) if isinstance(v, float) else v)
                                                        for k, v in best.as_dict().items()}))

    # thermal drift: the reference pose through the session
    if len(refs_all) >= 2:
        ref_p = np.array([p for _, p in refs_all])
        eff['thermal'] = float(np.linalg.norm(ref_p[-1] - ref_p[0]) * 1e3)
    else:
        eff['thermal'] = 0.0
    sizes = {k: v for k, v in eff.items() if k in LEVERS}
    dominant = max(sizes, key=sizes.get)
    for k, v in sizes.items():
        lines.append('  %-10s %.3f mm%s' % (k, v, '   <- dominant: ' + LEVERS[k] if k == dominant else ''))

    # confirm with one full run, then the free-approach number
    P, over, _ = iso_run(ctx, poses, full_cycles, best, mocap, rng)
    res = iso_rp(P)
    rp = res['rp'] * 1e3
    dr = drift(P) * 1e3
    free = Trial(**{**best.as_dict(), 'approach': 0})
    Pf, _, _ = iso_run(ctx, poses, short_cycles, free, mocap, rng)
    rp_free = iso_rp(Pf)['rp'] * 1e3
    if gains_settable:
        _apply_gains(ctx, baseline)      # leave the drives as found; the tuned gains are reported
    lines.append('confirmation (%d cycles): RP %s mm, drift %s mm' % (
        full_cycles, np.round(rp, 3).tolist(), np.round(dr, 3).tolist()))
    lines.append('free approach direction: RP %s mm' % np.round(rp_free, 3).tolist())
    passed = bool(np.all(rp < RP_MM) and np.all(dr < DRIFT_MM))
    metrics = {'rp_mm': rp.tolist(), 'rp_free_mm': rp_free.tolist(), 'drift_mm': dr.tolist(),
               'baseline_rp_mm': rp0.tolist(), 'effects_mm': eff, 'dominant': dominant,
               'tuning': history, 'best': best.as_dict(), 'gains_settable': gains_settable,
               'time_since_power_on_s': ctx.client.time_since_power_on()}
    return StageResult('mocap-repeat', passed, metrics,
                       {'validation': {'repeatability_rp_mm': float(np.max(rp))}}, lines, flags,
                       joint_config_updates={'repeatability_tuning': best.as_dict()})
