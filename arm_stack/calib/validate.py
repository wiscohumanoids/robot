"""`arm validate`: release gate (spec: Validation and repeatability).

The arm is released for data collection only after all four tests pass warm;
the report is saved beside the calibration file (the CLI does that).

  Repeatability  ISO 9283: warm up 20 min, 5 poses x 30 cycles from a fixed home,
                 flange measured by the mocap cluster (static 3 sigma checked
                 under 0.1 mm first). RP < 0.5 mm per pose, no steady drift.
                 Without mocap: three dial indicators on a cube on the flange
                 (hw/indicator_cube.py), one pose per fixture setup; the sim
                 uses it automatically, hardware asks first.
  Pivot          stage 8 rerun at the end of the warm session: spread < 3 mm
                 and within 1 mm of the cold result.
  Hold           slow up/down sweeps through 10 poses not used in the stage 5
                 fit (residual < 5% of peak gravity torque per joint), then
                 gravity comp only (Mode T, no stiffness) in each: drift < 2 deg in 5 s.
  Replay         3 recorded teleop trajectories in Mode P, each twice: RMS joint
                 error < 0.5 deg, run-to-run variation < 0.1 deg.

The time since power on is logged with every run (the drives expose no
temperature over EtherCAT).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from ..control.motion import Trace, move_to, stream
from ..hw import indicator_cube
from ..hw.client import Command
from . import mocap_repeat, pivot
from .common import DEG, StageContext, StageResult, random_poses, safe_home
from .gravity import RESIDUAL_FRACTION, measure_pose

HOLD_DRIFT = 2 * DEG
HOLD_S = 5.0
REPLAY_RMS = 0.5 * DEG
REPLAY_VAR = 0.1 * DEG
PIVOT_DELTA_MM = 1.0


def synthetic_teleop(model, rng, duration=8.0, dt=0.001, count=3) -> List[np.ndarray]:
    """Smooth teleop-like trajectories (sums of slow sines inside the limits),
    used when no recorded sessions are given."""
    out = []
    home = safe_home(model)
    lo, hi = model.lower + 15 * DEG, model.upper - 15 * DEG
    t = np.arange(0, duration, dt)
    for _ in range(count):
        q = np.repeat(home[None], len(t), axis=0)
        for j in range(model.n):
            amp = 0.25 * (hi[j] - lo[j])
            for f, a in zip(rng.uniform(0.05, 0.4, 3), rng.uniform(0.2, 0.5, 3)):
                q[:, j] += a * amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        ramp = np.minimum(1, t / 1.0)[:, None] * np.minimum(1, (duration - t) / 1.0)[:, None]
        q = home + ramp * (q - home)
        out.append(np.clip(q, lo, hi))
    return out


def replay(ctx, traj: np.ndarray, dt: float):
    client = ctx.client
    move_to(client, traj[0], max_speed=0.5, settle=0.5)
    stride = max(1, int(round(client.cmd_dt / dt)))
    q = traj[::stride]
    qd = np.gradient(q, client.cmd_dt, axis=0)
    tr = Trace()
    stream(client, q, qd, trace=tr)
    a = tr.arrays()
    return a['q'] - a['q_cmd']


def hold_test(ctx, n_poses, exclude: Sequence[np.ndarray]):
    model = ctx.model
    rng = np.random.default_rng(2024)
    peak_g = model.peak_gravity_torque()
    peak = np.array([j.peak_torque_nm for j in ctx.arm.joints])
    poses = []
    for q in random_poses(model, n_poses * 5, rng, margin=12 * DEG, spread=0.8):
        if all(np.max(np.abs(q - e)) > 5 * DEG for e in exclude):
            poses.append(q)
        if len(poses) == n_poses:
            break
    worst = np.zeros(model.n)
    drifts = []
    client = ctx.client
    for q in poses:
        clamp = np.minimum(peak, np.abs(model.gravity(q)) * 1.5 + 0.25 * peak)
        tau, q_meas = measure_pose(ctx, q, clamp)
        worst = np.maximum(worst, np.abs(tau - model.gravity(q_meas)))
        move_to(client, q, max_speed=0.4, settle=0.3)
        s0 = client.state()
        zeros = np.zeros(model.n)
        s = s0
        for _ in range(int(HOLD_S / client.cmd_dt)):
            client.send(Command(mode='T', q=s0.q.copy(), kp=zeros, kd=zeros))
            s = client.step(client.cmd_dt)
        drifts.append(float(np.max(np.abs(s.q - s0.q))))
        move_to(client, q, max_speed=0.4, settle=0.1)
    frac = worst / np.maximum(peak_g, 1e-6)
    return frac, np.array(drifts), poses


def repeatability_source(ctx: StageContext):
    """(source, why-not): the mocap if there is one, else the indicator cube
    (sim: automatically; hardware: if the operator has the fixture ready)."""
    mocap = ctx.client.mocap()
    if mocap is not None:
        return mocap, None
    if not ctx.sim:
        try:
            answer = input('No mocap. Measure repeatability with the dial-indicator cube now '
                           '(one fixture setup per ISO pose, readings typed per cycle)? [y/N] ')
        except EOFError:
            answer = ''
        if not answer.strip().lower().startswith('y'):
            return None, 'no mocap and the indicator cube declined'
    return indicator_cube.for_context(ctx), None


def run(ctx: StageContext, replay_trajectories: Optional[List[np.ndarray]] = None,
        replay_dt: float = 0.001, warm_s: float = None) -> StageResult:
    lines, flags = [], []
    results = {}
    client = ctx.client
    t_on = client.time_since_power_on()
    lines.append('time since power on at start: %s' % ('%.0f s' % t_on if not math.isnan(t_on) else 'unknown'))

    # 1. repeatability (includes the warm-up): mocap, else the dial-indicator cube
    source, why = repeatability_source(ctx)
    if source is not None:
        rep = mocap_repeat
        res = rep.iso_measure(ctx, source, 5 if ctx.quick else 30, np.random.default_rng(11), warm_s)
        rp, dr, noise, ok = res['rp_mm'], res['drift_mm'], res['noise_3sigma_m'], res['passed']
        results['repeatability'] = ok
        lines.append('repeatability (%s): 3 sigma %.3f mm; RP %s mm (pass < %.1f); drift %s mm -> %s' % (
            res['source'], noise * 1e3, np.round(rp, 3).tolist(), rep.RP_MM, np.round(dr, 3).tolist(),
            'PASS' if ok else 'FAIL'))
        rp_max = float(np.max(rp))
    else:
        results['repeatability'] = False
        rp_max = None
        lines.append('repeatability: %s -> NOT RUN' % why)
        flags.append('repeatability needs mocap or the dial-indicator cube (arm repeat --indicators)')

    # 2. hold
    exclude = [np.array(q) for q in ((ctx.calibration.get('validation') or {}).get('gravity_fit_poses') or [])]
    frac, drifts, _ = hold_test(ctx, 5 if ctx.quick else 10, exclude)
    ok = bool(np.all(frac < RESIDUAL_FRACTION) and np.all(drifts < HOLD_DRIFT))
    results['hold'] = ok
    lines.append('hold: worst residual %.1f%% of peak gravity (pass < %.0f%%), drift max %.2f deg in %.0f s '
                 '(pass < %.0f) -> %s' % (100 * frac.max(), 100 * RESIDUAL_FRACTION, drifts.max() / DEG, HOLD_S,
                                          HOLD_DRIFT / DEG, 'PASS' if ok else 'FAIL'))

    # 3. replay tracking
    if replay_trajectories is None:
        replay_trajectories = synthetic_teleop(ctx.model, np.random.default_rng(5),
                                               duration=4.0 if ctx.quick else 10.0)
        flags.append('replay used synthetic teleop trajectories (pass --replay SESSION.mcap for recorded ones)')
    rms_runs = []
    for traj in replay_trajectories:
        runs = [replay(ctx, traj, replay_dt) for _ in range(2)]
        rms_runs.append([float(np.sqrt(np.mean(e ** 2))) for e in runs])
    rms_runs = np.array(rms_runs)
    rms = float(rms_runs.max())
    var = float(np.max(np.abs(rms_runs[:, 0] - rms_runs[:, 1])))
    ok = bool(rms < REPLAY_RMS and var < REPLAY_VAR)
    results['replay'] = ok
    lines.append('replay: RMS %.3f deg (pass < %.1f), run-to-run %.3f deg (pass < %.1f) -> %s' % (
        rms / DEG, REPLAY_RMS / DEG, var / DEG, REPLAY_VAR / DEG, 'PASS' if ok else 'FAIL'))

    # 4. pivot, at the end of the warm session
    cold = (ctx.calibration.get('validation') or {}).get('pivot_spread_mm')
    pr = pivot.run(ctx)
    warm = pr.metrics.get('spread_mm', float('nan'))
    near = cold is not None and abs(warm - cold) < PIVOT_DELTA_MM
    ok = bool(pr.passed and near)
    results['pivot'] = ok
    lines.append('pivot (warm): spread %.2f mm, cold %s -> %s' % (
        warm, '%.2f mm' % cold if cold is not None else 'not measured (run arm pivot first)',
        'PASS' if ok else 'FAIL'))

    passed = all(results.values())
    lines.append('time since power on at end: %s' % (
        '%.0f s' % client.time_since_power_on() if not math.isnan(client.time_since_power_on()) else 'unknown'))
    validation = {'hold_drift_deg': float(drifts.max() / DEG), 'replay_rms_deg': rms / DEG}
    if rp_max is not None:
        validation['repeatability_rp_mm'] = rp_max
    return StageResult('validate', passed,
                       {'tests': results, 'pivot_warm_mm': warm, 'pivot_cold_mm': cold,
                        'hold_residual_frac': frac.tolist(), 'hold_drift_deg': (drifts / DEG).tolist(),
                        'replay_rms_deg': (rms_runs / DEG).tolist(), 'repeatability_rp_mm': rp_max,
                        'time_since_power_on_s': client.time_since_power_on()},
                       {'validation': validation}, lines, flags)
