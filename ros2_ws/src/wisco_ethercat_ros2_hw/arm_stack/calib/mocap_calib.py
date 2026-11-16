"""`arm mocap-calib`: kinematic calibration from a flange marker cluster (spec: Mocap path).

1. Measure the mocap: 3 sigma of a still cluster, and the length-bar error,
   which caps the accuracy any fit can reach.
2. The arm drives itself through 50 to 100 poses in Mode P (half of them with a
   hanging load), settling before each sample.
3. One nonlinear least-squares fit (Levenberg-Marquardt) for joint zero
   offsets, joint-origin position and axis errors (the kinematic overlay), the
   base-to-mocap and marker-to-flange transforms, and per-joint compliance and
   backlash, so gravity deflection is modeled instead of folded into the
   geometry. Redundant parameters are removed first (greedy rank selection on
   the scaled Jacobian, in priority order), and the condition number of the
   kept set is reported.
   J1 and J7 zero offsets fold into the base and flange transforms, so they
   come out as unobservable and keep their jig values.
4. Refine in rounds: score fresh held-out poses, move the worst into the fit
   set, refit; stop when a round's gain is inside the measurement repeat spread.
Pass: residual RMS on 20 held-out poses under twice the length-bar error.
Writes zero_offsets_rad, kinematic_overlay, joint_stiffness_nm_per_rad.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pinocchio as pin

from ..control.motion import move_to
from ..hw.client import Command, Payload
from . import kinfit
from .common import DEG, StageContext, StageResult, feasible_poses, mount_payload, safe_home
from .kinfit import BACKLASH_EPS, compose_overlay, mean_rotation, rms  # noqa: F401 (re-exported)

ROT_WEIGHT = 0.05             # m per rad: orientation residuals count like 5 cm of lever
LOAD_KG = 1.0


@dataclass
class Sample:
    q: np.ndarray             # encoder angles (motor side), rad
    tau: np.ndarray           # measured joint torque, Nm
    pose: pin.SE3             # marker cluster in the mocap frame
    loaded: bool


def measure(ctx: StageContext, q_goal, mocap, loaded, settle=0.6, average=0.15) -> Sample:
    client = ctx.client
    move_to(client, q_goal, max_speed=0.8, settle=settle)
    n = max(1, int(average / client.cmd_dt))
    Q, T, P, R = [], [], [], []
    for _ in range(n):
        client.send(Command(mode='P', q=np.asarray(q_goal, float)))
        s = client.step(client.cmd_dt)
        M = mocap.marker_pose()
        Q.append(s.q)
        T.append(s.tau)
        P.append(M.translation)
        R.append(M.rotation)
    return Sample(np.mean(Q, axis=0), np.mean(T, axis=0),
                  pin.SE3(mean_rotation(np.array(R)), np.mean(P, axis=0)), loaded)


class KinematicFit(kinfit.KinematicModel):
    """Parameter vector (all increments on the current calibration):
    dzero[n], (dxyz[3], drot[3]) per joint, base_in_mocap twist[6],
    marker_in_flange twist[6], compliance[n] (rad/Nm), backlash[n] (rad)."""

    def __init__(self, model, frame='flange'):
        j = kinfit.joint_blocks(model.joint_names)
        super().__init__(model, frame, [
            ('dzero',) + j['dzero'], ('overlay',) + j['overlay'],
            ('base', ['base:%d' % i for i in range(6)], np.full(6, 1e-2)),
            ('marker', ['marker:%d' % i for i in range(6)], np.full(6, 1e-2)),
            ('compliance',) + j['compliance'], ('backlash',) + j['backlash']])

    se3 = staticmethod(kinfit.se3)

    def predict(self, x, samples: List[Sample], base0: pin.SE3, marker0: pin.SE3):
        self.set_geometry(x)
        base = base0 * self.se3(x[self.sl['base']])
        marker = marker0 * self.se3(x[self.sl['marker']])
        return [base * self.frame_pose(self.link_angles(x, s.q, s.tau)) * marker for s in samples]

    def residuals(self, x, samples, base0, marker0):
        r = []
        for s, M in zip(samples, self.predict(x, samples, base0, marker0)):
            r.append(M.translation - s.pose.translation)
            r.append(ROT_WEIGHT * pin.log3(M.rotation.T @ s.pose.rotation))
        return np.concatenate(r)


def initial_transforms(fit: KinematicFit, samples: List[Sample]):
    """Base-to-mocap by Kabsch on flange positions, then marker-to-flange in closed form."""
    fit.set_geometry(np.zeros(fit.size))
    F = [fit.model.fk(s.q, fit.frame) for s in samples]
    base = kinfit.kabsch(np.array([f.translation for f in F]), np.array([s.pose.translation for s in samples]))
    Rm = mean_rotation(np.array([(base.rotation @ f.rotation).T @ s.pose.rotation
                                 for f, s in zip(F, samples)]))
    # p_meas = R_b (p_f + R_f t) + t_b  ->  linear in t
    M = np.vstack([base.rotation @ f.rotation for f in F])
    y = np.concatenate([s.pose.translation - base.act(f.translation) for f, s in zip(F, samples)])
    t = np.linalg.lstsq(M, y, rcond=None)[0]
    return base, pin.SE3(Rm, t)


PRIORITY = ('base', 'marker', 'dzero', 'compliance', 'backlash', 'overlay')


def select_parameters(fit: KinematicFit, J: np.ndarray, tol: float = 2e-3):
    """Greedy independent-column selection in PRIORITY order on the scaled
    Jacobian (kinfit.select_parameters). Returns (kept indices, condition
    number of the kept set)."""
    return kinfit.select_parameters(fit, J, PRIORITY, tol)


def solve(fit: KinematicFit, samples, base0, marker0, keep, x0=None):
    return kinfit.solve(fit, (samples, base0, marker0), keep, x0)


def per_pose_mm(fit, x, samples, base0, marker0) -> np.ndarray:
    P = fit.predict(x, samples, base0, marker0)
    return np.array([np.linalg.norm(M.translation - s.pose.translation)
                     for M, s in zip(P, samples)]) * 1e3


def collect(ctx, poses, mocap, loaded_share=0.5) -> List[Sample]:
    """The first part of the poses unloaded, the rest with the test mass on the last link."""
    samples = []
    split = int(round(len(poses) * (1 - loaded_share)))
    link = ctx.model.link_names()[-1]
    for k, q in enumerate(poses):
        if k == split:
            mount_payload(ctx, Payload(link=link, mass_kg=LOAD_KG, offset_m=(0, 0, 0)))
        samples.append(measure(ctx, q, mocap, loaded=k >= split))
    if split < len(poses):
        mount_payload(ctx, None)
    return samples


def run(ctx: StageContext, n_fit: int = None, n_test: int = None, rounds: int = None) -> StageResult:
    mocap = ctx.client.mocap()
    if mocap is None:
        return StageResult('mocap-calib', False, {}, {}, ['no mocap source on this client'])
    arm, model = ctx.arm, ctx.model
    n_fit = n_fit or (36 if ctx.quick else 70)
    n_test = n_test or (12 if ctx.quick else 20)
    rounds = rounds or (2 if ctx.quick else 4)
    lines, flags = [], []

    noise = mocap.static_noise_3sigma_m()
    bar = mocap.length_bar_error_m() if hasattr(mocap, 'length_bar_error_m') else None
    if bar is None:
        bar = (arm.raw.get('mocap') or {}).get('length_bar_error_m')
    if bar is None:
        return StageResult('mocap-calib', False, {}, {}, [
            'length-bar error unknown: set mocap.length_bar_error_m in the joint config'])
    threshold_mm = 2 * bar * 1e3
    lines.append('mocap: still-cluster 3 sigma %.3f mm, length-bar error %.3f mm -> pass < %.3f mm'
                 % (noise * 1e3, bar * 1e3, threshold_mm))
    if noise > 1e-4:
        flags.append('mocap static noise above 0.1 mm')

    max_torque = [j.max_torque_nm for j in arm.joints]
    pool = list(feasible_poses(model, n_fit + n_test * rounds, ctx.rng, max_torque, extra_mass=LOAD_KG))
    if len(pool) < n_fit + n_test:
        return StageResult('mocap-calib', False, {}, {}, ['only %d feasible poses' % len(pool)])
    fit_poses, pool = pool[:n_fit], pool[n_fit:]
    samples = collect(ctx, fit_poses, mocap)
    # measurement repeat spread: one pose reached three times from home
    home = safe_home(model)
    rep = []
    for _ in range(3):
        move_to(ctx.client, home, max_speed=0.8, settle=0.1)
        rep.append(measure(ctx, fit_poses[0], mocap, False))
    rep_p = np.array([s.pose.translation for s in rep])
    repeat_mm = float(np.max(np.linalg.norm(rep_p - rep_p.mean(axis=0), axis=1)) * 1e3)

    fit = KinematicFit(model)
    base0, marker0 = initial_transforms(fit, samples)
    x = np.zeros(fit.size)
    keep, cond = select_parameters(fit, fit.jacobian(x, samples, base0, marker0))
    lines.append('%d samples (%d loaded), %d of %d parameters identifiable, condition number %.1f'
                 % (len(samples), sum(s.loaded for s in samples), len(keep), fit.size, cond))
    if cond > 1e3:
        flags.append('condition number %.0f: add poses or drop parameters before trusting the fit' % cond)
    x = solve(fit, samples, base0, marker0, keep)
    history = []
    prev = None
    for r in range(rounds):
        test_poses, pool = pool[:n_test], pool[n_test:]
        if len(test_poses) < n_test:
            break
        test = collect(ctx, test_poses, mocap, loaded_share=0.0)
        held = per_pose_mm(fit, x, test, base0, marker0)
        history.append({'round': r, 'fit_rms_mm': rms(per_pose_mm(fit, x, samples, base0, marker0)),
                        'held_out_rms_mm': rms(held), 'samples': len(samples)})
        lines.append('round %d: fit RMS %.3f mm on %d samples, held-out RMS %.3f mm on %d'
                     % (r, history[-1]['fit_rms_mm'], len(samples), rms(held), len(test)))
        if prev is not None and prev - rms(held) < repeat_mm:
            break
        prev = rms(held)
        if r == rounds - 1:
            break
        worst = np.argsort(held)[-max(2, n_test // 4):]
        samples += [test[i] for i in worst]
        x = solve(fit, samples, base0, marker0, keep, x0=x)

    final = history[-1]['held_out_rms_mm'] if history else float('nan')
    passed = bool(final < threshold_mm and cond < 1e4)
    lines.append('repeat spread %.3f mm; final held-out RMS %.3f mm' % (repeat_mm, final))

    names = arm.names
    observed = set(fit.labels[i] for i in keep)
    dzero = x[fit.sl['dzero']]
    old_zero = ctx.calibration.get('zero_offsets_rad') or {}
    new_zero = {}
    for i, n in enumerate(names):
        if 'dzero:' + n in observed:
            # q_true = q_meas + d  and  q_meas = raw - zero  ->  zero_new = zero - d
            new_zero[n] = float((old_zero.get(n) or 0.0) - dzero[i])
            lines.append('  %-15s zero correction %+.4f deg' % (n, dzero[i] / DEG))
        else:
            lines.append('  %-15s zero not observable (folds into base/flange), keeps the jig value' % n)
    c = x[fit.sl['compliance']]
    stiffness = {n: float(1.0 / c[i]) for i, n in enumerate(names)
                 if 'compliance:' + n in observed and c[i] > 1e-7}
    if stiffness:
        lines.append('  stiffness Nm/rad: ' + ', '.join('%s %.0f' % kv for kv in stiffness.items()))
    overlay = compose_overlay(ctx.calibration.get('kinematic_overlay'), names,
                              x[fit.sl['overlay']].reshape(len(names), 6))
    writes = {'zero_offsets_rad': new_zero, 'kinematic_overlay': overlay}
    if stiffness:
        writes['joint_stiffness_nm_per_rad'] = stiffness
    metrics = {'threshold_mm': threshold_mm, 'held_out_rms_mm': final, 'rounds': history,
               'condition_number': cond, 'parameters': [fit.labels[i] for i in keep],
               'repeat_spread_mm': repeat_mm, 'mocap_noise_3sigma_mm': noise * 1e3,
               'length_bar_error_mm': bar * 1e3,
               'zero_corrections_rad': {n: float(dzero[i]) for i, n in enumerate(names)
                                        if 'dzero:' + n in observed},
               'backlash_rad': {n: float(v) for n, v in zip(names, x[fit.sl['backlash']])}}
    return StageResult('mocap-calib', passed, metrics, writes, lines, flags)
