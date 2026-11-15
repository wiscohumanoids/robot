"""`arm plate`: kinematic calibration from a multi-divot plate, without mocap.

A machined plate holds N divots (6 to 9) at known positions in the plate frame
(`plate:` in the joint config, from the plate drawing).

1. The probe tip (tool0) is hand guided (Mode T, gravity comp, no stiffness)
   into each divot in several varied configurations, then again with the test
   mass hanging from the last link, so the joint torques span more than the
   arm's own weight gives (joint stiffness).
2. One Levenberg-Marquardt fit, from tip positions only: the plate pose in the
   base frame (6), the probe-tip offset in tool0 (3), joint zero offsets, a
   rank-selected subset of the kinematic overlay, joint compliance (when the
   loaded pass is in) and backlash. The residual is FK(tip) - T_plate * divot,
   so the plate's machined divot spacing is the only length reference. The
   parameterization and the greedy rank selection are mocap-calib's
   (calib/kinfit.py). J1's zero folds into the plate pose and J7's into the
   probe-tip offset (tool0 sits ~160 mm off the J7 axis, so J7's zero, the
   probe's angle about it and the wrist tilts trade against each other near
   one plate), so both come out unobservable, as in mocap-calib. J1 keeps its
   jig value; J7 gets the tool0-equivalent value (below).
3. Every parameter's observability (the share of its Jacobian column that the
   higher-priority kept ones cannot produce) and 1 sigma are reported, with
   the condition number of the kept set.
4. Validation on held-out divots (never in the fit): their tips' RMS error
   against the fitted plate, and the distance of every divot pair that
   includes a held-out one, measured vs drawing, against the plate tolerance.
Pass: held-out RMS (tips and pair distances) under 0.5 mm and condition
number below 1e4. Writes zero_offsets_rad, kinematic_overlay and
joint_stiffness_nm_per_rad, as mocap-calib does.

The probe-tip offset is then folded into J7 for storage (fold_tip): its
rotation about the J7 axis into J7's zero, its part along the axis into J7's
origin, both exact at tool0, so the stored calibration reproduces the fit at
the tip; the held-out check runs on the stored parameters. The written J7 zero
is the tool0-equivalent one, not a physical J7 zero. The radial part has no
calibration field: it is reported and flagged over 0.5 mm (fix tool0_joint).
`plate.fit_tip: false` uses the URDF tip as is and fits J7's zero directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pinocchio as pin

from ..hw.client import Payload
from . import kinfit, pivot
from .common import DEG, StageContext, StageResult, mount_payload

HELD_OUT_MM = 0.5
COND_MAX = 1e4
COND_WARN = 1e3
LOAD_KG = 1.0
# Tip positions alone see less than mocap's full poses: a term whose column is under 3%
# independent of the higher-priority ones costs more variance than it removes bias
# (at 1e-2, sim fits trade J2's zero against weak overlay terms by up to 0.5 deg).
SELECT_TOL = 3e-2
ZERO_SIGMA_MAX = 0.05 * DEG   # a zero correction is written only if its 1 sigma is under this
TIP_FLAG_MM = 0.5
# plate and probe tip first: they absorb J1's and J7's zeros, as base/marker do in mocap-calib
PRIORITY = ('plate', 'tip', 'dzero', 'compliance', 'overlay', 'backlash')


@dataclass
class Plate:
    divots: np.ndarray            # (N, 3) divot seats in the plate frame, m
    tolerance_mm: float
    held_out: List[int]
    cone: float                   # probe axis within this of the plate normal, rad
    pose_hint: pin.SE3            # rough plate frame in base_link
    needs_drawing: bool
    fit_tip: bool = True          # fit a probe-tip offset, stored folded into J7 (fold_tip)

    @property
    def fit_divots(self) -> List[int]:
        return [k for k in range(len(self.divots)) if k not in self.held_out]

    @staticmethod
    def from_config(raw: dict) -> Optional['Plate']:
        cfg = raw.get('plate') or {}
        if not cfg.get('divots_m'):
            return None
        pose = cfg.get('pose_in_base') or {}
        return Plate(divots=np.array(cfg['divots_m'], float),
                     tolerance_mm=float(cfg.get('tolerance_mm', 0.02)),
                     held_out=[int(k) for k in cfg.get('held_out') or []],
                     cone=float(cfg.get('approach_cone_deg', 60.0)) * DEG,
                     pose_hint=pin.SE3(pin.rpy.rpyToMatrix(*[float(v) for v in pose.get('rpy', [0, 0, 0])]),
                                       np.array(pose.get('xyz', [0, 0, 0]), float)),
                     needs_drawing=bool(cfg.get('needs_drawing', False)),
                     fit_tip=bool(cfg.get('fit_tip', True)))


@dataclass
class PlateSample:
    q: np.ndarray                 # encoder angles (motor side), rad
    tau: np.ndarray               # measured joint torque, Nm
    divot: int
    loaded: bool


class PlateFit(kinfit.KinematicModel):
    """Parameter vector: dzero[n], (dxyz[3], drot[3]) per joint, plate-in-base
    twist[6] (increment on plate0), probe-tip offset in tool0[3],
    compliance[n] (rad/Nm, absolute), backlash[n] (rad, absolute)."""

    def __init__(self, model, divots: np.ndarray):
        j = kinfit.joint_blocks(model.joint_names)
        super().__init__(model, model.tool_frame, [
            ('dzero',) + j['dzero'], ('overlay',) + j['overlay'],
            ('plate', ['plate:%s' % a for a in ('rx', 'ry', 'rz', 'x', 'y', 'z')], np.full(6, 1e-2)),
            ('tip', ['tip:%s' % a for a in 'xyz'], np.full(3, 1e-3)),
            ('compliance',) + j['compliance'], ('backlash',) + j['backlash']])
        self.divots = np.asarray(divots, float)

    def tips(self, x, samples: Sequence[PlateSample]) -> np.ndarray:
        self.set_geometry(x)
        t = x[self.sl['tip']]
        return np.array([self.frame_pose(self.link_angles(x, s.q, s.tau)).act(t) for s in samples])

    def plate(self, x, plate0: pin.SE3) -> pin.SE3:
        return plate0 * kinfit.se3(x[self.sl['plate']])

    def targets(self, x, samples, plate0) -> np.ndarray:
        P = self.plate(x, plate0)
        return np.array([P.act(self.divots[s.divot]) for s in samples])

    def residuals(self, x, samples, plate0):
        return (self.tips(x, samples) - self.targets(x, samples, plate0)).ravel()


def initial_plate(fit: PlateFit, x0, samples) -> pin.SE3:
    """Plate pose by Kabsch from the drawing's divots to the mean FK tip per divot."""
    tips = fit.tips(x0, samples)
    ks = sorted(set(s.divot for s in samples))
    mean = np.array([tips[[s.divot == k for s in samples]].mean(axis=0) for k in ks])
    return kinfit.kabsch(fit.divots[ks], mean)


def sim_plate_pose(ctx: StageContext, plate: Plate) -> pin.SE3:
    """Sim: where the plate really is (the hint plus a fixed few-mm, sub-degree setup error)."""
    sim = ctx.client
    T = getattr(sim, 'plate_in_base', None)
    if T is None:
        r = np.random.default_rng(77)
        T = plate.pose_hint * kinfit.se3(np.concatenate([r.normal(0, 0.01, 3), r.normal(0, 0.003, 3)]))
        sim.plate_in_base = T
    return T


def collect(ctx: StageContext, plate: Plate, divots: Sequence[int], per_divot: int,
            loaded: bool = False) -> List[PlateSample]:
    client = ctx.client
    out = []
    if loaded:
        mount_payload(ctx, Payload(link=ctx.model.link_names()[-1], mass_kg=LOAD_KG, offset_m=(0, 0, 0)))
    T = sim_plate_pose(ctx, plate) if ctx.sim else None
    if T is not None:
        truth = client.truth
        down = -T.rotation[:, 2]
        cos_cone = math.cos(plate.cone)

        def accept(q):
            return float(truth.fk(q).rotation[:, 2] @ down) >= cos_cone
    for k in divots:
        if T is not None:
            targets, _ = pivot.sim_divot_poses(ctx, per_divot, T.act(plate.divots[k]), accept=accept)
        else:
            targets = [None] * per_divot
        d_mm = np.round(plate.divots[k] * 1e3, 1).tolist()
        for j, q_true in enumerate(targets):
            s = client.hand_guide('plate%s: divot %d at %s mm on the plate, configuration %d/%d: seat the '
                                  'probe tip, vary the elbow and wrist from the last one, then confirm'
                                  % (' (mass hanging)' if loaded else '', k, d_mm, j + 1, len(targets)), q_true)
            out.append(PlateSample(np.array(s.q, float), np.array(s.tau, float), k, loaded))
    if loaded:
        mount_payload(ctx, None)
    return out


def fit_plate(fit: PlateFit, data, x0, exclude, tol):
    """Select, solve, re-linearize at the solution (the start can be degrees off),
    then drop zero offsets whose 1 sigma is over ZERO_SIGMA_MAX and refit."""
    keep, _ = kinfit.select_parameters(fit, fit.jacobian(x0, *data), PRIORITY, tol, exclude)
    x = kinfit.solve(fit, data, keep, x0, max_iter=80)
    J = fit.jacobian(x, *data)
    keep2, cond, share = kinfit.select_parameters(fit, J, PRIORITY, tol, exclude, details=True)
    if keep2 != keep:
        keep = keep2
        x = kinfit.solve(fit, data, keep, x, max_iter=80)
        J = fit.jacobian(x, *data)
        _, cond, share = kinfit.select_parameters(fit, J, PRIORITY, tol, exclude, details=True)
    fit_err = np.linalg.norm(fit.residuals(x, *data).reshape(-1, 3), axis=1) * 1e3
    sigma = kinfit.parameter_sigma(fit, J, keep, kinfit.rms(fit_err) * 1e-3)
    zeros = range(fit.size)[fit.sl['dzero']]
    weak = [i for i in keep if i in zeros and sigma[i] > ZERO_SIGMA_MAX]
    if weak:
        keep = [i for i in keep if i not in weak]
        x1 = x.copy()
        x1[weak] = x0[weak]
        x = kinfit.solve(fit, data, keep, x1, max_iter=80)
        J = fit.jacobian(x, *data)
        cond = kinfit.condition(fit, J, keep)
        fit_err = np.linalg.norm(fit.residuals(x, *data).reshape(-1, 3), axis=1) * 1e3
        sigma = kinfit.parameter_sigma(fit, J, keep, kinfit.rms(fit_err) * 1e-3)
    return x, keep, cond, share, sigma, fit_err, weak


def fold_tip(fit: PlateFit, x):
    """Fold the fitted probe-tip offset into the last joint, as far as the
    calibration can hold it: its rotation about the joint axis is that joint's
    zero (q + theta turns tool0 so), its part along the axis is the joint's
    origin (a shift along the axis commutes with the rotation). Both are exact
    at tool0 to first order. The radial part (the tip's distance from the axis)
    has no calibration field. Returns (x as stored, theta or None, radial part
    in tool0, m)."""
    tip = x[fit.sl['tip']]
    m = fit.model.model
    F = m.frames[fit.fid]
    parent = F.parentJoint if hasattr(F, 'parentJoint') else F.parent
    if parent != fit.jid[-1]:
        return x, None, tip.copy()
    a = fit.model.axis_in_child(fit.n - 1)
    t = F.placement.translation
    d = F.placement.rotation @ tip                     # tip offset in the joint frame
    w = np.cross(a, t)
    if w @ w < 1e-6:
        return x, None, tip.copy()
    theta = float(w @ d / (w @ w))
    along = float(a @ d)
    radial = d - theta * w - along * a
    fit.set_geometry(x)
    R = m.jointPlacements[fit.jid[-1]].rotation          # with the fitted axis tilt
    xs = x.copy()
    xs[fit.sl['tip']] = 0.0
    xs[fit.sl['dzero'].start + fit.n - 1] += theta
    ov = xs[fit.sl['overlay']].reshape(fit.n, 6).copy()
    ov[-1, :3] += R @ a * along
    xs[fit.sl['overlay']] = ov.ravel()
    return xs, theta, F.placement.rotation.T @ radial


def pair_errors(fit: PlateFit, x, samples, held_out) -> List[dict]:
    """Measured vs drawing distance for every divot pair with a held-out divot (mean tip per divot)."""
    tips = fit.tips(x, samples)
    ks = sorted(set(s.divot for s in samples))
    mean = {k: tips[[s.divot == k for s in samples]].mean(axis=0) for k in ks}
    out = []
    for a in ks:
        for b in ks:
            if b <= a or (a not in held_out and b not in held_out):
                continue
            meas = float(np.linalg.norm(mean[a] - mean[b]))
            ref = float(np.linalg.norm(fit.divots[a] - fit.divots[b]))
            out.append({'pair': [a, b], 'drawing_mm': ref * 1e3, 'error_mm': (meas - ref) * 1e3})
    return out


def run(ctx: StageContext, per_divot: int = None, loaded_per_divot: int = None,
        held_per_divot: int = None, plate: Optional[Plate] = None, tol: float = SELECT_TOL) -> StageResult:
    plate = plate or Plate.from_config(ctx.arm.raw)
    if plate is None:
        return StageResult('plate', False, {}, {}, ['no plate geometry: add plate.divots_m to the joint config'])
    lines, flags = [], []
    fit_divots, held = plate.fit_divots, [k for k in plate.held_out if k < len(plate.divots)]
    if len(fit_divots) < 3 or not held:
        return StageResult('plate', False, {}, {}, ['need >= 3 fitted divots and >= 1 held out (plate.held_out)'])
    if plate.needs_drawing:
        (lines if ctx.sim else flags).append(
            'plate geometry is a placeholder (plate.needs_drawing): use the drawing\'s divot coordinates')
    per_divot = per_divot or (5 if ctx.quick else 8)
    if loaded_per_divot is None:
        loaded_per_divot = 2 if ctx.quick else 3
    held_per_divot = held_per_divot or (4 if ctx.quick else 6)

    samples = collect(ctx, plate, fit_divots, per_divot)
    if loaded_per_divot:
        if not ctx.sim:
            flags.append('loaded pass on hardware: gravity comp does not carry the mass, so it adds '
                         'configurations; stiffness comes from the arm\'s own gravity torques')
        samples += collect(ctx, plate, fit_divots, loaded_per_divot, loaded=True)
    test = collect(ctx, plate, held, held_per_divot)
    loaded = any(s.loaded for s in samples)
    short = [k for k in fit_divots if sum(s.divot == k for s in samples) < 3]
    if short:
        flags.append('divots %s have fewer than 3 configurations (reach / approach cone)' % short)

    model, names = ctx.model, ctx.arm.names
    fit = PlateFit(model, plate.divots)
    x0 = np.zeros(fit.size)
    k_prior = ctx.calibration.get('joint_stiffness_nm_per_rad') or {}
    x0[fit.sl['compliance']] = [1.0 / k_prior[n] if k_prior.get(n) else 0.0 for n in names]
    plate0 = initial_plate(fit, x0, samples)
    exclude = (() if loaded else ('compliance',)) + (() if plate.fit_tip else ('tip',))
    data = (samples, plate0)
    x, keep, cond, share, sigma, fit_err, weak = fit_plate(fit, data, x0, exclude, tol)
    for i in weak:
        lines.append('  %s: 1 sigma over %.2f deg, left at the jig value and refitted without it'
                     % (fit.labels[i], ZERO_SIGMA_MAX / DEG))
    lines.append('%d samples on divots %s (%d loaded), %d of %d parameters identifiable, condition number %.0f '
                 '(pass < %.0e)' % (len(samples), fit_divots, sum(s.loaded for s in samples), len(keep), fit.size,
                                    cond, COND_MAX))
    if cond > COND_WARN:
        flags.append('condition number %.0f: add configurations per divot before trusting the fit' % cond)

    # what is stored: the tip offset folded into J7, its radial part dropped
    xs, j7_theta, radial = fold_tip(fit, x) if plate.fit_tip else (x, None, np.zeros(3))
    tip = x[fit.sl['tip']]
    tip_sigma = float(np.sqrt(np.nansum(sigma[fit.sl['tip']] ** 2)))
    if j7_theta is not None:
        lines.append('probe tip offset in tool0 %s mm (1 sigma %.3f): %+.4f deg about J7 -> its zero, the part '
                     'along J7 -> its origin, %.3f mm radial (not storable)'
                     % (np.round(tip * 1e3, 3).tolist(), tip_sigma * 1e3, j7_theta / DEG,
                        np.linalg.norm(radial) * 1e3))
    if np.linalg.norm(radial) > max(TIP_FLAG_MM * 1e-3, 3 * tip_sigma):
        flags.append('probe tip %.2f mm off the URDF radially to J7: fix tool0_joint and rerun'
                     % (np.linalg.norm(radial) * 1e3))

    # held-out divots, with the parameters as stored
    held_err = np.linalg.norm(fit.tips(xs, test) - fit.targets(xs, test, plate0), axis=1) * 1e3
    pairs = pair_errors(fit, xs, samples + test, held)
    pair_rms = kinfit.rms([p['error_mm'] for p in pairs])
    held_rms = kinfit.rms(held_err)
    lines.append('fit RMS %.3f mm; held-out divots %s: tip RMS %.3f mm, pair distance RMS %.3f mm over %d pairs '
                 '(pass < %.1f mm; plate tolerance %.3f mm)'
                 % (kinfit.rms(fit_err), held, held_rms, pair_rms, len(pairs), HELD_OUT_MM, plate.tolerance_mm))
    if pair_rms < 2 * plate.tolerance_mm:
        lines.append('  pair distances agree to within 2x the plate tolerance: the plate limits this check')

    P = fit.plate(x, plate0)
    hint_mm = np.linalg.norm(P.translation - plate.pose_hint.translation) * 1e3
    lines.append('plate origin at %s mm in base_link (%.1f mm from the configured hint)'
                 % (np.round(P.translation * 1e3, 1).tolist(), hint_mm))

    observed = set(fit.labels[i] for i in keep)
    dzero = xs[fit.sl['dzero']]
    old_zero = ctx.calibration.get('zero_offsets_rad') or {}
    new_zero = {}
    observability = {fit.labels[i]: float(share[i]) for i in range(fit.size)}
    for i, n in enumerate(names):
        lab = 'dzero:' + n
        if i == len(names) - 1 and j7_theta is not None and lab not in observed:
            new_zero[n] = float((old_zero.get(n) or 0.0) - dzero[i])
            lines.append('  %-15s zero correction %+.4f deg from the probe tip (the J7 zero and the probe angle '
                         'about it are one parameter: exact at tool0, not a physical J7 zero)' % (n, dzero[i] / DEG))
        elif lab in observed:
            new_zero[n] = float((old_zero.get(n) or 0.0) - dzero[i])
            lines.append('  %-15s zero correction %+.4f deg (1 sigma %.4f, observability %.0f%%)'
                         % (n, dzero[i] / DEG, sigma[fit.labels.index(lab)] / DEG, 100 * share[fit.labels.index(lab)]))
        else:
            j = fit.labels.index(lab)
            why = ('1 sigma over %.2f deg' % (ZERO_SIGMA_MAX / DEG) if j in weak else
                   'folds into the plate pose / probe tip')
            lines.append('  %-15s zero not observable (observability %.1f%%: %s), keeps the jig value'
                         % (n, 100 * share[j], why))
    ov_kept = [lab for lab in observed if lab.split(':')[0] in kinfit.OVERLAY_KEYS]
    ov_share = [share[fit.labels.index(lab)] for lab in ov_kept]
    lines.append('  kinematic overlay: %d of %d terms identifiable%s' % (
        len(ov_kept), 6 * len(names), ', weakest observability %.1f%%' % (100 * min(ov_share)) if ov_share else ''))
    c = xs[fit.sl['compliance']]
    stiffness = {n: float(1.0 / c[i]) for i, n in enumerate(names)
                 if loaded and 'compliance:' + n in observed and c[i] > 1e-7}
    if stiffness:
        lines.append('  stiffness Nm/rad: ' + ', '.join('%s %.0f' % kv for kv in stiffness.items()))
    elif not loaded:
        lines.append('  no loaded pass: joint stiffness kept from the calibration')

    passed = bool(held_rms < HELD_OUT_MM and pair_rms < HELD_OUT_MM and cond < COND_MAX)
    overlay = kinfit.compose_overlay(ctx.calibration.get('kinematic_overlay'), names,
                                     xs[fit.sl['overlay']].reshape(len(names), 6))
    writes = {'zero_offsets_rad': new_zero, 'kinematic_overlay': overlay}
    if stiffness:
        writes['joint_stiffness_nm_per_rad'] = stiffness
    metrics = {
        'held_out_rms_mm': held_rms, 'pair_distance_rms_mm': pair_rms, 'pairs': pairs,
        'fit_rms_mm': kinfit.rms(fit_err), 'condition_number': cond, 'threshold_mm': HELD_OUT_MM,
        'parameters': [fit.labels[i] for i in keep], 'observability': observability,
        'sigma': {fit.labels[i]: float(sigma[i]) for i in keep},
        'samples': len(samples), 'held_out_samples': len(test), 'held_out_divots': held,
        'plate_in_base': {'xyz': P.translation.tolist(), 'rpy': pin.rpy.matrixToRpy(P.rotation).tolist()},
        'tip_offset_m': tip.tolist(), 'tip_offset_sigma_m': tip_sigma, 'tip_radial_m': radial.tolist(),
        'j7_zero_from_tip_rad': j7_theta,
        'plate_tolerance_mm': plate.tolerance_mm,
        'zero_corrections_rad': {n: float(dzero[i]) for i, n in enumerate(names) if 'dzero:' + n in observed},
        'backlash_rad': {n: float(v) for n, v in zip(names, xs[fit.sl['backlash']])}}
    return StageResult('plate', passed, metrics, writes, lines, flags)
