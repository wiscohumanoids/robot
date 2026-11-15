"""`arm indicator`: Phase 1 single-joint bring-up check (after `arm bringup`).

Per joint:

  1. Angle: command 0 deg, then 90 deg (Mode P), and read the joint's real angle
     at both with an external reference: a protractor / digital inclinometer
     on the lever, or a dial indicator on a known radius. The operator types
     the readings on hardware; the sim reads its true link angle. Pass: the
     measured 90 deg step is within 0.5 deg of the commanded one. This catches
     a wrong counts_per_rev (motor- vs output-side encoder, gear ratio) and a
     reversed direction, which the drive's own readings cannot see.
     A joint whose range cannot fit 0 -> 90 deg uses the largest step that fits.
  2. Torque sign: hold the joint where gravity loads it most (the lever
     horizontal on the rig), sweeping up and down through the pose so friction
     cancels, and compare the measured 0x6077 torque with the model gravity.
     Same sign = torque sign confirmed; the ratio checks the 0x6077 scale
     (per-mille of rated current) to within a factor of 2.
  3. Units: reports how 0x6077 is converted (per-mille of rated current x Kt)
     and where each number comes from.
  4. Nulls: lists every null left in the joint's row of the hardware table
     (joint config). Phase 1 gate: none left. On hardware a null fails the
     stage; the sim only lists them.
"""

from __future__ import annotations

import numpy as np

from ..control.motion import move_to
from ..units import Units
from .common import DEG, StageContext, StageResult, safe_home
from .gravity import measure_pose

TOL = 0.5 * DEG
STEP = 90 * DEG
MARGIN = 5 * DEG
SCALE_RANGE = (0.5, 2.0)     # measured / model gravity torque
MIN_COUNTS = 5               # gravity torque needed for a sign test, in 0x6077 counts


def test_angles(model, j):
    """(start, end) commanded angles for joint j: 0 -> +-90 deg, or the largest
    step that fits inside the limits."""
    lo, hi = model.lower[j] + MARGIN, model.upper[j] - MARGIN
    a = float(np.clip(0.0, lo, hi))
    if a + STEP <= hi:
        return a, a + STEP
    if a - STEP >= lo:
        return a, a - STEP
    span = min(STEP, hi - lo)
    mid = 0.5 * (lo + hi)
    return mid - span / 2, mid + span / 2


def read_angle(ctx: StageContext, j: int, label: str) -> float:
    """The joint's real angle (rad): sim truth, or the operator's reading."""
    truth = getattr(ctx.client, 'true_joint_angles', None)
    if ctx.sim and truth is not None:
        return float(truth()[j])
    while True:
        text = input('%s: %s. Read the protractor / indicator on the lever, in degrees: '
                     % (ctx.arm.names[j], label))
        try:
            return float(text) * DEG
        except ValueError:
            print('  not a number: %r' % text)


def row_nulls(arm):
    """{joint: [keys]} of every None left in each joint row (nested dicts too)."""
    out = {}
    for row in arm.raw.get('joints') or []:
        keys = []
        for k, v in row.items():
            if v is None:
                keys.append(k)
            elif isinstance(v, dict):
                keys += ['%s.%s' % (k, kk) for kk, vv in v.items() if vv is None]
        out[row['name']] = keys
    return out


def sign_pose(model, j, home, samples=25):
    """Pose (joint j swept, others at home) where j carries the most gravity."""
    best, best_g = home.copy(), abs(model.gravity(home)[j])
    lo, hi = model.lower[j] + MARGIN + 3 * DEG, model.upper[j] - MARGIN - 3 * DEG
    for v in np.linspace(lo, hi, samples):
        q = home.copy()
        q[j] = v
        g = abs(model.gravity(q)[j])
        if g > best_g:
            best, best_g = q, g
    return best


def run(ctx: StageContext) -> StageResult:
    client, model, arm = ctx.client, ctx.model, ctx.arm
    lines, flags, metrics = [], [], {}
    units = Units(arm, ctx.calibration)
    count_nm = np.abs(units.nm_per_permille())
    peak = np.array([j.peak_torque_nm for j in arm.joints])
    home = safe_home(model)
    nulls = row_nulls(arm)
    ok = True
    move_to(client, home, max_speed=0.3)
    for j, spec in enumerate(arm.joints):
        name = spec.name
        m = {}
        # ---- 1. commanded step vs external reading
        a, b = test_angles(model, j)
        qa, qb = home.copy(), home.copy()
        qa[j], qb[j] = a, b
        sa = move_to(client, qa, max_speed=0.3, settle=1.0)
        ra = read_angle(ctx, j, 'at the %.1f deg command' % (a / DEG))
        sb = move_to(client, qb, max_speed=0.3, settle=1.0)
        rb = read_angle(ctx, j, 'at the %.1f deg command' % (b / DEG))
        cmd_step, read_step, enc_step = b - a, rb - ra, sb.q[j] - sa.q[j]
        err = read_step - cmd_step
        angle_ok = abs(err) < TOL
        ratio = read_step / cmd_step
        m.update({'commanded_deg': [a / DEG, b / DEG], 'read_deg': [ra / DEG, rb / DEG],
                  'encoder_step_deg': enc_step / DEG, 'error_deg': err / DEG, 'angle_scale': ratio})
        lines.append('%-15s commanded %+.1f -> %+.1f deg; indicator step %+.2f deg, encoder step %+.2f deg: '
                     'error %+.3f deg (pass < %.1f)%s' % (name, a / DEG, b / DEG, read_step / DEG,
                                                          enc_step / DEG, err / DEG, TOL / DEG,
                                                          '' if angle_ok else '  <-- FAIL'))
        if abs(cmd_step) < STEP - 1e-6:
            flags.append('%s: range only allows a %.0f deg step' % (name, abs(cmd_step) / DEG))
        if not angle_ok:
            if ratio < 0:
                flags.append('%s: the joint moved the other way (indicator/command %.3f): direction '
                             'is reversed' % (name, ratio))
            else:
                flags.append('%s: indicator/command = %.4f: counts_per_rev is off by that factor (%d now; '
                             'motor- vs output-side encoder, gear ratio?)' % (name, ratio, spec.counts_per_rev))

        # ---- 2. torque sign and 0x6077 scale against gravity
        q_s = sign_pose(model, j, home)
        g_model = float(model.gravity(q_s)[j])
        clamp = np.minimum(peak, np.abs(model.gravity(q_s)) * 1.5 + 0.25 * peak)
        tau, q_meas = measure_pose(ctx, q_s, clamp)
        tau_j = float(tau[j])
        permille = tau_j / units.nm_per_permille()[j]       # back to 0x6077 counts
        amps = permille * spec.rated_current_a / 1000.0
        testable = abs(g_model) > max(MIN_COUNTS * count_nm[j], 0.01 * peak[j])
        scale = tau_j / g_model if g_model else float('nan')
        if testable:
            sign_ok = bool(np.sign(tau_j) == np.sign(g_model))
            scale_ok = bool(sign_ok and SCALE_RANGE[0] < scale < SCALE_RANGE[1])
        else:
            sign_ok = scale_ok = None
            flags.append('%s: gravity torque %.3f Nm too small to check the torque sign; hang a test '
                         'mass and confirm by hand' % (name, g_model))
        m.update({'sign_pose_deg': float(q_s[j] / DEG), 'gravity_model_nm': g_model, 'measured_nm': tau_j,
                  'measured_permille': float(permille), 'measured_a': float(amps), 'torque_scale': scale,
                  'sign_ok': sign_ok, 'scale_ok': scale_ok})
        verdict = {True: 'sign confirmed', False: 'REVERSED', None: 'not testable'}[sign_ok]
        lines.append('%-15s at %+.1f deg: model gravity %+.3f Nm, 0x6077 %+.1f counts = %+.3f A = %+.3f Nm '
                     '(x%.2f) -> %s%s' % (name, q_s[j] / DEG, g_model, permille, amps, tau_j, scale, verdict,
                                          '' if scale_ok in (True, None) or sign_ok is False
                                          else ', 0x6077 SCALE OFF'))
        if sign_ok is False:
            flags.append('%s: measured torque opposes gravity: torque sign (direction or 0x6077 sign) is '
                         'reversed' % name)
        elif scale_ok is False:
            flags.append('%s: 0x6077 torque is x%.2f the model gravity: check its units (per-mille of rated '
                          'current), rated_current_a and Kt' % (name, scale))

        # ---- 3. units in use
        rated_src = 'model table (null in joint row)' if 'rated_current_a' in spec.nulls else 'joint row'
        lines.append('%-15s 0x6077 = per-mille of rated current: 1 count = %.3f A (rated %.2f A, %s) x Kt '
                     '%.4f Nm/A (%s) = %.5f Nm; counts_per_rev %d%s' % (
                         name, spec.rated_current_a / 1000.0, spec.rated_current_a, rated_src, units.kt[j],
                         units.kt_source[j], count_nm[j], spec.counts_per_rev,
                         ' (model table)' if 'counts_per_rev' in spec.nulls else ''))
        m.update({'nm_per_count': float(count_nm[j]), 'rated_current_a': spec.rated_current_a,
                  'kt_nm_per_a': float(units.kt[j]), 'kt_source': units.kt_source[j],
                  'counts_per_rev': spec.counts_per_rev})

        # ---- 4. nulls left in the hardware table
        left = nulls.get(name, [])
        m['nulls'] = left
        lines.append('%-15s hardware table nulls: %s' % (name, ', '.join(left) if left else 'none'))
        if left:
            flags.append('%s: %d null(s) left in %s (%s); Phase 1 gate needs them resolved'
                         % (name, len(left), arm.path, ', '.join(left)))
        joint_ok = angle_ok and sign_ok is not False and scale_ok is not False
        ok &= joint_ok
        metrics[name] = m
        move_to(client, home, max_speed=0.3)

    any_nulls = any(nulls.get(n) for n in arm.names)
    gate = bool(ok and not any_nulls)
    lines.append('Phase 1 gate (90 deg on the indicator, torque sign, 0x6077 units, no nulls): %s'
                 % ('MET' if gate else 'NOT MET'))
    passed = bool(ok and (ctx.sim or not any_nulls))
    metrics['phase1_gate'] = gate
    return StageResult('indicator', passed, metrics, {}, lines, flags)
