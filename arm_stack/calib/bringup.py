"""Stage 1, `arm bringup`: slaves in OP, DC locked, directions match the URDF.

Direction cannot be seen from the drive alone: a reversed actuator flips its
torque and its encoder together, so the host's own readings stay consistent.
Gravity is the external reference: hold a pose where the joint carries
gravity torque and compare the measured holding torque with the model. A
reversed joint holds with the wrong sign. Each jog runs at a 10% torque clamp.
On hardware the operator also confirms each jog by eye.
"""

from __future__ import annotations

import numpy as np

from ..control.motion import hold_and_average, move_to
from .common import DEG, StageContext, StageResult, safe_home

TORQUE_CLAMP_FRACTION = 0.10


def gravity_test_pose(model, joint, home, rng, tries=400):
    """A pose near home where `joint` carries clear gravity torque."""
    best, best_g = home.copy(), abs(model.gravity(home)[joint])
    for _ in range(tries):
        q = home.copy()
        idx = rng.choice(model.n, size=min(3, model.n), replace=False)
        for i in set(idx) | {joint}:
            span = model.upper[i] - model.lower[i]
            q[i] = np.clip(home[i] + rng.uniform(-0.25, 0.25) * span, model.lower[i] + 5 * DEG,
                           model.upper[i] - 5 * DEG)
        g = abs(model.gravity(q)[joint])
        if g > best_g:
            best, best_g = q, g
    return best, best_g


def run(ctx: StageContext, soak_s: float = None) -> StageResult:
    client, model, arm = ctx.client, ctx.model, ctx.arm
    lines, flags = [], []
    s = client.state()
    op = bool(np.all((s.status & 0x006F) == 0x0027))
    lines.append('all %d slaves in Operation enabled: %s' % (arm.n, 'yes' if op else 'NO'))

    peak = np.array([j.peak_torque_nm for j in arm.joints])
    clamp = TORQUE_CLAMP_FRACTION * peak
    home = safe_home(model)
    move_to(client, home, max_speed=0.3, torque_clamp=np.maximum(clamp, np.abs(model.gravity(home)) * 1.5))

    mismatched = {}
    details = {}
    for j, name in enumerate(arm.names):
        q_test, g_mag = gravity_test_pose(model, j, home, ctx.rng)
        g_model = model.gravity(q_test)
        tclamp = np.maximum(clamp, np.abs(g_model) * 1.5 + 0.1 * peak)
        move_to(client, q_test, max_speed=0.2, torque_clamp=tclamp)
        avg = hold_and_average(client, 0.5, q_test, torque_clamp=tclamp, gravity_ff=False)
        measured = avg['tau'][j]
        # a small jog: does the joint follow at the clamp?
        jog = q_test.copy()
        jog[j] += 3 * DEG if q_test[j] + 3 * DEG < model.upper[j] else -3 * DEG
        after = move_to(client, jog, max_speed=0.2, torque_clamp=tclamp)
        followed = abs(after.q[j] - jog[j]) < 1.0 * DEG
        agree = np.sign(measured) == np.sign(g_model[j]) if g_mag > 0.05 * peak[j] else None
        details[name] = {'gravity_model_nm': float(g_model[j]), 'holding_nm': float(measured),
                         'followed_jog': bool(followed)}
        verdict = {True: 'matches', False: 'REVERSED', None: 'not testable by gravity'}[agree]
        lines.append('%-15s model g %+6.2f Nm, held with %+6.2f Nm -> %s%s' % (
            name, g_model[j], measured, verdict, '' if followed else ', did NOT follow the jog'))
        if agree is False:
            mismatched[name] = -arm.joints[j].direction
        if agree is None:
            flags.append('%s: confirm direction by eye (gravity torque too small to test)' % name)
        move_to(client, home, max_speed=0.3, torque_clamp=np.maximum(clamp, np.abs(model.gravity(home)) * 1.5))

    soak = soak_s if soak_s is not None else (5.0 if ctx.sim else 600.0)
    s0 = client.state()
    end = client.hold(soak)
    faults = int(np.sum((end.status & 0x0008) != 0))
    lines.append('held %.0f s at %.0f Hz: %d faults, watchdog fallback %s' % (
        soak, 1.0 / client.dt, faults, 'fired' if end.fallback else 'not fired'))
    passed = op and not mismatched and faults == 0 and all(d['followed_jog'] for d in details.values())
    if mismatched:
        lines.append('reversed joints: %s; rerun with --write to fix their direction in %s'
                     % (', '.join(mismatched), arm.path))
    return StageResult('bringup', passed, {'joints': details, 'faults': faults, 'soak_s': soak,
                                           'start_t': s0.t}, {}, lines, flags,
                       joint_config_updates={'direction': mismatched})
