"""`arm float` and `arm gravity-check` in sim (control/float_mode.py, calib/gravity_check.py).

The sim arm has friction with stiction, a 2-cycle command delay, current
noise and joint compliance; the calibration's friction model matches it
unless a test scales it. An operator is an external force at tool0
(SimArm.apply_external_force)."""

import math
import shutil

import numpy as np
import pytest

from arm_stack.calib import gravity_check as gc
from arm_stack.calib.common import StageContext
from arm_stack.cli import build_model, update_joint_config
from arm_stack.config import empty_calibration, load_arm
from arm_stack.control.float_mode import (FloatConfig, configured_friction_gains, coulomb_ramp, parse_lock,
                                          parse_payload, push_profile, run_float)
from arm_stack.control.law import payload_gravity
from arm_stack.hw.client import Command, Payload
from arm_stack.sim.arm_sim import ARMATURE, SimArm, SimErrors

DEG = math.pi / 180
Q_START = np.array([0.3, 0.6, 0.2, 0.9, 0.1, 0.1, 0.2])
COULOMB = np.array([0.40, 0.40, 0.30, 0.25, 0.10, 0.08, 0.08])
VISCOUS = np.array([0.10, 0.10, 0.05, 0.05, 0.02, 0.01, 0.01])
PAYLOAD = (0.5, 0.0, 0.0, 0.03)


@pytest.fixture(scope='module')
def arm():
    return load_arm()


def make(arm, coulomb=COULOMB, model_scale=None, com_offsets=None, stiction=0.02):
    """Sim arm at Q_START held in Mode P; calibration friction = truth x model_scale."""
    n = arm.n
    err = SimErrors(coulomb_nm=coulomb, viscous_nm_s=VISCOUS, stiction_extra_nm=np.full(n, stiction),
                    delay_cycles=2, backlash_rad=np.full(n, 0.05 * DEG), com_offsets_m=com_offsets,
                    stiffness_nm_per_rad=np.array([4000, 4000, 1500, 1500, 1500, 600, 600], float))
    cal = empty_calibration(arm)
    scale = np.ones(n) if model_scale is None else np.asarray(model_scale, float)
    cal['friction'] = {nm: {'coulomb_nm': float(c * s), 'viscous_nm_s_per_rad': float(v)}
                       for nm, c, v, s in zip(arm.names, coulomb, VISCOUS, scale)}
    cal['pace']['armature'] = {j.name: ARMATURE[j.model] for j in arm.joints}
    sim = SimArm(arm, cal, err, seed=4)
    sim.set_true_state(Q_START)
    sim.hold(0.3)
    return sim, cal, build_model(arm, cal)


def window(res, t0, t1):
    t = res.trace['t']
    return (t >= t0) & (t <= t1)


# --------------------------------------------------------------------------- pieces
def test_parsers_and_ramp(arm):
    assert parse_payload('1.5@0.01,0,-0.02') == (1.5, 0.01, 0.0, -0.02)
    assert parse_payload('0.3') == (0.3, 0.0, 0.0, 0.0)
    assert parse_payload(None) is None
    with pytest.raises(ValueError):
        parse_payload('1@0,0')
    assert parse_lock('shoulder_pitch,3', arm.names).tolist() == [True, False, False, True, False, False, False]
    assert parse_lock('0000011', arm.names).tolist() == [False] * 5 + [True, True]
    with pytest.raises(ValueError):
        parse_lock('nope', arm.names)
    r = coulomb_ramp(np.array([0.0, 0.015, -0.015, 0.035, 0.06, -0.2]))
    assert r[:3].tolist() == [0.0, 0.0, 0.0]          # deadband
    assert 0.4 < r[3] < 0.6                           # smooth ramp
    assert r[4:].tolist() == [1.0, -1.0]


def test_float_config_is_safe_by_construction(arm):
    _, _, model = make(arm)
    cfg = FloatConfig.from_arm(arm, model, friction_gain=1.7)
    assert np.all(cfg.friction_gain < 1.0)                       # never all of the friction
    assert cfg.torque_clamp.tolist() == [j.max_torque_nm for j in arm.joints] == [60, 60, 34, 34, 10, 10, 10]   # model peaks
    assert np.allclose(cfg.vel_limit_fraction * cfg.max_vel, 1.0)    # 0.5 x bring-up 2 rad/s


def test_payload_gravity_matches_inertia_on_the_truth_model(arm):
    """Python's J^T m g (the law) == Pinocchio with the mass in link7's inertia (the
    C++ GravityModel / the sim truth)."""
    sim, _, model = make(arm)
    q = np.array([0.5, 1.0, -0.3, 1.2, 0.4, 0.3, -0.6])
    g0 = sim.truth.gravity(q)
    sim.set_payload(Payload('tool0', PAYLOAD[0], PAYLOAD[1:]))
    assert sim.truth.gravity(q) - g0 == pytest.approx(payload_gravity(model, q, PAYLOAD), abs=1e-9)


def test_law_payload_is_sticky_and_used_by_the_fallback(arm):
    """Same semantics as ArmMock.PayloadInGravityAndWatchdogFallback (C++)."""
    sim, _, model = make(arm)
    law = sim.law
    q, qd = Q_START, np.full(arm.n, 0.1)
    g, gp = model.gravity(q), model.gravity(q) + payload_gravity(model, q, PAYLOAD)
    out = law.compute(Command(mode='P', q=q, payload=PAYLOAD), 0.0, q, qd, 'P')
    assert out.tau == pytest.approx(gp)
    out = law.compute(Command(mode='P', q=q), 0.0, q, qd, 'P')          # no block: kept
    assert out.tau == pytest.approx(gp)
    out = law.compute(None, 1e9, q, qd, 'P')                            # watchdog fallback
    assert out.fallback and out.tau == pytest.approx(gp - law.kd_fallback * qd)
    out = law.compute(Command(mode='P', q=q, payload=(0.0, 0.0, 0.0, 0.0)), 0.0, q, qd, 'P')
    assert out.tau == pytest.approx(g) and law.payload is None          # kg 0 clears


def test_wire_payload_block():
    from arm_stack.hw.ros_client import encode_command
    n = 7
    assert len(encode_command(Command(mode='T'), n)) == 3 + 6 * n          # old layout unchanged
    d = encode_command(Command(mode='T', payload=PAYLOAD), n)
    assert len(d) == 3 + 6 * n + 4 and d[-4:] == list(PAYLOAD)


# --------------------------------------------------------------------------- float
def test_no_drift_without_push(arm):
    sim, _, model = make(arm)
    res = run_float(sim, FloatConfig.from_arm(arm, model), duration=3.0)
    assert np.max(res.travel) < 0.2 * DEG, np.degrees(res.travel)
    assert not res.damping_triggered
    assert res.final.mode == 'P' and not res.final.fallback           # exit: Mode P hold
    assert np.max(np.abs(res.q_end - res.q_start)) < 0.2 * DEG


def test_follows_push_and_stops_when_released(arm):
    sim, _, model = make(arm)
    p0 = sim.true_tool_position()
    positions = {}

    def operator(client, t, s):
        push(client, t, s)
        if abs(t - 1.3) < 5e-4:
            positions['release'] = client.true_tool_position()
    push = push_profile([0.0, 6.0, 0.0], 0.3, 1.3)
    res = run_float(sim, FloatConfig.from_arm(arm, model), duration=3.0, operator=operator)
    moved = np.linalg.norm(positions['release'] - p0)
    assert moved > 0.02, moved                                         # it follows the hand
    after = window(res, 2.0, 3.0)
    assert np.max(np.abs(res.trace['qd'][after])) < 0.01               # and stops when let go
    drift = np.ptp(res.trace['q'][after], axis=0)
    assert np.max(drift) < 0.1 * DEG, np.degrees(drift)
    assert res.final.mode == 'P'
    # friction help was on while moving, never at rest
    print('damping triggered:', res.damping_triggered, 'peak speed fraction', res.max_speed_fraction)
    moving = (np.abs(res.trace['qd']) > 0.06) & ~res.trace['damping'][:, None]
    assert moving.sum() > 100
    assert np.mean(np.abs(res.trace['tau_ff'][moving]) > 0) > 0.95
    assert np.all(res.trace['tau_ff'][after] == 0)


def test_velocity_limit_switches_to_damping(arm):
    elbow = arm.names.index('elbow')

    def shove(client, t, s):                    # the operator throws the forearm: 2.5 Nm for 0.7 s
        tau = np.zeros(arm.n)
        if 0.2 <= t <= 0.9:
            tau[elbow] = 2.5
        client.apply_external_torque(tau)

    def fast(vel_limit_fraction):
        sim, _, model = make(arm)
        cfg = FloatConfig.from_arm(arm, model, vel_limit_fraction=vel_limit_fraction)
        return run_float(sim, cfg, duration=2.0, operator=shove)
    res = fast(0.5)
    off = fast(100.0)                                                  # same shove, no velocity limit
    assert res.damping_triggered and not off.damping_triggered
    on = [e for e in res.events if e['event'] == 'float_damping_on'][0]
    assert on['joint'] == elbow and abs(on['qd']) > on['limit'] == pytest.approx(1.0)
    k = np.flatnonzero(res.trace['damping'])
    assert np.all(res.trace['tau_ff'][k] == 0)                         # no friction help while damping
    assert res.max_speed_fraction[elbow] < 0.75                        # held near the 1 rad/s limit
    assert res.max_speed_fraction[elbow] < 0.6 * off.max_speed_fraction[elbow]
    assert any(e['event'] == 'float_damping_off' for e in res.events)  # floats again once slow
    assert not res.trace['damping'][-1]


def test_torque_never_exceeds_bring_up_clamp(arm):
    sim, _, model = make(arm)
    cfg = FloatConfig.from_arm(arm, model)
    # a hard shove that saturates the wrist, into the walls and the safety damping
    res = run_float(sim, cfg, duration=2.0, operator=push_profile([0.0, 300.0, 100.0], 0.2, 0.8))
    tau = np.abs(res.trace['tau_cmd'])
    assert np.all(tau <= cfg.torque_clamp + 1e-9)
    assert np.max(res.max_torque_fraction) == pytest.approx(1.0)        # and the clamp was reached


def test_watchdog_falls_back_to_gravity_comp_with_payload(arm):
    sim, _, model = make(arm)
    sim.set_payload(Payload('tool0', PAYLOAD[0], PAYLOAD[1:]))
    run_float(sim, FloatConfig.from_arm(arm, model, payload=PAYLOAD), duration=0.5, hold_s=0.0)
    # the float loop dies mid-session (no exit hold): last command is Mode T
    sim.send(Command(mode='T', q=sim.state().q, kp=np.zeros(arm.n), kd=np.zeros(arm.n), friction_ff_gain=0.0))
    q0 = sim.state().q.copy()
    s = sim.step(0.04)
    assert not s.fallback
    s = sim.step(0.03)
    assert s.fallback and s.mode == 'T'                                # 50 ms watchdog
    s = sim.step(1.5)
    assert np.max(np.abs(s.q - q0)) < 0.2 * DEG                        # payload still compensated
    assert sim.law.payload == PAYLOAD


def test_locked_joints_hold(arm):
    sim, _, model = make(arm)
    lock = parse_lock('shoulder_pitch,shoulder_roll', arm.names)
    cfg = FloatConfig.from_arm(arm, model, lock=lock)
    res = run_float(sim, cfg, duration=2.5, operator=push_profile([0.0, 5.0, 5.0], 0.3, 1.3))
    assert np.max(res.travel[lock]) < 1.5 * DEG, np.degrees(res.travel)       # peak, while pushed
    assert np.max(np.abs(res.q_end - res.q_start)[lock]) < 0.1 * DEG          # integrator: back on the pose
    assert np.max(res.travel[~lock]) > 3 * DEG                                # the rest floated


def test_payload_is_compensated(arm):
    sim, _, model = make(arm)
    sim.set_payload(Payload('tool0', PAYLOAD[0], PAYLOAD[1:]))
    with_payload = run_float(sim, FloatConfig.from_arm(arm, model, payload=PAYLOAD), duration=2.0)
    assert np.max(with_payload.travel) < 0.2 * DEG
    sim.set_true_state(Q_START)
    sim.hold(0.3)
    without = run_float(sim, FloatConfig.from_arm(arm, model, payload=(0.0, 0.0, 0.0, 0.0)), duration=1.0)
    assert np.max(without.travel) > 2 * DEG                            # the same mass, uncompensated, falls


# --------------------------------------------------------------------------- gravity-check
def test_gravity_check_flags_an_injected_com_error(arm):
    com = np.zeros((arm.n, 3))
    com[3] = [0.05, 0.0, 0.04]                      # elbow link, off its axis (~0.3 Nm)
    sim, cal, model = make(arm, coulomb=COULOMB * 0.3, com_offsets=com, stiction=0.005)
    ctx = StageContext(arm=arm, client=sim, calibration=cal, model=model, sim=True, quick=True)
    r = gc.run(ctx, n_poses=3, bisect=False)
    print(r.report())
    elbow = arm.names.index('elbow')
    d = r.metrics['drift']
    assert not r.passed
    assert d['flagged'][elbow] and any('elbow' in f for f in r.flags)
    # the estimate matches the injected error where the elbow drifted
    checked = 0
    for p in d['per_pose']:
        if not p['moved'][elbow]:
            continue
        q = np.array(p['q'])
        true = sim.truth.gravity(q) - model.gravity(q)
        assert p['dtau'][elbow] == pytest.approx(true[elbow], rel=0.3)
        checked += 1
    assert checked >= 1
    # the wrist (no error, Fc above 5% of its tiny gravity) is not flagged
    assert not any(d['flagged'][5:])


def test_gravity_check_with_payload(arm):
    """--payload puts the mass in the RT gravity: clean with it, flagged without it."""
    def check(payload):
        sim, cal, model = make(arm, coulomb=COULOMB * 0.5)
        sim.set_payload(Payload('tool0', 0.4, (0.0, 0.0, 0.03)))
        ctx = StageContext(arm=arm, client=sim, calibration=cal, model=model, sim=True, quick=True)
        return gc.run(ctx, payload=payload, n_poses=2, drift_s=1.5, bisect=False, mount_sim_payload=False)
    good = check((0.4, 0.0, 0.0, 0.03))
    assert good.passed and not any(good.metrics['drift']['flagged']), good.report()
    bad = check(None)                                  # no --payload: none in the RT gravity
    assert not bad.passed and any(bad.metrics['drift']['flagged']), bad.report()


def test_gravity_check_bisects_friction_gain_and_writes_joint_config(arm, tmp_path):
    scale = np.ones(arm.n)
    elbow = arm.names.index('elbow')
    scale[elbow] = 1.25                              # model Coulomb 25% high: creeps above gain 0.8
    sim, cal, model = make(arm, model_scale=scale, stiction=0.0)
    ctx = StageContext(arm=arm, client=sim, calibration=cal, model=model, sim=True, quick=True)
    r = gc.run(ctx, n_poses=1, drift_s=1.0, joints=[elbow], bisect_iters=4)
    print(r.report())
    fg = r.metrics['friction_gain']['elbow']
    assert 0.65 <= fg['max_no_creep'] <= 0.85, fg
    assert any(ok for _, ok in fg['trials']) and not all(ok for _, ok in fg['trials'])
    rec = r.joint_config_updates['float_friction_gain']['elbow']
    assert rec == pytest.approx(0.9 * fg['max_no_creep'], abs=0.01)
    # written with the bringup mechanism, and float picks it up
    path = str(tmp_path / 'arm01.yaml')
    shutil.copy(arm.path, path)
    done = update_joint_config(path, r.joint_config_updates)
    assert done == ['elbow float_friction_gain -> %.2f' % rec]
    update_joint_config(path, {'float_friction_gain': {'elbow': 0.42}})     # second time: replaces
    raw = open(path).read()
    assert raw.count('float_friction_gain') == 1 and '# J4' in raw          # comments kept
    a2 = load_arm(path.replace('\\', '/'))
    assert configured_friction_gains(a2)[elbow] == pytest.approx(0.42)
    assert configured_friction_gains(a2)[0] == pytest.approx(0.6)          # unset: default


# --------------------------------------------------------------------------- CLI
def test_cli_float_sim_records_a_teach_session(tmp_path):
    pytest.importorskip('mcap')
    from arm_stack.cli import main
    from arm_stack.log.recorder import load_session
    path = str(tmp_path / 'teach.mcap')
    assert main(['float', '--sim', '--fresh', '--duration', '1.0', '--record', path, '--lock', 'wrist_yaw',
                 '--payload', '0.2@0,0,0.02', '--friction-gain', '0.5', '--damping', '0.05']) == 0
    sess = load_session(path)
    names = [e['event'] for e in sess.events]
    assert 'float_start' in names and 'float_end' in names
    start = sess.events[names.index('float_start')]
    assert start['lock'][-1] == 1 and start['payload'] == [0.2, 0.0, 0.0, 0.02]
    assert len(sess.state['q']) > 1000                                # every cycle with --record


def test_write_config_applies_repeatability_tuning(tmp_path):
    arm = load_arm()
    path = str(tmp_path / 'arm01.yaml')
    shutil.copy(arm.path, path)
    tuning = {'kp_scale': 1.8057117, 'kd_scale': 1.0, 'friction_ff_gain': 0.762, 'approach': -1, 'settle_s': 0.566}
    done = update_joint_config(path, {'repeatability_tuning': tuning})
    assert done and done[0].startswith('repeatability_tuning')
    update_joint_config(path, {'repeatability_tuning': dict(tuning, approach=1)})   # second time: replaces
    raw = open(path).read()
    assert raw.count('repeatability_tuning:') == 1 and '# J4' in raw
    got = load_arm(path.replace('\\', '/')).raw['repeatability_tuning']
    assert got['approach'] == 1 and got['kp_scale'] == pytest.approx(1.806, abs=1e-3)
