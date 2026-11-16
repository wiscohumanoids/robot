"""Phases 1 and 2 on the single-joint lever rig (config/arms/bench01.yaml) in sim.

Phase 1: one sim actuator with known errors (zero offset, Kt, lever COM,
friction and stiction, delay, backlash, hard stops past the URDF limits) goes
through bringup, indicator, zero and limits, each writing to a scratch store
as `arm <stage> --sim --arm bench01` would. `arm indicator` must also catch a
reversed direction and a wrong counts_per_rev, and the full-arm stages skip.

Phase 2: `arm rig --sim --quick --arm bench01 --serial ...` runs zero, kt,
gravity, friction and latency through the CLI; its sim errors are seeded from
the serial, so the test rebuilds them and checks each recovered value and that
the serial is in every calibration version and the rig summary.

Quick mode; about two and a half minutes.
"""

import math
import os
import zlib

import numpy as np
import pytest
import yaml

DEG = math.pi / 180
SERIAL = 'RMD-TEST-0042'


@pytest.fixture(scope='module')
def rig(tmp_path_factory):
    from arm_stack.config import CalibrationStore, load_arm
    from arm_stack.sim.arm_sim import SimArm, SimErrors
    arm = load_arm('bench01')
    store = CalibrationStore(arm, str(tmp_path_factory.mktemp('calib')))
    err = SimErrors.random(arm.n, seed=4)
    sim = SimArm(arm, store.latest(), err, seed=4)
    return {'arm': arm, 'store': store, 'sim': sim, 'err': err, 'results': {}}


def run(rig, name, client=None, **kw):
    import importlib

    from arm_stack.calib.common import StageContext
    from arm_stack.cli import MODULES, build_model
    arm, store = rig['arm'], rig['store']
    sim = client or rig['sim']
    cal = store.latest() if client is None else client.calibration
    sim.reload_calibration(cal)
    ctx = StageContext(arm=arm, client=sim, calibration=cal, model=build_model(arm, cal), sim=True,
                       quick=True, rng=np.random.default_rng(3))
    r = importlib.import_module('arm_stack.calib.' + MODULES[name]).run(ctx, **kw)
    print(r.report())
    if client is None and r.writes and r.passed:
        store.write(name, r.writes)
    rig['results'][name] = r
    return r


def _fresh_sim(rig, **errors):
    from arm_stack.config import empty_calibration
    from arm_stack.sim.arm_sim import SimArm, SimErrors
    return SimArm(rig['arm'], empty_calibration(rig['arm']), SimErrors(**errors), seed=5)


# --------------------------------------------------------------------------- Phase 1
def test_bench_model_is_one_joint_lever(rig):
    from arm_stack.cli import build_model
    arm = rig['arm']
    assert arm.n == 1 and arm.names == ['bench_joint']
    m = build_model(arm, rig['store'].latest())
    assert m.link_names() == ['lever']
    # lever level at q = 0, hanging at +90 deg; the hook (tool0) sits 0.25 m out on the lever
    hook = m.fk([0.0]).translation - m.fk([0.0], 'lever').translation
    assert np.allclose(hook, arm.raw['rig']['hook_offset_m'], atol=1e-9)
    assert m.tool_position([90 * DEG])[2] < m.tool_position([0.0])[2] - 0.2
    assert abs(m.gravity([0.0])[0]) > 10 * abs(m.gravity([90 * DEG])[0])


def test_bringup(rig):
    assert run(rig, 'bringup', soak_s=1.0).passed


def test_indicator(rig):
    r = run(rig, 'indicator')
    assert r.passed, r.report()
    m = r.metrics['bench_joint']
    assert abs(m['error_deg']) < 0.5
    assert abs(abs(m['commanded_deg'][1] - m['commanded_deg'][0]) - 90) < 1e-6
    assert m['sign_ok'] and m['scale_ok']
    # with nominal Kt in use, the 0x6077 torque over gravity is 1 / (Kt error), give or take the lever COM error
    assert abs(m['torque_scale'] - 1 / rig['sim'].kt_scale[0]) < 0.1
    # every null in the joint row is listed, and the gate is not met while any is left
    assert {'serial', 'gear_ratio', 'counts_per_rev', 'encoder_side', 'rated_current_a',
            'max_torque_nm', 'max_vel_rad_s'} <= set(m['nulls'])
    assert r.metrics['phase1_gate'] is False


def test_indicator_catches_reversed_direction(rig):
    r = run(rig, 'indicator', client=_fresh_sim(rig, direction_flip=np.array([True])))
    assert not r.passed
    m = r.metrics['bench_joint']
    assert m['angle_scale'] < 0 and m['sign_ok'] is False


def test_indicator_catches_counts_per_rev(rig):
    # encoder scale 1% off: a 90 deg command is 0.9 deg short on the protractor
    r = run(rig, 'indicator', client=_fresh_sim(rig, counts_scale=np.array([1.01])))
    assert not r.passed
    assert abs(r.metrics['bench_joint']['angle_scale'] - 1 / 1.01) < 0.002


def test_zero_recovers_offset(rig):
    r = run(rig, 'zero')
    assert r.passed
    assert abs(r.writes['zero_offsets_rad']['bench_joint'] - rig['sim'].true_zero[0]) < 0.1 * DEG


def test_limits_find_stops(rig):
    r = run(rig, 'limits')
    assert r.passed, r.report()
    lo, hi = r.metrics['hard_stops_rad']['bench_joint']
    assert abs(lo - rig['sim'].stop_lower[0]) < 1.0 * DEG
    assert abs(hi - rig['sim'].stop_upper[0]) < 1.0 * DEG


def test_full_arm_stages_skip(rig):
    from arm_stack.cli import run_stage
    for name in ('pivot', 'mocap-calib', 'mocap-repeat', 'validate'):
        r = run_stage(name, None, rig['arm'], rig['store'], rig['sim'])
        assert r.skipped and r.passed and 'assembled arm' in r.lines[0]
        assert 'SKIP' in r.report()


# --------------------------------------------------------------------------- Phase 2
@pytest.fixture(scope='module')
def rig_run(tmp_path_factory):
    from arm_stack.cli import main
    root = tmp_path_factory.mktemp('rig')
    rc = main(['rig', '--sim', '--quick', '--arm', 'bench01', '--serial', SERIAL, '--no-log',
               '--calib-root', str(root)])
    d = root / 'bench01' / SERIAL
    versions = sorted((p for p in os.listdir(d) if '_v' in p and not p.startswith('rig_')),
                      key=lambda p: int(p.split('_v')[1].split('.')[0]))
    summaries = [p for p in os.listdir(d) if p.startswith('rig_')]
    return {'rc': rc, 'dir': d, 'versions': versions,
            'cal': yaml.safe_load(open(d / versions[-1])),
            'summary': yaml.safe_load(open(d / summaries[0])) if summaries else None}


def _rig_truth():
    """The sim errors `arm rig --sim --serial SERIAL` (seed 0) injects."""
    from arm_stack.config import load_arm
    from arm_stack.sim.arm_sim import SimArm, SimErrors
    arm = load_arm('bench01')
    seed = zlib.crc32(SERIAL.encode()) % 100000
    from arm_stack.config import empty_calibration
    return SimArm(arm, empty_calibration(arm), SimErrors.random(arm.n, seed=seed), seed=seed), arm


def test_rig_passes_and_records_serial(rig_run):
    assert rig_run['rc'] == 0
    assert len(rig_run['versions']) == 5           # zero, kt, gravity, friction, latency
    for v in rig_run['versions']:
        cal = yaml.safe_load(open(rig_run['dir'] / v))
        assert cal['actuator_serial'] == SERIAL and cal['actuator_model'] == 'x4_36'
    s = rig_run['summary']
    assert s['actuator_serial'] == SERIAL and s['actuator_model'] == 'x4_36'
    assert s['gate'] == {'kt_linear_3_loads': True, 'friction_r2_above_0_9': True, 'met': True}


def test_rig_kt_on_the_hook_masses(rig_run):
    sim, arm = _rig_truth()
    j = arm.joints[0]
    s = rig_run['summary']
    assert abs(s['kt_output_nm_per_a'] / (j.kt_nominal_nm_per_a * sim.kt_scale[0]) - 1) < 0.03
    assert s['kt_loads_kg'] == arm.raw['rig']['test_masses_kg'][j.model]
    assert s['kt_linearity'] < 0.05


def test_rig_gravity_fits_lever_com(rig_run):
    sim, _ = _rig_truth()
    true_com = sim._truth_inertias[0].lever
    com = np.array(rig_run['cal']['link_inertials']['lever']['com_m'])
    # x and z (in the plane the lever swings in) are identifiable; y is not
    assert abs(com[0] - true_com[0]) < 0.003
    assert abs(com[2] - true_com[2]) < 0.003


def test_rig_friction(rig_run):
    sim, _ = _rig_truth()
    f = rig_run['summary']['friction']
    assert rig_run['summary']['friction_r2'] > 0.9
    assert abs(f['coulomb_nm'] - sim.coulomb[0]) < 0.15 * sim.coulomb[0] + 0.03
    assert abs(f['viscous_nm_s_per_rad'] - sim.viscous[0]) < 0.3 * sim.viscous[0] + 0.02


def test_rig_latency(rig_run):
    sim, _ = _rig_truth()
    true_ms = sim.err.delay_cycles * sim.dt * 1e3
    assert abs(rig_run['summary']['latency_ms'] - true_ms) < 1.0 + 0.5 * true_ms
