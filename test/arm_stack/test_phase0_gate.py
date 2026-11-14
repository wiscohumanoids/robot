"""Phase 0 gate (spec: Milestones): each procedure recovers its injected errors
in sim within its pass threshold.

One sim arm with known errors (zero offsets, Kt, COM offsets, friction and
stiction, delay, backlash, compliance, hard stops past the URDF limits,
kinematic errors) goes through the stages in order, each writing to a scratch
calibration store as `arm <stage> --sim` would.

This takes several minutes, so ctest runs it only with ARM_PHASE0_GATE=1:
    ARM_PHASE0_GATE=1 python3 -m pytest test/arm_stack/test_phase0_gate.py -v
(`arm all --sim --quick --fresh` runs the same chain from the command line.)
"""

import math
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('ARM_PHASE0_GATE') != '1',
                                reason='slow; set ARM_PHASE0_GATE=1')

DEG = math.pi / 180


@pytest.fixture(scope='module')
def rig(tmp_path_factory):
    from arm_stack.config import CalibrationStore, load_arm
    from arm_stack.sim.arm_sim import SimArm, SimErrors
    arm = load_arm()
    store = CalibrationStore(arm, str(tmp_path_factory.mktemp('calib')))
    err = SimErrors.random(arm.n, seed=1)
    err.kinematic = {'elbow': {'dxyz': [0.0008, -0.0005, 0.0004], 'drot': [0.002, 0.0, -0.0015]},
                     'shoulder_yaw': {'dxyz': [-0.0006, 0.0, 0.0007], 'drot': [0.0, 0.0015, 0.0]}}
    sim = SimArm(arm, store.latest(), err, seed=1)
    return {'arm': arm, 'store': store, 'sim': sim, 'err': err, 'results': {}}


def run(rig, name, **kw):
    import importlib

    from arm_stack.calib.common import StageContext
    from arm_stack.cli import MODULES, build_model
    arm, store, sim = rig['arm'], rig['store'], rig['sim']
    cal = store.latest()
    sim.reload_calibration(cal)
    ctx = StageContext(arm=arm, client=sim, calibration=cal, model=build_model(arm, cal), sim=True,
                       quick=True, rng=np.random.default_rng(3))
    r = importlib.import_module('arm_stack.calib.' + MODULES[name]).run(ctx, **kw)
    print(r.report())
    if r.writes and r.passed:
        store.write(name, r.writes)
    rig['results'][name] = r
    return r


def test_bringup(rig):
    assert run(rig, 'bringup', soak_s=2.0).passed


def test_zero_recovers_offsets(rig):
    r = run(rig, 'zero')
    assert r.passed
    got = np.array([r.writes['zero_offsets_rad'][n] for n in rig['arm'].names])
    assert np.max(np.abs(got - rig['sim'].true_zero)) < 0.1 * DEG


def test_limits_find_stops_past_urdf(rig):
    r = run(rig, 'limits')
    assert r.passed, r.report()
    stops = r.metrics['hard_stops_rad']
    for i, n in enumerate(rig['arm'].names):
        lo, hi = stops[n]
        assert abs(hi - rig['sim'].stop_upper[i]) < 1.0 * DEG, n
        assert abs(lo - rig['sim'].stop_lower[i]) < 1.0 * DEG, n


def test_kt_recovers_scale(rig):
    r = run(rig, 'kt')
    assert r.passed, r.report()
    arm, sim = rig['arm'], rig['sim']
    for i, j in enumerate(arm.joints):
        got = r.writes['kt_output_nm_per_a'].get(j.name)
        if got is None:
            continue
        true = j.kt_nominal_nm_per_a * sim.kt_scale[i]
        assert abs(got / true - 1) < 0.05, j.name


def test_gravity_fit(rig):
    assert run(rig, 'gravity').passed


def test_friction_fit(rig):
    r = run(rig, 'friction')
    assert r.passed, r.report()


def test_latency(rig):
    r = run(rig, 'latency')
    true_ms = rig['err'].delay_cycles * rig['sim'].dt * 1e3
    assert abs(r.writes['latency_ms'] - true_ms) < 1.0 + 0.5 * true_ms


def test_pivot(rig):
    assert run(rig, 'pivot').passed


def test_mocap_calib_recovers_kinematics(rig):
    r = run(rig, 'mocap-calib')
    assert r.passed, r.report()


def test_pace(rig):
    # after mocap-calib, as in `arm all`: the replay uses its joint stiffness
    r = run(rig, 'pace')
    assert r.passed, r.report()


def test_mocap_repeat(rig):
    r = run(rig, 'mocap-repeat')
    assert r.passed, r.report()
    assert max(r.metrics['rp_mm']) < 0.5


def test_validate(rig):
    r = run(rig, 'validate')
    assert r.passed, r.report()
