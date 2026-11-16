"""Calibration and validation without mocap: the divot-plate kinematic fit
(`arm plate`) and ISO 9283 repeatability on the dial-indicator cube
(`arm repeat --indicators`, `arm validate` without mocap), in sim, quick mode.
"""

import math
import time

import numpy as np
import pytest

from arm_stack.hw.indicator_cube import (CubeConfig, SimIndicatorCube, parse_readings_mm, position_to_readings,
                                         quantize, readings_to_position)

DEG = math.pi / 180


# --------------------------------------------------------------------------- cube math
def _rotation(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def test_cube_readings_invert_on_a_rotated_stand():
    axes = _rotation([0.3, -0.5, 0.9])           # rows: three orthogonal indicator axes
    dp = np.array([12e-6, -40e-6, 7e-6])
    r = position_to_readings(dp, axes)
    assert np.allclose(r, [axes[i] @ dp for i in range(3)])
    assert np.allclose(readings_to_position(r, axes), dp, atol=1e-15)
    # a skewed stand (not quite orthogonal) still inverts
    skew = axes + np.array([[0, 0.02, 0], [0, 0, 0], [0.01, 0, 0]])
    skew /= np.linalg.norm(skew, axis=1, keepdims=True)
    assert np.allclose(readings_to_position(position_to_readings(dp, skew), skew), dp, atol=1e-15)


def test_cube_readings_on_the_default_stand_are_the_coordinates():
    dp = np.array([0.0123e-3, -0.004e-3, 0.0e-3])
    assert np.allclose(readings_to_position(dp, np.eye(3)), dp)
    assert np.allclose(quantize([1.4e-6, -2.6e-6, 0.49e-6], 1e-6), [1e-6, -3e-6, 0.0])
    assert np.allclose(parse_readings_mm('0.003, -0.012 0.000'), [3e-6, -12e-6, 0.0])
    with pytest.raises(ValueError):
        parse_readings_mm('0.003 0.001')


def test_typed_readings_become_a_position():
    from arm_stack.hw.indicator_cube import OperatorIndicatorCube

    class Client:
        sent = 0

        def send(self, cmd):
            Client.sent += 1
    answers = iter(['', 'oops', '0.010 0 -0.005'])

    def ask(prompt):
        time.sleep(0.03)                                # the operator takes a moment
        return next(answers)
    cube = OperatorIndicatorCube(CubeConfig(axes=np.eye(3)[[1, 0, 2]]), ask=ask)
    ctx = type('Ctx', (), {'client': Client()})()
    cube.setup(ctx, 0, np.zeros(7))                    # Enter once the stand is zeroed
    p = cube.position(ctx.client, np.zeros(7))         # one bad line, then x y z in mm
    assert np.allclose(p, [0.0, 10e-6, -5e-6])          # axes rows: y, x, z
    assert Client.sent > 0                              # the hold kept being sent while waiting


class _FakeSim:
    """Just enough of SimArm for SimIndicatorCube: a truth model and link angles."""

    def __init__(self, truth, q):
        self.truth = truth
        self.q = np.array(q, float)

    def true_joint_angles(self):
        return self.q.copy()


def test_sim_cube_reads_the_flange_motion_on_its_axes():
    from arm_stack.calib.common import safe_home
    from arm_stack.config import load_arm
    from arm_stack.model.robot import ArmModel
    arm = load_arm()
    truth = ArmModel(arm.urdf, arm.names)
    q = safe_home(truth) + 0.2
    fake = _FakeSim(truth, q)
    cfg = CubeConfig(axes=_rotation([0.0, 0.0, 0.4]))
    cube = SimIndicatorCube(fake, cfg, noise_m=0.0)
    cube.setup(None, 0, q)
    assert np.allclose(cube.read(None, None), 0.0)
    c0 = truth.fk(q, 'flange').act(cfg.center_in_flange_m)
    fake.q = q + np.array([0, 2e-4, 0, -1e-4, 0, 0, 0])
    c1 = truth.fk(fake.q, 'flange').act(cfg.center_in_flange_m)
    r = cube.read(None, None)
    assert np.allclose(r, quantize(cfg.axes @ (c1 - c0), 1e-6))
    assert np.linalg.norm(cube.position(None, None) - (c1 - c0)) < 1e-6        # 1 um resolution
    assert cube.static_noise_3sigma_m() < 1e-12


# --------------------------------------------------------------------------- sim stages
def _axial_free_overlay(model, names, rng, dxyz=0.0008, drot=0.002):
    """Kinematic errors as in test_phase0_gate, but with each axis tilt
    orthogonal to the joint's own axis and its parent's: a tilt about either
    is physically a zero offset (of that joint or the parent), which no tip
    or pose measurement can tell apart."""
    out = {}
    for k in range(1, len(names)):
        # drot is applied in joint k's own frame (placement @ exp(drot)): the parent's
        # axis goes into that frame through the nominal placement (link frames are not
        # parallel at q = 0)
        R = model.nominal.jointPlacements[model.jid[k]].rotation
        a, b = model.axis_in_child(k), R.T @ model.axis_in_child(k - 1)
        t = np.cross(a, b)
        if np.linalg.norm(t) < 1e-3:
            t = np.cross(a, [1.0, 0.0, 0.0])
        t /= np.linalg.norm(t)
        out[names[k]] = {'dxyz': (rng.uniform(-1, 1, 3) * dxyz).tolist(),
                         'drot': (t * rng.uniform(0.5, 1.0) * rng.choice([-1, 1]) * drot).tolist()}
    return out


def _context(arm, store, sim, quick=True):
    from arm_stack.calib.common import StageContext
    from arm_stack.cli import build_model
    cal = store.latest()
    sim.reload_calibration(cal)
    return StageContext(arm=arm, client=sim, calibration=cal, model=build_model(arm, cal), sim=True,
                        quick=quick, rng=np.random.default_rng(3))


@pytest.fixture(scope='module')
def rig(tmp_path_factory):
    """A sim arm with no mocap: random zero offsets (5 deg), compliance, backlash,
    kinematic errors; the calibration holds jig zeros 0.5 deg off the truth."""
    from arm_stack.config import CalibrationStore, load_arm
    from arm_stack.model.robot import ArmModel
    from arm_stack.sim.arm_sim import SimArm, SimErrors
    arm = load_arm()
    store = CalibrationStore(arm, str(tmp_path_factory.mktemp('calib')))
    err = SimErrors.random(arm.n, seed=2)
    err.kinematic = _axial_free_overlay(ArmModel(arm.urdf, arm.names), arm.names, np.random.default_rng(12))
    sim = SimArm(arm, store.latest(), err, seed=2, mocap_enabled=False)
    jig = sim.true_zero + np.random.default_rng(5).uniform(-0.5, 0.5, arm.n) * DEG
    store.write('zero', {'zero_offsets_rad': {n: float(v) for n, v in zip(arm.names, jig)}})
    return {'arm': arm, 'store': store, 'sim': sim, 'jig': jig}


def test_plate_recovers_zero_offsets(rig):
    from arm_stack.calib import plate
    arm, sim = rig['arm'], rig['sim']
    assert sim.mocap() is None
    r = plate.run(_context(arm, rig['store'], sim))
    print(r.report())
    assert r.passed, r.report()
    m = r.metrics
    assert m['held_out_rms_mm'] < 0.5 and m['pair_distance_rms_mm'] < 0.5
    assert m['condition_number'] < 1e4
    observed = m['zero_corrections_rad']
    # J1 folds into the plate pose and J7 into the probe tip; the rest must be seen
    assert set(arm.names[1:6]) <= set(observed), observed
    assert arm.names[0] not in observed and arm.names[-1] not in observed
    for i, n in enumerate(arm.names):
        if n in observed:
            got = r.writes['zero_offsets_rad'][n]
            assert abs(got - sim.true_zero[i]) < 0.1 * DEG, (n, (got - sim.true_zero[i]) / DEG)
    assert arm.names[0] not in r.writes['zero_offsets_rad']          # keeps the jig value
    # J7 is written tool0-equivalent (probe tip folded in); the held-out check above ran on the
    # parameters as stored, so the written calibration reproduces the fit at the tip
    assert m['j7_zero_from_tip_rad'] is not None and arm.names[-1] in r.writes['zero_offsets_rad']
    assert np.linalg.norm(m['tip_radial_m']) < 0.5e-3
    assert set(r.writes) >= {'zero_offsets_rad', 'kinematic_overlay', 'joint_stiffness_nm_per_rad'}
    # the fit is written like mocap-calib's
    rig['store'].write('plate', r.writes)


def test_cube_repeatability(rig):
    from arm_stack.calib import repeat
    # the stage's quick run is 5 poses x 6 cycles; 4 cycles keep the test short
    r = repeat.run(_context(rig['arm'], rig['store'], rig['sim']), indicators=True, cycles=4, warm_s=0.5)
    print(r.report())
    assert r.passed, r.report()
    assert r.metrics['source'].startswith('indicator cube') and len(r.metrics['rp_mm']) == 5
    assert max(r.metrics['rp_mm']) < 0.5
    assert r.metrics['noise_3sigma_mm'] < 0.01
    assert r.writes['validation']['repeatability_rp_mm'] < 0.5


def test_validate_uses_the_cube_without_mocap(rig, monkeypatch):
    from arm_stack.calib import validate
    from arm_stack.hw.indicator_cube import OperatorIndicatorCube
    ctx = _context(rig['arm'], rig['store'], rig['sim'])
    # sim: the cube, automatically (validate runs the same iso_measure as test_cube_repeatability)
    source, why = validate.repeatability_source(ctx)
    assert isinstance(source, SimIndicatorCube) and why is None
    # hardware without mocap asks first: yes -> typed readings, no -> NOT RUN
    ctx.sim = False
    monkeypatch.setattr('builtins.input', lambda prompt='': 'y')
    assert isinstance(validate.repeatability_source(ctx)[0], OperatorIndicatorCube)
    monkeypatch.setattr('builtins.input', lambda prompt='': 'n')
    assert validate.repeatability_source(ctx) == (None, 'no mocap and the indicator cube declined')


def test_chain_picks_the_stages_by_source(rig):
    from arm_stack.calib.common import chain_skip

    class WithMocap:
        def mocap(self):
            return object()
    sim = rig['sim']
    assert chain_skip('plate', sim) is None and chain_skip('repeat', sim) is None
    assert 'no mocap' in chain_skip('mocap-calib', sim)
    assert 'no repeatability tuning' in chain_skip('mocap-repeat', sim)
    assert 'mocap present' in chain_skip('plate', WithMocap())
    assert chain_skip('mocap-calib', WithMocap()) is None and chain_skip('pivot', sim) is None
