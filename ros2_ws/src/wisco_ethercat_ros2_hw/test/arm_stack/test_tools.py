"""Optimizers, ISO 9283 math, the recorder and the exports."""

import os

import numpy as np
import pytest

from arm_stack.calib.mocap_repeat import drift, iso_rp
from arm_stack.pace.bo import GPBO, Space
from arm_stack.pace.cmaes import CMAES


def test_cmaes_finds_shifted_sphere_inside_box():
    target = np.array([0.3, -0.2, 0.7, 0.1])
    es = CMAES(np.zeros(4), 0.3, lower=-np.ones(4), upper=np.ones(4), popsize=12, seed=1)
    for _ in range(150):
        X = es.ask()
        assert np.all(X >= -1) and np.all(X <= 1)
        es.tell(X, np.sum((X - target) ** 2, axis=1))
    assert np.allclose(es.best_x, target, atol=1e-3)


def test_gpbo_minimizes_1d():
    space = Space({'x': ('lin', -2.0, 3.0)})
    bo = GPBO(space, n_init=4, seed=0)
    for _ in range(20):
        u = bo.ask()
        x = space.to_params(u)['x']
        bo.tell(u, (x - 1.3) ** 2)
    u, y = bo.best()
    assert abs(space.to_params(u)['x'] - 1.3) < 0.1


def test_space_choice_and_log():
    s = Space({'a': ('choice', [1, -1]), 'k': ('log', 0.5, 2.0)})
    assert s.to_params(np.array([0.1, 0.0])) == {'a': 1, 'k': pytest.approx(0.5)}
    assert s.to_params(np.array([0.9, 1.0])) == {'a': -1, 'k': pytest.approx(2.0)}


def test_iso_rp_matches_definition():
    rng = np.random.default_rng(0)
    P = rng.normal(0, 1e-4, (30, 5, 3)) + np.arange(5)[None, :, None]
    r = iso_rp(P)
    l = np.linalg.norm(P - P.mean(axis=0), axis=2)
    S = np.sqrt(((l - l.mean(axis=0)) ** 2).sum(axis=0) / 29)
    assert np.allclose(r['rp'], l.mean(axis=0) + 3 * S)


def test_drift_sees_trend_not_noise():
    rng = np.random.default_rng(1)
    c = 30
    trend = np.linspace(0, 1e-3, c)[:, None, None] * np.array([1, 0, 0])[None, None]
    P = rng.normal(0, 1e-5, (c, 2, 3)) + trend
    d = drift(P)
    assert d[0] == pytest.approx(1e-3, rel=0.1)
    assert drift(rng.normal(0, 1e-5, (c, 2, 3))).max() < 1e-4


@pytest.fixture
def sim():
    from arm_stack.config import empty_calibration, load_arm
    from arm_stack.sim.arm_sim import SimArm
    arm = load_arm()
    return arm, SimArm(arm, empty_calibration(arm))


def test_recorder_round_trip_and_rejection(tmp_path, sim):
    pytest.importorskip('mcap')
    from arm_stack.control.motion import move_to
    from arm_stack.log.export import export_pace
    from arm_stack.log.recorder import Recorder, RecordingClient, SessionRejected, load_session
    arm, s = sim
    with pytest.raises(SessionRejected):
        Recorder(str(tmp_path / 'bad.mcap'), {'arm': arm.arm})
    path = str(tmp_path / 's.mcap')
    rec = Recorder(path, {'arm': arm.arm, 'joint_names': arm.names, 'calibration_version': 3})
    c = RecordingClient(s, rec)
    q = s.state().q.copy()
    q[3] += 0.2
    move_to(c, q, max_speed=0.5, settle=0.1)
    c.close()
    sess = load_session(path)
    assert sess.meta['calibration_version'] == 3
    assert sess.state['q'].shape[1] == arm.n
    assert len(sess.cmd) > 100
    assert {'q', 'counts_raw', 'tau_raw_permille'} <= set(sess.state)
    out = export_pace(path, str(tmp_path / 'pace'))
    d = np.load(out)
    assert d['command'].shape == d['position'].shape
    assert d['sample_rate_hz'] > 100
    assert os.path.exists(str(tmp_path / 'pace.json'))
