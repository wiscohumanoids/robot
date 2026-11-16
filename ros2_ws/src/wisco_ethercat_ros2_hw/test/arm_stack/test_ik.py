"""IK solvers (arm_stack.model.ik): recovery of FK targets, joint limits,
null-space posture, the streaming tracker and unreachable targets."""

import math
import os

import numpy as np
import pinocchio as pin
import pytest

from arm_stack.model import ik
from arm_stack.model.robot import ArmModel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF = os.path.join(ROOT, 'urdf', 'arm01', 'arm01.urdf')
NAMES = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'forearm_yaw',
         'wrist_pitch', 'wrist_yaw']
DEG = math.pi / 180


@pytest.fixture(scope='module')
def model():
    return ArmModel(URDF, NAMES)


def _targets(model, n, seed):
    rng = np.random.default_rng(seed)
    Q = rng.uniform(model.lower, model.upper, (n, model.n))
    return Q, [model.fk(q) for q in Q], rng


def _inside(model, q):
    return bool(np.all(q >= model.lower) and np.all(q <= model.upper))


def test_multistart_recovers_fk_targets_from_nominal(model):
    Q, T, rng = _targets(model, 200, 1)
    mid = ik.mid_posture(model)
    ok = 0
    for t in T:
        r = ik.solve(model, t, mid, method='multistart', base='lm', n_starts=40, rng=rng)
        ok += r.success
        assert _inside(model, r.q)
        if r.success:
            M = model.fk(r.q)
            assert np.linalg.norm(M.translation - t.translation) < 1e-4
            assert np.linalg.norm(pin.log3(M.rotation.T @ t.rotation)) < 0.1 * DEG
    assert ok / len(T) >= 0.99


@pytest.mark.parametrize('method', ['clik', 'lm', 'dls'])
def test_local_solvers_recover_from_near_seed(model, method):
    Q, T, rng = _targets(model, 200, 2)
    ok = 0
    for q, t in zip(Q, T):
        q0 = np.clip(q + rng.uniform(-0.1, 0.1, model.n), model.lower, model.upper)
        r = ik.solve(model, t, q0, method=method)
        ok += r.success
        assert _inside(model, r.q)
    assert ok / len(T) >= 0.95


@pytest.mark.parametrize('method', ['dls', 'lm', 'clik', 'multistart'])
def test_joint_limits_respected(model, method):
    rng = np.random.default_rng(3)
    for _ in range(25):
        # seeds outside the box, targets partly outside the workspace
        q0 = rng.uniform(model.lower - 1.0, model.upper + 1.0)
        p = rng.uniform(-1.2, 1.2, 3)
        T = pin.SE3(pin.exp3(rng.normal(0, 2, 3)), p)
        r = ik.solve(model, T, q0, method=method, max_iters=60, n_starts=3)
        assert _inside(model, r.q)
        assert r.iters >= 0 and r.time >= 0


def test_position_only_and_free_spin(model):
    Q, T, rng = _targets(model, 40, 4)
    mid = ik.mid_posture(model)
    for t in T:
        r = ik.solve(model, t.translation, mid, method='multistart', base='lm', n_starts=10, rng=rng)
        assert r.success
        assert np.linalg.norm(model.tool_position(r.q) - t.translation) < 1e-4
        r = ik.solve(model, t, mid, method='multistart', base='lm', mask='free_z', n_starts=20, rng=rng)
        assert r.success
        M = model.fk(r.q)
        assert np.linalg.norm(M.translation - t.translation) < 1e-4
        # the tool z axis matches, the spin about it is free
        assert np.arccos(np.clip(M.rotation[:, 2] @ t.rotation[:, 2], -1, 1)) < 0.1 * DEG


def _loop_path(model, steps=300, loops=3, seed=5):
    rng = np.random.default_rng(seed)
    mid = ik.mid_posture(model)
    half = 0.5 * (model.upper - model.lower)
    harm = np.array([1, 2, 1, 3, 2, 1, 2])
    ph = rng.uniform(0, 2 * np.pi, model.n)
    t = np.arange(steps * loops) / steps
    Q = mid + 0.3 * half * np.sin(2 * np.pi * harm * t[:, None] + ph)
    return Q, [model.fk(q) for q in Q]


def test_null_space_posture_holds_the_elbow(model):
    steps = 300
    Q, path = _loop_path(model, steps)

    def run(method, **kw):
        q = Q[0].copy()
        starts = []
        for i, T in enumerate(path):
            if i % steps == 0:
                starts.append(q.copy())
            r = ik.solve(model, T, q, method=method, **kw)
            assert r.success
            q = r.q
        return [model.fk(s, 'elbow').translation for s in starts], starts

    E, S = run('clik', min_iters=1)
    # same tool pose at every loop start: the elbow comes back (cyclic)
    assert np.linalg.norm(E[2] - E[1]) < 2e-3
    assert np.max(np.abs(S[2] - S[1])) < 1 * DEG
    # the null-space motion does not disturb the tool
    q = Q[0] + np.array([0, 0, 0.3, 0, -0.3, 0, 0])
    q = np.clip(q, model.lower, model.upper)
    T = model.fk(q)
    r = ik.solve(model, T, q, method='clik', min_iters=20, k_rest=0.1)
    assert r.success
    assert np.linalg.norm(r.q - ik.mid_posture(model)) < np.linalg.norm(q - ik.mid_posture(model))


def test_elbow_pull_moves_the_elbow(model):
    q = ik.mid_posture(model)
    T = model.fk(q)
    e0 = model.fk(q, 'elbow').translation
    d = np.array([0.0, 0.0, 1.0])
    r = ik.solve(model, T, q, method='clik', k_rest=0.0, elbow_dir=d, k_elbow=0.5, min_iters=30)
    assert r.success
    assert (model.fk(r.q, 'elbow').translation - e0) @ d > 5e-3


def test_tracker_is_continuous_on_a_path(model):
    dt = 0.01
    Q, path = _loop_path(model, steps=1000, loops=1, seed=6)     # 10 s loop: under the speed cap
    tr = ik.IKTracker(model, Q[0], dt=dt)
    prev = tr.q.copy()
    lim = tr.max_vel * dt + 1e-12
    reach = 0
    for T in path:
        out = tr.update(T)
        assert _inside(model, out.q)
        assert np.all(np.abs(out.q - prev) <= lim)
        reach += out.reachable
        prev = out.q.copy()
    assert reach / len(path) >= 0.99
    assert out.pos_err < 1e-4


def test_unreachable_target_gives_closest_pose_and_flag(model):
    q0 = ik.mid_posture(model)
    M0 = model.fk(q0)
    far = pin.SE3(M0.rotation, np.array([0.0, 0.0, 2.0]))
    r = ik.solve(model, far, q0, method='lm')
    assert not r.success and _inside(model, r.q)
    assert r.pos_err > 1.0
    r2 = ik.solve(model, far, q0, method='multistart', n_starts=5)
    assert not r2.success and _inside(model, r2.q)
    tr = ik.IKTracker(model, q0, dt=0.01, max_vel=50.0)
    for _ in range(50):
        out = tr.update(far.translation)          # position only
    assert not out.reachable
    assert _inside(model, out.q)
    p = model.tool_position(out.q)
    # stretched toward the target: closer than where it started
    assert np.linalg.norm(p - far.translation) < np.linalg.norm(M0.translation - far.translation) - 0.05
    # and the reported error is the real distance
    assert out.pos_err == pytest.approx(np.linalg.norm(p - far.translation), rel=1e-6)
    # back to a reachable target: it recovers
    for _ in range(100):
        out = tr.update(M0)
    assert out.reachable


def test_batch_kinematics_and_solver(model):
    bk = ik.BatchKinematics(model)
    rng = np.random.default_rng(8)
    Q = rng.uniform(model.lower, model.upper, (300, model.n))
    p, R, J = bk.fk(Q, jacobian=True)
    for i in range(0, 300, 37):
        M = model.fk(Q[i])
        assert np.allclose(M.translation, p[i], atol=1e-12)
        assert np.allclose(M.rotation, R[i], atol=1e-12)
        assert np.allclose(model.jacobian(Q[i]), J[i], atol=1e-12)
    Q0 = np.clip(Q + rng.uniform(-0.1, 0.1, Q.shape), model.lower, model.upper)
    Qs, ok, pe, re = ik.solve_batch(bk, p, R, Q0)
    assert ok.mean() >= 0.97
    assert np.all(Qs >= model.lower) and np.all(Qs <= model.upper)


def test_arm_model_ik_wrapper(model):
    q = np.clip(ik.mid_posture(model) + 0.2, model.lower, model.upper)
    T = model.fk(q)
    r = model.ik(T, q + 0.05)
    assert r.success
    r = model.ik(T, q + 0.05, method='lm')
    assert r.success
    # the position-only DLS used by sim procedures is still there and unchanged
    p = model.ik_position(T.translation, q + 0.05)
    assert np.linalg.norm(model.tool_position(p) - T.translation) < 1e-5


def test_bad_inputs(model):
    with pytest.raises(ValueError):
        ik.solve(model, np.eye(3), ik.mid_posture(model))
    with pytest.raises(ValueError):
        ik.solve(model, np.eye(4), ik.mid_posture(model), method='nope')
    with pytest.raises(ValueError):
        ik.solve(model, np.eye(4), ik.mid_posture(model), weights=np.zeros(6))
