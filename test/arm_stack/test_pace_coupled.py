"""PACE coupled replay (JointReplayBackend with hold_gains).

With the held joints made effectively rigid, the coupled replay must reduce to
the per-joint one; with realistic hold gains the chirped joint drags the others,
so the two must differ (slightly, at this pose) as the chirp speeds up.
"""

import numpy as np
import pytest

pytest.importorskip('pinocchio')

from arm_stack.config import load_arm  # noqa: E402
from arm_stack.cli import build_model  # noqa: E402
from arm_stack.config import empty_calibration  # noqa: E402
from arm_stack.calib.common import safe_home  # noqa: E402
from arm_stack.pace.backend import JointLog, JointReplayBackend  # noqa: E402


def _setup(j=1, T=400, dt=0.004, f1=6.0):
    arm = load_arm()
    model = build_model(arm, empty_calibration(arm))
    hold = safe_home(model)
    t = np.arange(T) * dt
    f = 0.5 + (f1 - 0.5) * t / t[-1]
    amp = 0.05
    q_cmd = hold[j] + amp * np.sin(2 * np.pi * np.cumsum(f) * dt)
    qd_cmd = np.gradient(q_cmd, dt)
    log = JointLog(j, dt, q_cmd, qd_cmd, q_cmd.copy(), hold, (hold[j], amp), (0.5, f1))
    P = np.array([[0.03, 0.2, 0.3, 0.0, 0.002, 300.0, 6.0]])
    return model, log, P


def test_rigid_hold_reduces_to_per_joint_replay():
    model, log, P = _setup()
    n = model.n
    single = JointReplayBackend(model, [log], substeps=4).simulate(0, P)
    # very stiff, critically damped hold on every other joint
    M = model.mass_matrix(log.q_hold)
    kp = np.full(n, 2e4)
    kd = 2 * np.sqrt(kp * np.diag(M))
    coupled = JointReplayBackend(model, [log], substeps=4, hold_gains=(kp, kd),
                                 hold_armature=np.full(n, 0.01)).simulate(0, P)
    assert np.all(np.isfinite(coupled))
    err = np.max(np.abs(coupled - single))
    assert err < 0.05 * log.sweep[1], err


def test_soft_hold_couples_the_joints():
    model, log, P = _setup()
    n = model.n
    single = JointReplayBackend(model, [log], substeps=4).simulate(0, P)
    coupled = JointReplayBackend(model, [log], substeps=4,
                                 hold_gains=(np.full(n, 200.0), np.full(n, 4.0)),
                                 hold_armature=np.full(n, 0.01)).simulate(0, P)
    assert np.all(np.isfinite(coupled))
    assert np.max(np.abs(coupled - single)) > 0.002 * log.sweep[1]


def test_population_rows_are_independent():
    model, log, P = _setup(T=150)
    n = model.n
    be = JointReplayBackend(model, [log], hold_gains=(np.full(n, 300.0), np.full(n, 6.0)),
                            hold_armature=np.full(n, 0.01))
    P2 = np.vstack([P, P * [3.0, 1, 1, 1, 1, 1, 1]])
    both = be.simulate(0, P2)
    alone = be.simulate(0, P)
    assert np.allclose(both[0], alone[0])
    assert not np.allclose(both[1], alone[0])


def test_compliance_softens_the_loop_and_reduces_when_rigid():
    model, log, P = _setup()
    n = model.n
    rigid = JointReplayBackend(model, [log], substeps=4).simulate(0, P)
    inf = JointReplayBackend(model, [log], substeps=4, stiffness=np.full(n, np.inf)).simulate(0, P)
    assert np.allclose(rigid, inf)
    soft = JointReplayBackend(model, [log], substeps=4, stiffness=np.full(n, 1500.0)).simulate(0, P)
    assert np.all(np.isfinite(soft))
    assert np.max(np.abs(soft - rigid)) > 0.01 * log.sweep[1]
    # the coupled replay with a rigid hold agrees with the per-joint one under compliance too
    M = model.mass_matrix(log.q_hold)
    kp = np.full(n, 2e4)
    kd = 2 * np.sqrt(kp * np.diag(M))
    stiff = np.where(np.arange(n) == log.joint, 1500.0, np.inf)
    single = JointReplayBackend(model, [log], substeps=4, stiffness=stiff).simulate(0, P)
    coupled = JointReplayBackend(model, [log], substeps=4, hold_gains=(kp, kd), hold_armature=np.full(n, 0.01),
                                 stiffness=stiff).simulate(0, P)
    assert np.max(np.abs(coupled - single)) < 0.05 * log.sweep[1]


def test_torque_clamp_limits_the_replay():
    model, log, P = _setup()
    n = model.n
    free = JointReplayBackend(model, [log], substeps=4).simulate(0, P)
    inf = JointReplayBackend(model, [log], substeps=4, torque_limit=np.full(n, np.inf)).simulate(0, P)
    assert np.allclose(free, inf)
    # a clamp just above the hold torque cuts the chirp's swing
    g = abs(model.gravity(log.q_hold)[log.joint])
    tight = JointReplayBackend(model, [log], substeps=4, torque_limit=np.full(n, g + 0.5)).simulate(0, P)
    assert np.all(np.isfinite(tight))
    swing = lambda q: np.ptp(q[0, len(q[0]) // 2:])  # noqa: E731
    assert swing(tight) < 0.9 * swing(free)
