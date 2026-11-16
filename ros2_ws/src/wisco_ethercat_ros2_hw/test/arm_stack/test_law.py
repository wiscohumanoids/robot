"""The Python law (sim) must match the C++ law (test/test_arm_law.cpp): same cases."""

import math

import numpy as np
import pytest

from arm_stack.control.law import ControlLaw
from arm_stack.hw.client import Command


class FakeModel:
    def __init__(self, g):
        self.g = np.asarray(g, float)

    def gravity(self, q):
        return self.g.copy()

    def friction(self, qd):
        return np.zeros(len(self.g))


G = [3.0, 0.5]
Q = np.zeros(2)


def law():
    return ControlLaw(FakeModel(G), max_torque=[10.0, 2.0], max_vel=[2.0, 4.0], soft_lower=[-1.0, -0.5],
                      soft_upper=[1.0, 0.5], kd_fallback=[0.5, 0.1])


def cmd(mode, **kw):
    base = dict(mode=mode, q=np.array([0.2, 0.1]), qd=np.zeros(2), kp=np.array([50.0, 10.0]),
                kd=np.array([1.0, 0.2]), tau_ff=np.zeros(2))
    base.update(kw)
    return Command(**base)


def test_no_command_is_gravity_comp_and_damping():
    out = law().compute(None, 1e9, Q, np.array([1.0, -2.0]), 'P')
    assert out.fallback and out.mode == 'T'
    assert out.tau == pytest.approx([2.5, 0.7])


def test_watchdog():
    assert law().compute(cmd('P'), 0.051, Q, Q, 'P').fallback
    assert not law().compute(cmd('P'), 0.049, Q, Q, 'P').fallback


def test_mode_p_target_and_offset():
    out = law().compute(cmd('P', tau_ff=np.array([0.5, 0.0])), 0.0, Q, Q, 'P')
    assert out.mode == 'P'
    assert out.q_target[0] == pytest.approx(0.2)
    assert out.tau == pytest.approx([3.5, 0.5])


def test_mode_p_soft_limits():
    out = law().compute(cmd('P', q=np.array([5.0, -5.0])), 0.0, Q, Q, 'P')
    assert out.limit_hit
    assert out.q_target == pytest.approx([1.0, -0.5])
    out = law().compute(cmd('P', q=np.array([5.0, -5.0]), ignore_soft_limits=True), 0.0, Q, Q, 'P')
    assert out.q_target[0] == pytest.approx(5.0)


def test_mode_t_pd_and_clamp():
    out = law().compute(cmd('T'), 0.0, Q, np.array([0.1, 0.0]), 'T')
    assert out.tau[1] == pytest.approx(1.5)
    assert out.tau[0] == pytest.approx(10.0) and out.clamped


def test_command_clamp_only_tightens():
    out = law().compute(cmd('T', kp=np.zeros(2), kd=np.zeros(2), tau_ff=np.array([0.0, 5.0]),
                            torque_clamp=np.array([1.0, 50.0])), 0.0, Q, Q, 'T')
    assert out.tau == pytest.approx([1.0, 2.0])


def test_outside_soft_limit_never_pushes_out():
    out = law().compute(cmd('T', q=np.array([2.0, 0.0]), ignore_soft_limits=True), 0.0,
                        np.array([1.02, 0.0]), Q, 'T')
    assert out.limit_hit and out.tau[0] < 0


def test_overspeed_damped():
    out = law().compute(cmd('T', kp=np.zeros(2), kd=np.zeros(2), gravity_ff=False), 0.0, Q,
                        np.array([3.0, 0.0]), 'T')
    assert out.tau[0] == pytest.approx(-2.5)


def test_friction_ff_gain_scales_friction_only():
    class M(FakeModel):
        def friction(self, qd):
            return np.array([1.0, 1.0])
    l = ControlLaw(M(G), [10, 10], [2, 4], [-1, -1], [1, 1], [0.5, 0.1])
    a = l.compute(cmd('P', friction_ff_gain=0.0), 0.0, Q, Q, 'P')
    b = l.compute(cmd('P', friction_ff_gain=1.0), 0.0, Q, Q, 'P')
    assert b.tau - a.tau == pytest.approx([1.0, 1.0])


def test_wire_layout_round_trip():
    from arm_stack.hw.ros_client import decode_state, encode_command
    n = 3
    c = Command(mode='T', q=np.array([1.0, 2.0, 3.0]), kp=np.array([5.0, 5.0, 5.0]),
                gravity_ff=True, ignore_soft_limits=True, friction_ff_gain=0.5)
    d = encode_command(c, n)
    assert len(d) == 3 + 6 * n
    assert d[:3] == [1.0, 3.0, 0.5]
    assert d[3:6] == [1.0, 2.0, 3.0]
    assert all(math.isnan(x) for x in d[6:9])       # qd unset
    state = [0.1, 1.0, 0.0, 1.0, 0.0] + list(range(5 * n))
    s = decode_state(state, n)
    assert s['mode'] == 'T' and s['limit_hit'] and not s['fallback']
    assert s['qd'].tolist() == [3, 4, 5]
