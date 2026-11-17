"""Relative-reference calibration: local sweeps, the gravity fit and its safety checks, on a fake arm."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from arm_stack.calib.relative import calibrate, fit_local
from arm_stack.config import load_arm
from arm_stack.hw.client import JointState


def test_local_fit_identifies_cross_joint_torque_and_rejects_bad_validation():
    rng = np.random.default_rng(12)
    center = rng.normal(size=7)
    radius = np.full(7, 0.02)
    q = center + rng.uniform(-1, 1, (40, 7))*radius
    qt = center + rng.uniform(-1, 1, (12, 7))*radius
    jac = rng.normal(size=(7, 7))*3
    bias = rng.normal(size=7)
    tau = bias+(q-center)@jac.T
    tt = bias+(qt-center)@jac.T
    fitted = fit_local(center, radius, q, tau, qt, tt, np.full(7, 0.01))
    assert np.asarray(fitted['jacobian_nm_per_rad']) == pytest.approx(jac)
    assert fitted['bias_nm'] == pytest.approx(bias)
    with pytest.raises(RuntimeError, match='residual'):
        fit_local(center, radius, q, tau, qt, tt+1, np.full(7, 0.01))
    with pytest.raises(RuntimeError, match='Insufficient actual'):
        fit_local(center, radius, np.tile(center, (40, 1)), tau, qt, tt, np.full(7, 0.01))


class FakeArm:
    """Fake arm whose holding torque is an affine function of its angles."""
    cmd_dt = 0.05

    def __init__(self, n, stuck=False):
        """Affine torque model; `stuck` makes the arm ignore commands."""
        self.q = np.zeros(n)
        self.t = 0.
        self.jac = np.eye(n)*2 + 0.3
        self.bias = np.linspace(0.1, 0.8, n)
        self.stuck = stuck

    def state(self):
        """State with the torque from the affine model."""
        return JointState(t=self.t, q=self.q.copy(), qd=self.q*0,
                          tau=self.bias+self.jac@self.q, raw_counts=self.q*0,
                          raw_permille=self.q*0, status=np.full(len(self.q), 0x27))

    def send(self, command):
        """Accept a Mode P command without gravity feed-forward."""
        assert command.mode == 'P' and not command.gravity_ff
        if not self.stuck:
            self.q = command.q.copy()

    def step(self, seconds):
        """Advance time."""
        self.t += seconds
        return self.state()


def test_complete_relative_identification_without_cad_angles():
    arm = load_arm()
    fake = FakeArm(arm.n)
    seed = {'soft_limits_rad': {n: [-0.017, 0.017] for n in arm.names}}
    fit, data = calibrate(arm, seed, np.zeros(arm.n), fake)
    assert np.asarray(fit['jacobian_nm_per_rad']) == pytest.approx(fake.jac)
    assert fit['bias_nm'] == pytest.approx(fake.bias)
    assert data['held_out_count'] >= 8
    assert fake.q == pytest.approx(np.zeros(arm.n))
    with pytest.raises(RuntimeError, match='did not follow'):
        calibrate(arm, seed, np.zeros(arm.n), FakeArm(arm.n, stuck=True))


def test_position_check_identifies_drifting_held_joint():
    arm = load_arm()

    class DriftingArm(FakeArm):
        def send(self, command):
            super().send(command)
            if abs(self.q[0]) > np.deg2rad(0.3):
                self.q[1] = np.deg2rad(0.4)

    seed = {'soft_limits_rad': {n: [-0.017, 0.017] for n in arm.names}}
    with pytest.raises(RuntimeError, match=f'{arm.names[1]} did not follow during the {arm.names[0]} check'):
        calibrate(arm, seed, np.zeros(arm.n), DriftingArm(arm.n))


@pytest.mark.parametrize('stale_reference', [False, True])
def test_relative_cli_saves_only_validated_results(tmp_path, monkeypatch, stale_reference):
    from pathlib import Path
    import yaml
    from arm_stack.calib import relative
    from arm_stack.cli import main
    from arm_stack.config import CalibrationStore
    arm = load_arm()
    store = CalibrationStore(arm, str(tmp_path))
    first = store.write('reference', {'coordinate_frame': 'relative',
                                    'zero_offsets_rad': {n: 0. for n in arm.names}})
    counts = np.zeros(arm.n)
    if stale_reference:
        old = store.latest()
        old.update(joint_config_hash='before-firmware-update',
                   commissioning={'method': 'passive_relative_reference'})
        Path(first).write_text(yaml.safe_dump(old))
        counts[:] = 2048
    fake = FakeArm(arm.n)
    fake.close = lambda: None
    closed = []

    class Reader:
        def require_idle_master(self, arm):
            pass
        def positions(self, arm):
            return nullcontext(SimpleNamespace(capture_all=lambda arm: counts))

        def disable_for_verification(self, arm):
            pass

    class Launch:
        def __init__(self, *args):
            self.process = SimpleNamespace(wait=lambda: 0, poll=lambda: None)
        def connect(self, arm, seed):
            return fake
        def close(self):
            closed.append(True)

    monkeypatch.setattr(relative, 'EncoderReader', Reader)
    monkeypatch.setattr(relative, 'ArmLaunch', Launch)
    monkeypatch.setattr('builtins.input', lambda _: 'yes')
    assert main(['autocal', '--calib-root', str(tmp_path)]) == 0
    cal = store.latest()
    assert cal['coordinate_frame'] == 'relative'
    if stale_reference:
        assert cal['commissioning']['reference_counts'] == counts.tolist()
        assert cal['joint_config_hash'] == arm.hash()
        assert cal['zero_offsets_rad'][arm.names[0]] == pytest.approx(2048 * 2*np.pi/arm.joints[0].counts_per_rev)
    assert cal['soft_limits_rad'][arm.names[0]] == pytest.approx(np.deg2rad([-7, 7]))
    assert cal['local_gravity']['validated']
    assert cal['commissioning']['relative_motion']
    assert cal['commissioning']['range_kind'] == 'local_test_envelope'
    assert cal['commissioning']['start_counts'] == counts.tolist()
    assert cal['link_inertials'] is None
    assert len(store.versions()) == 2 and closed
    assert first in store.versions()
    latest = store.latest_path()
    def fail(*args):
        raise RuntimeError('Simulated feedback fault')
    monkeypatch.setattr(relative, 'calibrate', fail)
    with pytest.raises(RuntimeError, match='feedback fault'):
        main(['autocal', '--calib-root', str(tmp_path)])
    assert store.latest_path() == latest
