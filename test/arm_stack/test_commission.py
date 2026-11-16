import argparse
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from arm_stack.calib.commission import EncoderReader, fit_encoder, run
from arm_stack.config import CalibrationStore, load_arm
from arm_stack.hw.ros_client import decode_state


def test_passive_commission_writes_measured_config_and_calibration(tmp_path, monkeypatch):
    arm = load_arm()
    table = yaml.safe_load(Path(arm.path).read_text())
    table['joints'] = table['joints'][:1]
    path = tmp_path / 'arm.yaml'
    path.write_text(yaml.safe_dump(table, sort_keys=False))
    args = argparse.Namespace(arm=str(path), write_config=True, sim=False,
                              calib_root=str(tmp_path / 'calib'))
    answers = iter(['0', '30', '', '', '', ''])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    cpr, zero = 131072, 0.25
    angles = [0, 30, 0, -20, 60, 0]

    class Reader:
        values = iter([-cpr * (math.radians(a) + zero) / (2 * math.pi) for a in angles])

        def require_idle(self, arm):
            pass

        def capture(self, arm, joint):
            return next(self.values)

    assert run(args, Reader()) == 0
    measured = load_arm(str(path))
    assert measured.joints[0].direction == -1
    assert measured.joints[0].counts_per_rev == cpr
    calibration = CalibrationStore(measured, args.calib_root).latest()
    name = measured.names[0]
    assert calibration['zero_offsets_rad'][name] == pytest.approx(zero)
    assert np.degrees(calibration['soft_limits_rad'][name]) == pytest.approx([-17, 57])
    assert calibration['joint_config_hash'] == measured.hash()
    assert calibration['link_inertials'] is None


def test_commission_refuses_active_master_before_reading_drives():
    class Reader(EncoderReader):
        def command(self, *args):
            assert args == ('master',)
            return 'Active: yes\nSlaves: 7'
    with pytest.raises(RuntimeError, match='master is active'):
        Reader().require_idle(load_arm())


@pytest.mark.parametrize('counts,angles', [([0, 1], [0, 0.5]), ([10, 10], [0, 30])])
def test_encoder_fit_refuses_insufficient_movement(counts, angles):
    with pytest.raises(ValueError):
        fit_encoder(counts[0], math.radians(angles[0]), counts[1], math.radians(angles[1]))


def test_extended_state_carries_real_status():
    values = [0., 0., 1., 0., 0.] + [0.] * 10 + [0x1250, 0x1237, 8, 8]
    assert decode_state(values, 2)['status'].tolist() == [0x1250, 0x1237]


def test_static_mass_fit_recovers_holding_torque_with_wrong_prior():
    from arm_stack.calib.gravity import fit_static_parameters
    from arm_stack.cli import build_model
    arm = load_arm()
    model = build_model(arm, {})
    truth = model.copy()
    inertials = {name: {'mass_kg': float(mass * 1.4),
                       'com_m': (moment / mass).tolist()}
                 for name, mass, moment in zip(model.link_names(), model.masses(),
                                               model.first_moments().reshape(-1, 3))}
    truth.apply_calibration({'link_inertials': inertials})
    rng = np.random.default_rng(7)
    poses = rng.uniform(model.lower + 0.1, model.upper - 0.1, size=(60, arm.n))
    torques = np.array([truth.gravity(q) for q in poses])
    masses, h, _, _ = fit_static_parameters(model, poses[:45], torques[:45], np.ones(arm.n))
    fitted = {name: {'mass_kg': float(masses[k]), 'com_m': (h[3*k:3*k+3] / masses[k]).tolist()}
              for k, name in enumerate(model.link_names())}
    model.apply_calibration({'link_inertials': fitted})
    error = np.array([model.gravity(q) - truth.gravity(q) for q in poses[45:]])
    assert np.max(np.abs(error)) < 0.1
