"""Passive commissioning, autocal and the motion monitor, on fake encoders and a simulated arm."""
import argparse
from contextlib import nullcontext
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
        method = 'passive_cyclic'

        def positions(self, arm):
            return nullcontext(self)

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


def test_resting_reference_saves_relative_zero_without_a_motion_client(tmp_path, monkeypatch):
    from arm_stack.calib.reference import run as reference
    from arm_stack.cli import build_model, main
    from arm_stack.calib.autocal import has_reference, prepare_calibration
    from arm_stack.units import Units
    arm = load_arm()
    store = CalibrationStore(arm, str(tmp_path))
    original = store.write('gravity', {'link_inertials': {'some_link': {'mass_kg': 1}},
                                     'gravity_validation': {'passed': True},
                                     'commissioning': {'relative_motion': True, 'travel_deg': 10}})
    original_bytes = Path(original).read_bytes()
    counts = np.arange(arm.n) * 25000 - 55000

    class Reader:
        def require_idle(self, arm):
            pass

        def positions(self, arm):
            return nullcontext(self)

        def capture_all(self, arm):
            return counts

    monkeypatch.setattr('builtins.input', lambda _: 'yes')
    assert reference(argparse.Namespace(arm=arm.path, sim=False, calib_root=str(tmp_path)), Reader()) == 0
    cal = store.latest()
    assert cal['coordinate_frame'] == 'relative'
    assert Units(arm, cal).counts_to_rad(counts) == pytest.approx(np.zeros(arm.n))
    assert cal['link_inertials'] is None and cal['gravity_validation'] is None
    assert all(pair == [None, None] for pair in cal['soft_limits_rad'].values())
    assert cal['commissioning']['verified'] is False
    assert 'relative_motion' not in cal['commissioning']
    assert 'travel_deg' not in cal['commissioning']
    assert not has_reference(arm, cal)  # not a reference for CAD-based autocal
    assert Path(original).read_bytes() == original_bytes
    with pytest.raises(ValueError, match='relative angles'):
        build_model(arm, cal)
    with pytest.raises(ValueError, match='not the CAD home'):
        prepare_calibration(arm, cal, counts, None, 3)
    for stage in ('gravity', 'bringup', 'limits'):
        with pytest.raises(SystemExit) as error:
            main([stage, '--calib-root', str(tmp_path)])
        assert error.value.code == 2
    # Autocal routes relative references to the independent local workflow.
    monkeypatch.setattr('arm_stack.calib.relative.run', lambda args: 42)
    assert main(['autocal', '--calib-root', str(tmp_path)]) == 42


def test_resting_reference_does_not_save_on_cancel_or_movement(tmp_path, monkeypatch):
    from arm_stack.calib.reference import run as reference
    args = argparse.Namespace(arm=load_arm().path, sim=False, calib_root=str(tmp_path))

    class Reader:
        def require_idle(self, arm):
            pass

        def positions(self, arm):
            return nullcontext(self)

        def capture_all(self, arm):
            raise RuntimeError('Arm moved')

    monkeypatch.setattr('builtins.input', lambda _: 'no')
    with pytest.raises(RuntimeError, match='cancelled'):
        reference(args, Reader())
    monkeypatch.setattr('builtins.input', lambda _: 'yes')
    with pytest.raises(RuntimeError, match='moved'):
        reference(args, Reader())
    assert CalibrationStore(load_arm(), str(tmp_path)).versions() == []


def test_capture_all_refuses_a_joint_settling_during_capture(monkeypatch):
    class Reader(EncoderReader):
        sample = -1

        def require_idle(self, arm):
            pass

        def upload(self, position, *args):
            if position == 0:
                self.sample += 1
            return 10000 * self.sample if position == 1 else 0

    monkeypatch.setattr('arm_stack.calib.commission.time.sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='moved during measurement'):
        Reader().capture_all(load_arm())


@pytest.mark.parametrize('problem', [None, 'enabled', 'moving', 'invalid'])
def test_passive_cyclic_capture_requires_settled_disabled_feedback(monkeypatch, problem):
    from types import SimpleNamespace
    from arm_stack.calib.commission import CyclicPositions
    arm = load_arm()
    start = np.arange(arm.n, dtype=float) * 1000
    calls = []
    def state():
        counts = start.copy()
        status = np.full(arm.n, 0x40)
        if calls:
            if problem == 'moving':
                counts[1] += 1000
            if problem == 'invalid':
                counts[1] = np.nan
            if problem == 'enabled':
                status[1] = 0x27
        calls.append(None)
        return SimpleNamespace(raw_counts=counts, status=status)
    monkeypatch.setattr('arm_stack.calib.commission.time.sleep', lambda _: None)
    positions = CyclicPositions(SimpleNamespace(state=state))
    if problem:
        with pytest.raises(RuntimeError, match={'moving': 'shoulder_roll moved', 'invalid': 'Invalid',
                                               'enabled': 'drive is enabled'}[problem]):
            positions.capture_all(arm)
    else:
        assert positions.capture_all(arm) == pytest.approx(start)
        assert len(calls) == 12


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


def test_autocal_anchors_all_encoders_from_one_pose():
    from arm_stack.calib.autocal import prepare_calibration
    from arm_stack.cli import build_model
    from arm_stack.config import empty_calibration
    from arm_stack.units import Units
    arm = load_arm()
    offsets = np.linspace(-0.05, 0.05, arm.n)
    counts = Units(arm).rad_to_counts(arm.home + offsets)
    seed, center = prepare_calibration(arm, empty_calibration(arm), counts, build_model(arm, {}), 3)
    assert center == pytest.approx(arm.home)
    assert np.array(list(seed['zero_offsets_rad'].values())) == pytest.approx(offsets, abs=5e-5)
    assert seed['commissioning']['mechanical_stops_measured'] is False
    assert seed['commissioning']['verified'] is False
    for i, name in enumerate(arm.names):
        low, high = seed['soft_limits_rad'][name]
        assert center[i] - 3 * math.pi / 180 - 1e-8 <= low <= center[i]
        assert center[i] <= high <= center[i] + 3 * math.pi / 180 + 1e-8


def test_autocal_accepts_older_zero_calibration_without_limits():
    from arm_stack.calib.autocal import prepare_calibration
    from arm_stack.cli import build_model
    from arm_stack.config import empty_calibration
    from arm_stack.units import Units
    arm = load_arm()
    previous = empty_calibration(arm)
    previous.update(version=1, joint_config_hash=arm.hash())
    counts = Units(arm).rad_to_counts(arm.home)
    seed, _ = prepare_calibration(arm, previous, counts, build_model(arm, {}), 3)
    assert seed['commissioning']['encoder_reference'] == 'existing_calibration'
    assert all(all(np.isfinite(pair)) for pair in seed['soft_limits_rad'].values())


def test_autocal_monitor_stops_on_fault_and_sustained_error():
    from types import SimpleNamespace
    from arm_stack.calib.autocal import MotionMonitor
    from arm_stack.hw.client import Command, JointState
    arm = SimpleNamespace(joints=[SimpleNamespace(name='test_joint', max_torque_nm=5)])
    state = JointState(t=0., q=np.array([0.]), qd=np.array([0.]), tau=np.array([0.]),
                       raw_counts=np.array([0.]), raw_permille=np.array([0.]),
                       status=np.array([0x1237]))
    inner = SimpleNamespace(state=lambda: state, send=lambda cmd: None)
    model = SimpleNamespace(lower=np.array([-0.1]), upper=np.array([0.1]))
    monitor = MotionMonitor(inner, arm, model)
    monitor.send(Command(q=np.array([0.08])))
    monitor.state()
    state.t = 0.31
    with pytest.raises(RuntimeError, match='tracking error'):
        monitor.state()
    state.status[:] = 0x1250
    with pytest.raises(RuntimeError, match='operation-enabled'):
        monitor.state()


def test_autocal_monitor_keeps_commands_inside_envelope():
    from types import SimpleNamespace
    from arm_stack.calib.autocal import MotionMonitor
    from arm_stack.hw.client import Command, JointState
    state = JointState(t=0., q=np.array([0.1001]), qd=np.array([0.]), tau=np.array([0.]),
                       raw_counts=np.array([0.]), raw_permille=np.array([0.]), status=np.array([0x1237]))
    sent = []
    inner = SimpleNamespace(state=lambda: state, send=sent.append)
    monitor = MotionMonitor(inner, SimpleNamespace(joints=[SimpleNamespace(max_torque_nm=5)]),
                            SimpleNamespace(lower=np.array([-0.1]), upper=np.array([0.1])))
    monitor.send(Command(q=np.array([0.1001])))
    assert sent[-1].q[0] == 0.1
    with pytest.raises(RuntimeError, match='outside'):
        monitor.send(Command(q=np.array([0.2])))


def test_autocal_sim_command_runs_without_per_joint_prompts(tmp_path, monkeypatch):
    from arm_stack.cli import main
    from arm_stack.sim import arm_sim
    original_sim = arm_sim.SimArm
    def heavier_arm(arm, calibration, seed=0):
        return original_sim(arm, calibration,
                            errors=arm_sim.SimErrors(mass_scale=np.full(arm.n, 1.25)), seed=seed)
    monkeypatch.setattr(arm_sim, 'SimArm', heavier_arm)
    def unexpected_prompt(prompt):
        raise AssertionError('Unexpected per-joint prompt: ' + prompt)
    monkeypatch.setattr('builtins.input', unexpected_prompt)
    assert main(['autocal', '--sim', '--quick', '--calib-root', str(tmp_path), '--no-log']) == 0
    calibration = CalibrationStore(load_arm(), str(tmp_path)).latest()
    assert calibration['stage'] == 'autocal'
    assert calibration['commissioning']['verified']
    assert calibration['gravity_validation']['passed']
    assert len(calibration['commissioning']['gravity_metrics']['holding_torques_nm']) > 0


def test_calibration_write_failure_preserves_previous_version(tmp_path, monkeypatch):
    import os
    store = CalibrationStore(load_arm(), str(tmp_path))
    first = store.write('zero', {})
    original = Path(first).read_bytes()
    def collide(source, destination):
        raise FileExistsError('Concurrent writer')
    monkeypatch.setattr(os, 'link', collide)
    with pytest.raises(FileExistsError):
        store.write('gravity', {})
    assert store.versions() == [first]
    assert Path(first).read_bytes() == original
    assert not list(Path(store.dir).glob('.calib_*'))


def test_autocal_abort_preserves_calibration(tmp_path, monkeypatch):
    from arm_stack.calib import autocal
    from arm_stack.cli import main
    def abort(*args, **kwargs):
        raise RuntimeError('Simulated drive fault')
    monkeypatch.setattr(autocal, 'calibrate', abort)
    with pytest.raises(RuntimeError, match='drive fault'):
        main(['autocal', '--sim', '--calib-root', str(tmp_path)])
    assert CalibrationStore(load_arm(), str(tmp_path)).versions() == []


def test_autocal_stops_only_its_owned_process_group(monkeypatch):
    import io
    import signal
    import subprocess
    from arm_stack.calib.autocal import ArmLaunch
    signals = []
    class Process:
        pid = 12345
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if len(signals) == 1:
                raise subprocess.TimeoutExpired('owned launch', timeout)
            self.returncode = 0
    owned = ArmLaunch.__new__(ArmLaunch)
    owned.process = Process()
    owned.stream = io.StringIO()
    monkeypatch.setattr('os.killpg', lambda pid, sig: signals.append((pid, sig)))
    owned.close()
    assert signals == [(12345, signal.SIGINT), (12345, signal.SIGTERM)]
    assert owned.stream.closed
