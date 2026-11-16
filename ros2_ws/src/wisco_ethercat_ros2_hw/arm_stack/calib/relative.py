"""Bounded position checks and empirical holding-torque identification, without CAD angles."""
import copy
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import yaml

from .autocal import MotionMonitor
from .commission import EncoderReader
from .common import DEG
from ..config import CalibrationStore, load_arm
from ..control.motion import hold_and_average, move_to
from ..hw.client import Command
from ..hw.launch import ArmLaunch
from ..units import Units


SWEEP_SPEED = 12 * DEG


def fit_local(center, radius, q, tau, test_q, test_tau, resolution):
    """Fit all cross-joint derivatives; independent poses must validate the fit."""
    center, radius = np.asarray(center), np.asarray(radius)
    q, tau = np.asarray(q), np.asarray(tau)
    test_q, test_tau = np.asarray(test_q), np.asarray(test_tau)
    if not all(np.all(np.isfinite(a)) for a in (center, radius, q, tau, test_q, test_tau)):
        raise RuntimeError('Nonfinite local gravity measurements.')
    if np.any(radius <= 0) or np.any(np.ptp(q, axis=0) < radius * 0.5):
        raise RuntimeError('Insufficient actual joint movement for identification.')
    x = np.column_stack([np.ones(len(q)), (q-center)/radius])
    coefficients, _, rank, sv = np.linalg.lstsq(x, tau, rcond=None)
    if rank != len(center)+1 or sv[0]/sv[-1] > 100:
        raise RuntimeError('Local gravity measurements do not identify every joint derivative.')
    prediction = np.column_stack([np.ones(len(test_q)), (test_q-center)/radius]) @ coefficients
    residual = np.max(np.abs(prediction-test_tau), axis=0)
    peak = np.max(np.abs(np.vstack([tau, test_tau])), axis=0)
    allowed = np.maximum(0.05*peak, 2*np.asarray(resolution))
    if np.any(residual > allowed):
        raise RuntimeError('Held-out gravity residual exceeds allowance: %s Nm versus %s Nm.' %
                           (np.round(residual, 4).tolist(), np.round(allowed, 4).tolist()))
    return {'kind': 'local_affine', 'center_rad': center.tolist(),
            'bias_nm': coefficients[0].tolist(),
            'jacobian_nm_per_rad': (coefficients[1:].T / radius).tolist(),
            'lower_rad': (center-radius).tolist(), 'upper_rad': (center+radius).tolist(),
            'max_residual_nm': residual.tolist(), 'allowed_residual_nm': allowed.tolist(),
            'validated': True}


def measure(client, q, radius, resolution, speed):
    passes = []
    for sign in (-1, 1):
        move_to(client, q + sign*0.1*radius, max_speed=speed, settle=0.3, gravity_ff=False)
        move_to(client, q, max_speed=speed, settle=1.0, gravity_ff=False)
        sample = hold_and_average(client, 0.6, q_hold=q, gravity_ff=False)
        if np.any(np.abs(sample['q']-q) > 0.15*DEG) or np.any(np.abs(sample['qd']) > 0.02):
            raise RuntimeError('A joint did not reach and settle at its target; check obstruction and feedback.')
        passes.append(sample)
    # Averages alone can hide friction. Reject large approach-dependent torque.
    difference = np.abs(passes[0]['tau']-passes[1]['tau']) / 2
    scale = np.maximum(np.abs(passes[0]['tau']), np.abs(passes[1]['tau']))
    if np.any(difference > np.maximum(0.1*scale, 3*resolution)):
        raise RuntimeError('Approach-dependent holding torque is too large for a reliable gravity fit.')
    return (passes[0]['q']+passes[1]['q'])/2, (passes[0]['tau']+passes[1]['tau'])/2


def calibrate(arm, seed, center, client):
    lower = np.array([seed['soft_limits_rad'][n][0] for n in arm.names])
    upper = np.array([seed['soft_limits_rad'][n][1] for n in arm.names])
    radius = (upper-lower)/2
    speed = min(SWEEP_SPEED, *(j.max_vel_rad_s for j in arm.joints))
    client = MotionMonitor(client, arm, SimpleNamespace(lower=lower, upper=upper), command_margin=0.05*radius)
    move_to(client, center, max_speed=speed, settle=0.5, gravity_ff=False)
    for i, name in enumerate(arm.names):
        print('Checking %s in the local envelope; other joints stay held.' % name, flush=True)
        for sign in (-1, 1):
            target = center.copy()
            target[i] += sign*0.7*radius[i]
            sample = move_to(client, target, max_speed=speed, settle=0.7, gravity_ff=False)
            errors = np.abs(sample.q - target)
            worst = int(np.argmax(errors))
            if errors[worst] > 0.15 * DEG:
                raise RuntimeError(
                    f'{arm.names[worst]} did not follow during the {name} check: '
                    f'target {(target[worst]-center[worst])/DEG:+.2f} deg from start, '
                    f'measured {(sample.q[worst]-center[worst])/DEG:+.2f} deg, '
                    f'error {errors[worst]/DEG:.2f} deg (limit 0.15), '
                    f'status 0x{sample.status[worst]:04x}')
        move_to(client, center, max_speed=speed, settle=0.3, gravity_ff=False)
    rng = np.random.default_rng(41)
    # Axis samples give a well-conditioned fit; coupled held-out poses test it.
    fit_poses = [center.copy()]
    for i in range(arm.n):
        for sign in (-1, 1):
            q = center.copy()
            q[i] += sign*0.7*radius[i]
            fit_poses.append(q)
    test_poses = center + rng.uniform(-0.7, 0.7, size=(max(8, arm.n+2), arm.n))*radius
    resolution = np.abs(Units(arm, seed).nm_per_permille())
    measurements = []
    for i, q in enumerate([*fit_poses, *test_poses]):
        print('Measuring settled holding torque %d/%d.' % (i+1, len(fit_poses)+len(test_poses)), flush=True)
        measurements.append(measure(client, q, radius, resolution, speed))
    q, tau = map(np.array, zip(*measurements))
    n = len(fit_poses)
    # Validity ends at the measured +/-0.7-radius range, not the larger motion guard.
    result = fit_local(center, 0.7*radius, q[:n], tau[:n], q[n:], tau[n:], resolution)
    move_to(client, center, max_speed=speed, settle=0.5, gravity_ff=False)
    return result, {'measured_poses_rad': q.tolist(), 'holding_torques_nm': tau.tolist(),
                    'fit_count': n, 'held_out_count': len(test_poses)}


def record_sweep(arm, center, lower, upper, client):
    radius = (upper-lower)/2
    monitor = MotionMonitor(client, arm, SimpleNamespace(lower=lower, upper=upper),
                            command_margin=0.05*radius, check_tracking=False)
    speed = min(SWEEP_SPEED, *(j.max_vel_rad_s for j in arm.joints))
    with tempfile.NamedTemporaryFile(mode='w', dir=Path.cwd(), prefix='arm_sweep_',
                                     suffix='.csv', newline='', delete=False) as output:
        print(f'Recording to {output.name}', flush=True)
        writer = csv.writer(output)
        writer.writerow(['controller_time_s', 'sweeping_joint'] + [
            f'{n}_{field}' for field in ('command_deg', 'measured_deg', 'velocity_deg_s',
                                         'torque_Nm', 'statusword') for n in arm.names])

        class Recorder:
            def add(self, state, q_cmd, qd_cmd):
                writer.writerow([state.t, name, *np.degrees(monitor.command), *np.degrees(state.q),
                                 *np.degrees(state.qd), *state.tau, *state.status])

        trace = Recorder()
        for i, name in enumerate(arm.names):
            print(f'Recording {i+1}/{arm.n}: {name}', flush=True)
            for fraction in (-0.7, 0.7, 0.0):
                target = center.copy()
                target[i] += fraction * radius[i]
                move_to(monitor, target, max_speed=speed, settle=0.3, gravity_ff=False, trace=trace)
                output.flush()
        print(f'Completed all {arm.n} commanded sweeps. Saved {output.name}', flush=True)


def run(args):
    if args.sim:
        raise ValueError('Use the relative calibration tests offline; this command reads the physical reference.')
    if args.force or args.quick:
        raise ValueError('Relative hardware calibration does not accept --force or --quick.')
    arm = load_arm(args.arm)
    store = CalibrationStore(arm, args.calib_root)
    previous = store.latest()
    joint_hash = arm.hash()
    reanchor = previous.get('joint_config_hash') != joint_hash
    metadata = previous.get('commissioning') or {}
    if reanchor and (metadata.get('method') != 'passive_relative_reference' or metadata.get('relative_motion')):
        raise ValueError('Joint configuration changed since reference capture; record arm reference again.')
    reader = EncoderReader()
    reader.require_idle_master(arm)
    travel = args.travel_deg * DEG
    recording = getattr(args, 'record_sweep', False)
    if recording:
        print('Recording commanded sweeps and measured feedback for every joint.')
    else:
        print('Relative autocal: smooth sweeps up to 12 deg/s, then settled holding-torque measurements.')
    if reanchor and not recording:
        print('Joint configuration changed. The current supported pose will supply the new relative zero on PASS.')
    print('Maximum travel is +/- %.2f degrees per joint about the current resting pose.' % (travel/DEG))
    print('This is NOT a hard-stop search. All joints need clearance both ways, including cables and tool.')
    if input('Confirm the arm is supported and clearance exists. Type yes to disable drives for capture, then enable holding: ').strip().lower() != 'yes':
        raise RuntimeError('Cancelled before motor enabling.')
    reader.disable_for_verification(arm)
    with reader.positions(arm) as positions:
        counts = positions.capture_all(arm)
    seed = copy.deepcopy(previous)
    if reanchor:
        seed['zero_offsets_rad'] = dict(zip(arm.names, Units(arm).counts_to_rad(counts).tolist()))
        seed['commissioning'].update(reference_counts=counts.tolist(), reference_pose_rad=[0.]*arm.n)
    center = Units(arm, seed).counts_to_rad(counts)
    lower, upper = center-travel, center+travel
    if (previous.get('commissioning') or {}).get('relative_motion'):
        old = np.array([previous['soft_limits_rad'][n] for n in arm.names])
        lower, upper = np.maximum(lower, old[:, 0]), np.minimum(upper, old[:, 1])
        if np.any(center-lower < 0.5*DEG) or np.any(upper-center < 0.5*DEG):
            raise RuntimeError('Current pose is too close to existing local bounds. No motors were enabled.')
        travel = float(np.min(np.minimum(center-lower, upper-center)))
        lower, upper = center-travel, center+travel
    seed.update(joint_config_hash=joint_hash, version=int(previous['version'])+1, stage='relative_autocal_pending',
                soft_limits_rad={n: [float(lower[i]), float(upper[i])] for i,n in enumerate(arm.names)},
                link_inertials=None, local_gravity=None, gravity_validation=None, friction=None, kinematic_overlay=None,
                commissioning=dict(seed.get('commissioning') or {}, relative_motion=True, verified=False))
    launch, client = None, None
    with tempfile.TemporaryDirectory(prefix='arm_relative_') as directory:
        file = Path(directory)/'pending.yaml'
        file.write_text(yaml.safe_dump(seed, sort_keys=False))
        try:
            launch = ArmLaunch(arm, file, Path(directory)/'launch.log')
            client = launch.connect(arm, seed)
            at_hold = client.state().q.copy()
            client.send(Command(mode='P', q=at_hold.copy(), gravity_ff=False))
            print('Position holding started. Remove any external load/support before measurements.', flush=True)
            input('Once holding is steady and you are clear of the arm, press Enter to start the sweeps: ')
            current = client.state().q
            drift = (current - center) / DEG
            outside = np.flatnonzero(~np.isfinite(drift) | (np.abs(drift) > 0.5))
            if outside.size and not recording:
                details = ', '.join(
                    f'{arm.names[i]} {drift[i]:+.2f}° from capture '
                    f'({(at_hold[i]-center[i])/DEG:+.2f}° at first powered reading)'
                    for i in outside)
                raise RuntimeError(f'Starting feedback differs by over 0.5° or is invalid: {details}. '
                                   'Sweep has not started.')
            if recording:
                record_sweep(arm, center, lower, upper, client)
                print('Holding position. Support the arm before Ctrl-C disables control.', flush=True)
                return launch.process.wait()
            fit, measurements = calibrate(arm, seed, center, client)
            if arm.hash() != joint_hash or store.latest() != previous:
                raise RuntimeError('Configuration changed during calibration; results were not saved.')
            limits = {n: [fit['lower_rad'][i], fit['upper_rad'][i]] for i,n in enumerate(arm.names)}
            path = store.write('relative_autocal', {
                'zero_offsets_rad': seed['zero_offsets_rad'], 'soft_limits_rad': limits, 'local_gravity': fit,
                'link_inertials': None, 'friction': None, 'kinematic_overlay': None,
                'pace': None, 'validation': None, 'latency_ms': None, 'joint_stiffness_nm_per_rad': None,
                'gravity_validation': {'passed': True, 'kind': 'local_affine', 'max_residual_nm': fit['max_residual_nm']},
                'commissioning': dict(seed['commissioning'], method='relative_local_identification',
                                      verified=True, local_checks_passed=True,
                                      range_kind='local_test_envelope', travel_deg=float(travel / DEG),
                                      start_counts=counts.tolist(), start_pose_rad=center.tolist(),
                                      gravity_metrics=measurements, mechanical_stops_measured=False),
            })
            print('PASS: saved %s. Local gravity loads on the next launch; CAD coordinates remain unknown.' % path, flush=True)
            print('Holding position. Support the arm before Ctrl-C disables control.', flush=True)
            returncode = launch.process.wait()
            if returncode:
                raise RuntimeError(f'Arm launch exited while holding, code {returncode}')
            return 0
        except Exception as error:
            print('Relative autocal stopped: %s' % error, flush=True)
            if client is not None and launch is not None and launch.process.poll() is None:
                try:
                    client.send(Command(mode='P', q=client.state().q.copy(), gravity_ff=False))
                except RuntimeError:
                    pass
                print('Controller stays holding. Support the arm before Ctrl-C.', flush=True)
                launch.process.wait()
            raise
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                if launch is not None:
                    launch.close()
