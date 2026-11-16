"""One reference pose, bounded automatic position checks, and a local gravity fit."""

import copy
from pathlib import Path
import tempfile

import numpy as np
import yaml

from .commission import EncoderReader
from .common import DEG, StageContext
from . import gravity
from ..config import CalibrationStore, load_arm
from ..control.motion import move_to
from ..hw.client import Command
from ..hw.launch import ArmLaunch
from ..units import Units


def has_reference(arm, calibration):
    if calibration.get('coordinate_frame') == 'relative':
        return False
    if not calibration.get('version') or calibration.get('joint_config_hash') != arm.hash():
        return False
    try:
        zeros = [float(calibration['zero_offsets_rad'][n]) for n in arm.names]
        return bool(np.all(np.isfinite(zeros)))
    except (KeyError, TypeError, ValueError):
        return False


def prepare_calibration(arm, previous, counts, nominal_model, travel_deg):
    """Anchor one pose and reserve a bounded envelope; scales remain configured."""
    if not np.isfinite(travel_deg) or not 0.5 <= travel_deg <= 10:
        raise ValueError('--travel-deg must be between 0.5 and 10 degrees for local autocal.')
    if np.asarray(counts).shape != (arm.n,) or not np.all(np.isfinite(counts)):
        raise ValueError('Invalid encoder readings.')
    if previous.get('coordinate_frame') == 'relative':
        raise ValueError('A hanging relative reference is not the CAD home pose. '
                         'Register the physical pose before running CAD-based autocal.')
    calibrated = has_reference(arm, previous)
    if calibrated:
        center = Units(arm, previous).counts_to_rad(counts)
        zeros = previous['zero_offsets_rad']
        limits = previous.get('soft_limits_rad') or {}
        lower, upper = nominal_model.lower.copy(), nominal_model.upper.copy()
        for i, name in enumerate(arm.names):
            low, high = limits.get(name) or [None, None]
            if low is not None:
                lower[i] = low
            if high is not None:
                upper[i] = high
    else:
        if arm.home is None:
            raise ValueError('Configure home_pose_rad to define a physical reference pose first.')
        center = arm.home.copy()
        units = Units(arm)
        offset = units.counts_to_rad(counts) - center
        zeros = dict(zip(arm.names, offset.tolist()))
        lower, upper = nominal_model.lower.copy(), nominal_model.upper.copy()
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError('Autocal needs finite configured motion bounds.')
    if np.any(center < lower - 1e-6) or np.any(center > upper + 1e-6):
        raise ValueError('Starting pose is outside the existing motion bounds.')
    lower = np.maximum(lower, center - travel_deg * DEG)
    upper = np.minimum(upper, center + travel_deg * DEG)
    if np.any(upper - lower < 0.4 * DEG):
        raise ValueError('Insufficient room around the starting pose for local calibration.')
    seed = copy.deepcopy(previous)
    seed.update(version=int(previous.get('version') or 0) + 1, arm_serial=arm.arm,
                joint_config_hash=arm.hash(), stage='autocal_pending',
                zero_offsets_rad=zeros,
                soft_limits_rad={n: [float(lower[i]), float(upper[i])] for i, n in enumerate(arm.names)},
                link_inertials=None, gravity_validation=None, friction=None, kinematic_overlay=None,
                commissioning={'method': 'automatic_local', 'verified': False,
                               'encoder_reference': 'existing_calibration' if calibrated else 'operator_confirmed_home',
                               'direction_and_scale': 'configured', 'torque_scale': 'configured_or_previous_kt',
                               'range_kind': 'local_test_envelope', 'travel_deg': travel_deg,
                               'reference_counts': np.asarray(counts).tolist(),
                               'reference_pose_rad': center.tolist(), 'mechanical_stops_measured': False})
    return seed, center


class MotionMonitor:
    """Stop automated motion on faults, excessive tracking error, or saturation."""

    def __init__(self, inner, arm, model, command_margin=0.0, *, check_tracking=True):
        self.check_tracking = check_tracking
        self.inner, self.arm, self.model = inner, arm, model
        self.command_lower = model.lower + command_margin
        self.command_upper = model.upper - command_margin
        self.command = None
        self.bad_since = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def state(self):
        state = self.inner.state()
        if not np.all((state.status & 0x6f) == 0x27):
            raise RuntimeError('Autocal stopped because a drive lost operation-enabled state.')
        if not all(np.all(np.isfinite(v)) for v in (state.q, state.qd, state.tau)):
            raise RuntimeError('Autocal stopped because drive feedback is invalid.')
        if np.any(state.q < self.model.lower - 0.5 * DEG) or np.any(state.q > self.model.upper + 0.5 * DEG):
            raise RuntimeError('Autocal stopped because a joint left the local motion envelope. '
                               'Positions (deg): %s; bounds: %s to %s' %
                               (np.round(state.q / DEG, 2).tolist(),
                                np.round(self.model.lower / DEG, 2).tolist(),
                                np.round(self.model.upper / DEG, 2).tolist()))
        limit = np.array([j.max_torque_nm for j in self.arm.joints])
        saturated = np.abs(state.tau) >= 0.95 * limit
        tracking = np.zeros_like(saturated)
        if self.check_tracking and self.command is not None:
            tracking = np.abs(state.q - self.command) > 2 * DEG
        if np.any(saturated | tracking):
            if self.bad_since is None:
                self.bad_since = state.t
            if state.t - self.bad_since > 0.3:
                details = []
                for i in np.flatnonzero(saturated | tracking):
                    name = self.arm.joints[i].name
                    if tracking[i]:
                        details.append(f'{name} tracking error: target {self.command[i]/DEG:.2f} deg, '
                                       f'measured {state.q[i]/DEG:.2f} deg')
                    if saturated[i]:
                        details.append(f'{name} torque saturation: measured {state.tau[i]:.2f} Nm, '
                                       f'limit {limit[i]:.2f} Nm')
                raise RuntimeError('Autocal stopped: ' + ', '.join(details))
        else:
            self.bad_since = None
        return state

    def send(self, command):
        if command.mode != 'P' or command.q is None:
            raise RuntimeError('Autocal only supports bounded position commands.')
        target = np.asarray(command.q, float)
        if not np.all(np.isfinite(target)) or np.any(target < self.model.lower - 0.5 * DEG) or np.any(target > self.model.upper + 0.5 * DEG):
            raise RuntimeError('Autocal target is outside its local motion envelope.')
        # A trajectory starts at measured position. Small servo overshoot at
        # an endpoint is allowed in feedback, while every command stays inside.
        target = np.clip(target, self.command_lower, self.command_upper)
        self.state()
        self.command = target.copy()
        command.q = target
        command.gravity_ff = False
        self.inner.send(command)

    def step(self, seconds=0.0):
        self.inner.step(seconds)
        return self.state()


def calibrate(arm, seed, center, client, quick=False):
    from ..cli import build_model
    model = build_model(arm, seed)
    margin = np.minimum(1 * DEG, 0.2 * (model.upper - model.lower))
    monitored = MotionMonitor(client, arm, model, command_margin=margin)
    center = np.clip(center, model.lower + margin, model.upper - margin)
    print('Establishing the interior holding pose.', flush=True)
    move_to(monitored, center, max_speed=0.05, settle=0.5, gravity_ff=False)
    for i, name in enumerate(arm.names):
        print('Checking %s; remaining joints are held by the motors.' % name, flush=True)
        for endpoint in (model.lower[i] + margin[i], model.upper[i] - margin[i]):
            target = center.copy()
            target[i] = endpoint
            state = move_to(monitored, target, max_speed=0.05, settle=0.5, gravity_ff=False)
            if abs(state.q[i] - target[i]) > 1 * DEG:
                raise RuntimeError('%s failed its local position check.' % name)
        move_to(monitored, center, max_speed=0.05, settle=0.5, gravity_ff=False)
    planning_model = model.copy()
    planning_model.lower = model.lower + margin
    planning_model.upper = model.upper - margin
    context = StageContext(arm=arm, client=monitored, model=planning_model, calibration=seed,
                           sim=hasattr(client, 'truth'), quick=quick,
                           motion_speed=0.05,
                           rng=np.random.default_rng(0))
    print('Fitting gravity inside the checked envelope, with separate validation poses.', flush=True)
    result = gravity.run(context, fit_masses=True, static=True)
    print(result.report(), flush=True)
    if not result.passed:
        raise RuntimeError('Gravity validation failed. Existing calibration files were preserved.')
    move_to(monitored, center, max_speed=0.05, settle=0.5, gravity_ff=False)
    return result


def run(args):
    from ..cli import build_model
    arm = load_arm(args.arm)
    joint_hash = arm.hash()
    if args.force:
        raise ValueError('autocal saves only passing results; --force is unsupported.')
    root = args.calib_root
    if args.sim and root is None:
        from ..cli import SIM_ROOT
        root = SIM_ROOT
    if args.sim and args.fresh:
        root = tempfile.mkdtemp(prefix='arm_autocal_sim_')
    store = CalibrationStore(arm, root)
    previous = store.latest()
    relative = previous.get('coordinate_frame') == 'relative'
    if getattr(args, 'record_sweep', False) and not relative:
        raise ValueError('--record-sweep requires a relative reference.')
    if args.travel_deg is None:
        args.travel_deg = 10.0 if relative else 3.0
    if not np.isfinite(args.travel_deg) or not 0.5 <= args.travel_deg <= 10:
        raise ValueError('--travel-deg must be between 0.5 and 10 degrees for local autocal.')
    if relative:
        from .relative import run
        return run(args)
    reader = None if args.sim else EncoderReader()
    if reader is not None:
        reader.require_idle(arm)
    valid_reference = has_reference(arm, previous)
    if not args.sim:
        if not valid_reference:
            print('Place the whole arm ONCE at the configured reference pose (degrees):')
            if arm.home is None:
                raise RuntimeError('home_pose_rad is missing from the joint configuration.')
            for name, value in zip(arm.names, arm.home):
                print('  %s %.2f' % (name, value / DEG))
            print('Use a rest/fixture for this initial pose while motor enabling is disabled.')
        print('Configured joint directions, encoder scales and torque scales will be retained.')
        print('Autocal will enable motors and test up to %.1f deg on each side within configured bounds.' % args.travel_deg)
        if input('Confirm the reference pose and clear motion area. Type yes to start: ').strip().lower() != 'yes':
            raise RuntimeError('Autocal cancelled before motor enabling.')
        with reader.positions(arm) as positions:
            counts = positions.capture_all(arm)
        client = None
    else:
        from ..sim.arm_sim import SimArm
        client = SimArm(arm, previous, seed=args.seed)
        client.place_in_jig(arm.home)
        counts = client.state().raw_counts
    if arm.hash() != joint_hash:
        raise RuntimeError('Joint configuration changed while collecting the reference pose.')
    seed, center = prepare_calibration(arm, previous, counts, build_model(arm, {}), args.travel_deg)
    seed['joint_config_hash'] = joint_hash
    launch = None
    with tempfile.TemporaryDirectory(prefix='arm_autocal_') as directory:
        calibration_file = Path(directory) / 'pending.yaml'
        calibration_file.write_text(yaml.safe_dump(seed, sort_keys=False))
        try:
            if args.sim:
                client.reload_calibration(seed)
            else:
                print('Starting the position controller.', flush=True)
                launch = ArmLaunch(arm, calibration_file, Path(directory) / 'launch.log')
                client = launch.connect(arm, seed)
                state = client.state()
                client.send(Command(mode='P', q=state.q.copy(), gravity_ff=False))
                print('Powered position holding is active. Carefully unload and remove the reference '
                      'support so the motors carry the arm during measurement.', flush=True)
                input('Once the arm holds steadily and the motion area is clear, press Enter to begin: ')
            result = calibrate(arm, seed, center, client, args.quick)
            # Stop sampling before committing the tested model. Runtime keeps the
            # initial position-hold configuration until the next normal launch.
            updates = dict(result.writes)
            metadata = dict(seed['commissioning'], verified=True,
                            local_checks_passed=True, gravity_metrics=result.metrics)
            updates.update(zero_offsets_rad=seed['zero_offsets_rad'],
                           soft_limits_rad=seed['soft_limits_rad'], commissioning=metadata,
                           friction=None, kinematic_overlay=None, latency_ms=None)
            if arm.hash() != seed['joint_config_hash']:
                raise RuntimeError('Joint configuration changed during autocal; results were not saved.')
            if store.latest() != previous:
                raise RuntimeError('Calibration store changed during autocal; results were not saved.')
            path = store.write('autocal', updates)
            print('Autocal PASS. Saved %s' % path, flush=True)
            print('The saved limits cover the tested local envelope; mechanical stops remain unmeasured.', flush=True)
            if launch is not None:
                print('Arm is holding position. Support it before Ctrl-C stops motor control. '
                      'The saved gravity fit loads on the next normal launch.', flush=True)
                returncode = launch.process.wait()
                if returncode:
                    raise RuntimeError('Arm launch exited while holding, code %d.' % returncode)
            return 0
        except Exception as error:
            print('Autocal failed: %s' % error, flush=True)
            if launch is not None and launch.process.poll() is None and client is not None:
                try:
                    client.send(Command(mode='P', q=client.state().q, gravity_ff=False))
                except RuntimeError:
                    pass
                print('The controller continues position holding. '
                      'Support the arm before Ctrl-C stops motor control.', flush=True)
                # Keep holding while the operator handles the failure.
                try:
                    launch.process.wait()
                except KeyboardInterrupt:
                    pass
            raise
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                if launch is not None:
                    launch.close()
