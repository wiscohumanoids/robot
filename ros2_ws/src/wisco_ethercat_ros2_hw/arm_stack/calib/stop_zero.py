"""Exercise mechanical-stop encoder registration on the simulated EtherCAT arm."""

import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .common import DEG, StageContext, safe_home
from .limits import find_stop, path_gravity
from ..config import CalibrationStore, load_arm
from ..control.motion import move_to


def stop_references(arm):
    """Stop angles per joint for arm01: CAD contact angles, and J1's configured range."""
    if arm.arm != 'arm01':
        raise ValueError('The CAD contact capture is for arm01.')
    # The CAD capture sits with the arm's description: robots/arm01/cad/ (config/ is its sibling).
    capture = json.loads((Path(arm.path).resolve().parent.parent / 'cad' / 'arm01_capture.json').read_text())
    stops = np.deg2rad([capture['contact_onset_deg'][n] for n in arm.names])
    limit = ET.parse(arm.urdf).find("./joint[@name='shoulder_pitch']/limit")
    stops[arm.names.index('shoulder_pitch')] = [float(limit.attrib[k]) for k in ('lower', 'upper')]
    return stops


def check_encoder(args, reader=None):
    """Compare all drive position readings across a small hand movement."""
    from .commission import EncoderReader

    arm = load_arm(args.arm)
    if args.sim or args.check_encoder not in arm.names:
        raise ValueError('Choose a physical joint from ' + ', '.join(arm.names))
    reader = reader or EncoderReader()
    reader.disable_for_verification(arm)
    with reader.positions(arm) as positions:
        input('With the supported arm stationary, press Enter for the baseline: ')
        before = positions.capture_all(arm)
        input(f'Move only {args.check_encoder} a few degrees by hand, hold it there, then press Enter: ')
        after = positions.capture_all(arm)
    for joint, first, last in zip(arm.joints, before, after):
        print(f'bus {joint.bus_position} {joint.name}: 0x6064 {first:.1f} -> {last:.1f}, '
              f'delta {last-first:+.1f} counts')
    return 0


def verify_stop(args, reader=None):
    """Measure a hand-seated stop independently of the saved encoder zero."""
    from .commission import EncoderReader, angle, fit_encoder

    if args.sim:
        raise ValueError('Stop verification measures the physical arm. For simulation use --stop-zero.')
    arm = load_arm(args.arm)
    stops = stop_references(arm)
    if args.verify_stop not in arm.names:
        raise ValueError('Choose a joint from ' + ', '.join(arm.names))
    i = arm.names.index(args.verify_stop)
    joint = arm.joints[i]
    reader = reader or EncoderReader()
    reader.disable_for_verification(arm)
    with reader.positions(arm) as positions:
        start_relative = joint.name == 'shoulder_pitch'
        print(f'{joint.name}: support the arm with drives disabled and hold the other joints '
              'in the intended homing posture throughout the check.')
        side = input('Which physical stop are you verifying (lower/upper)? ').strip().lower()
        if side not in ('lower', 'upper'):
            raise ValueError('Choose lower or upper.')
        if start_relative:
            input('Set J1 at the starting position and press Enter to record it as 0 deg: ')
            ca = positions.capture(arm, joint)
            direction, cpr = joint.direction, joint.counts_per_rev
            zero = direction * ca * 2 * np.pi / cpr
            print(f'J1 start recorded. Configured range {stops[i, 0]/DEG:+.0f} to {stops[i, 1]/DEG:+.0f} deg.')
            a = b = 0.0
            return_pose = 'the starting position (0 deg)'
        else:
            print('Use an independent angle reference aligned to the URDF axis.')
            a = angle('Place at a known interior angle A. Enter measured URDF degrees: ')
            ca = positions.capture(arm, joint)
            b = angle('Move to a second interior angle B at least 5 degrees away. Enter measured degrees: ')
            cb = positions.capture(arm, joint)
            direction, cpr, zero = fit_encoder(ca, a, cb, b)
            if direction != joint.direction:
                raise ValueError('Measured encoder direction disagrees with the joint configuration.')
            return_pose = 'interior angle A'
        hits = []
        for attempt in range(2):
            input(f'Approach {side} stop by hand ({attempt+1}/2). Confirm the intended stop faces '
                  'touch, with cables and stand clear. Press Enter when stationary: ')
            counts = positions.capture(arm, joint)
            if start_relative and abs(counts - ca) <= 2:
                raise RuntimeError(f'{joint.name} bus {joint.bus_position} encoder shows no measurable movement '
                                   f'(start {ca:.1f}, stop {counts:.1f} counts). '
                                   f'Run arm autocal --check-encoder {joint.name} to compare all drives.')
            hits.append(direction * counts * 2 * np.pi / cpr - zero)
            if attempt == 0:
                input(f'Move back to {return_pose} and press Enter: ')
                repeat = positions.capture(arm, joint)
                if abs(repeat - ca) * 360 / cpr > 0.2:
                    raise ValueError('Reference position did not repeat within 0.2 deg.')
    reference = stops[i, int(side == 'upper')]
    label = 'configured endpoint' if start_relative else 'conditional CAD angle'
    print(f'{joint.name} {side}: hand-seated readings {hits[0]/DEG:.2f}, {hits[1]/DEG:.2f} deg; '
          f'{label} {reference/DEG:.2f} deg; '
          f'mean difference {(np.mean(hits)-reference)/DEG:+.2f} deg.')
    if abs(hits[1] - hits[0]) > 0.2 * DEG:
        raise ValueError('Contact measurements did not repeat within 0.2 deg.')
    if (side == 'lower' and max(hits) >= min(a, b)) or (side == 'upper' and min(hits) <= max(a, b)):
        raise ValueError('Selected stop is on the wrong side of the measured interior angles.')
    print('Report only. No calibration or motor settings written.')
    return 0


def run(args):
    """`arm autocal --stop-zero --sim`: zero each joint against its modelled mechanical stop on the simulated arm.
    """
    if not args.sim:
        raise ValueError('Powered stop-zero requires arm-specific physical stop angles, '
                         'approach paths and contact loads. arm01.urdf lists operating ranges '
                         'and marks J1 stops unverified. Use --sim to exercise the modeled stops.')
    if args.force or args.travel_deg is not None:
        raise ValueError('Stop-zero uses physical references and accepts neither --force nor --travel-deg.')
    from ..cli import build_model
    from ..sim.arm_sim import SimArm, SimErrors

    arm = load_arm(args.arm)
    model = build_model(arm, {})
    home = safe_home(model)
    stops = stop_references(arm)
    start = home.copy()
    start[arm.names.index('shoulder_pitch')] = 0.0
    print('J1 starts at 0 deg with its configured operating range.')
    # Encoder zeros are recovered from the stops, independently of the starting pose.
    bias = np.random.default_rng(args.seed).uniform(-2 * DEG, 2 * DEG, arm.n)
    errors = SimErrors(zero_offsets_rad=bias, hard_stop_margin_rad=np.zeros(arm.n))
    client = SimArm(arm, {}, errors, q0=start, seed=args.seed)
    client.stop_lower, client.stop_upper = stops.T.copy()
    start_encoders = client.state().q.copy()
    ctx = StageContext(arm, client, {}, model, sim=True, quick=args.quick)
    root = args.calib_root
    if root and Path(root).resolve() == (Path(arm.path).resolve().parent.parent / 'calibration').resolve():
        raise ValueError('Use a simulation calibration directory.')
    if args.fresh or not root:
        root = tempfile.mkdtemp(prefix='arm_stop_zero_')
    store = CalibrationStore(arm, root)
    previous = store.latest()
    zeros, contacts = {}, {}
    peak = np.array([joint.peak_torque_nm for joint in arm.joints])
    for i, name in enumerate(arm.names):
        # The loaded shoulder needs the normal approach speed for the contact dwell.
        approach_pose = start if name == 'shoulder_pitch' else home
        ctx.quick = args.quick and name != 'shoulder_pitch'
        lower, upper = stops[i]
        direction = -1 if approach_pose[i] - lower <= upper - approach_pose[i] else 1
        reference = float(lower if direction == -1 else upper)
        clamp = path_gravity(model, approach_pose, i, reference) + 0.08 * peak
        print(f'{name}: approaching simulated stop at {reference/DEG:+.2f} deg', flush=True)
        hits = []
        for _ in range(2):
            hit = find_stop(ctx, i, direction, approach_pose, clamp, reference=reference)
            if hit is None or abs(hit - reference) > 4 * DEG:
                raise RuntimeError(f'{name}: no contact near the expected reference. Nothing saved.')
            hits.append(hit)
        if abs(hits[0] - hits[1]) > 0.25 * DEG:
            raise RuntimeError(f'{name}: stop did not repeat within 0.25 deg. Nothing saved.')
        zeros[name] = float(np.mean(hits) - reference)
        contacts[name] = {'reference_rad': reference, 'encoder_angles_rad': hits}
        move_to(client, home, max_speed=0.3)
    start_error = start_encoders - np.array([zeros[n] for n in arm.names]) - start
    worst = int(np.argmax(np.abs(start_error)))
    print(f'Independent simulated start-pose check: maximum error '
          f'{abs(start_error[worst])/DEG:.3f} deg ({arm.names[worst]})')
    if abs(start_error[worst]) > 0.5 * DEG:
        raise RuntimeError('Simulated zero error exceeds 0.5 deg. Nothing saved.')
    if store.latest() != previous:
        raise RuntimeError('Calibration changed during stop-zero. Nothing saved.')
    path = store.write('stop_zero', {
        'coordinate_frame': 'urdf', 'zero_offsets_rad': zeros,
        'soft_limits_rad': {name: [None, None] for name in arm.names},
        'commissioning': {'method': 'simulated_stop_zero', 'references': contacts,
                          'relative_motion': False, 'local_checks_passed': False,
                          'range_kind': 'unmeasured', 'verified': False},
        'local_gravity': None, 'link_inertials': None, 'gravity_validation': None,
        'kinematic_overlay': None, 'friction': None, 'joint_stiffness_nm_per_rad': None,
        'pace': None, 'latency_ms': None, 'validation': None,
    })
    print(f'Saved simulated stop-referenced encoder offsets to {path}')
    return 0
