"""Record an arbitrary resting pose without enabling or commanding motors."""

import numpy as np

from .commission import EncoderReader
from ..config import CalibrationStore, load_arm
from ..units import Units


def run(args, reader=None):
    """`arm reference`: record the resting encoder pose as the relative zero; nothing is enabled or moved."""
    if args.sim:
        raise ValueError('arm reference reads physical encoders; it does not use the simulator.')
    arm = load_arm(args.arm)
    joint_hash = arm.hash()
    store = CalibrationStore(arm, args.calib_root)
    previous = store.latest()
    reader = reader or EncoderReader()
    reader.require_idle(arm)
    print('Leave the arm launch stopped and motor torque disabled. Electronics stay powered.')
    print('Let the arm rest steadily with tool, cables and base clear. No particular elbow angle is required.')
    print('This records relative zero only: no motor motion, CAD registration, gravity fit or travel measurement.')
    if previous.get('version'):
        print('A new reference will replace the active coordinates; earlier files remain in the history.')
    if input('Type yes to save this resting pose as relative zero: ').strip().lower() != 'yes':
        raise RuntimeError('Reference cancelled; nothing saved.')
    with reader.positions(arm) as positions:
        counts = np.asarray(positions.capture_all(arm), float)
    if counts.shape != (arm.n,) or not np.all(np.isfinite(counts)):
        raise ValueError('Invalid encoder readings; nothing saved.')
    reader.require_idle(arm)
    if arm.hash() != joint_hash or store.latest() != previous:
        raise RuntimeError('Configuration or calibration changed during capture; nothing saved.')
    # Use the configured encoder direction and output scale, with no old zero.
    zeros = Units(arm).counts_to_rad(counts)
    path = store.write('reference', {
        'coordinate_frame': 'relative', 'local_gravity': None,
        'zero_offsets_rad': dict(zip(arm.names, zeros.tolist())),
        'soft_limits_rad': {n: [None, None] for n in arm.names},
        'link_inertials': None, 'gravity_validation': None, 'friction': None,
        'kinematic_overlay': None, 'joint_stiffness_nm_per_rad': None,
        'pace': None, 'validation': None, 'latency_ms': None,
        'commissioning': {
            'method': 'passive_relative_reference', 'verified': False,
            'encoder_reference': 'arbitrary_resting_pose',
            'reference_counts': counts.tolist(), 'reference_pose_rad': [0.] * arm.n,
            'direction_and_scale': 'configured', 'mechanical_stops_measured': False,
            'range_kind': 'unmeasured', 'references': None, 'gravity_metrics': None,
            'local_checks_passed': False,
        },
    })
    print('Saved %s' % path)
    for name, count in zip(arm.names, counts):
        print('  %s: %.1f counts -> 0.00 relative degrees' % (name, count))
    print('No motors were enabled. To start position holding only:')
    print('  ros2 launch wisco_ethercat_ros2_hw arm01.launch.py enable_motion:=true')
    print('Holding latches the actual startup position; it does not move back to the saved reference.')
    print('Motion commands and CAD gravity remain disabled. Do not re-zero routinely on each startup.')
    return 0
