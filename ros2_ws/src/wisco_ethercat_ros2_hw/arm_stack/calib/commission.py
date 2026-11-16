"""Passive commissioning over read-only EtherCAT SDO uploads, with ROS stopped."""

import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from ..config import CalibrationStore, load_arm


class EncoderReader:
    def command(self, *args):
        result = subprocess.run(['ethercat', *args], capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or 'EtherCAT command failed')
        return result.stdout

    def require_idle(self, arm):
        master = self.command('master')
        if not re.search(r'Active:\s+no\b', master):
            raise RuntimeError('Stop ros2 launch before commissioning; the master is active.')
        count = re.search(r'Slaves:\s+(\d+)', master)
        if not count or int(count.group(1)) != arm.n:
            raise RuntimeError('Expected exactly %d responding drives.' % arm.n)
        for joint in arm.joints:
            if self.upload(joint.bus_position, 'uint16', '0x6041') & 4:
                raise RuntimeError('%s is enabled. Disable the drives before commissioning.' % joint.name)

    def upload(self, position, kind, index):
        output = self.command('upload', '-p', str(position), '-t', kind, index, '0')
        return int(output.split()[-1])

    def capture(self, arm, joint):
        self.require_idle(arm)
        values = []
        for _ in range(20):
            values.append(self.upload(joint.bus_position, 'int32', '0x6064'))
            time.sleep(0.025)
        self.require_idle(arm)
        if np.ptp(values) > joint.counts_per_rev * 0.2 / 360:
            raise RuntimeError('%s moved during measurement; hold it stationary and retry.' % joint.name)
        return float(np.mean(values))


def angle(prompt):
    while True:
        try:
            value = float(input(prompt))
            if math.isfinite(value):
                return math.radians(value)
        except ValueError:
            pass
        print('Enter a finite angle in degrees relative to the robot model zero pose.')


def fit_encoder(count_a, angle_a, count_b, angle_b):
    if abs(angle_b - angle_a) < math.radians(5):
        raise ValueError('The two reference angles must differ by at least 5 degrees.')
    slope = (count_b - count_a) / (angle_b - angle_a)
    if not math.isfinite(slope) or abs(count_b - count_a) < 10:
        raise ValueError('Insufficient encoder movement. Check the bus position and joint mapping.')
    counts_per_rev = round(abs(slope) * 2 * math.pi)
    direction = 1 if slope > 0 else -1
    zero = direction * count_a * 2 * math.pi / counts_per_rev - angle_a
    return direction, counts_per_rev, zero


def atomic_text(path, text):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.chmod(temporary, path.stat().st_mode & 0o777)
    temporary.replace(path)


def run(args, reader=None):
    if not args.write_config:
        raise RuntimeError('Use arm commission --write-config to save measured encoder scales and directions.')
    if args.sim:
        raise RuntimeError('arm commission reads physical SDOs. Use arm all --sim for simulation.')
    arm = load_arm(args.arm)
    reader = reader or EncoderReader()
    reader.require_idle(arm)
    print('Support the arm and keep the drives disabled. This command only reads SDOs.')
    print('Measure physical joint angles relative to the URDF zero pose with an angle reference.')
    print('For each joint, support the other joints and move only the named joint.')
    original = Path(arm.path).read_text()
    updated = original
    zeros, limits, references = {}, {}, {}
    for joint in arm.joints:
        print('\n%s, EtherCAT bus position %d' % (joint.name, joint.bus_position))
        a = angle('Place at reference angle A, then enter its measured degrees: ')
        count_a = reader.capture(arm, joint)
        b = angle('Place at reference angle B (at least 5 deg away), enter measured degrees: ')
        count_b = reader.capture(arm, joint)
        direction, cpr, zero = fit_encoder(count_a, a, count_b, b)
        input('Return to reference angle A and press Enter to verify repeatability: ')
        repeat = reader.capture(arm, joint)
        if abs(repeat - count_a) * 360 / cpr > 0.2:
            raise RuntimeError('Reference A repeatability exceeds 0.2 degrees; calibration was not saved.')
        print('Measured direction %+d, counts/output revolution %d, zero %.4f deg' %
              (direction, cpr, math.degrees(zero)))
        input('Manually place at the LOWER usable endpoint, allowing mechanical/cable clearance. Enter: ')
        low = direction * reader.capture(arm, joint) * 2 * math.pi / cpr - zero
        input('Manually place at the UPPER usable endpoint, allowing mechanical/cable clearance. Enter: ')
        high = direction * reader.capture(arm, joint) * 2 * math.pi / cpr - zero
        inset = math.radians(3)
        if high - low <= 2 * inset:
            raise RuntimeError('Endpoints are reversed or less than 6 degrees apart; calibration was not saved.')
        limits[joint.name] = [low + inset, high - inset]
        zeros[joint.name] = zero
        references[joint.name] = {'angles_rad': [a, b], 'counts': [count_a, count_b],
                                  'direction': direction, 'counts_per_rev': cpr}
        print('Software limits %.2f to %.2f degrees (3 deg inside the measured endpoints).' %
              tuple(math.degrees(v) for v in limits[joint.name]))
        input('Return this joint to a supported pose INSIDE those software limits, then press Enter: ')
        rest = direction * reader.capture(arm, joint) * 2 * math.pi / cpr - zero
        if not limits[joint.name][0] <= rest <= limits[joint.name][1]:
            raise RuntimeError('%s is outside its selected software limits; calibration was not saved.' % joint.name)
        pattern = re.compile(r'(^[ \t]*-[ \t]+name:[ \t]*' + re.escape(joint.name) +
                             r'\b.*?)(?=^[ \t]*-[ \t]+name:|^\S|\Z)',
                             re.M | re.S)
        def replace(match):
            block = match.group(0)
            for key, value in [('direction', direction), ('counts_per_rev', cpr)]:
                block, count = re.subn(r'(^[ \t]+' + key + r':[ \t]*)[^\s#]+',
                                       lambda m: m.group(1) + str(value), block, flags=re.M)
                if count != 1:
                    raise RuntimeError('Expected one %s field for %s' % (key, joint.name))
            return block
        updated, count = pattern.subn(replace, updated)
        if count != 1:
            raise RuntimeError('Cannot locate joint %s in the configuration.' % joint.name)
    reader.require_idle(arm)
    if Path(arm.path).read_text() != original:
        raise RuntimeError('Joint configuration changed during measurement. Calibration was not saved.')
    atomic_text(arm.path, updated)
    try:
        calibrated_arm = load_arm(args.arm)
        path = CalibrationStore(calibrated_arm, args.calib_root).write('commission', {
            'zero_offsets_rad': zeros, 'soft_limits_rad': limits,
            'commissioning': {'method': 'passive_sdo', 'references': references},
            # Recommissioning invalidates coordinate-dependent dynamics.
            'link_inertials': None, 'friction': None, 'kt_output_nm_per_a': None,
            'kinematic_overlay': None, 'latency_ms': None,
            'gravity_validation': None,
        })
    except BaseException:
        atomic_text(arm.path, original)
        raise
    print('\nSaved measured calibration to %s' % path)
    print('Start arm01.launch.py enable_motion:=true for powered position holding and checks.')
    return 0
