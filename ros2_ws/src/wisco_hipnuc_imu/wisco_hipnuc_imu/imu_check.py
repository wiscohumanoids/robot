"""Bench check of the HiPNUC IMU straight on its serial port (stop the driver first; no ROS needed).

    ros2 run wisco_hipnuc_imu imu_check [--port /dev/hipnuc_imu] [--baud 230400] [--seconds 10] [--mounted]

Hold the board (or the robot) still. Reports the frame rate, CRC errors, the board's coordinate setting and status
warnings, the gravity reading in the board's axes, and the gyro / accelerometer noise (use these for the driver's
covariances). With --mounted (board on the robot, robot standing upright or hung level), it also rotates the reading
into imu_link with the config's mount and checks that gravity lands on +z: that is the check that the printed board
axes match the CAD (robots/wisco_v1/hardware_observations.md, "IMU orientation").

Pass: rate within 10 % of --rate, no CRC errors, |gravity| within 2 % of 9.81, attitude converged; with --mounted, the
rotated gravity within --max-tilt-deg of +z. Exit status 1 on failure.
"""
import argparse
import os
import sys
import time

import numpy as np
import yaml

from wisco_hipnuc_imu.imu_math import MountConverter
from wisco_hipnuc_imu.protocol import Decoder

AXES = ('+x', '-x', '+y', '-y', '+z', '-z')


def up_axis(acc):
    """The device axis that points up (specific force at rest points up), e.g. '-y'."""
    i = int(np.argmax(np.abs(acc)))
    return ('+' if acc[i] > 0 else '-') + 'xyz'[i]


def summarize(samples, elapsed, decoder, mount_wxyz=None, rate=None, max_tilt_deg=5.0):
    """Statistics and pass/fail from decoded samples."""
    if len(samples) < 10:
        return {'pass': False, 'reason': f'only {len(samples)} samples: check the port, baud rate and HI91 output'}
    acc = np.array([s.acc for s in samples])
    gyr = np.array([s.gyr for s in samples])
    g = acc.mean(axis=0)
    out = {'samples': len(samples), 'rate_hz': round(len(samples) / elapsed, 1), 'crc_errors': decoder.crc_errors,
           'coord': samples[-1].coord, 'warnings': sorted({w for s in samples[-20:] for w in s.warnings()}),
           'gravity_device_axes': np.round(g, 3).tolist(), 'up_axis_device': up_axis(g),
           'gravity_norm': round(float(np.linalg.norm(g)), 3),
           'gyro_std_rad_s': np.round(gyr.std(axis=0), 5).tolist(), 'acc_std_m_s2': np.round(acc.std(axis=0), 4).tolist(),
           'gyro_mean_rad_s': np.round(gyr.mean(axis=0), 5).tolist(), 'temperature_c': samples[-1].temperature_c}
    reasons = []
    if rate and abs(out['rate_hz'] - rate) > 0.1 * rate:
        reasons.append(f"rate {out['rate_hz']} Hz, expected {rate:g}")
    if decoder.crc_errors:
        reasons.append(f'{decoder.crc_errors} CRC errors (cable, baud rate)')
    if abs(out['gravity_norm'] - 9.81) > 0.02 * 9.81:
        reasons.append(f"|gravity| {out['gravity_norm']} m/s^2 (moving, or accelerometer scale)")
    if 'attitude not converged' in out['warnings']:
        reasons.append('attitude not converged (keep it still longer)')
    if mount_wxyz is not None:
        g_link = MountConverter(mount_wxyz).vector(g)
        tilt = float(np.degrees(np.arccos(np.clip(g_link[2] / np.linalg.norm(g_link), -1, 1))))
        out['gravity_imu_link'] = np.round(g_link, 3).tolist()
        out['tilt_from_up_deg'] = round(tilt, 2)
        if tilt > max_tilt_deg:
            reasons.append(f'rotated gravity {tilt:.1f} deg from +z: mount_wxyz or the board COORD setting is wrong '
                           f"(board up axis {out['up_axis_device']})")
    out['pass'] = not reasons
    out['reason'] = '; '.join(reasons) or 'ok'
    return out


def main():
    """Read the port for --seconds and print the summary."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', default='/dev/hipnuc_imu')
    ap.add_argument('--baud', type=int, default=230400)
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--rate', type=float, default=200.0, help='expected output rate, Hz')
    ap.add_argument('--mounted', action='store_true', help='board on the robot, robot upright: check the mount')
    ap.add_argument('--config', default='', help="driver config with mount_wxyz (default: the package's hipnuc.yaml)")
    ap.add_argument('--max-tilt-deg', type=float, default=5.0)
    a = ap.parse_args()
    import serial
    mount = None
    if a.mounted:
        if not a.config:
            source = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'hipnuc.yaml')
            if os.path.isfile(source):
                a.config = source
            else:                                   # installed layout: share/<pkg>/config
                from ament_index_python.packages import get_package_share_directory
                a.config = os.path.join(get_package_share_directory('wisco_hipnuc_imu'), 'config', 'hipnuc.yaml')
        mount = yaml.safe_load(open(a.config))['imu']['ros__parameters']['mount_wxyz']
    dec, samples = Decoder(), []
    with serial.Serial(a.port, a.baud, timeout=0.05) as s:
        t0 = time.monotonic()
        while time.monotonic() - t0 < a.seconds:
            samples += dec.feed(s.read(s.in_waiting or 1))
        elapsed = time.monotonic() - t0
    res = summarize(samples, elapsed, dec, mount, a.rate, a.max_tilt_deg)
    for k, v in res.items():
        print(f'{k:22s} {v}')
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
