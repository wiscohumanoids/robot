#!/usr/bin/env python3
"""Measure a mocap pose stream before using it: rate, dropouts, and the noise of a still rigid body.

    ros2 run wisco_ethercat_ros2_hw mocap_check.py /vrpn_mocap/<rigid body>/pose [--seconds 10]

Keep the rigid body still while it runs. Reports the message rate and its jitter, gaps longer than three frame
periods (occlusions, network drops), the position spread as 3 sigma per axis in mm, the orientation spread in
degrees, and the largest jump between frames. Pass: rate within 10 % of the expected rate (--rate), no gap longer
than 50 ms, position 3 sigma under --max-noise-mm (default 0.5 mm). The arm calibrations cap their accuracy at the
length-bar error; this check is the quick look at the live stream.
"""
import argparse
import sys
import time

import numpy as np


def summarize(stamps, positions, quats, rate=None, max_noise_mm=0.5):
    """Statistics and a pass flag from arrival times (s), positions (m, N x 3) and quaternions (x, y, z, w)."""
    stamps, positions, quats = np.asarray(stamps), np.asarray(positions), np.asarray(quats)
    if len(stamps) < 10:
        return {'samples': len(stamps), 'pass': False, 'reason': 'fewer than 10 messages'}
    dt = np.diff(stamps)
    hz = 1.0 / np.median(dt)
    gaps = dt[dt > 3.0 / hz]
    noise_mm = 3.0 * positions.std(axis=0) * 1000.0
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    q *= np.sign(q @ q[0])[:, None]                       # one hemisphere
    angles = 2 * np.degrees(np.arccos(np.clip(np.abs(q @ q.mean(axis=0) / np.linalg.norm(q.mean(axis=0))), 0, 1)))
    jump_mm = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).max() * 1000.0)
    out = {'samples': int(len(stamps)), 'rate_hz': float(hz), 'jitter_ms': float(dt.std() * 1000.0),
           'gaps': int(len(gaps)), 'longest_gap_ms': float(dt.max() * 1000.0),
           'noise_3sigma_mm': noise_mm.round(3).tolist(), 'orientation_3sigma_deg': float(3 * angles.std()),
           'max_jump_mm': jump_mm}
    reasons = []
    if rate and abs(hz - rate) > 0.1 * rate:
        reasons.append(f'rate {hz:.1f} Hz, expected {rate:g}')
    if dt.max() > 0.05:
        reasons.append(f'gap of {dt.max() * 1000:.0f} ms')
    if noise_mm.max() > max_noise_mm:
        reasons.append(f'position noise {noise_mm.max():.2f} mm (3 sigma) > {max_noise_mm}')
    out['pass'] = not reasons
    out['reason'] = '; '.join(reasons) or 'ok'
    return out


def main():
    """Collect for --seconds, then print the summary; exit 1 on failure."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('topic')
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--rate', type=float, help='expected frame rate, Hz')
    ap.add_argument('--max-noise-mm', type=float, default=0.5)
    a = ap.parse_args()
    import rclpy
    from geometry_msgs.msg import PoseStamped
    rclpy.init()
    node = rclpy.create_node('mocap_check')
    stamps, positions, quats = [], [], []

    def on_pose(m):
        """Record the arrival time, position and orientation."""
        p, q = m.pose.position, m.pose.orientation
        stamps.append(time.monotonic())
        positions.append((p.x, p.y, p.z))
        quats.append((q.x, q.y, q.z, q.w))

    node.create_subscription(PoseStamped, a.topic, on_pose, 50)
    end = time.monotonic() + a.seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    s = summarize(stamps, positions, quats, a.rate, a.max_noise_mm)
    for k, v in s.items():
        print(f'{k:24s} {v}')
    sys.exit(0 if s['pass'] else 1)


if __name__ == '__main__':
    main()
