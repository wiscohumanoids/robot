"""HiPNUC driver logic without hardware: frames and CRC, stream recovery, the HI91 layout, the mount rotation,
board-clock stamps, the bench check, and the config's mount against robots/wisco_v1/model/robot_numbers.yaml.
"""
import math
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml

from wisco_hipnuc_imu import imu_math, protocol
from wisco_hipnuc_imu.imu_check import summarize, up_axis
from wisco_hipnuc_imu.stamping import DeviceClock

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[2]
MOUNT = yaml.safe_load((PKG / 'config' / 'hipnuc.yaml').read_text())['imu']['ros__parameters']['mount_wxyz']


# ---- protocol -----------------------------------------------------------------------------------------------------

def test_crc_is_xmodem():
    """CRC-16/XMODEM check value: '123456789' -> 0x31C3."""
    assert protocol.crc16(b'123456789') == 0x31C3


def test_frame_layout_matches_the_protocol():
    """Sync, little-endian length counting the payload only, 82-byte HI91 frame, CRC over header and payload."""
    f = protocol.encode_hi91()
    assert f[:2] == b'\x5a\xa5' and struct.unpack_from('<H', f, 2)[0] == 76 and len(f) == 82 and f[6] == 0x91
    assert struct.unpack_from('<H', f, 4)[0] == protocol.crc16(f[6:], protocol.crc16(f[:4]))


def test_hi91_round_trip_and_units():
    """Device units in, SI out: G -> m/s^2, deg/s -> rad/s, uT -> T, deg -> rad, quaternion untouched."""
    q = (0.9238795, 0.0, 0.3826834, 0.0)
    f = protocol.encode_hi91(acc_g=(0.0, -1.0, 0.0), gyr_dps=(90.0, 0.0, -45.0), quat_wxyz=q, status=0x0088,
                             euler_deg=(1.0, 2.0, 3.0), mag_ut=(20.0, 0.0, -40.0), temperature_c=-5,
                             system_time_ms=123456)
    (s,) = protocol.Decoder().feed(f)
    assert s.acc == pytest.approx((0.0, -protocol.G, 0.0))
    assert s.gyr == pytest.approx((math.pi / 2, 0.0, -math.pi / 4))
    assert s.mag == pytest.approx((20e-6, 0.0, -40e-6))
    assert s.euler_rad == pytest.approx(tuple(math.radians(v) for v in (1, 2, 3)))
    assert s.quat_wxyz == pytest.approx(q, abs=1e-6)
    assert (s.temperature_c, s.system_time_ms, s.coord) == (-5.0, 123456, 0)
    assert s.warnings() == ['gyro bias not converged', 'attitude not converged']


def test_stream_split_garbage_and_corruption():
    """Frames split byte by byte, noise between frames, a corrupted frame and a fake sync are all handled."""
    good = [protocol.encode_hi91(system_time_ms=i) for i in range(4)]
    bad = bytearray(protocol.encode_hi91(system_time_ms=99))
    bad[40] ^= 0xFF
    stream = b'\x00\x13\x5a' + good[0] + bytes(bad) + b'\x5a\xa5\xff\xff' + good[1] + b'junk' + good[2] + good[3]
    d = protocol.Decoder()
    out = []
    for b in stream:                          # worst case: one byte per read
        out += d.feed(bytes([b]))
    assert [s.system_time_ms for s in out] == [0, 1, 2, 3]
    assert d.crc_errors == 1 and d.frames == 4 and d.skipped_bytes > 0


def test_other_packets_are_counted_not_decoded():
    """A valid frame of another packet type (e.g. HI81) is skipped and counted."""
    d = protocol.Decoder()
    assert d.feed(protocol.encode_frame(bytes([0x81]) + bytes(103)) + protocol.encode_hi91()) != []
    assert d.unknown == 1


# ---- mount and orientation ----------------------------------------------------------------------------------------

def test_mount_matches_robot_numbers():
    """The driver's mount is the CAD mount in robot_numbers.yaml (and the MJCF imu site)."""
    numbers = yaml.safe_load((REPO / 'robots/wisco_v1/model/robot_numbers.yaml').read_text())
    assert MOUNT == pytest.approx(numbers['sensors']['imu']['quat_wxyz'])


def test_mount_axes_are_the_documented_ones():
    """Board x -> robot right (-y), y -> down (-z), z -> forward (+x)."""
    c = imu_math.MountConverter(MOUNT)
    assert c.vector([1, 0, 0]) == pytest.approx([0, -1, 0], abs=1e-12)
    assert c.vector([0, 1, 0]) == pytest.approx([0, 0, -1], abs=1e-12)
    assert c.vector([0, 0, 1]) == pytest.approx([1, 0, 0], abs=1e-12)


def test_robot_upright_reads_gravity_on_plus_z():
    """Board upright on the pelvis: its y axis points down, so it reads -1 G on y; in imu_link that is +g on z."""
    (s,) = protocol.Decoder().feed(protocol.encode_hi91(acc_g=(0.0, -1.0, 0.0)))
    assert imu_math.MountConverter(MOUNT).vector(s.acc) == pytest.approx([0, 0, protocol.G], abs=1e-9)


def _q_nav_device_for(q_world_link, mount, heading=0.0):
    """The device orientation the board would report for a given imu_link orientation (and nav heading)."""
    return imu_math.quat_mul(imu_math.quat_mul(imu_math.yaw_quat(heading), q_world_link), mount)


@pytest.mark.parametrize('roll,pitch', [(0.0, 0.0), (0.2, 0.0), (0.0, -0.3), (0.15, 0.25)])
def test_orientation_keeps_tilt_and_zeroes_heading(roll, pitch):
    """Whatever the board's heading, the published orientation has the robot's true tilt and starts at yaw 0."""
    mount = np.asarray(MOUNT)
    q_true = imu_math.quat_mul(imu_math.yaw_quat(0.0),
                               imu_math.quat_mul([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0],
                                                 [math.cos(roll / 2), math.sin(roll / 2), 0, 0]))
    c = imu_math.MountConverter(mount, zero_yaw=True)
    q = c.orientation(_q_nav_device_for(q_true, mount, heading=1.234))
    R_true, R = imu_math.quat_to_mat(q_true), imu_math.quat_to_mat(q)
    assert R[2] == pytest.approx(R_true[2], abs=1e-9)                       # same gravity direction in the body
    assert imu_math.yaw_of(q) == pytest.approx(0.0, abs=1e-9)
    later = imu_math.quat_mul(imu_math.yaw_quat(0.5), q_true)              # the robot turns 0.5 rad left
    assert imu_math.yaw_of(c.orientation(_q_nav_device_for(later, mount, heading=1.234))) == pytest.approx(0.5)


def test_zero_quaternion_means_no_orientation():
    """A zero quaternion (not available) gives None, so the node flags orientation as unknown."""
    assert imu_math.MountConverter(MOUNT).orientation((0, 0, 0, 0)) is None


# ---- timestamps ---------------------------------------------------------------------------------------------------

def test_stamps_follow_board_spacing_not_arrival_jitter():
    """Arrival jitter of up to 4 ms disappears; stamps are never later than arrival."""
    rng = np.random.default_rng(0)
    clock, t0 = DeviceClock(), 1000.0
    stamps, arrivals = [], []
    for k in range(400):                                  # 2 s at 200 Hz
        board_ms = 50_000 + 5 * k
        arrival = t0 + 0.005 * k + 0.001 + rng.uniform(0, 0.004)
        arrivals.append(arrival)
        stamps.append(clock.stamp(board_ms, arrival))
    sent = t0 + 0.005 * np.arange(400) + 0.001                 # when each frame really left the board
    assert np.abs(np.array(stamps[50:]) - sent[50:]).max() < 5e-5        # within 0.05 ms once settled
    assert (np.array(arrivals) - sent).max() > 3e-3                      # arrival alone was off by up to 4 ms
    assert all(s <= a + 1e-12 for s, a in zip(stamps, arrivals))


def test_board_reboot_and_wrap():
    """A board reboot restarts the mapping; the 24 h wrap of the board clock is followed."""
    c = DeviceClock()
    c.stamp(86_399_990, 10.0)
    assert c.stamp(5, 10.015) == pytest.approx(10.015, abs=1e-6)          # wrapped: +15 ms, continuous
    assert c.resets == 0
    c.stamp(3_000_000, 10.02)                                              # jump: reboot or gap
    assert c.resets == 1


# ---- bench check --------------------------------------------------------------------------------------------------

def _still(n=400, acc_g=(0.0, -1.0, 0.0), noise=0.002, seed=0):
    """Decoded samples of a still board."""
    rng = np.random.default_rng(seed)
    d = protocol.Decoder()
    out = []
    for k in range(n):
        out += d.feed(protocol.encode_hi91(acc_g=tuple(np.array(acc_g) + rng.normal(0, noise, 3)),
                                           gyr_dps=tuple(rng.normal(0, 0.05, 3)), system_time_ms=5 * k))
    return out, d


def test_bench_check_passes_a_correct_mount_and_flags_a_wrong_one():
    """Upright board on the robot: passes with the CAD mount; a board turned the other way fails the tilt check."""
    samples, d = _still()
    ok = summarize(samples, 2.0, d, MOUNT, rate=200)
    assert ok['pass'], ok['reason']
    assert ok['up_axis_device'] == '-y' and ok['tilt_from_up_deg'] < 1.0
    flipped, d2 = _still(acc_g=(0.0, 1.0, 0.0))
    bad = summarize(flipped, 2.0, d2, MOUNT, rate=200)
    assert not bad['pass'] and 'mount_wxyz' in bad['reason']


def test_bench_check_flags_rate_and_too_few_samples():
    """Half the expected rate fails; nothing received fails with a hint."""
    samples, d = _still()
    assert 'rate' in summarize(samples, 4.0, d, rate=200)['reason']
    assert not summarize([], 1.0, protocol.Decoder())['pass']
    assert up_axis(np.array([0.1, 0.0, -9.8])) == '-z'
