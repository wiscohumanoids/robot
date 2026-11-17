"""HiPNUC serial protocol: frame sync, CRC check and the HI91 IMU packet. No ROS.

Follows HiPNUC's own decoder (github.com/hipnuc/products, c/hipnuc/hipnuc_decoder.c and hipnuc_data.h, Apache-2.0):

    frame    5A A5 | length u16 | crc u16 | payload (length bytes, first byte = packet tag)
    crc      CRC-16/XMODEM (poly 0x1021, init 0, no reflection, no final XOR) over the 4 bytes before the CRC field
             (sync + length) and then the payload
    numbers  little-endian

HI91 (tag 0x91, 76-byte payload), offsets in the payload:

    0  tag u8          1  main_status u16     3  temperature i8 (deg C)    4  pressure f32 (Pa)
    8  system_time u32 (ms)   12  acc f32[3] (G)   24  gyr f32[3] (deg/s)   36  mag f32[3] (uT)
    48 roll, pitch, yaw f32 (deg)   60  quat f32[4] (w, x, y, z; device body to navigation frame)

Everything here is in the device's output axes (set by its COORD setting: 0 = x right, y forward, z up, the factory
setting; 4 = x forward, y left, z up). Rotating into the robot's axes is the caller's job (imu_math.py).
"""

import struct
from dataclasses import dataclass

SYNC = b'\x5a\xa5'
HEADER = 6
MAX_PAYLOAD = 506                   # HIPNUC_FRAME_MAX 512 - header
TAG_HI91 = 0x91
HI91_SIZE = 76
G = 9.80665                         # the device reports acceleration in standard gravities

# main_status bits (hipnuc_data.h); a set convergence bit is a warning
STATUS_COORD_MASK = 0x7             # output coordinate setting: 0 or 4
STATUS_GYRO_BIAS_NOT_CONVERGED = 1 << 3
STATUS_MAG_DISTURBANCE = 1 << 4
STATUS_ACC_SATURATED = 1 << 5
STATUS_GYR_SATURATED = 1 << 6
STATUS_ATTITUDE_NOT_CONVERGED = 1 << 7
STATUS_STATIC = 1 << 9
STATUS_MAG_AIDING = 1 << 10


def _crc_table():
    """CRC-16/XMODEM lookup table."""
    table = []
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
        table.append(crc & 0xFFFF)
    return table


_TABLE = _crc_table()


def crc16(data, crc=0):
    """CRC-16/XMODEM of `data`, continuing from `crc`."""
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _TABLE[((crc >> 8) ^ b) & 0xFF]
    return crc


@dataclass
class Hi91:
    """One HI91 sample, SI units, device output axes."""
    status: int
    temperature_c: float
    pressure_pa: float
    system_time_ms: int
    acc: tuple                      # m/s^2, specific force (at rest: +9.81 along the device's up axis)
    gyr: tuple                      # rad/s
    mag: tuple                      # tesla
    euler_rad: tuple                # roll, pitch, yaw
    quat_wxyz: tuple                # device body to navigation frame

    @property
    def coord(self):
        """The device's output coordinate setting (0: x right, y forward, z up; 4: x forward, y left, z up)."""
        return self.status & STATUS_COORD_MASK

    def warnings(self):
        """Status warnings, as short strings."""
        flags = ((STATUS_GYRO_BIAS_NOT_CONVERGED, 'gyro bias not converged'),
                 (STATUS_ATTITUDE_NOT_CONVERGED, 'attitude not converged'),
                 (STATUS_ACC_SATURATED, 'accelerometer saturated'),
                 (STATUS_GYR_SATURATED, 'gyroscope saturated'),
                 (STATUS_MAG_DISTURBANCE, 'magnetic disturbance'))
        return [text for bit, text in flags if self.status & bit]


def decode_hi91(payload):
    """An HI91 payload (tag byte first) as Hi91."""
    if len(payload) != HI91_SIZE or payload[0] != TAG_HI91:
        raise ValueError('not an HI91 payload')
    status, temp, pressure, t_ms = struct.unpack_from('<HbfI', payload, 1)
    f = struct.unpack_from('<16f', payload, 12)
    deg = 3.141592653589793 / 180.0
    return Hi91(status=status, temperature_c=float(temp), pressure_pa=pressure, system_time_ms=t_ms,
                acc=tuple(v * G for v in f[0:3]), gyr=tuple(v * deg for v in f[3:6]),
                mag=tuple(v * 1e-6 for v in f[6:9]), euler_rad=tuple(v * deg for v in f[9:12]),
                quat_wxyz=tuple(f[12:16]))


def encode_frame(payload):
    """A complete frame around `payload` (for tests and the bench simulator)."""
    head = SYNC + struct.pack('<H', len(payload))
    return head + struct.pack('<H', crc16(payload, crc16(head))) + bytes(payload)


def encode_hi91(acc_g=(0.0, 0.0, 1.0), gyr_dps=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0), status=0,
                euler_deg=(0.0, 0.0, 0.0), mag_ut=(0.0, 0.0, 0.0), temperature_c=25, pressure_pa=101325.0,
                system_time_ms=0):
    """An HI91 frame in device units (G, deg/s, uT, deg), as the board sends it."""
    payload = struct.pack('<BHbfI', TAG_HI91, status, temperature_c, pressure_pa, system_time_ms)
    payload += struct.pack('<16f', *acc_g, *gyr_dps, *mag_ut, *euler_deg, *quat_wxyz)
    return encode_frame(payload)


class Decoder:
    """Incremental frame decoder: feed it bytes as they arrive, get complete packets back.

    Resynchronizes on the next 5A A5 after garbage, an impossible length or a CRC error, and counts what it dropped.
    """

    def __init__(self):
        """Empty buffer, zeroed counters."""
        self.buf = bytearray()
        self.frames = 0             # good frames
        self.crc_errors = 0
        self.unknown = 0            # good frames of a packet type this driver does not decode
        self.skipped_bytes = 0

    def feed(self, data):
        """Add bytes; return the decoded packets (Hi91) completed by them, in order."""
        self.buf += data
        out = []
        while True:
            start = self.buf.find(SYNC)
            if start < 0:
                keep = 1 if self.buf[-1:] == SYNC[:1] else 0       # a sync byte may be split across reads
                self.skipped_bytes += len(self.buf) - keep
                del self.buf[:len(self.buf) - keep]
                return out
            if start:
                self.skipped_bytes += start
                del self.buf[:start]
            if len(self.buf) < HEADER:
                return out
            length, crc = struct.unpack_from('<HH', self.buf, 2)
            if length == 0 or length > MAX_PAYLOAD:
                self.skipped_bytes += 2
                del self.buf[:2]
                continue
            if len(self.buf) < HEADER + length:
                return out
            payload = bytes(self.buf[HEADER:HEADER + length])
            if crc16(payload, crc16(self.buf[:4])) != crc:
                self.crc_errors += 1
                self.skipped_bytes += 2
                del self.buf[:2]
                continue
            del self.buf[:HEADER + length]
            self.frames += 1
            if payload[0] == TAG_HI91 and length == HI91_SIZE:
                out.append(decode_hi91(payload))
            else:
                self.unknown += 1
