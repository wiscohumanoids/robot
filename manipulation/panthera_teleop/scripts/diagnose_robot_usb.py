#!/usr/bin/env python3
"""Read USB topology and power status; never configure devices or open motors."""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def attr(path, name):
    try:
        return (path / name).read_text().strip()
    except OSError:
        return None


def max_power(path):
    """Read the first configuration descriptor even for unconfigured devices."""
    try:
        data = (path / 'descriptors').read_bytes()
    except OSError:
        return None
    offset = 0
    while offset + 2 <= len(data):
        length, kind = data[offset:offset + 2]
        if length < 2 or offset + length > len(data):
            break
        if kind == 2 and length >= 9:
            # The devices in this diagnostic are all USB 2.0 high-speed devices.
            return data[offset + 8] * 2
        offset += length
    return None


class Control(ctypes.Structure):
    _fields_ = [('request_type', ctypes.c_uint8), ('request', ctypes.c_uint8),
                ('value', ctypes.c_uint16), ('index', ctypes.c_uint16),
                ('length', ctypes.c_uint16), ('timeout', ctypes.c_uint32),
                ('data', ctypes.c_void_p)]


def hub_status(path):
    """Standard and class GET_STATUS requests only. Requires USB device access."""
    dev = f"/dev/bus/usb/{int(attr(path, 'busnum')):03d}/{int(attr(path, 'devnum')):03d}"
    fd = os.open(dev, os.O_RDWR)
    try:
        results = []
        for request_type, length in ((0x80, 2), (0xa0, 4)):
            buf = ctypes.create_string_buffer(length)
            ctrl = Control(request_type, 0, 0, 0, length, 1000, ctypes.addressof(buf))
            ioctl = (3 << 30) | (ctypes.sizeof(Control) << 16) | (ord('U') << 8)
            fcntl.ioctl(fd, ioctl, ctrl)
            results.append(int.from_bytes(buf.raw, 'little'))
        return dict(device_status_hex=hex(results[0]), currently_self_powered=bool(results[0] & 1),
                    hub_status_hex=hex(results[1]), local_power_lost=bool(results[1] & 1),
                    overcurrent=bool(results[1] & 2))
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hub-status', action='store_true',
                        help='Read live hub GET_STATUS (normally requires sudo)')
    args = parser.parse_args()
    devices = []
    for path in sorted(Path('/sys/bus/usb/devices').iterdir()):
        vendor, product = attr(path, 'idVendor'), attr(path, 'idProduct')
        if (vendor, product) not in (('caf1', 'ffff'), ('14cd', '8601'), ('05e3', '0610'), ('32e4', '9230')):
            continue
        resolved = path.resolve()
        pci = next((p for p in reversed(resolved.parts) if p.startswith('0000:')), None)
        ttys = sorted((tty.name for tty in Path('/sys/class/tty').glob('ttyACM*')
                       if resolved in tty.resolve().parents),
                      key=lambda name: int(name.removeprefix('ttyACM')))
        item = dict(usb_path=path.name, pci_controller=pci, identity=f'{vendor}:{product}',
                    product=attr(path, 'product'), configuration=attr(path, 'bConfigurationValue'),
                    requested_current_ma=max_power(path), serial_ports=ttys,
                    autosuspend=attr(path, 'power/control'))
        if args.hub_status and vendor in ('14cd', '05e3'):
            try:
                item['live_power_status'] = hub_status(path)
            except OSError as exc:
                item['live_power_status'] = dict(error=str(exc))
        devices.append(item)
    result = dict(saved_arm_mapping=json.loads((ROOT / 'config/robot-configs/arm-mapping.json').read_text()),
                  devices=devices)
    journal = subprocess.run(['journalctl', '-k', '-b', '--no-pager', '-o', 'short-iso'],
                             capture_output=True, text=True)
    result['recent_power_errors'] = [line for line in journal.stdout.splitlines()
                                     if 'insufficient available bus power' in line][-6:]
    result['power_error_note'] = ('These are historical messages from this boot. '
                                  'Use configuration and serial_ports to assess the current connection.')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
