#!/usr/bin/env python3
"""Explicitly configure the known robot hub's children for this connection only.

This bypasses automatic USB power-budget selection, not electrical protection.
It does not open motor ports or send motor commands. Reconnection undoes it.
"""
import argparse
import fcntl
import os
from pathlib import Path
import time

from diagnose_robot_usb import ROOT, attr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Write configuration 1 (requires sudo)')
    args = parser.parse_args()
    if args.apply and os.geteuid() != 0:
        parser.error('--apply requires sudo')
    with (ROOT / 'logs/control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hub = Path('/sys/bus/usb/devices/1-1')
        if (attr(hub, 'idVendor'), attr(hub, 'idProduct')) != ('14cd', '8601'):
            raise SystemExit('Expected robot hub is not at 1-1; no changes made')
        if '0000:06:00.3' not in hub.resolve().parts:
            raise SystemExit('Unexpected USB controller; no changes made')
        expected = [('1-1.1', 'caf1', 'ffff'), ('1-1.2', 'caf1', 'ffff'),
                    ('1-1.4', '32e4', '9230')]
        devices = []
        for name, vendor, product in expected:
            path = hub.parent / name
            if (attr(path, 'idVendor'), attr(path, 'idProduct')) != (vendor, product):
                raise SystemExit(f'Unexpected device at {name}; no changes made')
            value = attr(path, 'bConfigurationValue')
            if value not in ('', '0', '1'):
                raise SystemExit(f'Unexpected configuration at {name}; no changes made')
            devices.append((path, value))
        if args.apply:
            # Keep this USB branch awake while configuring and checking drivers.
            for path in [hub, hub.parent / 'usb1'] + [p for p, _ in devices]:
                (path / 'power/control').write_text('on\n')
        for path, value in devices:
            if value == '1':
                print(f'{path.name}: already configured')
            elif args.apply:
                (path / 'bConfigurationValue').write_text('1\n')
                if attr(path, 'bConfigurationValue') != '1':
                    raise SystemExit(f'{path.name}: configuration did not take effect')
                print(f'{path.name}: configured')
            else:
                print(f'{path.name}: would set configuration 1')
        if not args.apply:
            print('Dry run only. Use sudo python3 scripts/configure_robot_hub.py --apply')
        else:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                ready = all((p.parent / f'{p.name}:1.0/driver').exists() for p, _ in devices)
                if ready:
                    print('Both board drivers and the overhead camera driver are bound')
                    return
                time.sleep(0.1)
            missing = [p.name for p, _ in devices
                       if not (p.parent / f'{p.name}:1.0/driver').exists()]
            raise SystemExit(f'Configuration written, but drivers failed to bind: {missing}. '
                             'Inspect journalctl -k; do not start motion yet.')


if __name__ == '__main__':
    main()
