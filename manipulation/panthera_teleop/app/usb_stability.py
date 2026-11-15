"""Keep robot USB devices and their upstream hubs awake during control."""
from pathlib import Path
import sys


def robot_usb_devices(tty, sys_tty=Path('/sys/class/tty')):
    """Resolve sysfs ancestry, never infer a board from its changing tty number."""
    resolved = (sys_tty / Path(tty).name).resolve(strict=True)
    return [parent for parent in resolved.parents
            if (parent / 'idVendor').is_file() and (parent / 'idProduct').is_file()]


def keep_usb_awake(tty, sys_tty=Path('/sys/class/tty')):
    blocked = []
    for device in robot_usb_devices(tty, sys_tty):
        control = device / 'power/control'
        if not control.exists():
            continue
        try:
            if control.read_text().strip() != 'on':
                control.write_text('on\n')
            if control.read_text().strip() != 'on':
                blocked.append(str(control))
        except OSError:
            blocked.append(str(control))
    if blocked:
        print('Robot USB autosuspend protection needs administrator setup: '
              'sudo ./scripts/enable-robot-usb.sh (with the controller stopped). '
              'Unprotected: ' + ', '.join(blocked), file=sys.stderr, flush=True)
    return blocked


def is_serial_disconnect(exc):
    """Recognize SDK transport errors without hiding feedback or motor faults."""
    while exc is not None:
        message = str(exc).lower()
        if any(marker in message for marker in (
                'bad file descriptor', 'device disconnected', 'portnotopened',
                'serialexception', 'serial port is not open')):
            return True
        if 'io exception' in message and ('input/output error' in message or '(5)' in message):
            return True
        exc = exc.__cause__
    return False
