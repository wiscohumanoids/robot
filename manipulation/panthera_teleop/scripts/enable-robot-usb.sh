#!/usr/bin/env bash
# Install targeted USB protection; run with the controller stopped.
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$EUID" -ne 0 ]]; then
  printf '%s\n' 'Run with sudo: sudo ./scripts/enable-robot-usb.sh' >&2
  exit 1
fi
controller_lock="$task_dir/logs/control.lock"
if [[ -e "$controller_lock" ]]; then
  exec 9<"$controller_lock"
  if ! flock -n 9; then
    printf '%s\n' 'Stop the robot controller before changing USB rules.' >&2
    exit 1
  fi
fi
install -m 0644 "$task_dir/config/99-panthera-robot-usb.rules" /etc/udev/rules.d/99-panthera-robot-usb.rules
udevadm control --reload-rules
# Apply only to robot boards/ports and the known robot hub.
udevadm trigger --action=change --subsystem-match=usb --attr-match=idVendor=caf1 --attr-match=idProduct=ffff
udevadm trigger --action=change --subsystem-match=usb --attr-match=idVendor=14cd --attr-match=idProduct=8601
udevadm settle
# Also keep the current upstream USB ancestry awake, including other hub models.
python3 - "$task_dir/app" <<'PY'
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, sys.argv[1])
from usb_stability import keep_usb_awake, robot_usb_devices
for tty in Path('/sys/class/tty').glob('ttyACM*'):
    devices = robot_usb_devices(tty)
    if devices and (devices[0] / 'idVendor').read_text().strip() == 'caf1' and (devices[0] / 'idProduct').read_text().strip() == 'ffff':
        subprocess.run(['udevadm', 'trigger', '--action=change', str(tty)], check=True)
        if keep_usb_awake(tty):
            raise SystemExit('Could not disable robot USB autosuspend')
PY
udevadm settle
printf '%s\n' 'Robot USB protection installed. Reconnect both robot USB cables before restarting control so modem-ignore rules take effect.'
