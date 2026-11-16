#!/usr/bin/env bash
# Grants USB access to this Quest 3S. Run through sudo or pkexec.
set -euo pipefail
if [[ "$EUID" -ne 0 ]]; then
  printf '%s\n' 'Run this script with sudo or pkexec.' >&2
  exit 1
fi
cat > /etc/udev/rules.d/70-panthera-quest.rules <<'RULE'
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="2833", ATTR{serial}=="3487C20GBC0231", GROUP="dialout", MODE="0660", TAG+="uaccess"
RULE
chmod 0644 /etc/udev/rules.d/70-panthera-quest.rules
udevadm control --reload-rules
udevadm trigger --action=add --subsystem-match=usb --attr-match=idVendor=2833 --attr-match=serial=3487C20GBC0231
udevadm settle
printf '%s\n' 'Quest USB access rule installed.'
