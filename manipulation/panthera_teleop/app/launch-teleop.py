"""Resolve the identified robot boards and launch the official dual-arm demo.

Without --run, only validate/generate configurations; no motor devices are opened.
The official demo moves both arms on startup and returns them to zero on exit.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import socket
import struct
import sys
import time

import pinocchio as pin
import yaml

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / 'work/Panthera-HT_SDK/panthera_python'
CONFIG = ROOT / 'config/robot-configs'


class RobotConnectionError(RuntimeError):
    """An expected hardware preflight failure, before opening motor devices."""


def wait_for_joint_feedback(robot, label, timeout=5.0):
    """Reject SDK placeholder readings before any startup trajectory is built."""
    deadline = time.monotonic() + timeout
    consecutive = 0
    while time.monotonic() < deadline:
        robot.send_get_motor_state_cmd()
        robot.motor_send_cmd()
        time.sleep(0.1)
        positions = robot.get_current_pos()
        valid = len(positions) == 6 and all(
            math.isfinite(q) and low <= q <= high
            for q, low, high in zip(positions, robot.joint_limits['lower'], robot.joint_limits['upper'])
        ) and all(m.get_current_motor_state().fault == 0 for m in robot.Motors)
        consecutive = consecutive + 1 if valid else 0
        if consecutive >= 5:
            print(f'{label}: valid joint feedback received; motor faults clear.', flush=True)
            return
    raise RuntimeError(f'{label}: joint feedback invalid or motor fault present; startup refused')


def resolve_board(mapping, sys_tty=Path('/sys/class/tty'),
                  sys_usb=Path('/sys/bus/usb/devices'), dev_root=Path('/dev')):
    found = []
    for tty in sys_tty.glob('ttyACM*'):
        resolved = tty.resolve()
        interface = f"{mapping['usb_path']}:1.0"
        if interface not in resolved.parts or mapping['pci_controller'] not in resolved.parts:
            continue
        usb_device = next((p for p in resolved.parents if p.name == mapping['usb_path']), None)
        if usb_device is None:
            continue
        if (usb_device/'idVendor').read_text().strip() != 'caf1' or (usb_device/'idProduct').read_text().strip() != 'ffff':
            raise RobotConnectionError('Unexpected device at the saved robot USB connection')
        found.append(dev_root/tty.name)
    if len(found) != 1:
        board = (sys_usb / mapping['usb_path']).resolve()
        if board.exists() and mapping['pci_controller'] in board.parts:
            try:
                vendor = (board / 'idVendor').read_text().strip()
                product = (board / 'idProduct').read_text().strip()
                configured = (board / 'bConfigurationValue').read_text().strip()
            except OSError:
                pass  # The device may have disappeared during enumeration.
            else:
                if (vendor, product) != ('caf1', 'ffff'):
                    raise RobotConnectionError('Unexpected device at the saved robot USB connection')
                if configured in ('', '0'):
                    raise RobotConnectionError(
                        f"Robot board at USB {mapping['usb_path']} is detected but USB is unconfigured; "
                        'no serial port is available. Check the robot hub external power supply '
                        'and USB cable, then reconnect the hub. Linux may have rejected the '
                        'configuration because of insufficient available bus power '
                        '(check journalctl -k).')
                raise RobotConnectionError(
                    f"Robot board at USB {mapping['usb_path']} is detected but its serial interface "
                    'is unavailable. Wait for enumeration or check the cdc_acm driver '
                    'and reconnect the robot USB cable.')
        raise RobotConnectionError(f"Expected one robot at USB {mapping['usb_path']}; found {len(found)}. Reconnect its USB cable to the original hub port.")
    device = found[0]
    # The vendor SDK uses prefix matching, so refuse ambiguous tty names.
    if list(device.parent.glob(device.name+'*')) != [device]:
        raise RobotConnectionError(f'SDK prefix matching is ambiguous for {device}; do not launch.')
    return device


def prepare():
    mappings = json.loads((CONFIG/'arm-mapping.json').read_text())
    configs = {}
    devices = {}
    for side in ('left', 'right'):
        device = resolve_board(mappings[side])
        devices[side] = str(device)
        motor = yaml.safe_load((SDK/'robot_param/motor_param/6dof_Panthera_params_follower.yaml').read_text())
        motor['robot']['Serial_Type'] = str(device)
        motor['robot']['CANboard']['No_1_CANboard']['CANport']['CANport_1']['serial_id'] = 1
        motor_path = CONFIG/f'{side}-motors.yaml'
        motor_path.write_text(yaml.safe_dump(motor, sort_keys=False))
        config = yaml.safe_load((SDK/'robot_param/Follower.yaml').read_text())
        config['robot'].update(name=side, robot_name=side, param_file=str(motor_path))
        model_path = (SDK/'robot_param'/config['urdf']['file_path']).resolve()
        config['urdf']['file_path'] = str(model_path)
        model = pin.buildModelFromUrdf(str(model_path))
        assert model.nq == 6 and model.existFrame(config['urdf']['end_effector_link'])
        configs[side] = CONFIG/f'{side}.yaml'
        configs[side].write_text(yaml.safe_dump(config, sort_keys=False))
        print(f'{side.title()} controller -> USB {mappings[side]["usb_path"]} -> {device}', flush=True)
    if len(set(devices.values())) != 2:
        raise RuntimeError('Left and right must address different robot devices')
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Enable the official demo; both arms move on startup and return to zero on exit')
    args = parser.parse_args()
    configs = prepare()
    if not args.run:
        print('Configurations checked. No motor devices opened; no motion commands sent.')
        return
    print('Checking live tracking from both controllers with both grip buttons released...', flush=True)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(('127.0.0.1', 5005))
        receiver.settimeout(1)
        deadline = time.monotonic() + 20
        valid_since = None
        while time.monotonic() < deadline:
            try:
                packet, _ = receiver.recvfrom(4096)
            except socket.timeout:
                valid_since = None
                continue
            if len(packet) != 160:
                valid_since = None
                continue
            values = struct.unpack('40f', packet)
            valid = all(math.isfinite(v) for v in values) and values[2] == 1 and values[22] == 1 and values[13] < 0.1 and values[33] < 0.1
            if not valid:
                valid_since = None
                continue
            now = time.monotonic()
            valid_since = now if valid_since is None else valid_since
            if now - valid_since >= 0.5:
                break
        else:
            raise RuntimeError('No stable two-controller tracking with grips released. No robot objects were created.')
    sys.path.insert(0, str(SDK/'scripts'))
    spec = importlib.util.spec_from_file_location('panthera_dual_vr', SDK/'scripts/7_vr_dual_cartesian_control.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.Panthera

    def configured_robot(config_path):
        filename = Path(config_path).name
        side = {'Leader.yaml': 'left', 'Follower.yaml': 'right'}.get(filename)
        if side is None:
            raise RuntimeError(f'Unexpected robot configuration: {filename}')
        # Recheck endpoint identity immediately before opening hardware.
        mapping = json.loads((CONFIG/'arm-mapping.json').read_text())[side]
        expected = yaml.safe_load((CONFIG/f'{side}-motors.yaml').read_text())['robot']['Serial_Type']
        if str(resolve_board(mapping)) != expected:
            raise RuntimeError('USB endpoints changed after preflight; restart the launcher')
        robot = original(str(configs[side]))
        wait_for_joint_feedback(robot, side)
        return robot

    module.Panthera = configured_robot
    print('Starting official dual-arm demo: automatic startup pose and exit-to-zero are enabled.', flush=True)
    module.main()


if __name__ == '__main__':
    main()
