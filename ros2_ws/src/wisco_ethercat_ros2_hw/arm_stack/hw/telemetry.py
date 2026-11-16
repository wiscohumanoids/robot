"""Live read-only EtherCAT encoder monitor."""
import csv
import math
from pathlib import Path
import sys
import time

from .launch import passive_feedback

from ..calib.commission import EncoderReader
from ..config import CalibrationStore, load_arm, load_yaml


def run(args):
    arm = load_arm(args.arm)
    if args.cyclic or args.log_angles or args.diagnose_encoders:
        return cyclic(arm, args.duration, args.calib_root, args.log_angles, args.diagnose_encoders, args.manual_shoulders)
    reader = EncoderReader()
    baseline = {}
    started = time.monotonic()
    sample = 0
    try:
        while True:
            lines = [' Bus  Joint                Counts   Δ counts   Δ degrees   Status']
            for joint in arm.joints:
                pos = joint.bus_position
                counts = reader.upload(pos, 'int32', '0x6064')
                status = reader.upload(pos, 'uint16', '0x6041')
                delta = counts - baseline.setdefault(pos, counts)
                degrees = joint.direction * delta * 360 / joint.counts_per_rev
                lines.append(f'{pos:4}  {joint.name:16} {counts:10} {delta:+10} {degrees:+11.3f}   0x{status:04x}')
            prefix = '\033[H\033[2J' if sys.stdout.isatty() else ''
            sample += 1
            elapsed = time.monotonic() - started
            print(f'{prefix}Sample {sample}   {elapsed:.1f} s\n' + '\n'.join(lines), flush=True)
            if args.duration is not None and time.monotonic() - started >= args.duration:
                break
            time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    return 0


def capture_angles(arm, client, output, calibration, manual_shoulders=False):
    manual = ('shoulder_roll', 'shoulder_yaw') if manual_shoulders else ()
    if any(name not in arm.names for name in manual):
        raise ValueError('Manual shoulder logging requires shoulder_roll and shoulder_yaw.')
    path = Path(output).expanduser().resolve()
    baseline = None
    try:
        stream = path.open('x', newline='')
    except FileExistsError:
        raise ValueError(f'{path} already exists. Choose a new CSV filename.') from None
    with stream:
        writer = csv.writer(stream)
        writer.writerow(['pose', 'controller_time_s', 'calibration_version', 'coordinate_frame',
                         *[f'{n}_deg' for n in arm.names],
                         *[f'{n}_from_start_deg' for n in arm.names],
                         *[f'{n}_external_from_start_deg' for n in manual]])
        stream.flush()
        print(f'Logging to {path}. Capture the starting pose first. Ctrl-C finishes.', flush=True)
        while True:
            try:
                label = input('Hold the pose still, type its label, and press Enter: ').strip()
            except EOFError:
                return 0
            if not label:
                continue
            measured = []
            for name in manual:
                while True:
                    try:
                        angle = float(input(f'{name} measured degrees from starting pose '
                                            '(first pose = 0; hold still): '))
                    except ValueError:
                        continue
                    except EOFError:
                        return 0
                    if math.isfinite(angle) and (baseline is not None or angle == 0):
                        measured.append(angle)
                        break
                    print('Enter a finite angle; use 0 for the starting pose.')
            state = client.state()
            angles = [math.degrees(q) for q in state.q]
            if not all(map(math.isfinite, angles)):
                raise RuntimeError('Invalid joint feedback. Pose not recorded.')
            if baseline is None:
                baseline = angles
            deltas = [q - start for q, start in zip(angles, baseline)]
            writer.writerow([label, state.t, calibration.get('version', 0),
                             calibration.get('coordinate_frame') or 'unspecified', *angles, *deltas, *measured])
            stream.flush()
            print(f'Saved {label}. Degrees from start: ' +
                  ', '.join(f'{n} {d:+.2f}' for n, d in zip(arm.names, deltas)), flush=True)


def diagnose_encoders(arm, client):
    """Compare CoE reads with the independently published cyclic positions."""
    reader = EncoderReader()
    targets = {'shoulder_roll', 'shoulder_yaw'} if arm.arm == 'arm01' else set(arm.names)

    def read(joint, kind, index):
        try:
            return reader.upload(joint.bus_position, kind, index)
        except RuntimeError as error:
            print(f'bus {joint.bus_position} {index}: {error}')
            return None

    def snapshot():
        raw = [read(j, 'int32', '0x6064') for j in arm.joints]
        first = {j.name: read(j, 'int32', '0x200b') for j in arm.joints if j.name in targets}
        return client.state(), raw, first

    for joint in arm.joints:
        if joint.name not in targets:
            continue
        print(f'Bus {joint.bus_position} {joint.name}: reported PDO layout')
        print(reader.command('pdos', '-p', str(joint.bus_position)))
        print(f'Drive error code (0x603f): {read(joint, "uint16", "0x603f")}')
        input('Hold the supported arm still and press Enter for the baseline: ')
        before, raw_before, first_before = snapshot()
        input(f'Move only {joint.name} a few degrees by hand, hold it there and press Enter: ')
        after, raw_after, first_after = snapshot()
        for i, j in enumerate(arm.joints):
            angle = math.degrees(after.q[i] - before.q[i])
            line = (f'bus {j.bus_position} {j.name}: cyclic change {angle:+.4f} deg, '
                    f'0x6064 {raw_before[i]} -> {raw_after[i]}')
            if j.name in targets:
                line += f', 0x200b {first_before[j.name]} -> {first_after[j.name]}'
            print(line)
    return 0


def cyclic(arm, duration, calib_root=None, log_angles=None, diagnose=False, manual_shoulders=False):
    if arm.control_mode != 'arm':
        raise ValueError('Cyclic telemetry requires an arm-mode joint configuration.')
    EncoderReader().require_idle(arm)
    print('Starting cyclic feedback with motor enabling disabled. Waiting for arm state...', flush=True)
    latest = CalibrationStore(arm, calib_root).latest_path()
    calibration = load_yaml(latest) if latest else {}
    try:
        with passive_feedback(arm, calibration) as client:
            if diagnose:
                return diagnose_encoders(arm, client)
            if log_angles:
                return capture_angles(arm, client, log_angles, calibration, manual_shoulders)
            first = client.state()
            started = time.monotonic()
            while True:
                state = client.state()
                lines = [f'Cyclic feedback   controller time {state.t:.3f} s',
                         ' Bus  Joint                Position °    Δ degrees   Status']
                for i, joint in enumerate(arm.joints):
                    lines.append(f'{joint.bus_position:4}  {joint.name:16} '
                                 f'{math.degrees(state.q[i]):11.3f} '
                                 f'{math.degrees(state.q[i]-first.q[i]):+12.3f}   0x{state.status[i]:04x}')
                print(('\033[H\033[2J' if sys.stdout.isatty() else '') + '\n'.join(lines), flush=True)
                if duration is not None and time.monotonic() - started >= duration:
                    break
                time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    return 0
