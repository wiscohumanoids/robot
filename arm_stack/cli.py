"""`arm <stage> [--sim]`: one command per calibration stage (spec: Calibration procedures).

  arm bringup | indicator | zero | limits | kt | gravity | friction | latency | pivot | plate
  arm mocap-calib | pace | mocap-repeat | repeat [--indicators] | validate
  arm all --sim [--no-mocap]    every stage in order on one sim arm (Phase 0 gate run);
                                with mocap: mocap-calib + mocap-repeat, without: plate +
                                the indicator cube (repeat, validate)
  arm rig --arm bench01 --serial SN [--model x4_36]
                                Phase 2 lever rig for one actuator: zero, kt, gravity
                                (lever COM), friction, latency, stored per serial
  arm float [--lock J,..] [--friction-gain G] [--damping D] [--payload KG@x,y,z]
            [--duration S] [--record [PATH]]
                                teach / hand-guide in gravity comp (control/float_mode.py)
  arm gravity-check [--payload KG@x,y,z]
                                float drift -> gravity residual per joint, and the largest
                                friction help that does not creep (float_friction_gain)
  arm status                    latest calibration version and what it holds
  arm export pace SESSION.mcap OUT
  arm export lerobot OUT SESSION.mcap [...]

Each stage prints its report and, if it passed, writes a new calibration
version (config/calib/<arm>/<date>_v<N>.yaml; --sim writes to a separate sim
store so sim runs never mix with the arm's files). --force writes a failing
stage too, for debugging. Every run is recorded to one MCAP session in
--log-dir with the calibration version in its meta.

Hardware runs need arm01.launch.py up first; --sim needs nothing but
Pinocchio and numpy. --arm takes a path or a name under config/arms/
(e.g. --arm bench01, the single-joint lever rig). Stages that need more
joints than the arm has (pivot, mocap, validate on the 1-joint rig) report
SKIP.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import importlib
import os
import re
import sys
import tempfile
import time
import zlib
from typing import Dict, List

import numpy as np
import yaml

from .config import CalibrationStore, calibration_hash, git_sha, load_arm
from .model.robot import ArmModel

STAGES = [
    ('bringup', 'bringup'), ('indicator', 'indicator'), ('zero', 'zero'), ('limits', 'limits'),
    ('kt', 'torque_constant'),
    ('gravity', 'gravity'), ('friction', 'friction'), ('latency', 'latency'), ('pivot', 'pivot'),
    # the kinematic stages (plate / mocap-calib) give PACE the joint stiffness
    ('plate', 'plate'), ('mocap-calib', 'mocap_calib'), ('pace', 'pace'), ('mocap-repeat', 'mocap_repeat'),
    ('repeat', 'repeat'), ('validate', 'validate'),
]
MODULES = dict(STAGES)
MODULES['gravity-check'] = 'gravity_check'     # a check, not part of `arm all`
# Stages that need an assembled arm: fewer joints -> SKIP with the reason.
MIN_JOINTS = {'pivot': 4, 'plate': 6, 'mocap-calib': 6, 'mocap-repeat': 6, 'repeat': 6, 'validate': 6}
# Phase 2 lever rig, per actuator: zero (lever level), Kt on the hook masses, the lever's
# COM (friction subtracts its gravity), friction, latency.
RIG_STAGES = ['zero', 'kt', 'gravity', 'friction', 'latency']
RIG_KT_LINEARITY = 0.05
RIG_FRICTION_R2 = 0.9
SIM_ROOT = os.path.join(os.path.expanduser('~'), '.cache', 'arm_stack', 'sim_calib')


def build_model(arm, calibration) -> ArmModel:
    model = ArmModel(arm.urdf, arm.names)
    model.apply_calibration(calibration, arm.link_masses_kg)
    soft = calibration.get('soft_limits_rad') or {}
    for i, n in enumerate(arm.names):
        lo, hi = (soft.get(n) or [None, None])
        if lo is not None:
            model.lower[i] = lo
        if hi is not None:
            model.upper[i] = hi
    return model


def make_client(args, arm, calibration):
    if args.sim:
        from .sim.arm_sim import SimArm, SimErrors
        # each actuator serial gets its own unit-to-unit errors
        seed = args.seed + (zlib.crc32(args.serial.encode()) % 100000 if args.serial else 0)
        errors = SimErrors.random(arm.n, seed=seed, scale=args.error_scale)
        return SimArm(arm, calibration, errors, seed=seed, mocap_enabled=not getattr(args, 'no_mocap', False))
    from .hw.ros_client import RosArmClient
    power_on = None
    if args.powered_at:
        # seconds ago the drives were powered, e.g. --powered-at 1200
        power_on = time.monotonic() - float(args.powered_at)
    client = RosArmClient(arm, calibration, power_on_monotonic=power_on)
    if getattr(args, 'no_mocap', False):
        client.mocap = lambda: None
    return client


def session_meta(arm, store, calibration, client, operator):
    return {
        'arm': arm.arm, 'joint_names': arm.names,
        'calibration_version': calibration.get('version'),
        'calibration_hash': calibration_hash(store.latest_path()),
        'joint_config_hash': arm.hash(), 'git_sha': git_sha(),
        'time_since_power_on_s': client.time_since_power_on(), 'operator': operator,
        'actuator_serial': store.serial, 'actuator_models': [j.model for j in arm.joints],
        'started': dt.datetime.now().isoformat(timespec='seconds'),
    }


def update_joint_config(path: str, updates: Dict[str, Dict]) -> List[str]:
    """Targeted edit of per-joint `direction:` (bringup) and `float_friction_gain:`
    (gravity-check) lines in the joint YAML, keeping comments. A missing
    float_friction_gain line is added under the joint's name. The arm-level
    `repeatability_tuning:` block (mocap-repeat) is replaced, or appended."""
    done = []
    if not any((updates.get(k) or {}) for k in ('direction', 'float_friction_gain', 'repeatability_tuning')):
        return done
    text = open(path).read()
    tuning = updates.get('repeatability_tuning') or {}
    if tuning:
        block = 'repeatability_tuning:   # written by `arm mocap-repeat --write-config`\n' + ''.join(
            '  %s: %s\n' % (k, ('%.4g' % v) if isinstance(v, float) else v) for k, v in tuning.items())
        pat = re.compile(r'^repeatability_tuning:[^\n]*\n(?:[ \t]+[^\n]*\n)*', re.M)
        text, k = pat.subn(lambda m: block, text, count=1)
        if not k:
            text = text.rstrip('\n') + '\n\n' + block
        done.append('repeatability_tuning -> %s' % ', '.join('%s %s' % kv for kv in tuning.items()))
    for joint, value in (updates.get('direction') or {}).items():
        pat = re.compile(r'(-\s*name:\s*%s\b(?:(?!\n\s*-\s*name:).)*?\n\s*direction:\s*)(-?1)' % re.escape(joint),
                         re.S)
        text, k = pat.subn(lambda m: m.group(1) + str(int(value)), text, count=1)
        if k:
            done.append('%s direction -> %+d' % (joint, value))
    for joint, value in (updates.get('float_friction_gain') or {}).items():
        v = '%.2f' % float(value)
        pat = re.compile(r'(-\s*name:\s*%s\b(?:(?!\n\s*-\s*name:).)*?\n\s*float_friction_gain:[ \t]*)([^\s#]+)'
                         % re.escape(joint), re.S)
        text, k = pat.subn(lambda m: m.group(1) + v, text, count=1)
        if not k:
            pat = re.compile(r'^([ \t]*-[ \t]*)(name:[ \t]*%s\b[^\n]*\n)' % re.escape(joint), re.M)
            text, k = pat.subn(lambda m: m.group(0) + ' ' * len(m.group(1)) + 'float_friction_gain: %s\n' % v,
                               text, count=1)
        if k:
            done.append('%s float_friction_gain -> %s' % (joint, v))
    with open(path, 'w') as f:
        f.write(text)
    return done


def run_stage(name, args, arm, store, client, recorder=None, extra=None):
    from .calib.common import StageContext, skipped
    need = MIN_JOINTS.get(name)
    if need and arm.n < need:
        result = skipped(name, 'skipped: needs an assembled arm (>= %d joints); %s has %d'
                         % (need, arm.arm, arm.n))
        print(result.report())
        if recorder is not None:
            recorder.event('stage_skipped', stage=name, reason=result.lines[0])
        return result
    calibration = store.latest()
    if args.sim and hasattr(client, 'reload_calibration'):
        client.reload_calibration(calibration)
    ctx = StageContext(arm=arm, client=client, calibration=calibration, model=build_model(arm, calibration),
                       sim=args.sim, quick=args.quick, log=recorder,
                       rng=np.random.default_rng(args.seed))
    mod = importlib.import_module('arm_stack.calib.' + MODULES[name])
    if recorder is not None:
        recorder.event('stage_start', stage=name, calibration_version=calibration.get('version'))
    t0 = time.monotonic()
    result = mod.run(ctx, **(extra or {}))
    print(result.report())
    print('  (%.1f s)' % (time.monotonic() - t0))
    if recorder is not None:
        recorder.event('stage_end', stage=name, passed=result.passed)
    path = None
    if result.writes and (result.passed or args.force):
        path = store.write(name, result.writes)
        print('  wrote %s' % path)
    elif result.writes:
        print('  not written (failed; --force to write anyway)')
    if result.joint_config_updates:
        if args.write_config and not args.sim:
            for line in update_joint_config(arm.path, result.joint_config_updates):
                print('  joint config: ' + line)
        elif any(result.joint_config_updates.values()):
            print('  joint config changes (apply with --write-config): %s' % result.joint_config_updates)
            if args.sim and result.joint_config_updates.get('repeatability_tuning'):
                # the operator applies the tuning before validating; in sim do it in memory
                arm.raw['repeatability_tuning'] = dict(result.joint_config_updates['repeatability_tuning'])
                print('  (sim) repeatability_tuning applied in memory for the stages that follow')
    if name == 'validate':
        target = path or store.latest_path()
        if target:
            report = os.path.splitext(target)[0] + '_validate_report.yaml'
            with open(report, 'w') as f:
                yaml.safe_dump({'passed': result.passed, 'lines': result.lines, 'flags': result.flags,
                                'metrics': _plain(result.metrics)}, f, sort_keys=False)
            print('  report %s' % report)
    return result


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def cmd_status(args):
    arm = load_arm(args.arm, model=args.model, serial=args.serial)
    store = CalibrationStore(arm, args.calib_root or (SIM_ROOT if args.sim else None), store_serial(arm))
    path = store.latest_path()
    if not path:
        print('%s: no calibration yet in %s' % (arm.arm, store.dir))
        return 0
    cal = store.latest()
    print('%s%s: v%s (%s, stage %s) %s' % (arm.arm, ' actuator %s' % store.serial if store.serial else '',
                                           cal.get('version'), cal.get('date'), cal.get('stage'), path))
    for key in ('zero_offsets_rad', 'soft_limits_rad', 'kt_output_nm_per_a', 'link_inertials', 'friction',
                'latency_ms', 'joint_stiffness_nm_per_rad', 'kinematic_overlay', 'pace', 'validation'):
        v = cal.get(key)
        filled = bool(v) and (not isinstance(v, dict) or any(x not in (None, [None, None], {}) for x in v.values()))
        print('  %-28s %s' % (key, 'set' if filled else '-'))
    return 0


def store_serial(arm):
    """Actuator serial for a 1-joint (bench) arm: --serial or the joint row's `serial`."""
    serials = [j.serial for j in arm.joints if j.serial]
    if not serials:
        return None
    if arm.n != 1:
        raise SystemExit('--serial / joint serial is for a 1-joint bench arm; %s has %d joints'
                         % (arm.arm, arm.n))
    return serials[0]


def run_rig(args, arm, store, client, recorder):
    """Phase 2 lever rig for the mounted actuator; writes a per-serial summary."""
    print('rig: %s actuator %s (%s): %s' % (arm.arm, store.serial, arm.joints[0].model, ', '.join(RIG_STAGES)))
    results = {}
    for name in RIG_STAGES:
        results[name] = run_stage(name, args, arm, store, client, recorder)
    cal = store.latest()
    joint = arm.names[0]
    kt_m = results['kt'].metrics.get(joint, {})
    fr_m = results['friction'].metrics.get(joint, {})
    loads = len(kt_m.get('masses_kg') or [])
    kt_ok = bool(results['kt'].passed and loads >= 3 and kt_m.get('linearity', 1.0) < RIG_KT_LINEARITY)
    fr_ok = bool(results['friction'].passed and fr_m.get('r2', 0.0) > RIG_FRICTION_R2)
    gate = bool(kt_ok and fr_ok and all(r.passed for r in results.values()))
    summary = {
        'actuator_serial': store.serial, 'actuator_model': arm.joints[0].model, 'arm': arm.arm,
        'joint': joint, 'date': dt.date.today().isoformat(), 'sim': bool(args.sim),
        'calibration_version': cal.get('version'), 'calibration_file': store.latest_path(),
        'kt_output_nm_per_a': (cal.get('kt_output_nm_per_a') or {}).get(joint),
        'kt_nominal_nm_per_a': arm.joints[0].kt_nominal_nm_per_a,
        'kt_linearity': kt_m.get('linearity'), 'kt_loads_kg': kt_m.get('masses_kg'),
        'kt_per_load': kt_m.get('per_load'),
        'friction': (cal.get('friction') or {}).get(joint), 'friction_r2': fr_m.get('r2'),
        'latency_ms': cal.get('latency_ms'),
        'stages': {n: 'PASS' if r.passed else 'FAIL' for n, r in results.items()},
        'gate': {'kt_linear_3_loads': kt_ok, 'friction_r2_above_0_9': fr_ok, 'met': gate},
    }
    os.makedirs(store.dir, exist_ok=True)
    path = os.path.join(store.dir, 'rig_%s_v%s.yaml' % (summary['date'], cal.get('version')))
    with open(path, 'w') as f:
        f.write('# Written by `arm rig` for actuator %s.\n' % store.serial)
        yaml.safe_dump(_plain(summary), f, sort_keys=False)
    print('RIG %s: Kt %s Nm/A (linear across %d loads: %s), friction R^2 %s (> 0.9: %s), latency %s ms -> '
          'Phase 2 gate %s' % (store.serial, _fmt(summary['kt_output_nm_per_a']), loads, 'yes' if kt_ok else 'NO',
                               _fmt(summary['friction_r2']), 'yes' if fr_ok else 'NO',
                               _fmt(summary['latency_ms']), 'MET' if gate else 'NOT MET'))
    print('  summary %s' % path)
    return 0 if gate else 1


def _fmt(v):
    return '-' if v is None else '%.4g' % v


def cmd_export(args):
    from .log import export
    if args.kind == 'pace':
        print(export.export_pace(args.paths[0], args.paths[1]))
    else:
        print(export.export_lerobot(args.paths[1:], args.paths[0], repo_id=args.repo_id, fps=args.fps))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='arm', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=[s for s, _ in STAGES] + ['gravity-check', 'float', 'all', 'rig', 'status',
                                                               'export'])
    p.add_argument('paths', nargs='*', help='export: kind and files')
    p.add_argument('--sim', action='store_true', help='run on the sim backend')
    p.add_argument('--quick', action='store_true', help='fewer poses and cycles (smoke runs)')
    p.add_argument('--arm', default='config/arms/arm01.yaml', help='joint config path or name (bench01)')
    p.add_argument('--serial', default=None, help='bench: serial of the actuator on the rig')
    p.add_argument('--model', default=None, help='bench: actuator model on the rig (x4_10, x4_36, x6_60)')
    p.add_argument('--calib-root', default=None, help='calibration store root (default config/calib)')
    p.add_argument('--log-dir', default=None, help='MCAP session directory (default ~/arm_logs)')
    p.add_argument('--no-log', action='store_true')
    p.add_argument('--seed', type=int, default=0, help='sim error seed and pose sampling')
    p.add_argument('--error-scale', type=float, default=1.0, help='sim: scale of the injected errors')
    p.add_argument('--force', action='store_true', help='write results even if the stage failed')
    p.add_argument('--write-config', action='store_true', help='apply joint config changes (bringup directions)')
    p.add_argument('--powered-at', default=None, help='seconds since the drives were powered on')
    p.add_argument('--operator', default=None)
    p.add_argument('--replay', nargs='*', default=None, help='validate: recorded teleop sessions to replay')
    p.add_argument('--repo-id', default='WiscoHumanoidsLab/arm01-teleop')
    p.add_argument('--fps', type=int, default=50)
    p.add_argument('--fresh', action='store_true', help='sim: start from an empty sim calibration store')
    p.add_argument('--payload', default=None, help='float, gravity-check: KG@x,y,z point mass in tool0 (m)')
    p.add_argument('--lock', default=None, help='float: joints held still (names, indices or 0/1 mask)')
    p.add_argument('--friction-gain', default=None, help='float: Coulomb help, fraction (< 1; default: '
                   'joint config float_friction_gain, else 0.6); one value or one per joint')
    p.add_argument('--damping', default=None, help='float: viscous damping, Nm s/rad (default 0)')
    p.add_argument('--duration', type=float, default=None, help='float: seconds (default: until Ctrl-C)')
    p.add_argument('--record', nargs='?', const='', default=None,
                   help='float: record every cycle to MCAP (PATH, or a _teach file in --log-dir)')
    p.add_argument('--no-mocap', action='store_true',
                   help='the client reports no mocap (sim: the plate + indicator-cube chain)')
    p.add_argument('--indicators', action='store_true', help='repeat: measure with the dial-indicator cube')
    args = p.parse_args(argv)

    if args.stage == 'status':
        return cmd_status(args)
    if args.stage == 'export':
        if len(args.paths) < 3 or args.paths[0] not in ('pace', 'lerobot'):
            p.error('arm export pace SESSION.mcap OUT | arm export lerobot OUT SESSION.mcap [...]')
        args.kind = args.paths[0]
        args.paths = args.paths[1:]
        return cmd_export(args)

    arm = load_arm(args.arm, model=args.model, serial=args.serial)
    serial = store_serial(arm)
    if args.stage == 'rig':
        if arm.n != 1:
            p.error('arm rig is the single-joint lever rig (--arm bench01); %s has %d joints' % (arm.arm, arm.n))
        if not serial:
            if not args.sim:
                p.error('arm rig needs the actuator serial: --serial SN (label on the housing)')
            serial = 'sim-%d' % args.seed
            arm = load_arm(args.arm, model=args.model, serial=serial)
    args.serial = serial
    root = args.calib_root or (SIM_ROOT if args.sim else None)
    if args.sim and args.fresh:
        root = tempfile.mkdtemp(prefix='arm_sim_calib_')
    store = CalibrationStore(arm, root, serial)
    calibration = store.latest()
    if calibration.get('version') in (None, 0) and not args.sim and args.stage not in (
            'bringup', 'indicator', 'zero', 'all', 'rig'):
        print('warning: no calibration file yet for %s; run arm bringup and arm zero first' % arm.arm)
    client = make_client(args, arm, calibration)
    recorder = None
    if not args.no_log or args.record is not None:
        from .log.recorder import Recorder, RecordingClient, SessionRejected
        log_dir = args.log_dir or os.path.join(os.path.expanduser('~'), 'arm_logs')
        name = '%s_%s_%s%s%s.mcap' % (arm.arm, dt.datetime.now().strftime('%Y%m%d_%H%M%S'), args.stage,
                                      '_teach' if args.record is not None else '', '_sim' if args.sim else '')
        meta = session_meta(arm, store, calibration, client, args.operator or getpass.getuser())
        if meta['calibration_version'] is None:
            meta['calibration_version'] = 0
        try:
            recorder = Recorder(args.record or os.path.join(log_dir, name), meta)
            # --record (teaching): every cycle, on the sim too
            client = RecordingClient(client, recorder, every=10 if args.sim and args.record is None else 1)
        except (ImportError, SessionRejected) as e:
            print('warning: not recording (%s)' % e)
            recorder = None

    extra = {}
    if args.stage == 'validate' and args.replay:
        from .log.recorder import command_array, load_session
        trajs = [command_array(load_session(path), 'q')['q'] for path in args.replay]
        extra = {'replay_trajectories': trajs, 'replay_dt': client.cmd_dt}
    if args.stage == 'gravity-check':
        from .control.float_mode import parse_payload
        extra = {'payload': parse_payload(args.payload)}
    elif args.stage == 'repeat':
        extra = {'indicators': args.indicators}
    try:
        if args.stage == 'float':
            from .control.float_mode import cli_run
            if args.sim and hasattr(client, 'reload_calibration'):
                client.reload_calibration(calibration)
            return cli_run(args, arm, build_model(arm, calibration), client, recorder)
        if args.stage == 'all':
            if not args.sim:
                p.error('arm all runs on the sim only; run hardware stages one at a time')
            ok = True
            from .calib.common import chain_skip, skipped
            for name, _ in STAGES:
                why = chain_skip(name, client)
                if why:
                    print(skipped(name, why).report())
                    continue
                r = run_stage(name, args, arm, store, client, recorder)
                ok &= r.passed
            print('ALL STAGES %s' % ('PASS' if ok else 'had failures'))
            return 0 if ok else 1
        if args.stage == 'rig':
            return run_rig(args, arm, store, client, recorder)
        r = run_stage(args.stage, args, arm, store, client, recorder, extra)
        return 0 if r.passed else 1
    finally:
        if recorder is not None:
            print('  session %s' % recorder.path)
        client.close()


if __name__ == '__main__':
    sys.exit(main())
