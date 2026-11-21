"""Local operator dashboard for a two-block stack with one physical Panthera arm."""

import argparse
import base64
import json
import math
import os
import secrets
import select
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

APP = Path(__file__).resolve().parent
DEFAULT_VISION_PYTHON = APP.parents[1] / 'panthera_sim/.venv-dino/bin/python'
LEASE_SECONDS = 4.0


class Detector:
    def __init__(self, python, config, check):
        self.proc = subprocess.Popen([str(python), '-u', str(APP / 'stack_detector_worker.py')],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            self.request({'config': config}, check)
        except BaseException:
            self.close()
            raise

    def request(self, payload, check):
        deadline = time.monotonic() + 120
        packet = memoryview(json.dumps(payload).encode() + b'\n')
        while packet:
            check()
            if time.monotonic() >= deadline:
                raise TimeoutError('Detector did not accept the camera frames within 120 seconds')
            if select.select([], [self.proc.stdin], [], .1)[1]:
                # A pipe can accept PIPE_BUF bytes atomically after reporting writable.
                count = os.write(self.proc.stdin.fileno(), packet[:select.PIPE_BUF])
                packet = packet[count:]
        while time.monotonic() < deadline:
            check()
            if select.select([self.proc.stdout], [], [], .1)[0]:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError('Detector exited. See the dashboard terminal for its error.')
                reply = json.loads(line)
                if 'error' in reply:
                    raise ValueError(reply['error'])
                return reply['result']
        raise TimeoutError('Detector did not respond within 120 seconds')

    def images(self, op, pair, q, check, **fields):
        frames = {name: base64.b64encode(frame['jpeg']).decode() for name, frame in pair.items()}
        return self.request(dict(frames, op=op, q=list(q), **fields), check)

    def locate(self, pair, q, check):
        return self.images('locate', pair, q, check)

    def calibrate(self, pair, state, check, *, finish):
        return self.images('finish' if finish else 'calibrate', pair, state[:6], check,
                           gripper=float(state[6]))

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc.stdout.close()


def hardware(arm, camera):
    from stack_hardware import StackHardware
    return StackHardware(arm, camera)


def native(sample):
    return np.r_[sample['position_rad'], sample['gripper']['position']]


def survey(hw):
    """Bracket a fresh camera pair with stationary encoder samples."""
    first = hw.read()
    after = time.monotonic_ns()
    deadline = time.monotonic() + 1
    while True:
        pair = hw.frames()
        if min(frame['monotonic_ns'] for frame in pair.values()) >= after:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('No fresh survey frame from both cameras')
        time.sleep(.015)
    last = hw.read()
    if (max(np.max(np.abs(s['velocity_rad_s'])) for s in (first, last)) > .035 or
            max(abs(s['gripper'].get('velocity', 0.)) for s in (first, last)) > .035 or
            np.max(np.abs(native(first) - native(last))) > .003):
        raise ValueError('Keep the arm and jaws still while capturing the camera pair')
    return pair, native(last)


def identify(blocks, selected):
    result = []
    for role in ('source', 'base'):
        point = selected[role]
        matches = [b for b in blocks if b['bbox_xyxy'][0] <= point['x'] < b['bbox_xyxy'][2]
                   and b['bbox_xyxy'][1] <= point['y'] < b['bbox_xyxy'][3]]
        if len(matches) != 1:
            raise ValueError(f'The {role} click must identify exactly one detected two-stud block')
        result.append(matches[0])
    if result[0] is result[1]:
        raise ValueError('Choose two different blocks')
    return result


def verify_targets(previous, current):
    for role, old, new in zip(('source', 'base'), previous, current):
        a, b = (np.asarray(item['studs_xyz']) for item in (old, new))
        drift = min(np.linalg.norm(a - row, axis=1).max() for row in (b, b[::-1]))
        if old['color'] != new['color'] or not np.isfinite(drift) or drift > .003:
            raise ValueError(f'{role.capitalize()} changed since preview. Reconnect and select again.')


def project_path(points, camera):
    """Project the planned tool-center path into the original overhead JPEG."""
    import cv2

    points = np.asarray(points, dtype=float)
    optical = np.linalg.inv(camera['world_from_camera'])
    rotation = cv2.Rodrigues(optical[:3, :3])[0]
    pixels, _ = cv2.projectPoints(points, rotation, optical[:3, 3],
                                  np.array(camera['camera_matrix'], dtype=float),
                                  np.array(camera['distortion'], dtype=float))
    depths = points @ optical[2, :3] + optical[2, 3]
    return [xy.round(1).tolist() if z > 0 and np.isfinite(xy).all() else None
            for xy, z in zip(pixels[:, 0], depths)]


class Controller:
    def __init__(self, configs, python, hw_factory=hardware, detector_factory=Detector, motion=None):
        self.configs, self.python = configs, python
        self.hw_factory, self.detector_factory, self.motion = hw_factory, detector_factory, motion
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.hw = self.opening_hw = self.detector = self.pair = self.plan = self.targets = None
        self.arm = None
        self.phase, self.message = 'disconnected', 'Connect an arm to start a survey.'
        self.overhead = None
        self.calibration_samples = 0
        self.calibration_details = ''
        self.calibration_ready = False
        self.selected = dict(source=None, base=None)
        self.details = ''
        self.overlay = []
        self.heartbeat = 0.
        self.operator = None
        if not configs:
            self.message = 'Start with --config and your measured bench calibration to enable an arm.'

    def status(self):
        with self.lock:
            return dict(phase=self.phase, message=self.message, arm=self.arm,
                        connected=self.hw is not None, running=self.busy(),
                        selected=self.selected.copy(), preview_ready=self.phase == 'ready',
                        preview_details=self.details, preview_path=self.overlay if self.phase == 'ready' else [],
                        available_arms=list(self.configs), calibration_mode='auto' if self.auto else 'fixed',
                        calibration_samples=self.calibration_samples, calibration_ready=self.calibration_ready,
                        calibration_details=self.calibration_details)

    @property
    def auto(self):
        return self.arm is not None and self.configs[self.arm]['vision'].get('auto_calibrate', False)

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def check(self):
        if self.cancel.is_set():
            raise RuntimeError('Stopped by operator')
        if self.operator is not None and time.monotonic() - self.heartbeat > LEASE_SECONDS:
            raise RuntimeError('Browser heartbeat lost')

    def start(self, action, job):
        self.phase, self.message = action, ''
        self.cancel.clear()

        def work():
            try:
                job()
            except Exception as exc:
                with self.lock:
                    retry = {'sampling': 'calibrating', 'previewing': 'preview_error'}
                    phase = retry.get(action, 'fault') if isinstance(exc, ValueError) else 'fault'
                    self.phase = 'stopped' if self.cancel.is_set() else phase
                    self.message, self.plan = str(exc), None
            finally:
                if action == 'executing' or self.cancel.is_set():
                    if self.hw:
                        try:
                            self.hw.stop()
                        except Exception as exc:
                            with self.lock:
                                self.phase, self.message = 'fault', f'Motor stop failed: {exc}'
                with self.lock:
                    self.thread = None

        self.thread = threading.Thread(target=work, name=action)
        self.thread.start()

    def connect(self, arm):
        cfg = self.configs[arm]
        hw = self.hw_factory(arm, cfg['wrist_camera'])
        vision = cfg['vision']
        if self.auto:
            vision = dict(vision, arm=arm, gripper_open_rad=cfg['gripper_open_rad'],
                          gripper_closed_rad=cfg['gripper_closed_rad'])
        detector = self.detector_factory(self.python, vision, self.check)
        try:
            with self.lock:
                self.check()
                self.opening_hw = hw
            hw.open()
            pair, initial = (None, None) if self.auto else survey(hw)
            with self.lock:
                self.check()
                self.hw, self.detector, self.pair, self.initial = hw, detector, pair, initial
                if self.auto:
                    self.phase, self.message = 'calibrating', 'Capture stationary views of the arm and blocks.'
                else:
                    self.overhead = vision['overhead']
                    self.phase, self.message = 'select', 'Select the block to lift and its base on the survey image.'
        except BaseException:
            try:
                hw.close()
            finally:
                detector.close()
            raise
        finally:
            with self.lock:
                self.opening_hw = None

    def calibration_capture(self, finish):
        pair, initial = survey(self.hw)
        result = self.detector.calibrate(pair, initial, self.check, finish=finish)
        details = [result['message']]
        for name, metric in result.get('metrics', {}).items():
            rms = metric['rms_px']
            quality = f' · {rms:.2f} px RMS' if rms is not None else ''
            details.append(f"{name.capitalize()} · {metric['inliers']} inliers{quality}")
        with self.lock:
            self.check()
            self.calibration_samples = result['samples']
            self.calibration_details = '\n'.join(details)
            if not finish:
                self.calibration_ready = result['ready']
            if finish and result['ready']:
                self.overhead = result['cameras']['overhead']
                self.pair, self.initial = pair, initial
                self.phase, self.message = 'select', 'Select the block to lift and its base on the verified survey image.'
            else:
                self.phase, self.message = 'calibrating', ''

    def preview(self):
        blocks = self.detector.locate(self.pair, self.initial[:6], self.check)
        source, base = identify(blocks, self.selected)
        cfg = self.configs[self.arm]
        plan = self.motion.build_plan(self.initial, self.hw.lower, self.hw.upper,
                                      source['studs_xyz'], base['studs_xyz'],
                                      gripper_open_rad=cfg['gripper_open_rad'],
                                      gripper_closed_rad=cfg['gripper_closed_rad'],
                                      table_z_m=cfg['vision']['table_z_m'])
        overlay = project_path(plan.tcp_path, self.overhead)

        def describe(block):
            coords = ', '.join(f'{v * 1000:.0f}' for v in block['center_xyz'])
            return f"{block['color']} ({coords})"

        with self.lock:
            self.check()
            self.plan, self.targets, self.overlay = plan, (source, base), overlay
            self.details = (f"Arm-base XYZ (mm) · {describe(source)} onto {describe(base)}. " +
                            ' → '.join(p.replace('_', ' ') for p in plan.phases))
            self.phase, self.message = 'ready', 'Check the selected blocks and clear the arm’s path.'

    def run(self):
        pair, initial = survey(self.hw)
        self.check()
        blocks = self.detector.locate(pair, initial[:6], self.check)
        verify_targets(self.targets, identify(blocks, self.selected))
        self.check()

        def phase(name):
            with self.lock:
                self.message = name.replace('_', ' ').capitalize()

        last_camera_check = 0.
        last_good_pair = time.monotonic()

        def command(target):
            nonlocal last_camera_check, last_good_pair
            self.check()
            now = time.monotonic()
            if now - last_camera_check >= .1:
                last_camera_check = now
                try:
                    self.hw.frames()
                    last_good_pair = now
                except RuntimeError as exc:
                    if now - last_good_pair >= .5:
                        raise RuntimeError(f'Lost operator camera view: {exc}') from exc
            self.hw.command(target[:6], target[6])

        plan = self.plan
        self.motion.execute_plan(plan, read=lambda: native(self.hw.read()), command=command,
                                 check_active=self.check, on_phase=phase)
        with self.lock:
            self.plan = None
            self.check()
            self.phase = 'holding'
            self.message = ('Retreat complete. Inspect the stack. Support the arm, then choose Stop motion '
                            'to release the motors.')
        target = list(plan.waypoints.values())[-1][-1]
        while True:
            command(target)
            time.sleep(1 / 30)

    def frame(self, name):
        with self.lock:
            if self.hw is None:
                raise ValueError('No connected camera')
            if self.phase in ('select', 'previewing', 'preview_error', 'ready'):
                return self.pair[name]['jpeg']
            return self.hw.frames()[name]['jpeg']

    def action(self, verb, args, operator=None):
        with self.lock:
            if verb == 'heartbeat':
                if self.operator is not None and operator != self.operator:
                    return
                self.heartbeat = time.monotonic()
            elif verb == 'stop':
                self.cancel.set()
                self.plan = None
                self.phase, self.message = 'stopped', 'Stop requested. Disconnect before starting another run.'
                hw = self.hw or self.opening_hw
                if hw:
                    hw.stop()
            elif self.busy():
                raise ValueError('A dashboard operation is already in progress')
            elif verb == 'disconnect':
                self.release()
            elif verb == 'connect':
                arm = args.get('arm')
                if self.hw or arm not in self.configs:
                    raise ValueError('Choose a configured arm while disconnected')
                if self.motion is None:
                    import stack_motion
                    self.motion = stack_motion
                self.arm = arm
                self.heartbeat = time.monotonic()
                self.start('connecting', lambda: self.connect(arm))
            elif verb in ('sample', 'finish_calibration') and self.phase == 'calibrating':
                finish = verb == 'finish_calibration'
                if finish and not self.calibration_ready:
                    raise ValueError('Capture more calibration views before finishing')
                self.start('sampling', lambda: self.calibration_capture(finish))
            elif verb == 'select' and self.phase in ('select', 'ready', 'preview_error'):
                role = args.get('role')
                x, y = args.get('x'), args.get('y')
                frame = self.pair['overhead']
                if (role not in self.selected or type(x) is not int or type(y) is not int or
                        not (0 <= x < frame['width'] and 0 <= y < frame['height'])):
                    raise ValueError('Choose a pixel inside the overhead image')
                self.selected[role] = dict(x=x, y=y)
                self.phase, self.message, self.plan, self.details = 'select', '', None, ''
            elif verb == 'preview' and self.phase in ('select', 'ready', 'preview_error') and all(self.selected.values()):
                self.plan, self.details = None, ''
                self.start('previewing', self.preview)
            elif verb == 'run' and self.phase == 'ready' and self.plan is not None:
                if not operator or len(operator) > 128:
                    raise ValueError('Run from the operator dashboard')
                self.heartbeat = time.monotonic()
                self.operator = operator
                self.start('executing', self.run)
            else:
                raise ValueError(f'Cannot {verb} while {self.phase}')

    def release(self):
        try:
            if self.hw:
                self.hw.close()
        finally:
            self.hw = None
            if self.detector:
                self.detector.close()
            self.detector = self.arm = self.plan = self.pair = self.targets = self.operator = None
            self.overhead = None
            self.calibration_samples, self.calibration_details, self.calibration_ready = 0, '', False
            self.selected, self.details = dict(source=None, base=None), ''
            self.phase, self.message = 'disconnected', 'Connect an arm to start a survey.'

    def shutdown(self):
        with self.lock:
            self.cancel.set()
            worker = self.thread
            hw = self.hw or self.opening_hw
        if hw:
            hw.stop()
        if worker:
            worker.join()
        self.release()


def handler(controller, token):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, value, mime='application/json'):
            body = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(code)
            for key, val in [('Content-Type', mime), ('Content-Length', str(len(body))),
                             ('Cache-Control', 'no-store'), ('X-Frame-Options', 'DENY'),
                             ('X-Content-Type-Options', 'nosniff')]:
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(body)

        def trusted(self):
            port = self.server.server_port
            return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')

        def do_GET(self):
            if not self.trusted():
                return self.reply(403, {})
            path = urlsplit(self.path).path
            if path == '/':
                return self.reply(200, (APP / 'web/stack/index.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/status':
                return self.reply(200, dict(controller.status(), token=token))
            if path in ('/api/frame/overhead', '/api/frame/wrist'):
                try:
                    image = controller.frame(path.split('/')[-1])
                except Exception as exc:
                    return self.reply(503, {'error': str(exc)})
                return self.reply(200, image, 'image/jpeg')
            self.reply(404, {})

        def do_POST(self):
            if (not self.trusted() or self.headers.get('Origin') not in (None, 'http://' + self.headers['Host'])
                    or self.headers.get('X-Session-Token') != token):
                return self.reply(403, {'error': 'Local dashboard token required'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('Invalid request size')
                args = json.loads(self.rfile.read(length))
                if not isinstance(args, dict) or not self.path.startswith('/api/'):
                    raise ValueError('Invalid dashboard request')
                controller.action(self.path[5:], args, self.headers.get('X-Operator-ID'))
            except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
                return self.reply(409, {'error': str(exc)})
            self.reply(200, controller.status())

        def log_message(self, *_):
            pass

    return Handler


def load_config(path):
    configs = json.loads(path.read_text())['arms']
    if not isinstance(configs, dict) or not configs or set(configs) - {'left', 'right'}:
        raise ValueError('Configure at least one of the left and right Panthera arms')
    for arm, cfg in configs.items():
        if type(cfg['vision'].get('auto_calibrate', False)) is not bool:
            raise ValueError(f'{arm}: vision.auto_calibrate must be true or false')
        if cfg['wrist_camera'] not in ('wrist_port2', 'wrist_port3'):
            raise ValueError(f'{arm}: choose the wrist camera from recording.json')
        angles = [float(cfg[k]) for k in ('gripper_open_rad', 'gripper_closed_rad')]
        if not all(map(math.isfinite, angles)) or angles[0] <= angles[1]:
            raise ValueError(f'{arm}: invalid measured open and closed gripper positions')
        for key in ('checkpoint', 'landmark_checkpoint'):
            if cfg['vision'].get(key):
                cfg['vision'][key] = str((path.parent / cfg['vision'][key]).resolve(strict=True))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Measured bench calibration JSON (enables hardware connections)')
    parser.add_argument('--vision-python', type=Path, default=DEFAULT_VISION_PYTHON)
    parser.add_argument('--port', type=int, default=8093)
    args = parser.parse_args()
    try:
        configs = load_config(args.config) if args.config else {}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(f'Could not load bench calibration {args.config}: {exc}')
    controller = Controller(configs, args.vision_python)
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler(controller, secrets.token_urlsafe(32))) as server:
        def interrupt(*_):
            controller.cancel.set()
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        print(f'Panthera stacking dashboard at http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            controller.shutdown()


if __name__ == '__main__':
    main()
