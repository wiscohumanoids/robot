"""Exclusive access to one Panthera follower and the two cameras used for stacking."""

import fcntl
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMAND_TIMEOUT_S = 0.5


class _DiscardFrames:
    def submit(self, kind, frame):
        pass


class StackHardware:
    def __init__(self, arm, wrist_camera):
        if arm not in ('left', 'right'):
            raise ValueError('Select the left or right arm')
        self.arm = arm
        self.wrist_camera = wrist_camera
        self._io = threading.RLock()
        self._stopped = threading.Event()
        self._stop_sent = False
        self._last_command = None
        self._lock_file = None
        self._config_dir = None
        self._robot = None
        self._cameras = None
        self._watcher = None
        self._stale = {}
        self._reason = 'Arm stopped. Reconnect to run another stack.'

    def open(self):
        """Open the selected devices under the shared controller lock."""
        from dashboard import Cameras
        from hand_guide import setup
        from usb_stability import keep_usb_awake
        import yaml

        if self._lock_file or self._stopped.is_set():
            raise RuntimeError('Start a new stacking connection')
        config = json.loads((ROOT / 'config/recording.json').read_text())
        self.contract = json.loads((ROOT / 'infra/inference/contract.json').read_text())
        if self.wrist_camera == 'overhead' or self.wrist_camera not in config['cameras']:
            raise ValueError('Select the calibrated wrist camera from config/recording.json')
        devices = config['cameras']
        if Path(devices['overhead']).resolve() == Path(devices[self.wrist_camera]).resolve():
            raise ValueError('Overhead and wrist cameras must be separate devices')
        self._frame_period_ns = round(1e9 / config['camera_fps'])
        (ROOT / 'logs').mkdir(exist_ok=True)
        self._lock_file = (ROOT / 'logs/control.lock').open('a')
        try:
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Another robot controller owns the arms') from exc
            self._cameras = Cameras(_DiscardFrames(), dict(config, cameras={
                'overhead': devices['overhead'], 'wrist': devices[self.wrist_camera]}))
            self._cameras.wait_ready()
            self._config_dir = TemporaryDirectory(prefix='panthera-stack-')
            directory = Path(self._config_dir.name)
            configs = setup.prepare((self.arm,), directory, round(COMMAND_TIMEOUT_S * 1000))
            expected = yaml.safe_load((directory / f'{self.arm}-motors.yaml').read_text())['robot']['Serial_Type']
            mapping = json.loads((setup.CONFIG / 'arm-mapping.json').read_text())
            if str(setup.resolve_board(mapping[self.arm])) != expected:
                raise RuntimeError('Robot USB mapping changed during connection')
            keep_usb_awake(expected)
            sdk_scripts = str(setup.SDK / 'scripts')
            if sdk_scripts not in sys.path:
                sys.path.insert(0, sdk_scripts)
            from Panthera_lib import Panthera
            with self._io:
                self._require_active()
                self._robot = Panthera(str(configs[self.arm]))
                setup.wait_for_joint_feedback(self._robot, self.arm)
                self._require_active()
                self.lower, self.upper = (np.r_[self._robot.joint_limits[key], self._robot.gripper_limits[key]]
                                          for key in ('lower', 'upper'))
                self.read()
                self._watcher = threading.Thread(target=self._watch_commands, daemon=True)
                self._watcher.start()
        except SystemExit as exc:
            self.close()
            raise RuntimeError('The Panthera SDK could not initialize the arm') from exc
        except BaseException:
            self.close()
            raise

    def _require_active(self):
        if self._stopped.is_set():
            raise RuntimeError(self._reason)

    def read(self):
        """Read feedback in SDK radians and the tool pose in the arm base frame."""
        from hand_guide import feedback
        from remote_inference import GRIPPER_ENCODER_TOLERANCE_RAD

        with self._io:
            self._require_active()
            if self._robot is None:
                raise RuntimeError('Connect the arm first')
            try:
                if self._last_command is None:
                    # The SDK getters read its last CAN reply. Ask for a new
                    # sample while stationary before using the wrist pose.
                    motors = self._robot.Motors
                    before = [motor.get_current_motor_state().time for motor in motors]
                    deadline = time.monotonic() + COMMAND_TIMEOUT_S
                    while True:
                        self._require_active()
                        self._robot.send_get_motor_state_cmd()
                        self._robot.motor_send_cmd()
                        time.sleep(0.005)
                        if all(motor.get_current_motor_state().time != old
                               for motor, old in zip(motors, before)):
                            break
                        if time.monotonic() >= deadline:
                            raise RuntimeError('No fresh arm feedback before command')
                _, arm = feedback(self._robot, self.arm, self._stale)
                opening = arm['gripper']['position']
                if (opening < self.lower[6] - GRIPPER_ENCODER_TOLERANCE_RAD
                        or opening > self.upper[6] + GRIPPER_ENCODER_TOLERANCE_RAD):
                    raise RuntimeError('Gripper feedback outside encoder tolerance')
                return arm
            except Exception:
                self.stop()
                raise

    def frames(self):
        """Return uncropped JPEG receipts with bounded age and overhead/wrist receipt skew."""
        cameras = self._cameras
        if cameras is None:
            raise RuntimeError('Connect the cameras first')
        with cameras.lock:
            if cameras.errors:
                raise RuntimeError(f'Camera capture failed: {dict(cameras.errors)}')
            frames = dict(cameras.latest)
        if not {'wrist', 'overhead'} <= frames.keys():
            raise RuntimeError('Waiting for both cameras')
        now = time.monotonic_ns()
        times = [frames[name]['monotonic_ns'] for name in ('wrist', 'overhead')]
        if any(now - stamp > self.contract['max_camera_age_s'] * 1e9 for stamp in times):
            raise RuntimeError('Camera frame is stale')
        if max(times) - min(times) > self._frame_period_ns:
            raise RuntimeError('Camera frame receipts differ by more than one frame period')
        return frames

    def command(self, q_rad, gripper_rad):
        """Send a bounded joint target and gripper encoder target to the selected arm."""
        with self._io:
            self._require_active()
            try:
                if (self._last_command is not None
                        and time.monotonic() - self._last_command > COMMAND_TIMEOUT_S):
                    raise RuntimeError('Stack command stream timed out')
                measured = self.read()
                q = np.asarray(q_rad, dtype=float)
                target = np.r_[q, gripper_rad]
                if (q.shape != (6,) or target.shape != (7,) or not np.isfinite(target).all()
                        or np.any(target < self.lower) or np.any(target > self.upper)):
                    raise ValueError('Stack target outside measured arm configuration limits')
                if np.max(np.abs(q - measured['position_rad'])) > self.contract['max_tracking_error_rad']:
                    raise RuntimeError('Stack target exceeds arm tracking limit')
                self._require_active()
                if not self._robot.Joint_Pos_Vel(q, np.full(6, self.contract['max_joint_speed_rad_s']),
                                                 max_tqu=self.contract['torque_limits_nm'], iswait=False):
                    raise RuntimeError('Panthera rejected the joint target')
                self._require_active()
                if not self._robot.gripper_control(float(gripper_rad), self.contract['gripper_speed_rad_s'],
                                                   self.contract['gripper_torque_limit_nm']):
                    raise RuntimeError('Panthera rejected the gripper target')
                self._last_command = time.monotonic()
            except Exception:
                self.stop()
                raise

    def _watch_commands(self):
        while not self._stopped.wait(COMMAND_TIMEOUT_S / 10):
            with self._io:
                if self._last_command is not None and time.monotonic() - self._last_command > COMMAND_TIMEOUT_S:
                    self._reason = 'Stack command stream timed out. Reconnect to run again.'
                    self.stop()
                    return

    def stop(self):
        """Latch out further motion and send the SDK stop to the selected controller."""
        self._stopped.set()
        with self._io:
            if self._robot is not None and not self._stop_sent:
                self._robot.set_stop()
                self._stop_sent = True

    def close(self):
        try:
            self.stop()
        finally:
            if self._watcher and self._watcher is not threading.current_thread():
                self._watcher.join()
            with self._io:
                self._robot = None
            try:
                if self._cameras is not None:
                    self._cameras.close()
                    self._cameras = None
            finally:
                if self._config_dir:
                    self._config_dir.cleanup()
                    self._config_dir = None
                if self._lock_file:
                    self._lock_file.close()
                    self._lock_file = None
