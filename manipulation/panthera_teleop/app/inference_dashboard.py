"""Loopback web controls for bounded, operator-stepped physical policy inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
import os
from pathlib import Path
import queue
import secrets
import signal
import sys
import threading
import time
from urllib.parse import urlsplit

import numpy as np

from remote_inference import (CONTRACT, ROOT, GRIPPER_ENCODER_TOLERANCE_RAD,
                              SmoothActionTrajectory, image_payload, request, validate_chunk)

from hitl_recording import RolloutStore
from expo_recording import ExplorationRolloutStore
from takeover_hold import TakeoverHold

LEASE_SECONDS = 2.0


class CollectionReset:
    """Collection's measured-start trajectory, capped to inference speeds."""
    def __init__(self, settings, lower, upper, initial, now):
        from neutral_return import NeutralReturn
        self.settings = dict(settings, release_delay_seconds=0,
                             max_joint_speed_rad_s=min(settings['max_joint_speed_rad_s'],
                                                       CONTRACT['max_joint_speed_rad_s']))
        limits = {s: (lower[o:o+6], upper[o:o+6]) for s, o in [('left', 0), ('right', 7)]}
        self.neutral = NeutralReturn(self.settings, limits, now)
        self.gripper_settings = settings['gripper_open']
        fraction = self.gripper_settings['travel_fraction']
        self.gripper_targets = lower[[6, 13]] + fraction * (upper[[6, 13]] - lower[[6, 13]])
        self.started = now

    def tick(self, now, dt, measured, target):
        arms = {s: dict(position_rad=measured[o:o+6], velocity_rad_s=[0]*6)
                for s, o in [('left', 0), ('right', 7)]}
        joints, settled, _ = self.neutral.tick(now, arms)
        # A reset created mid-loop can be newer than this tick's timestamp.
        if joints is None:
            return target.copy(), False
        result = target.copy()
        for side, offset in [('left', 0), ('right', 7)]:
            result[offset:offset+6] = joints[side]
        speed = min(self.gripper_settings['speed_rad_s'], CONTRACT['gripper_speed_rad_s'])
        result[[6, 13]] += np.clip(self.gripper_targets-target[[6, 13]], -speed*dt, speed*dt)
        opened = np.all(np.abs(measured[[6, 13]]-self.gripper_targets) <=
                        self.gripper_settings['arrival_tolerance_rad'])
        if not opened and now-self.started > self.gripper_settings['timeout_seconds']:
            raise RuntimeError('Collection reset grippers did not reach their opening; reset stopped')
        commanded_open = np.allclose(result[[6, 13]], self.gripper_targets, atol=1e-8, rtol=0)
        return result, bool(settled and opened and commanded_open)


def action_count(value):
    if type(value) is not int or not 1 <= value <= 50:
        raise ValueError('Actions per step must be an integer from 1 to 50')
    return value


def blend_count(value):
    if type(value) is not int or not 0 <= value <= 50:
        raise ValueError('Blend actions must be an integer from 0 to 50')
    return value


def blend_predictions(actions, previous_tail, count):
    """Crossfade aligned absolute targets, then let the bounded path retime them.

    Observations are taken at stopped step boundaries, so old row N aligns with
    new row 0. No wall-clock latency offset applies while the robot is holding.
    This is client-side overlap blending, not model-side RTC guidance.
    """
    count = blend_count(count)
    result = np.asarray(actions, dtype=float).copy()
    if previous_tail is None:
        return result, 0
    overlap = min(count, len(previous_tail), len(result))
    if overlap:
        u = np.arange(1, overlap + 1, dtype=float) / overlap
        weight = (u*u*(3 - 2*u))[:, None]
        result[:overlap] = (1 - weight) * previous_tail[:overlap] + weight * result[:overlap]
    return result, overlap


class ActionStep(SmoothActionTrajectory):
    """Blend exactly N policy waypoints, then stop and hold the last target."""
    def __init__(self, actions, count, started):
        selected = actions[:action_count(count)]
        if len(selected) != count:
            raise ValueError('Prediction contains fewer actions than requested')
        super().__init__(selected, started)


class RobotHardware:
    """All SDK/camera access belongs to the single control worker."""
    def __init__(self):
        self.robots = {}
        self.cameras = None
        self.lock_file = None
        self.stale = {}
        self.hold_targets = {}
        self.check_active = lambda: None

    def open(self):
        from hand_guide import setup
        from dashboard import Cameras
        from usb_stability import keep_usb_awake
        import yaml

        (ROOT / 'logs').mkdir(exist_ok=True)
        self.lock_file = (ROOT / 'logs/control.lock').open('a')
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another robot controller is running; stop it first')
        configs = setup.prepare()
        self.configs = configs
        self.check_active()
        self.config = json.loads((ROOT / 'config/recording.json').read_text())
        self.cameras = Cameras(type('Sink', (), {'submit': lambda *_: None})(), self.config)
        self.cameras.wait_ready()
        self.check_active()
        sys.path.insert(0, str(setup.SDK / 'scripts'))
        from Panthera_lib import Panthera
        mappings = json.loads((setup.CONFIG / 'arm-mapping.json').read_text())
        for side in ('left', 'right'):
            self.check_active()
            expected = yaml.safe_load((setup.CONFIG / f'{side}-motors.yaml').read_text())['robot']['Serial_Type']
            if str(setup.resolve_board(mappings[side])) != expected:
                raise RuntimeError('Robot USB mapping changed during startup')
            keep_usb_awake(expected)
            self.robots[side] = Panthera(str(configs[side]))
            setup.wait_for_joint_feedback(self.robots[side], side)
            self.check_active()
        self.lower = np.concatenate([np.r_[self.robots[s].joint_limits['lower'],
                                           self.robots[s].gripper_limits['lower']]
                                     for s in ('left', 'right')])
        self.upper = np.concatenate([np.r_[self.robots[s].joint_limits['upper'],
                                           self.robots[s].gripper_limits['upper']]
                                     for s in ('left', 'right')])

    def read(self):
        from hand_guide import feedback
        arms = {s: feedback(r, s, self.stale)[1] for s, r in self.robots.items()}
        measured = np.asarray([v for s in ('left', 'right') for v in
                               arms[s]['position_rad'] + [arms[s]['gripper']['position']]])
        for i in (6, 13):
            if not self.lower[i] - GRIPPER_ENCODER_TOLERANCE_RAD <= measured[i] <= self.upper[i] + GRIPPER_ENCODER_TOLERANCE_RAD:
                raise RuntimeError('Gripper feedback outside encoder tolerance')
        camera_state = self.cameras.status()
        return measured, arms, camera_state

    def images(self):
        with self.cameras.lock:
            return {n: image_payload(self.cameras.latest[n]['jpeg'], n)
                    for n in CONTRACT['image_crops_xyxy']}

    def expo_images(self):
        """Return each exact inference JPEG together with its source frame receipt."""
        with self.cameras.lock:
            frames = {n: self.cameras.latest[n] for n in CONTRACT['image_crops_xyxy']}
            return ({n: image_payload(frame['jpeg'], n) for n, frame in frames.items()},
                    {n: {k: v for k, v in frame.items() if k != 'jpeg'}
                     for n, frame in frames.items()})

    def command(self, target):
        for offset, side in ((0, 'left'), (7, 'right')):
            self.command_arm(side, offset, target)

    def command_arm(self, side, offset, target, hold_gripper=True):
        robot = self.robots[side]
        if not robot.Joint_Pos_Vel(target[offset:offset + 6],
                                   np.full(6, CONTRACT['max_joint_speed_rad_s']),
                                   max_tqu=CONTRACT['torque_limits_nm'], iswait=False):
            raise RuntimeError(f'{side}: joint command rejected')
        if hold_gripper and not robot.gripper_control(float(target[offset + 6]),
                                     CONTRACT['gripper_speed_rad_s'],
                                     CONTRACT['gripper_torque_limit_nm']):
            raise RuntimeError(f'{side}: gripper command rejected')

    def recording_context(self, checkpoint):
        from hand_guide import setup, TORQUE_LIMIT
        import yaml
        return dict(mode='human_in_the_loop', checkpoint=checkpoint, contract=CONTRACT,
                    torque_limits_nm=TORQUE_LIMIT.tolist(),
                    calibration=json.loads((ROOT / 'config/calibration/setup.json').read_text()),
                    arm_mapping=json.loads((setup.CONFIG / 'arm-mapping.json').read_text()),
                    robot_configs={s: yaml.safe_load(p.read_text()) for s, p in self.configs.items()},
                    motor_configs={s: yaml.safe_load((setup.CONFIG / f'{s}-motors.yaml').read_text())
                                   for s in self.configs},
                    urdf_sources={s: Path(yaml.safe_load(p.read_text())['urdf']['file_path']).read_text()
                                  for s, p in self.configs.items()})

    def guide(self, arms, target, released):
        from hand_guide import ZERO
        from reset_grippers import command_gripper
        for side, robot in self.robots.items():
            if not released[side]:
                self.command_arm(side, 0 if side == 'left' else 7, target, hold_gripper=False)
                command_gripper(robot, arms[side])
                arms[side]['control_mode'] = 'position_hold'
                continue
            torque = np.asarray(arms[side]['gravity_model_torque_nm'])
            if not robot.pos_vel_tqe_kp_kd(ZERO, ZERO, torque, ZERO, ZERO):
                raise RuntimeError(f'{side}: gravity command rejected')
            command_gripper(robot, arms[side])
            arms[side]['control_mode'] = 'gravity_assist'
            arms[side]['gravity_torque_command_nm'] = torque.tolist()
            arms[side]['command'] = dict(mode='gravity_assist', torque_nm=torque.tolist())

    def validate_reset_path(self, initial, targets):
        from hand_guide import validate_return_path
        for side, offset in [('left', 0), ('right', 7)]:
            validate_return_path(self.robots[side], initial[offset:offset+6], targets[side], side)

    def hold_available(self):
        """Preserve even a partially opened session after a startup fault."""
        from hand_guide import feedback
        errors = []
        for side, robot in self.robots.items():
            try:
                if side not in self.hold_targets:
                    arm = feedback(robot, side, self.stale)[1]
                    q = np.asarray(arm['position_rad'], dtype=float)
                    if (not np.isfinite(q).all() or np.any(q < robot.joint_limits['lower']) or
                            np.any(q > robot.joint_limits['upper'])):
                        raise RuntimeError('Cannot establish a valid measured hold')
                    self.hold_targets[side] = q
                if not robot.Joint_Pos_Vel(self.hold_targets[side],
                                           np.full(6, CONTRACT['max_joint_speed_rad_s']),
                                           max_tqu=CONTRACT['torque_limits_nm'], iswait=False):
                    raise RuntimeError('Hold command rejected')
            except BaseException as exc:
                errors.append(f'{side}: {exc}')
        if errors:
            raise RuntimeError('; '.join(errors))

    def close(self):
        from hand_guide import stop_robots
        stop_robots(self.robots)
        try:
            if self.cameras:
                self.cameras.close()
        finally:
            self.robots.clear()
            if self.lock_file:
                self.lock_file.close()


class Controller:
    def __init__(self, url, hardware_factory=RobotHardware, log_path=None, data_dir=None, expo_record_dir=None,
                 expo_objective='gear_and_pin', expo_prefix_length=1):
        self.url = url
        self.data_dir = Path(data_dir) if data_dir else ROOT / 'data/hitl'
        self.recorder = None
        self.expo_record_dir = Path(expo_record_dir) if expo_record_dir else None
        self.expo_recorder = None
        if expo_objective not in ('gear_insertion', 'gear_and_pin'):
            raise ValueError('Unknown EXPO training objective')
        self.expo_objective = expo_objective
        self.expo_prefix_length = action_count(expo_prefix_length)
        self.collection_commands = queue.Queue(maxsize=1)
        self.hardware_factory = hardware_factory
        self.log_path = log_path or ROOT / 'logs/inference-steps.jsonl'
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.commands = queue.Queue(maxsize=1)
        self.worker = None
        self.hardware = None
        self.heartbeat_at = 0
        self.state = dict(mode='stopped', error=None, checkpoint=None, measured=None,
                          target=None, cameras={}, prediction=None, action_index=0,
                          step_count=1, steps_completed=0, latency_s=None,
                          playing=False, actions_per_step=1, blend_actions=10,
                          blended_actions=0,
                          hold_status='unverified', control_source='hold', rollout_active=False,
                          recording=None, collection_pending=False, handoff_ready={}, handoff_baseline_ready=False, handoff_gripper_enabled=True)
        if self.expo_record_dir is not None:
            self.state.update(actions_per_step=self.expo_prefix_length, step_count=self.expo_prefix_length,
                              blend_actions=0, expo_recording_enabled=True, expo_prefix_length=self.expo_prefix_length,
                              expo_objective=expo_objective)

    def check_expo_configuration(self, count, blend):
        if self.expo_record_dir is not None and (count != self.expo_prefix_length or blend != 0):
            raise ValueError(f'EXPO collection requires exactly {self.expo_prefix_length} actions per step and 0 blended actions')

    @staticmethod
    def expo_images(hardware, cameras):
        if hasattr(hardware, 'expo_images'):
            return hardware.expo_images()
        return hardware.images(), dict(camera_frame_metadata_exact=False, status=cameras)

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def snapshot(self):
        with self.lock:
            state = json.loads(json.dumps(self.state))
            state['connected'] = self.worker is not None and self.worker.is_alive()
            if self.recorder:
                state['recording'] = self.recorder.snapshot()
            if self.expo_recorder:
                state['expo_recording'] = self.expo_recorder.snapshot()
            return state

    def heartbeat(self):
        with self.lock:
            self.heartbeat_at = time.monotonic()

    def connect(self, reset_pose=True):
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                raise RuntimeError('A robot session is already active')
            self.start_session(reset_pose=reset_pose)

    def start_session(self, reset_pose):
        """Called with self.lock held, after the previous worker has exited."""
        self.stop_event.clear()
        self.pause_event.clear()
        self.commands = queue.Queue(maxsize=1)
        self.collection_commands = queue.Queue(maxsize=1)
        self.heartbeat_at = time.monotonic()
        self.state.update(mode='connecting', error=None, prediction=None,
                          measured=None, target=None, cameras={}, action_index=0,
                          playing=False, blended_actions=0,
                          hold_status='unverified', control_source='hold', rollout_active=False,
                          collection_pending=False, handoff_ready={}, handoff_baseline_ready=False)
        self.worker = threading.Thread(target=self.run, args=(reset_pose,),
                                       daemon=True, name='policy-control')
        self.worker.start()

    def configure(self, count, blend):
        count, blend = action_count(count), blend_count(blend)
        self.check_expo_configuration(count, blend)
        with self.lock:
            if self.state['mode'] not in ('holding', 'queued', 'predicting', 'executing'):
                raise RuntimeError('Connect and wait for the arms to hold before configuring playback')
            self.state.update(actions_per_step=count, blend_actions=blend)

    def submit(self, kind, count=1, blend=10):
        if kind not in ('step', 'play', 'predict', 'reset'):
            raise ValueError('Unknown inference command')
        count = action_count(count)
        blend = blend_count(blend)
        if kind in ('step', 'play'):
            self.check_expo_configuration(count, blend)
        with self.lock:
            if kind == 'reset' and self.state['rollout_active']:
                raise RuntimeError('Finish the rollout before resetting')
            if (self.state['mode'] != 'holding' or self.state['playing'] or self.stop_event.is_set() or
                    self.pause_event.is_set()):
                raise RuntimeError('Connect and wait for the arms to hold before taking a step')
            if time.monotonic() - self.heartbeat_at > LEASE_SECONDS:
                raise RuntimeError('Browser connection is stale; motion remains paused')
            if (self.expo_record_dir is not None and kind in ('step', 'play') and
                    not self.state['rollout_active']):
                raise RuntimeError('Record an EXPO trial before playing or stepping the policy')
            try:
                self.commands.put_nowait((kind, count))
            except queue.Full:
                raise RuntimeError('A command is already pending')
            self.state['mode'] = 'queued'
            self.state['error'] = None
            self.state['playing'] = kind == 'play'
            if kind in ('step', 'play'):
                self.state.update(actions_per_step=count, blend_actions=blend)

    def collection(self, action, payload=None):
        payload = payload or {}
        if self.expo_record_dir is not None and any(payload.get(key, self.expo_objective) != self.expo_objective
                for key in ('objective', 'training_objective')):
            raise ValueError('Trial objective must match the configured EXPO collection objective')
        if action not in ('begin', 'intervene', 'resume', 'finish'):
            raise ValueError('Unknown collection action')
        if action == 'begin':
            for key in ('operator', 'scene'):
                if not isinstance(payload.get(key), str) or not 1 <= len(payload[key].strip()) <= 120:
                    raise ValueError(f'Enter an {key} name (1–120 characters)')
        if action == 'finish':
            if type(payload.get('reset_after_finish', False)) is not bool:
                raise ValueError('reset_after_finish must be a boolean')
            milestones = payload.get('milestones', {})
            if (not isinstance(milestones, dict) or
                    any(key not in ('gear_pickup', 'gear_transport', 'gear_insertion', 'gear_seating',
                                    'pin_pickup', 'pin_insertion', 'pin_pushing') or value not in ('success', 'failure')
                        for key, value in milestones.items())):
                raise ValueError('Choose success or failure for known task milestones')
            if payload.get('label') not in ('kept', 'review', 'discarded'):
                raise ValueError('Choose keep, review, or discard')
            if type(payload.get('success')) is not bool:
                raise ValueError('Mark success or failure')
            if type(payload.get('quality')) is not int or not 1 <= payload['quality'] <= 5:
                raise ValueError('Quality must be an integer from 1 to 5')
            if not isinstance(payload.get('notes', ''), str) or len(payload.get('notes', '')) > 1000:
                raise ValueError('Notes must be at most 1000 characters')
            if 'review_approved' in payload and not (self.expo_record_dir is not None and
                    payload.get('label') == 'review' and payload.get('success') is False and
                    payload['review_approved'] is True):
                raise ValueError('Review approval is available only for an explicitly failed EXPO trial')
        with self.lock:
            if self.state['collection_pending']:
                raise RuntimeError('A collection change is already pending')
            if self.state['mode'] not in ('holding', 'queued', 'predicting', 'executing', 'takeover_hold', 'human'):
                raise RuntimeError('Connect and wait for the collection pose')
            if time.monotonic() - self.heartbeat_at > LEASE_SECONDS or self.pause_event.is_set():
                raise RuntimeError('Browser connection is stale or motion is stopping')
            if action == 'begin' and (self.state['mode'] != 'holding' or self.state['rollout_active']):
                raise RuntimeError('Begin from hold with no active rollout')
            if action == 'finish' and not self.state['rollout_active']:
                raise RuntimeError('No correction or rollout is recording')
            if action == 'resume' and self.state['mode'] not in ('takeover_hold', 'human'):
                raise RuntimeError('Resume is available during human correction')
            if action == 'intervene' and self.state['mode'] in ('takeover_hold', 'human'):
                raise RuntimeError('Human correction is already active')
            if action in ('begin', 'finish') and (self.recorder is None or self.recorder.failure):
                raise RuntimeError('Rollout recorder is unavailable')
            try:
                self.collection_commands.put_nowait((action, payload))
                self.state['collection_pending'] = True
            except queue.Full:
                raise RuntimeError('A collection change is already pending')

    def stop(self):
        """Cancel motion, retaining the connection and powered position hold."""
        with self.lock:
            self.state['playing'] = False
            self.pause_event.set()

    def shutdown(self):
        """Release resources only for a deliberate controller shutdown."""
        self.stop_event.set()

    def check_active(self):
        if self.stop_event.is_set():
            raise RuntimeError('Operator stopped the session')

    def clear_commands(self):
        while True:
            try:
                self.commands.get_nowait()
            except queue.Empty:
                return

    def jpeg(self, name):
        hardware = self.hardware
        return hardware.cameras.jpeg(name) if hardware and hardware.cameras else None

    def collection_reset(self, hardware, measured):
        if 'neutral_return' not in hardware.config:
            return None
        reset = CollectionReset(hardware.config['neutral_return'], hardware.lower,
                                hardware.upper, measured, time.monotonic())
        hardware.validate_reset_path(measured, reset.neutral.targets)
        self.update(mode='resetting', prediction=None, action_index=0)
        return reset

    def run(self, reset_pose=True):
        hardware = self.hardware_factory()
        hardware.check_active = self.check_active
        self.hardware = hardware
        pool = ThreadPoolExecutor(max_workers=1)
        future = step = None
        previous_tail = None
        target = None
        sequence = 0
        sample_sequence = 0
        last_policy_clock = None
        expo_feedback_pending = False
        human = rollout = False
        handoff = None
        released = {}
        takeover_details = {}
        completed = self.state['steps_completed']
        try:
            health = request(self.url, '/health')
            if health['status'] != 'ready':
                raise RuntimeError(health.get('error') or 'AWS inference is not ready')
            if health['contract'] != CONTRACT:
                raise RuntimeError('AWS inference has a different robot contract')
            checkpoint = health['checkpoint']
            if (self.expo_record_dir is not None and health.get('chunk_size') is not None and
                    health['chunk_size'] != self.expo_prefix_length):
                raise RuntimeError('EXPO server chunk size differs from the fixed collection prefix length')
            if self.expo_record_dir is not None and (
                    health.get('training_objective') is not None and
                    health['training_objective'] != self.expo_objective or
                    health.get('rl_model_id') is not None and
                    health.get('training_objective') != self.expo_objective):
                raise RuntimeError('EXPO server training objective differs from the collection objective')
            expo_policy_identity = {key: health[key] for key in
                                    ('checkpoint_id', 'processor_id', 'normalizer_hash', 'rl_model_id', 'training_objective')
                                    if health.get(key) is not None}
            if self.expo_record_dir is not None:
                self.expo_recorder = ExplorationRolloutStore(self.expo_record_dir,
                    dict(checkpoint=checkpoint, checkpoint_id=health.get('checkpoint_id'),
                         processor_id=health.get('processor_id'), contract=CONTRACT,
                         policy_bundle_id=health.get('policy_bundle_id'), server_url=self.url,
                         policy_identity=expo_policy_identity, server_health=health,
                         training_objective=self.expo_objective,
                         fixed_prefix_length=self.expo_prefix_length,
                         simulated=getattr(hardware, 'simulated', False)))
            self.update(checkpoint=checkpoint)
            self.check_active()
            hardware.open()
            self.check_active()
            if hardware.cameras is not None:
                self.recorder = RolloutStore(self.data_dir, hardware.config,
                                            hardware.recording_context(checkpoint))
                hardware.cameras.store = self.recorder
            measured, arms, cameras = hardware.read()
            target = measured.copy()
            target[[6, 13]] = np.clip(target[[6, 13]], hardware.lower[[6, 13]], hardware.upper[[6, 13]])
            self.update(measured=measured.tolist(), target=target.tolist(), cameras=cameras,
                        simulation=getattr(hardware, 'simulated', False))
            reset = self.collection_reset(hardware, measured) if reset_pose else None
            if reset is None:
                self.update(mode='holding')
            last_tick = last_publish = time.monotonic()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a') as log:
                while not self.stop_event.is_set():
                    now = time.monotonic()
                    dt = now - last_tick
                    last_tick = now
                    self.check_active()
                    measured, arms, cameras = hardware.read()
                    measured_at = time.monotonic_ns()
                    expo_next_images = None
                    expo_next_observation_id = None
                    # This feedback reflects the preceding command, including
                    # the final waypoint read on the tick after completion.
                    policy_clock = last_policy_clock
                    last_policy_clock = None
                    with self.lock:
                        browser_lost = time.monotonic() - self.heartbeat_at > LEASE_SECONDS
                    if self.pause_event.is_set() or browser_lost:
                        if self.expo_recorder:
                            self.expo_recorder.boundary('browser_connection_lost' if browser_lost else 'operator_stop')
                            if browser_lost:
                                self.expo_recorder.end(dict(label='review', success=None, complete=False,
                                                            reason='browser_connection_lost'))
                        expo_feedback_pending = False
                        policy_clock = None
                        if future is not None:
                            future.cancel()
                            sequence += 1
                        if human:
                            target = np.clip(measured, hardware.lower, hardware.upper)
                        human = False
                        handoff = None
                        released = {}
                        self.update(handoff_ready={}, handoff_baseline_ready=False)
                        if browser_lost and rollout:
                            self.recorder.submit('rollout_end', dict(label='review', success=False,
                                quality=1, complete=False, reason='browser_connection_lost'))
                            rollout = False
                        self.update(rollout_active=rollout)
                        while not self.collection_commands.empty():
                            self.collection_commands.get_nowait()
                        future = step = reset = None
                        previous_tail = None
                        self.clear_commands()
                        self.pause_event.clear()
                        self.update(mode='holding', playing=False, blended_actions=0,
                                    prediction=None, action_index=0, control_source='hold', collection_pending=False,
                                    error='Browser connection lost; motion canceled, arms remain held'
                                    if browser_lost else 'Motion stopped; arms remain held')
                    try:
                        if dt > 0.1:
                            raise RuntimeError('Local control loop missed its deadline; motion canceled')
                        if (rollout or human or reset is not None or step is not None or future is not None or not self.commands.empty()) and not all(
                                c['ok'] and c['age_s'] < CONTRACT['max_camera_age_s'] for c in cameras.values()):
                            raise RuntimeError('Camera stream missing or stale; motion canceled')
                        if not human and np.max(np.abs(target - measured)) > CONTRACT['max_tracking_error_rad']:
                            target = np.clip(measured, hardware.lower, hardware.upper)
                            raise RuntimeError('Arm tracking deviation; holding measured pose')
                        if self.recorder and self.recorder.failure:
                            raise RuntimeError(f'Recording failed: {self.recorder.failure}')
                        if self.expo_recorder and self.expo_recorder.failure:
                            raise RuntimeError(f'EXPO recording failed: {self.expo_recorder.failure}')
                        if expo_feedback_pending:
                            next_images, next_cameras = self.expo_images(hardware, cameras)
                            expo_next_images = (next_images, next_cameras)
                            expo_next_observation_id = self.expo_recorder.completed(
                                measured.tolist(), next_images, measured_at, next_cameras)
                            expo_feedback_pending = False
                        change = None
                        try:
                            change = self.collection_commands.get_nowait()
                        except queue.Empty:
                            pass
                        if change:
                            policy_clock = None
                            action, details = change
                            if self.expo_recorder:
                                if action == 'begin':
                                    self.expo_recorder.begin(details, measured_at)
                                elif action == 'finish':
                                    self.expo_recorder.end(dict(details, complete=True))
                                else:
                                    self.expo_recorder.boundary(action)
                            expo_feedback_pending = False
                            if future is not None:
                                future.cancel()
                                sequence += 1
                            future = step = None
                            previous_tail = None
                            self.clear_commands()
                            target = np.clip(measured, hardware.lower, hardware.upper)
                            if action == 'begin':
                                self.recorder.submit('rollout_begin', dict(
                                    sample=dict(monotonic_ns=measured_at),
                                    details=dict(details, actions_per_step=self.state['actions_per_step'],
                                                 blend_actions=self.state['blend_actions'])))
                                rollout = True
                            elif action == 'intervene':
                                settings = hardware.config.get('takeover_hold', hardware.config.get('neutral_return'))
                                if settings is None:
                                    raise RuntimeError('Handle-pull takeover is not configured')
                                handoff = TakeoverHold({side: target[offset:offset+6]
                                    for side, offset in (('left', 0), ('right', 7))}, settings, hardware.config, now)
                                released = dict(handoff.ready)
                                takeover_details = details
                                if hasattr(hardware, 'prepare_handoff'):
                                    hardware.prepare_handoff(now)
                                human = True
                            elif action == 'resume':
                                human = False
                                # Explicit Play after regripping avoids restarting while hands are on the arms.
                            elif action == 'finish':
                                human = rollout = False
                                self.recorder.submit('rollout_end', details)
                            if not human:
                                handoff = None
                                released = {}
                            self.update(mode='takeover_hold' if human else 'holding', playing=False,
                                        handoff_ready=released, handoff_baseline_ready=False,
                                        rollout_active=rollout, control_source='hold',
                                        prediction=None, action_index=0, error=None, collection_pending=False)
                            if (action == 'finish' and details.get('reset_after_finish', False)
                                    and not self.pause_event.is_set()):
                                self.check_active()
                                reset = self.collection_reset(hardware, measured)
                                if reset is None:
                                    raise RuntimeError('Recording finished, but collection reset is not configured')
                        if human:
                            previous_ready = released.copy()
                            released = handoff.tick(now, arms)
                            for side, offset in (('left', 0), ('right', 7)):
                                # Waiting grippers are passive, just like normal collection.
                                target[offset+6] = np.clip(measured[offset+6], hardware.lower[offset+6], hardware.upper[offset+6])
                                if released[side]:
                                    target[offset:offset+7] = np.clip(measured[offset:offset+7],
                                        hardware.lower[offset:offset+7], hardware.upper[offset:offset+7])
                            if any(released.values()) and not rollout and self.recorder:
                                names = {key: (takeover_details.get(key, '').strip()[:120] or 'unspecified')
                                         if isinstance(takeover_details.get(key, ''), str) else 'unspecified'
                                         for key in ('operator', 'scene')}
                                self.recorder.submit('rollout_begin', dict(sample=dict(monotonic_ns=measured_at),
                                    details=dict(names, recording_start='human_takeover',
                                        handoff='handle_pull_or_gripper', handoff_settings=dict(handoff.settings, **handoff.motion_settings),
                                        actions_per_step=self.state['actions_per_step'],
                                        blend_actions=self.state['blend_actions'])))
                                rollout = True
                            for side in released:
                                if released[side] and not previous_ready[side]:
                                    event = dict(event=handoff.diagnostics[side]['trigger']+'_release', side=side, monotonic_ns=measured_at,
                                                 **handoff.diagnostics[side])
                                    log.write(json.dumps(event) + '\n')
                                    log.flush()
                                    if rollout:
                                        self.recorder.submit('handoff_event', event)
                            self.update(mode='human' if any(released.values()) else 'takeover_hold',
                                rollout_active=rollout, handoff_ready=released,
                                handoff_baseline_ready=len(handoff.baselines) == len(released))
                        if reset is not None:
                            target, reset_done = reset.tick(now, dt, measured, target)
                            if reset_done:
                                log.write(json.dumps(dict(unix_ns=time.time_ns(), event='collection_reset_completed',
                                                          target=target.tolist())) + '\n')
                                log.flush()
                                reset = None
                                self.update(mode='holding')
                        if not human and reset is None and future is None and step is None:
                            command = None
                            try:
                                command = self.commands.get_nowait()
                            except queue.Empty:
                                with self.lock:
                                    if self.state['playing'] and not self.pause_event.is_set():
                                        command = ('play', self.state['actions_per_step'])
                            if command is not None:
                                kind, count = command
                                with self.lock:
                                    blend = self.state['blend_actions']
                                    if kind == 'play':
                                        count = self.state['actions_per_step']
                                if kind == 'reset':
                                    reset = self.collection_reset(hardware, measured)
                                    if reset is None:
                                        raise RuntimeError('Collection reset is not configured')
                                else:
                                    if self.expo_recorder and kind in ('step', 'play'):
                                        images, request_cameras = (expo_next_images if expo_next_images is not None
                                                                  else self.expo_images(hardware, cameras))
                                    else:
                                        images = hardware.images()
                                    observed_at = time.monotonic()
                                    inference_body = dict(sequence=sequence, state=measured.tolist(), images=images)
                                    if self.expo_recorder and kind in ('step', 'play'):
                                        observation_id = self.expo_recorder.requested(sequence, measured.tolist(), images,
                                            measured_at, request_cameras, expo_next_observation_id)
                                        if observation_id is not None:
                                            inference_body['observation_id'] = observation_id
                                    future = pool.submit(request, self.url, '/infer',
                                                         inference_body, 3)
                                    self.update(mode='predicting', step_count=count, prediction=None, action_index=0)
                        if future is not None:
                            if now - observed_at > CONTRACT['max_observation_age_s']:
                                raise RuntimeError('Inference observation expired')
                            if future.done():
                                result = future.result()
                                if self.expo_recorder and kind in ('step', 'play'):
                                    self.expo_recorder.responded(result)
                                if result['checkpoint'] != checkpoint:
                                    raise RuntimeError('AWS checkpoint changed; reconnect before moving')
                                if self.expo_recorder and any(result.get(key) != value
                                        for key, value in expo_policy_identity.items()):
                                    raise RuntimeError('EXPO policy identity changed; reconnect before moving')
                                latency = time.monotonic() - observed_at
                                log.write(json.dumps(dict(unix_ns=time.time_ns(), kind=kind, count=count,
                                                          latency_s=latency, measured=measured.tolist(),
                                                          response=result)) + '\n')
                                log.flush()
                                raw_actions = validate_chunk(result, sequence, target, hardware.lower, hardware.upper,
                                                         action_count=count if kind == 'step' or self.expo_recorder else None)
                                if (self.expo_recorder and result.get('chunk_size') is not None and
                                        result['chunk_size'] != self.expo_prefix_length):
                                    raise RuntimeError('EXPO response chunk size differs from the fixed collection prefix length')
                                tail = None
                                if kind == 'play' and previous_tail is not None:
                                    old_tail, old_observed_at = previous_tail
                                    if now - old_observed_at <= CONTRACT['max_observation_age_s']:
                                        tail = old_tail
                                actions, overlap = blend_predictions(raw_actions, tail, blend if kind == 'play' else 0)
                                self.update(prediction=actions.tolist(), latency_s=latency,
                                            blended_actions=overlap)
                                if kind == 'play':
                                    log.write(json.dumps(dict(unix_ns=time.time_ns(), event='play_blend',
                                                              sequence=sequence, count=count,
                                                              blend_actions=blend, blended_actions=overlap,
                                                              actions=actions.tolist())) + '\n')
                                    log.flush()
                                sequence += 1
                                future = None
                                if kind in ('step', 'play'):
                                    step = ActionStep(actions, count, now)
                                    if self.expo_recorder:
                                        self.expo_recorder.accepted(result, actions[:count].tolist())
                                    self.update(mode='executing')
                                else:
                                    self.update(mode='holding')
                        if step is not None:
                            target, done = step.tick(now, dt, target)
                            progress, rate = step.policy_clock(now)
                            last_policy_clock = (sequence - 1, progress, rate)
                            if done:
                                if self.expo_recorder and self.expo_recorder.active:
                                    expo_feedback_pending = True
                                completed += 1
                                self.update(mode='holding', action_index=count, steps_completed=completed)
                                log.write(json.dumps(dict(unix_ns=time.time_ns(), event='step_completed',
                                                          count=count, sequence=sequence - 1)) + '\n')
                                log.flush()
                                step = None
                                previous_tail = (actions[count:].copy(), observed_at) if kind == 'play' else None
                            else:
                                self.update(action_index=step.index + 1)
                    except Exception as exc:
                        if self.expo_recorder:
                            try:
                                self.expo_recorder.boundary('control_fault: ' + str(exc))
                                self.expo_recorder.end(dict(label='review', success=None, complete=False,
                                                            reason='control_fault: ' + str(exc)))
                            except RuntimeError:
                                pass
                        expo_feedback_pending = False
                        policy_clock = last_policy_clock = None
                        if future is not None:
                            future.cancel()
                        if human:
                            target = np.clip(measured, hardware.lower, hardware.upper)
                        human = False
                        handoff = None
                        released = {}
                        self.update(handoff_ready={}, handoff_baseline_ready=False)
                        if rollout and self.recorder and not self.recorder.failure:
                            self.recorder.submit('rollout_end', dict(label='review', success=False,
                                quality=1, complete=False, reason='control_fault: ' + str(exc)))
                        rollout = False
                        self.update(rollout_active=False, control_source='hold', collection_pending=False)
                        future = step = reset = None
                        previous_tail = None
                        self.clear_commands()
                        while not self.collection_commands.empty():
                            self.collection_commands.get_nowait()
                        sequence += 1
                        self.update(mode='holding', playing=False, blended_actions=0,
                                    error=str(exc) + '; command rejected, arms remain held')
                    if self.stop_event.is_set():
                        break
                    if human:
                        hardware.guide(arms, target, released)
                    else:
                        hardware.command(target)
                        if self.expo_recorder and last_policy_clock is not None:
                            self.expo_recorder.command(target.tolist(), time.monotonic_ns(), last_policy_clock[1])
                    source = 'human' if human and any(released.values()) else 'policy' if policy_clock else 'hold'
                    self.update(hold_status=('gravity_assist' if all(released.values()) else 'partial_hold')
                                if source == 'human' else 'commanded', control_source=source)
                    if self.recorder:
                        for side, offset in (('left', 0), ('right', 7)):
                            if not released.get(side, False):
                                arms[side]['gravity_torque_command_nm'] = None
                                arms[side]['command'] = dict(mode='joint_position_velocity',
                                    position_rad=target[offset:offset+6].tolist(),
                                    speed_limit_rad_s=CONTRACT['max_joint_speed_rad_s'],
                                    torque_limit_nm=CONTRACT['torque_limits_nm'])
                                if not human:
                                    arms[side]['gripper']['command'] = dict(mode='position_velocity',
                                        position_rad=float(target[offset+6]),
                                        speed_limit_rad_s=CONTRACT['gripper_speed_rad_s'],
                                        torque_limit_nm=CONTRACT['gripper_torque_limit_nm'])
                        self.recorder.submit('sample', dict(monotonic_ns=measured_at,
                            unix_ns=time.time_ns(), sequence=sample_sequence, arms=arms,
                            control_source=source, command_target_rad=None if source == 'human' else target.tolist(),
                            control_source_by_arm={side: 'human' if released.get(side, False) else
                                'policy' if source == 'policy' else 'hold' for side in ('left', 'right')},
                            camera_frames=hardware.cameras.recording_frames(),
                            policy_time_s=policy_clock[1] if policy_clock else None,
                            policy_clock_rate=policy_clock[2] if policy_clock else None,
                            control_compute_s=time.monotonic()-now,
                            inference_sequence=policy_clock[0] if policy_clock else sequence-1,
                            actions_per_step=self.state['actions_per_step'], blend_actions=self.state['blend_actions']))
                        sample_sequence += 1
                    if now - last_publish >= 0.05:
                        self.update(measured=measured.tolist(), target=target.tolist(), cameras=cameras)
                        last_publish = now
                    time.sleep(max(0, 1 / hardware.config['control_hz'] - (time.monotonic() - now)))
        except BaseException as exc:
            if self.expo_recorder:
                try:
                    self.expo_recorder.boundary('control_fault: ' + str(exc))
                    self.expo_recorder.end(dict(label='review', success=None, complete=False,
                                                reason='control_fault: ' + str(exc)))
                except RuntimeError:
                    pass
            self.update(mode='fault', playing=False, error=str(exc), rollout_active=False,
                        control_source='hold', handoff_ready={}, handoff_baseline_ready=False)
            if self.recorder:
                try:
                    # Never wait for disk flush before establishing powered fault hold.
                    self.recorder.submit('rollout_end', dict(label='review', success=False,
                        quality=1, complete=False, reason='control_fault: ' + str(exc)))
                except RuntimeError:
                    pass  # Writer failure preserves Review/pending for recovery.
            print(f'Policy session fault: {exc}', flush=True)
            # A failed policy, camera, or log must never tear down an established
            # powered hold. Keep SDK objects and the control lock alive.
            if target is not None or getattr(hardware, 'robots', {}):
                self.update(mode='fault_hold', hold_status='unverified',
                            error=f'{exc}; motion blocked. Reconnect after supporting both arms.')
                while not self.stop_event.is_set():
                    try:
                        if target is None:
                            hardware.hold_available()
                        else:
                            hardware.command(target)
                        self.update(hold_status='commanded')
                    except BaseException as hold_error:
                        self.update(hold_status='failed', error=f'{exc}; hold command failed: {hold_error}. '
                                    'Powered hold is not reliable. Support both arms, check USB/power, '
                                    'then use Reconnect.')
                    self.stop_event.wait(1 / hardware.config['control_hz'])
        finally:
            try:
                if self.expo_recorder:
                    self.expo_recorder.close('session_stopped')
                if self.recorder:
                    self.recorder.close('session_stopped')
                hardware.close()
            except BaseException as exc:
                self.update(mode='fault', error=f'Robot cleanup failed: {exc}')
            finally:
                self.hardware = None
                pool.shutdown(wait=False, cancel_futures=True)
                if self.snapshot()['mode'] != 'fault':
                    self.update(mode='stopped', hold_status='unverified')


def session_process(pipe, url, hardware_factory, health_request=None, data_dir=None, expo_record_dir=None,
                    expo_objective='gear_and_pin', expo_prefix_length=1):
    """SDK faults, native aborts and blocked calls stay outside the web process."""
    # Ctrl-C reaches the whole foreground process group. Only the web process
    # decides whether shutdown may release a connected arm session.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if health_request is not None:
        global request
        request = health_request
    controller = Controller(url, hardware_factory, data_dir=data_dir, expo_record_dir=expo_record_dir,
                            expo_objective=expo_objective, expo_prefix_length=expo_prefix_length)
    try:
        while True:
            method, args = pipe.recv()
            if method == 'shutdown':
                controller.shutdown()
                if controller.worker:
                    controller.worker.join(timeout=2)
                # Native destructors can send on invalid serial descriptors.
                # Process exit closes all descriptors without running them again.
                os._exit(0)
            try:
                result = getattr(controller, method)(*args)
                pipe.send((True, result))
            except Exception as exc:
                pipe.send((False, str(exc)))
    except (EOFError, BrokenPipeError):
        # Parent disappeared: do not silently release an existing powered hold.
        controller.stop()
        if controller.worker:
            controller.worker.join()


class SessionSupervisor:
    def __init__(self, url, hardware_factory=RobotHardware, health_request=None, data_dir=None, expo_record_dir=None,
                 expo_objective='gear_and_pin', expo_prefix_length=1):
        self.url = url
        self.hardware_factory = hardware_factory
        self.health_request = health_request
        self.data_dir = data_dir
        self.expo_record_dir = expo_record_dir
        self.expo_objective = expo_objective
        self.expo_prefix_length = expo_prefix_length
        self.lock = threading.RLock()
        self.recovery_cancel = threading.Event()
        self.recovery_worker = None
        self.process = self.pipe = None
        self.state = Controller(url, expo_record_dir=expo_record_dir, expo_objective=expo_objective,
                                expo_prefix_length=expo_prefix_length).snapshot()
        self.spawn()

    def spawn(self):
        context = multiprocessing.get_context('spawn')
        self.pipe, child_pipe = context.Pipe()
        self.process = context.Process(target=session_process,
                                       args=(child_pipe, self.url, self.hardware_factory,
                                             self.health_request, self.data_dir, self.expo_record_dir,
                                             self.expo_objective, self.expo_prefix_length), daemon=True)
        self.process.start()
        child_pipe.close()

    def rpc(self, method, *args):
        with self.lock:
            if self.process is None or not self.process.is_alive():
                raise RuntimeError('Robot session exited. Support both arms and use Reconnect.')
            try:
                self.pipe.send((method, args))
                if not self.pipe.poll(1):
                    # Never consume a late reply as the answer to another command.
                    self.pipe.close()
                    raise RuntimeError('Robot session is unresponsive. Support both arms and use Reconnect.')
                ok, result = self.pipe.recv()
            except (OSError, EOFError, ValueError) as exc:
                raise RuntimeError('Robot session connection failed. Support both arms and use Reconnect.') from exc
            if not ok:
                raise RuntimeError(result)
            return result

    def snapshot(self):
        with self.lock:
            recovering = self.recovery_worker is not None and self.recovery_worker.is_alive()
            if not recovering:
                try:
                    self.state = self.rpc('snapshot')
                except RuntimeError as exc:
                    self.state.update(mode='fault', playing=False, hold_status='unverified', error=str(exc),
                                      connected=self.process is not None and self.process.is_alive())
            state = json.loads(json.dumps(self.state))
            state['recovering'] = recovering
            return state

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def connect(self):
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                raise RuntimeError('Connection recovery is already running')
            self.rpc('connect')

    def heartbeat(self):
        with self.lock:
            if not (self.recovery_worker and self.recovery_worker.is_alive()):
                self.rpc('heartbeat')

    def configure(self, count, blend):
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                raise RuntimeError('Wait for recovery before configuring playback')
            self.rpc('configure', count, blend)

    def submit(self, kind, count=1, blend=10):
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                raise RuntimeError('Wait for recovery before taking a step')
            self.rpc('submit', kind, count, blend)

    def collection(self, action, payload=None):
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                raise RuntimeError('Wait for recovery before collecting')
            self.rpc('collection', action, payload)

    def jpeg(self, name):
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                return None
            try:
                return self.rpc('jpeg', name)
            except RuntimeError:
                return None

    def stop(self):
        self.recovery_cancel.set()
        with self.lock:
            if not (self.recovery_worker and self.recovery_worker.is_alive()):
                self.rpc('stop')

    def recover(self, arms_supported=False):
        if arms_supported is not True:
            raise ValueError('Support both arms before reconnecting: powered hold will be released')
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                raise RuntimeError('Connection recovery is already running')
            if self.snapshot()['mode'] not in ('fault', 'fault_hold', 'holding', 'stopped'):
                raise RuntimeError('Stop motion before reconnecting')
            self.recovery_cancel.clear()
            self.state.update(mode='recovering', prediction=None, hold_status='unverified',
                              error='Releasing old connection; keep both arms supported')
            self.recovery_worker = threading.Thread(target=self.finish_recovery, daemon=True)
            self.recovery_worker.start()

    def release_process(self):
        # Used only after explicit support confirmation, or ordinary shutdown of
        # a disconnected dashboard. Never automatically replace a powered session.
        if self.process and self.process.is_alive():
            try:
                self.pipe.send(('shutdown', ()))
            except (OSError, ValueError):
                pass
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=3)
            if self.process.is_alive():
                raise RuntimeError('Old robot process did not exit; reconnection blocked')
        if self.pipe:
            self.pipe.close()

    def finish_recovery(self):
        try:
            # HTTP Stop remains responsive while waiting for native cleanup.
            self.release_process()
            with self.lock:
                self.spawn()
                if self.recovery_cancel.is_set():
                    self.state.update(mode='stopped', connected=False,
                                      error='Recovery canceled; keep both arms supported')
                    return
                self.rpc('connect', False)
        except Exception as exc:
            self.update(mode='fault', error=f'Reconnection failed: {exc}', hold_status='unverified')

    def shutdown(self):
        self.recovery_cancel.set()
        if self.recovery_worker:
            self.recovery_worker.join(timeout=8)
        self.release_process()


def make_server(controller, port):
    token = secrets.token_urlsafe(32)
    html_path = Path(__file__).parent / 'web/inference.html'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, body, mime='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def trusted(self):
            return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')

        def do_GET(self):
            if not self.trusted():
                return self.send(403, {})
            path = urlsplit(self.path).path
            if path == '/':
                return self.send(200, html_path.read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/status':
                return self.send(200, dict(controller.snapshot(), token=token, fps=CONTRACT['fps'],
                                           speed_limit=CONTRACT['max_joint_speed_rad_s'],
                                           gripper_speed_limit=CONTRACT['gripper_speed_rad_s']))
            if path.startswith('/camera/'):
                name = path.rsplit('/', 1)[-1]
                if name not in CONTRACT['image_crops_xyxy']:
                    return self.send(404, {})
                jpeg = controller.jpeg(name)
                return self.send(200 if jpeg else 503, jpeg or b'', 'image/jpeg')
            self.send(404, {})

        def do_POST(self):
            if (not self.trusted() or self.headers.get('X-Session-Token') != token or
                    self.headers.get('Origin') not in (None, f'http://127.0.0.1:{port}', f'http://localhost:{port}')):
                return self.send(403, {'error': 'Invalid session'})
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 2048:
                    raise ValueError('Invalid request size')
                self.connection.settimeout(2)
                payload = json.loads(self.rfile.read(size))
                path = urlsplit(self.path).path
                if path in ('/api/stop', '/api/pause'):
                    controller.stop()
                elif path == '/api/heartbeat':
                    controller.heartbeat()
                elif path == '/api/connect':
                    controller.connect()
                elif path == '/api/recover':
                    controller.recover(payload.get('arms_supported', False))
                elif path == '/api/reset':
                    controller.submit('reset')
                elif path.startswith('/api/collection/'):
                    controller.collection(path.rsplit('/', 1)[-1], payload)
                elif path == '/api/configure':
                    controller.configure(payload.get('count'), payload.get('blend'))
                elif path in ('/api/predict', '/api/step', '/api/play'):
                    controller.submit(path.rsplit('/', 1)[-1], payload.get('count', 1), payload.get('blend', 10))
                else:
                    return self.send(404, {})
                self.send(202, {'accepted': True})
            except (ValueError, TypeError, AttributeError, RuntimeError, queue.Full) as exc:
                self.send(409, {'error': str(exc)})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--data-dir', type=Path, help='Raw rollout directory (default data/hitl)')
    parser.add_argument('--expo-record-dir', type=Path,
                        help='Optional separate EXPO decision ledger; fixed action prefix and 0 blending')
    parser.add_argument('--expo-prefix-length', type=int, choices=range(1, 51), default=17,
                        help='Fixed EXPO action chunk length for this session (default 17); cannot change during a session')
    parser.add_argument('--expo-objective', choices=('gear_and_pin', 'gear_insertion'), default='gear_and_pin',
                        help='Fixed EXPO outcome objective; base pi0.5 task prompt is unchanged')
    parser.add_argument('--demo', action='store_true', help='Simulated policy, arms and cameras; no devices opened')
    args = parser.parse_args()
    if args.demo:
        from hitl_demo import DemoHardware, demo_request
        import tempfile
        data_dir = args.data_dir or Path(tempfile.mkdtemp(prefix='panthera-hitl-demo-'))
        controller = SessionSupervisor(args.url, DemoHardware, demo_request, data_dir, args.expo_record_dir,
                                       args.expo_objective, args.expo_prefix_length)
        print(f'SIMULATION — recordings: {data_dir}', flush=True)
    else:
        controller = SessionSupervisor(args.url, data_dir=args.data_dir, expo_record_dir=args.expo_record_dir,
                                       expo_objective=args.expo_objective, expo_prefix_length=args.expo_prefix_length)
    server = make_server(controller, args.port)
    def shutdown(*_):
        state = controller.snapshot()
        if not args.demo and (state['connected'] or state.get('recovering')):
            try:
                controller.stop()
            except RuntimeError:
                pass
            controller.update(error='Shutdown deferred: arms remain connected and held')
            return
        controller.shutdown()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f'Policy step controls: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        controller.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
