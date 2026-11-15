"""SSH-forwarded policy client and locally bounded physical-arm controller."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import io
import json
from pathlib import Path
import signal
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'infra/inference/contract.json').read_text())
sys.path.insert(0, str(ROOT / 'infra/inference'))
from action_limits import GRIPPER_ENCODER_TOLERANCE_RAD, clamp_gripper_actions


def request(base, path, body=None, timeout=3):
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    req = Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            details = json.loads(exc.read(4096))
        except (ValueError, OSError):
            details = {}
        message = details.get('error') if isinstance(details, dict) else None
        if isinstance(message, str) and message:
            raise RuntimeError(f'Inference service HTTP {exc.code}: {message[:1000]}') from exc
        raise


def image_payload(jpeg, name):
    with Image.open(io.BytesIO(jpeg)) as source:
        if source.size != (640, 480):
            raise RuntimeError(f'{name}: expected 640x480 camera input')
        image = source.convert('RGB').crop(tuple(CONTRACT['image_crops_xyxy'][name]))
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=95, subsampling=0)
        return base64.b64encode(output.getvalue()).decode()


def validate_chunk(response, sequence, initial, lower, upper, physical=True, action_count=None):
    if response['sequence'] != sequence or response['fps'] != CONTRACT['fps']:
        raise RuntimeError('Response sequence/rate mismatch')
    if physical and response.get('mock'):
        raise RuntimeError('Mock predictions cannot control physical arms')
    actions = np.asarray(response['actions'], dtype=float)
    if (actions.ndim != 2 or actions.shape[1] != 14 or
            not 1 <= len(actions) <= 50 or not np.isfinite(actions).all()):
        raise RuntimeError('Malformed action chunk')
    if action_count is not None:
        if type(action_count) is not int or not 1 <= action_count <= len(actions):
            raise RuntimeError('Prediction contains fewer actions than requested or invalid action count')
    actions, _ = clamp_gripper_actions(actions, lower[[6, 13]], upper[[6, 13]])
    selected = actions if action_count is None else actions[:action_count]
    violations = np.argwhere((selected < lower) | (selected > upper))
    if len(violations):
        action, index = violations[0]
        side = 'left' if index < 7 else 'right'
        joint = 'gripper' if index % 7 == 6 else f'joint{index % 7 + 1}'
        raise RuntimeError(f'Policy target outside robot joint/gripper limits: '
                           f'action {action + 1}, {side}.{joint}={actions[action, index]:.6f} rad '
                           f'(allowed {lower[index]:.6f} to {upper[index]:.6f})')
    # Raw predictions are waypoints, not instantaneous motor commands. Large
    # in-limit changes are slowed by SmoothActionTrajectory before execution.
    return actions


class SmoothActionTrajectory:
    """Interpolate waypoints with a bounded, continuous-velocity quintic path.

    Same-direction waypoints share velocity; reversals, holds and the two path
    endpoints have zero velocity. Acceleration is zero at every knot. Monotone
    Bezier control points prevent overshoot, including at physical joint limits.
    The entire path is retimed when necessary rather than dropping waypoints.
    """
    def __init__(self, actions, started):
        self.actions = np.asarray(actions, dtype=float).copy()
        if (self.actions.ndim != 2 or self.actions.shape[1] != 14 or
                not 1 <= len(self.actions) <= 50 or not np.isfinite(self.actions).all()):
            raise ValueError('Expected 1 to 50 finite 14D waypoints')
        self.started = started
        self.index = 0
        self.origin = None
        self.duration = None
        self.ends = None
        self.speed = np.full(14, CONTRACT['max_joint_speed_rad_s'])
        self.speed[[6, 13]] = CONTRACT['gripper_speed_rad_s']
        self.acceleration = np.full(14, CONTRACT['max_joint_acceleration_rad_s2'])
        self.acceleration[[6, 13]] = CONTRACT['gripper_acceleration_rad_s2']

    @staticmethod
    def _derivative_bound(values):
        # Subdivide the derivative curves three times to tighten their convex
        # hull bound. This avoids slowing a simple ramp by a coarse hull bound.
        polygons = values[:, None, :, :]
        degree = values.shape[1] - 1
        for _ in range(3):
            work = polygons
            left, right = [work[:, :, 0]], [work[:, :, -1]]
            for _ in range(degree):
                work = (work[:, :, :-1] + work[:, :, 1:]) / 2
                left.append(work[:, :, 0])
                right.append(work[:, :, -1])
            polygons = np.concatenate([np.stack(left, axis=2),
                                       np.stack(right[::-1], axis=2)], axis=1)
        return np.max(np.abs(polygons), axis=(1, 2))

    def _plan(self, target):
        self.origin = target.copy()
        points = np.vstack([self.origin, self.actions])
        delta = np.diff(points, axis=0)
        distance = np.abs(delta)
        durations = np.maximum(1 / CONTRACT['fps'], np.max(distance / self.speed, axis=1))
        slopes = delta / durations[:, None]
        velocity = np.zeros_like(points)
        # Common knot tangents are capped by BOTH adjacent secants. This keeps
        # control points ordered, even with uneven waypoint spacing.
        same_direction = slopes[:-1] * slopes[1:] > 0
        velocity[1:-1] = np.where(same_direction, np.sign(slopes[:-1]) *
                                  np.minimum(np.abs(slopes[:-1]), np.abs(slopes[1:])), 0.)
        controls = np.stack([points[:-1],
                             points[:-1] + velocity[:-1] * durations[:, None] / 5,
                             points[:-1] + 2 * velocity[:-1] * durations[:, None] / 5,
                             points[1:] - 2 * velocity[1:] * durations[:, None] / 5,
                             points[1:] - velocity[1:] * durations[:, None] / 5,
                             points[1:]], axis=1)
        # Derivatives lie in the convex hull of derivative control points; the
        # bounds cover the whole curve, not just control-loop samples. Uniform
        # retiming preserves common knot velocities and zero knot acceleration.
        peak_speed = np.max(self._derivative_bound(5 * np.diff(controls, axis=1)) /
                            durations[:, None] / self.speed)
        peak_acceleration = np.max(self._derivative_bound(20 * np.diff(controls, n=2, axis=1)) /
                                   durations[:, None]**2 / self.acceleration)
        scale = max(1., float(peak_speed), float(np.sqrt(peak_acceleration)))
        self.durations = durations * scale
        self.ends = np.cumsum(self.durations)
        self.controls = controls
        self.duration = float(self.ends[-1])

    def remaining(self, now):
        return float('inf') if self.ends is None else max(0., self.started + self.duration - now)

    def policy_clock(self, now):
        """Policy seconds and their local rate along the safety-retimed path."""
        if self.ends is None:
            return 0., 0.
        elapsed = max(0., now - self.started)
        segment = min(int(np.searchsorted(self.ends, elapsed, side='right')), len(self.actions) - 1)
        beginning = self.ends[segment - 1] if segment else 0.
        fraction = float(np.clip((elapsed - beginning) / self.durations[segment], 0., 1.))
        return (segment + fraction) / CONTRACT['fps'], 1 / (CONTRACT['fps'] * self.durations[segment])

    def tick(self, now, dt, target):
        if not np.isfinite(dt) or not 0 <= dt <= 0.1:
            raise RuntimeError('Local control loop missed its deadline')
        if self.index >= len(self.actions):
            return target.copy(), True
        if self.origin is None:
            self._plan(target)
        elapsed = max(0., now - self.started)
        segment = min(int(np.searchsorted(self.ends, elapsed, side='right')), len(self.actions) - 1)
        beginning = self.ends[segment - 1] if segment else 0.
        u = np.clip((elapsed - beginning) / self.durations[segment], 0., 1.)
        # De Casteljau evaluation retains the no-overshoot property numerically.
        work = self.controls[segment].copy()
        for size in range(5, 0, -1):
            work[:size] = (1 - u) * work[:size] + u * work[1:size + 1]
        desired = work[0]
        # Bound the actual emitted command, including jitter or delayed ticks.
        bound = np.minimum(self.speed * dt, CONTRACT['max_target_jump_rad'])
        result = target + np.clip(desired - target, -bound, bound)
        self.index = segment
        if elapsed >= self.duration and np.allclose(result, self.actions[-1], rtol=0, atol=1e-10):
            self.index = len(self.actions)
        return result, self.index == len(self.actions)


class InferenceChunkStream:
    """One in-flight request and one validated chunk, with bounded lookahead."""
    def __init__(self):
        self.future = self.pending = None
        self.sequence = 0
        self.latency = .3

    def needs_request(self, now, trajectory):
        lead = min(CONTRACT['max_observation_age_s'] / 2, max(.1, self.latency * 1.5 + .05))
        return (self.future is None and self.pending is None and
                (trajectory is None or trajectory.remaining(now) <= lead))

    def submit(self, pool, url, state, images, now):
        self.observed_at = now
        self.future = pool.submit(request, url, '/infer',
                                  dict(sequence=self.sequence, state=state.tolist(), images=images), 3)

    def poll(self, now, target, lower, upper):
        if self.future is None:
            return None
        if now - self.observed_at > CONTRACT['max_observation_age_s']:
            raise RuntimeError('Inference observation expired')
        if not self.future.done():
            return None
        result = self.future.result()
        actions = validate_chunk(result, self.sequence, target, lower, upper)
        self.pending = (actions, self.observed_at)
        self.latency = now - self.observed_at
        self.sequence += 1
        self.future = None
        return result

    def activate(self, now):
        if self.pending is None:
            return None
        actions, observed_at = self.pending
        if now - observed_at > CONTRACT['max_observation_age_s']:
            raise RuntimeError('Buffered inference observation expired')
        self.pending = None
        return SmoothActionTrajectory(actions, now)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8765')
    p.add_argument('--run', action='store_true', help='Enable physical arms and cameras')
    p.add_argument('--seconds', type=float, default=30, help='Bounded physical session duration')
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        p.error('--seconds must be positive and finite')
    health = request(args.url, '/health')
    print(json.dumps(health, indent=2), flush=True)
    if health['contract'] != CONTRACT:
        raise RuntimeError('Server/client deployment contracts differ')
    if health['status'] not in ('ready', 'mock'):
        return 2
    if not args.run:
        images = {}
        for name in CONTRACT['image_crops_xyxy']:
            buffer = io.BytesIO()
            Image.new('RGB', (640, 480)).save(buffer, format='JPEG')
            images[name] = image_payload(buffer.getvalue(), name)
        started = time.monotonic()
        result = request(args.url, '/infer', dict(sequence=0, state=[0.0]*14, images=images), timeout=60)
        validate_chunk(result, 0, np.zeros(14), np.full(14, -np.inf), np.full(14, np.inf), physical=False)
        print(json.dumps(dict(probe=True, mock=result['mock'], actions=len(result['actions']),
                              latency_s=time.monotonic()-started, checkpoint=result['checkpoint'])))
        return 0
    if health['status'] != 'ready':
        raise RuntimeError('Physical execution requires a loaded real policy')
    from hand_guide import setup, feedback, stop_robots
    from dashboard import Cameras
    from usb_stability import keep_usb_awake
    import yaml
    configs = setup.prepare()
    lock_path = ROOT / 'logs/control.lock'
    lock_path.parent.mkdir(exist_ok=True)
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    robots = {}
    cameras = None
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    class CameraSink:
        def submit(self, *_):
            pass

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        config = json.loads((ROOT / 'config/recording.json').read_text())
        cameras = Cameras(CameraSink(), config)
        cameras.wait_ready()
        sys.path.insert(0, str(setup.SDK / 'scripts'))
        from Panthera_lib import Panthera
        for side in ('left', 'right'):
            mapping = json.loads((setup.CONFIG / 'arm-mapping.json').read_text())[side]
            expected = yaml.safe_load((setup.CONFIG / f'{side}-motors.yaml').read_text())['robot']['Serial_Type']
            if str(setup.resolve_board(mapping)) != expected:
                raise RuntimeError('USB mapping changed')
            keep_usb_awake(expected)
            robots[side] = Panthera(str(configs[side]))
            setup.wait_for_joint_feedback(robots[side], side)
        lower = np.concatenate([np.r_[robots[s].joint_limits['lower'], robots[s].gripper_limits['lower']]
                                for s in ('left', 'right')])
        upper = np.concatenate([np.r_[robots[s].joint_limits['upper'], robots[s].gripper_limits['upper']]
                                for s in ('left', 'right')])
        stale = {}
        target = None
        trajectory = None
        stream = InferenceChunkStream()
        started = last_tick = time.monotonic()
        deadline = started + args.seconds
        print('PHYSICAL POLICY CONTROL ACTIVE. Ctrl-C stops both arms.', flush=True)
        with (ROOT / 'logs/inference-actions.jsonl').open('a') as log:
            while not stopping and time.monotonic() < deadline:
                now = time.monotonic()
                dt = now - last_tick
                last_tick = now
                if dt > 0.1:
                    raise RuntimeError('Local control loop missed its deadline')
                arms = {s: feedback(r, s, stale)[1] for s, r in robots.items()}
                measured = np.asarray([v for s in ('left', 'right') for v in
                                       arms[s]['position_rad'] + [arms[s]['gripper']['position']]])
                # Recorded feedback can be slightly negative at the closed stop.
                # Bound that encoder tolerance without sending negative targets.
                for i in (6, 13):
                    if not lower[i]-GRIPPER_ENCODER_TOLERANCE_RAD <= measured[i] <= upper[i]+GRIPPER_ENCODER_TOLERANCE_RAD:
                        raise RuntimeError('Gripper feedback outside encoder tolerance')
                if target is None:
                    target = measured.copy()
                    target[[6, 13]] = np.clip(target[[6, 13]], lower[[6, 13]], upper[[6, 13]])
                if np.max(np.abs(target-measured)) > CONTRACT['max_tracking_error_rad']:
                    raise RuntimeError('Arm tracking deviation')
                if not all(c['ok'] and c['age_s'] < CONTRACT['max_camera_age_s']
                           for c in cameras.status().values()):
                    raise RuntimeError('Camera stream missing or stale')
                if stream.needs_request(now, trajectory):
                    with cameras.lock:
                        frames = {n: image_payload(cameras.latest[n]['jpeg'], n)
                                  for n in CONTRACT['image_crops_xyxy']}
                    stream.submit(pool, args.url, measured.copy(), frames, now)
                result = stream.poll(now, target, lower, upper)
                if result is not None:
                    log.write(json.dumps(dict(unix_ns=time.time_ns(), latency_s=stream.latency,
                                              response=result))+'\n')
                    log.flush()
                if trajectory is None:
                    trajectory = stream.activate(now)
                if trajectory is not None:
                    target, done = trajectory.tick(now, dt, target)
                    if done:
                        trajectory = stream.activate(now)
                for offset, side in ((0, 'left'), (7, 'right')):
                    r = robots[side]
                    if not r.Joint_Pos_Vel(target[offset:offset+6],
                                           np.full(6, CONTRACT['max_joint_speed_rad_s']),
                                           max_tqu=CONTRACT['torque_limits_nm'], iswait=False):
                        raise RuntimeError(f'{side}: joint command rejected')
                    if not r.gripper_control(float(target[offset+6]), CONTRACT['gripper_speed_rad_s'],
                                             CONTRACT['gripper_torque_limit_nm']):
                        raise RuntimeError(f'{side}: gripper command rejected')
                time.sleep(max(0, 1/config['control_hz']-(time.monotonic()-now)))
    finally:
        stop_robots(robots)
        if cameras:
            cameras.close()
        pool.shutdown(wait=False, cancel_futures=True)
        lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
