"""Hardware-free exercise of policy/human handoffs and the raw recording path."""
import io
import json
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

from remote_inference import ROOT, CONTRACT


class DemoCameras:
    def __init__(self, config):
        self.store = type('Sink', (), {'submit': lambda *_: None})()
        self.stop = threading.Event()
        self.latest = {}
        self.latest_frames = {}
        self.lock = threading.Lock()
        for name in config['cameras']:
            picture = Image.new('RGB', (640, 480), '#20344e')
            ImageDraw.Draw(picture).text((35, 35), f'SIMULATION / {name}', fill='white')
            stream = io.BytesIO()
            picture.save(stream, format='JPEG')
            self.latest[name] = stream.getvalue()
        self.thread = threading.Thread(target=self.capture, daemon=True)
        self.thread.start()

    def capture(self):
        sequence = 0
        while not self.stop.is_set():
            timestamp = time.monotonic_ns()
            try:
                for name, jpeg in self.latest.items():
                    payload = dict(camera=name, device='simulation', sequence=sequence,
                        monotonic_ns=timestamp, unix_ns=time.time_ns(), width=640, height=480, jpeg=jpeg)
                    with self.lock:
                        self.latest_frames[name] = payload
                    self.store.submit('frame', payload)
            except RuntimeError:
                return
            sequence += 1
            self.stop.wait(.05)

    def jpeg(self, name):
        return self.latest.get(name)

    def recording_frames(self):
        with self.lock:
            return dict(self.latest_frames)

    def close(self):
        self.stop.set()
        self.thread.join(2)


class DemoHardware:
    simulated = True
    def __init__(self):
        self.config = json.loads((ROOT / 'config/recording.json').read_text())
        self.config['takeover_hold'] = self.config.pop('neutral_return')
        self.pull_torque = {s: [0.]*6 for s in ('left', 'right')}
        self.auto_pull = True
        self.handoff_started = None
        self.config['minimum_free_bytes'] = 0
        self.lower = np.full(14, -1.)
        self.upper = np.full(14, 2.)
        self.position = np.zeros(14)
        self.cameras = None

    def open(self):
        self.cameras = DemoCameras(self.config)

    def prepare_handoff(self, now):
        self.handoff_started = now
        self.pull_torque = {s: [0.]*6 for s in ('left', 'right')}

    def read(self):
        if self.auto_pull and self.handoff_started is not None:
            elapsed = time.monotonic() - self.handoff_started
            for side, delay in (('left', 1.), ('right', 1.5)):
                self.pull_torque[side] = [.9 if elapsed >= delay else 0.] + [0.]*5
        arms = {side: dict(position_rad=self.position[offset:offset+6].tolist(),
            velocity_rad_s=[0.]*6, measured_torque_nm=list(self.pull_torque[side]), gravity_model_torque_nm=[0.]*6,
            gripper=dict(position=float(self.position[offset+6]), velocity=0., torque=0.))
            for side, offset in (('left', 0), ('right', 7))}
        cameras = {name: dict(ok=True, age_s=0.) for name in self.config['cameras']}
        return self.position.copy(), arms, cameras

    def images(self):
        return {}

    def command(self, target):
        self.position = target.copy()

    def guide(self, arms, target, released):
        for side, offset in (('left', 0), ('right', 7)):
            arms[side]['gripper'].update(control_mode='passive', command=[0.]*5)
            if released[side]:
                self.position[offset:offset+7] += .0003
            else:
                self.position[offset:offset+6] = target[offset:offset+6]

    def recording_context(self, checkpoint):
        return dict(mode='human_in_the_loop_simulation', simulated=True,
                    checkpoint=checkpoint, contract=CONTRACT)

    def close(self):
        if self.cameras:
            self.cameras.close()


def demo_request(base, path, body=None, timeout=3):
    if path == '/health':
        return dict(status='ready', checkpoint='simulation', contract=CONTRACT)
    state = np.asarray(body['state'])
    direction = -1 if state[0] > .2 else 1
    actions = [np.clip(state + direction * .002 * (i+1), -.5, 1.5).tolist() for i in range(50)]
    return dict(sequence=body['sequence'], fps=CONTRACT['fps'], checkpoint='simulation', mock=False,
                actions=actions)
