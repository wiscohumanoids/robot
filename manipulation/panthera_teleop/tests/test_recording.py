"""Exercise episode boundaries, classification, recovery and control guards without motors."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'app'))
from recording import EpisodeStore
from dashboard import Dashboard
import hand_guide
import numpy as np
import urllib.request
import urllib.error


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = json.loads((ROOT/'config/recording.json').read_text())
        self.config['minimum_free_bytes'] = 0
        self.config['neutral_return']['enabled'] = False
        self.store = EpisodeStore(self.temp.name, self.config, {'test': True})
        self.t = time.monotonic_ns()
        self.sequence = 0

    def tearDown(self):
        self.store.close('test_finished')
        self.temp.cleanup()

    def sample(self, q=0.0, velocity=0.0, dt=0.05):
        self.t += int(dt*1e9)
        self.sequence += 1
        sample = dict(monotonic_ns=self.t, unix_ns=time.time_ns(), sequence=self.sequence,
                      arms={side: dict(position_rad=[q]*6, velocity_rad_s=[velocity]*6,
                                       gripper=dict(position=0.0, velocity=0.0))
                            for side in ('left', 'right')})
        self.store.submit('sample', sample)
        self.store.events.join()

    def trigger(self):
        self.sample()
        for _ in range(5):
            self.sample(.03, .1)
        self.assertEqual(self.store.snapshot()['phase'], 'recording')
        return self.store.snapshot()['active']

    def decision(self, label, identifier):
        self.store.submit('decision', dict(label=label, episode_id=identifier))
        self.store.events.join()

    def test_jitter_does_not_trigger(self):
        for i in range(40):
            self.sample(.001*(i%3))
        self.assertIsNone(self.store.snapshot()['active'])

    def test_pre_roll_classification_and_rearm(self):
        for label in ('kept', 'review', 'discarded'):
            identifier = self.trigger()
            self.store.submit('frame', dict(camera='overhead', monotonic_ns=self.t,
                                            unix_ns=time.time_ns(), jpeg=b'fake-jpeg'))
            self.store.events.join()
            self.decision(label, identifier)
            directory = Path(self.temp.name)/label/identifier
            meta = json.loads((directory/'metadata.json').read_text())
            states = [json.loads(s) for s in (directory/'states.jsonl').read_text().splitlines()]
            self.assertTrue(meta['complete'])
            self.assertEqual(meta['label'], label)
            self.assertLess(states[0]['monotonic_ns'], meta['trigger_monotonic_ns'])
            self.assertEqual(meta['camera_frames']['overhead'], 1)
            self.assertTrue((directory/'images/overhead/0000000.jpg').exists())
            # Continued motion after labeling must not create an accidental second take.
            for i in range(15):
                self.sample(.03+i*.025, .1)
            self.assertIsNone(self.store.snapshot()['active'])
            for _ in range(20):
                self.sample(0.0, 0.0)
            self.assertEqual(self.store.snapshot()['phase'], 'armed')

    def test_stale_decision_cannot_finish_next_episode(self):
        identifier = self.trigger()
        self.decision('kept', 'stale-id')
        self.assertEqual(self.store.snapshot()['active'], identifier)

    def test_unfinished_stop_is_review(self):
        identifier = self.trigger()
        self.store.close('operator_stop')
        meta = json.loads((Path(self.temp.name)/'review'/identifier/'metadata.json').read_text())
        self.assertFalse(meta['complete'])
        self.assertEqual(meta['end_reason'], 'operator_stop')

    def test_crash_recovery(self):
        pending = Path(self.temp.name)/'pending'/'interrupted'
        pending.mkdir()
        (pending/'metadata.json').write_text('{"id":"interrupted"}')
        self.store.close('test')
        self.store = EpisodeStore(self.temp.name, self.config, {})
        meta = json.loads((Path(self.temp.name)/'review/interrupted/metadata.json').read_text())
        self.assertEqual(meta['end_reason'], 'recovered_after_interruption')
        self.assertFalse(meta['complete'])

    def test_http_requires_token_and_matching_episode(self):
        import socket
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        server = Dashboard(self.store, None, port, True)
        try:
            identifier = self.trigger()
            data = json.dumps(dict(label='kept', episode_id=identifier)).encode()
            request = urllib.request.Request(server.url+'/api/decision', data=data,
                                             headers={'Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 403)
            request.add_header('X-Session-Token', server.token)
            with urllib.request.urlopen(request) as response:
                self.assertEqual(response.status, 202)
            self.store.events.join()
            self.assertIsNone(self.store.snapshot()['active'])
        finally:
            server.close()


class ControlTests(unittest.TestCase):
    def fake(self):
        states = [SimpleNamespace(position=0.0, velocity=0.0, torque=0.0, time=1.0, fault=0, mode=0)
                  for _ in range(7)]
        motors = [SimpleNamespace(get_current_motor_state=lambda s=s:s, get_motor_id=lambda i=i:i+1)
                  for i, s in enumerate(states)]
        robot = SimpleNamespace(get_current_state=lambda:states[:6],
                                get_current_state_gripper=lambda:states[-1], Motors=motors,
                                joint_limits={'lower':[-1]*6, 'upper':[1]*6},
                                gripper_limits={'lower':0.0, 'upper':2.0},
                                get_Gravity=lambda q:np.full(6, 99.0),
                                forward_kinematics=lambda q:dict(position=[0.0]*3, rotation=np.eye(3)))
        return robot, states

    def test_torque_caps_and_passive_commands(self):
        robot, states = self.fake()
        torque, arm = hand_guide.feedback(robot, 'left', {})
        np.testing.assert_equal(torque, hand_guide.TORQUE_LIMIT)
        self.assertEqual(arm['command']['kp'], [0.0]*6)
        self.assertEqual(arm['command']['kd'], [0.0]*6)
        self.assertEqual(arm['gripper']['command'], [0.0]*5)

    def test_fault_nonfinite_and_limit_rejection(self):
        for field, value in [('fault',1), ('position',2), ('velocity',float('nan')), ('torque',float('nan'))]:
            robot, states = self.fake()
            setattr(states[0], field, value)
            with self.assertRaises(RuntimeError):
                hand_guide.feedback(robot, 'left', {})

    def test_stale_feedback_rejected(self):
        robot, _ = self.fake()
        stale = {('left',i):(1.0,time.monotonic()-1) for i in range(7)}
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            hand_guide.feedback(robot, 'left', stale)

    def test_gripper_placeholder_rejected(self):
        robot, states = self.fake()
        states[-1].position = 999.0
        with self.assertRaisesRegex(RuntimeError, 'gripper position'):
            hand_guide.feedback(robot, 'left', {})

    def test_passive_closed_gripper_encoder_offset_preserved(self):
        robot, states = self.fake()
        states[-1].position = -0.0094
        _, arm = hand_guide.feedback(robot, 'left', {})
        self.assertEqual(arm['gripper']['position'], -0.0094)
        self.assertEqual(arm['gripper']['command'], [0.0]*5)

    def test_stop_attempts_both_arms(self):
        called=[]
        def fail():
            called.append('left')
            raise RuntimeError('disconnected')
        hand_guide.stop_robots(dict(left=SimpleNamespace(set_stop=fail),
                                    right=SimpleNamespace(set_stop=lambda:called.append('right'))))
        self.assertEqual(called, ['left','right'])


if __name__ == '__main__':
    unittest.main()
