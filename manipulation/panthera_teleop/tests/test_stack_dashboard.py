"""Exercise a physical run's ownership, revalidation and stop paths without devices."""

import sys
import time
import unittest
from unittest.mock import Mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import stack_dashboard as dashboard


class Hardware:
    def __init__(self, arm, camera):
        self.commands = []
        self.stopped = False
        self.closed = False
        self.lower = np.zeros(7)
        self.upper = np.ones(7)

    def open(self):
        pass

    def frames(self):
        frame = dict(jpeg=b'jpeg', width=640, height=480, monotonic_ns=time.monotonic_ns())
        return dict(overhead=frame, wrist=frame)

    def read(self):
        return dict(position_rad=[0.] * 6, velocity_rad_s=[0.] * 6, gripper={'position': .5})

    def command(self, q, g):
        assert not self.stopped
        self.commands.append((list(q), g))

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class Detector:
    def __init__(self, *args):
        self.offset = 0

    def locate(self, frames, q, check):
        check()
        return [dict(bbox_xyxy=[x, 5, x+20, 25], color=color, center_xyz=[x/500, 0, .03],
                     studs_xyz=[[x/500+self.offset, 0, .05], [x/500+self.offset, .032, .05]])
                for x, color in [(40, 'red'), (180, 'blue')]]

    def close(self):
        pass


class Motion:
    @staticmethod
    def build_plan(*_, **kwargs):
        assert kwargs['table_z_m'] == 0.
        return SimpleNamespace(phases=('Retreat',), tcp_path=[[0., 0., 1.], [.1, .1, 1.]], waypoints={'Retreat': [np.zeros(7)]})

    @staticmethod
    def execute_plan(plan, *, read, command, check_active, on_phase):
        check_active()
        on_phase('Retreat')
        command(read())


class DashboardTest(unittest.TestCase):
    def setUp(self):
        cfg = dict(wrist_camera='wrist_port3', gripper_open_rad=1, gripper_closed_rad=0,
                   vision={'table_z_m': 0, 'overhead': {
                       'camera_matrix': [[300, 0, 300], [0, 300, 200], [0, 0, 1]],
                       'distortion': [0.] * 5, 'world_from_camera': np.eye(4).tolist()}})
        self.panel = dashboard.Controller({'right': cfg}, None, Hardware, Detector, Motion)
        self.addCleanup(self.panel.shutdown)
        self.panel.action('connect', {'arm': 'right'})
        self.wait_for('select')

    def wait_for(self, phase):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            state = self.panel.status()
            if state['phase'] == phase and (phase == 'holding' or not state['running']):
                return
            time.sleep(.005)
        self.fail(f'Expected {phase}, saw {self.panel.status()}')

    def preview(self):
        for role, x in [('source', 45), ('base', 185)]:
            self.panel.action('select', dict(role=role, x=x, y=10))
        self.panel.action('preview', {})
        self.wait_for('ready')
        self.assertFalse(self.panel.hw.commands)
        self.assertEqual(self.panel.status()['preview_path'], [[300., 200.], [330., 230.]])

    def test_preview_required_and_reselection_invalidates_plan(self):
        with self.assertRaises(ValueError):
            self.panel.action('run', {}, 'tab-A')
        self.preview()
        self.panel.action('select', dict(role='base', x=45, y=10))
        with self.assertRaises(ValueError):
            self.panel.action('run', {}, 'tab-A')
        self.panel.action('preview', {})
        self.wait_for('preview_error')
        self.preview()

    def test_block_moved_after_preview(self):
        self.preview()
        self.panel.detector.offset = .006
        self.panel.action('run', {}, 'tab-A')
        self.wait_for('fault')
        self.assertIn('changed since preview', self.panel.message)
        self.assertFalse(self.panel.hw.commands)
        self.assertTrue(self.panel.hw.stopped)

    def test_lost_operator_stops_hold(self):
        self.preview()
        self.panel.action('run', {}, 'tab-A')
        self.wait_for('holding')
        with self.panel.lock:
            self.panel.heartbeat = time.monotonic() - dashboard.LEASE_SECONDS - 1
            self.panel.action('heartbeat', {}, 'tab-B')
        self.wait_for('fault')
        self.assertIn('heartbeat lost', self.panel.message)
        self.assertTrue(self.panel.hw.stopped)

    def test_camera_loss_stops_motion(self):
        self.preview()
        self.panel.action('run', {}, 'tab-A')
        self.wait_for('holding')

        def lost_camera():
            raise RuntimeError('stale')

        self.panel.hw.frames = lost_camera
        self.wait_for('fault')
        self.assertIn('Lost operator camera view', self.panel.message)
        self.assertTrue(self.panel.hw.stopped)

    def test_stop_releases_hold_and_latches(self):
        self.preview()
        self.panel.action('run', {}, 'tab-A')
        self.wait_for('holding')
        self.panel.action('stop', {})
        self.wait_for('stopped')
        hw = self.panel.hw
        self.assertTrue(hw.commands)
        self.assertTrue(hw.stopped)
        with self.assertRaises(ValueError):
            self.panel.action('run', {}, 'tab-A')
        self.panel.action('disconnect', {})
        self.assertTrue(hw.closed)
        self.assertFalse(self.panel.status()['connected'])

    def test_auto_fit_requires_verified_camera_pair_and_is_scoped_to_connection(self):
        panel = self.panel
        panel.action('disconnect', {})
        fixed_overhead = panel.configs['right']['vision']['overhead']
        panel.configs['right']['vision'] = dict(auto_calibrate=True, table_z_m=0,
                                                overhead={'fovy_deg': 80}, wrist={'fovy_deg': 70})
        panel.detector_factory = Mock(side_effect=Detector)
        panel.action('connect', {'arm': 'right'})
        self.wait_for('calibrating')
        worker_config = panel.detector_factory.call_args.args[1]
        self.assertEqual(worker_config['arm'], 'right')
        self.assertEqual(worker_config['gripper_open_rad'], 1)
        self.assertNotIn('arm', panel.configs['right']['vision'])
        self.assertEqual(panel.status()['calibration_mode'], 'auto')
        for action, args in [('finish_calibration', {}), ('select', dict(role='source', x=4, y=9)),
                             ('preview', {}), ('run', {})]:
            with self.assertRaises(ValueError):
                panel.action(action, args)
        fresh = dict(ready=True, samples=3, message='Both cameras fitted')
        validation_miss = dict(ready=False, samples=3, message='Wrist landmarks missing')
        verified = dict(fresh, cameras={'overhead': fixed_overhead})
        panel.detector.calibrate = Mock(side_effect=[fresh, validation_miss, verified])
        for action in ('sample', 'finish_calibration'):
            panel.action(action, {})
            self.wait_for('calibrating')
            self.assertIsNone(panel.overhead)
            self.assertIsNone(panel.pair)
            self.assertTrue(panel.calibration_ready)
        panel.action('finish_calibration', {})
        self.wait_for('select')
        receipt, encoders, _ = panel.detector.calibrate.call_args.args
        self.assertIs(panel.pair, receipt)
        self.assertEqual(encoders[-1], .5)
        self.assertIs(panel.overhead, fixed_overhead)
        self.assertTrue(panel.detector.calibrate.call_args.kwargs['finish'])
        self.preview()
        self.assertFalse(panel.hw.commands)
        panel.action('disconnect', {})
        self.assertIsNone(panel.overhead)
        self.assertFalse(panel.calibration_ready)
        self.assertEqual(panel.calibration_samples, 0)
