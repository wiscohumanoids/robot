"""Policy-step timing, stop behavior and web authorization without robot devices."""
import json
import os
import signal
from pathlib import Path
import sys
import threading
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import inference_dashboard as dashboard


class FakeHardware:
    def __init__(self):
        self.lower = np.full(14, -1.)
        self.upper = np.full(14, 1.)
        self.config = dict(control_hz=150)
        self.cameras = None
        self.commands = []
        self.position = np.zeros(14)
        self.closed = threading.Event()

    def open(self):
        pass

    def read(self):
        return self.position.copy(), {}, {}

    def images(self):
        return {}

    def command(self, target):
        self.position = target.copy()
        self.commands.append(target.copy())

    def close(self):
        self.closed.set()


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('Timed out waiting for controller state')
        time.sleep(.005)


def fake_request(base, path, body=None, timeout=3):
    if path == '/health':
        return dict(status='ready', contract=dashboard.CONTRACT, checkpoint='saved')
    return dict(sequence=body['sequence'], fps=20, checkpoint='saved', mock=False,
                actions=[[.01] * 14] * 50)


class RecoveryHardware(FakeHardware):
    def __init__(self):
        super().__init__()
        self.position[:] = .2
        self.config['neutral_return'] = json.loads(
            (dashboard.ROOT / 'config/recording.json').read_text())['neutral_return']

    def validate_reset_path(self, *_):
        raise RuntimeError('Recovery must not reset')


class HangingCloseHardware(RecoveryHardware):
    def close(self):
        time.sleep(30)


class SupervisorTests(unittest.TestCase):
    def create(self, hardware=RecoveryHardware):
        supervisor = dashboard.SessionSupervisor('unused', hardware, fake_request)
        self.addCleanup(supervisor.shutdown)
        return supervisor

    def test_recovery_requires_confirmation_and_reopens_without_reset_or_replay(self):
        supervisor = self.create()
        old_pid = supervisor.process.pid
        for value in (False, None, 1, 'true'):
            with self.assertRaises(ValueError):
                supervisor.recover(value)
        supervisor.recover(True)
        wait_for(lambda: supervisor.snapshot()['mode'] == 'holding' and
                 supervisor.snapshot()['hold_status'] == 'commanded', 5)
        state = supervisor.snapshot()
        self.assertNotEqual(supervisor.process.pid, old_pid)
        self.assertIsNone(state['prediction'])
        self.assertEqual(state['steps_completed'], 0)
        np.testing.assert_allclose(state['target'], .2)
        self.assertEqual(state['hold_status'], 'commanded')

    def test_native_crash_keeps_supervisor_usable_and_can_recover(self):
        supervisor = self.create()
        supervisor.process.kill()
        supervisor.process.join(2)
        self.assertEqual(supervisor.snapshot()['mode'], 'fault')
        self.assertEqual(supervisor.snapshot()['hold_status'], 'unverified')
        supervisor.recover(True)
        wait_for(lambda: supervisor.snapshot()['mode'] == 'holding', 5)

    def test_foreground_interrupt_does_not_kill_robot_session(self):
        supervisor = self.create()
        supervisor.snapshot()  # Wait until the spawned child has installed handlers.
        os.kill(supervisor.process.pid, signal.SIGINT)
        os.kill(supervisor.process.pid, signal.SIGTERM)
        self.assertEqual(supervisor.snapshot()['mode'], 'stopped')
        self.assertTrue(supervisor.process.is_alive())

    def test_unresponsive_session_does_not_block_recovery_or_reuse_late_replies(self):
        supervisor = self.create()
        supervisor.snapshot()
        old_process = supervisor.process
        os.kill(old_process.pid, signal.SIGSTOP)
        self.assertEqual(supervisor.snapshot()['mode'], 'fault')
        self.assertEqual(supervisor.snapshot()['mode'], 'fault')
        supervisor.recover(True)
        wait_for(lambda: supervisor.snapshot()['mode'] == 'holding', 7)
        self.assertFalse(old_process.is_alive())

    def test_stuck_cleanup_is_replaced_only_after_old_process_exits(self):
        supervisor = self.create(HangingCloseHardware)
        supervisor.recover(True)
        wait_for(lambda: supervisor.snapshot()['mode'] == 'holding', 5)
        old_process = supervisor.process
        supervisor.recover(True)
        with self.assertRaises(RuntimeError):
            supervisor.recover(True)
        with self.assertRaises(RuntimeError):
            supervisor.connect()
        wait_for(lambda: supervisor.snapshot()['mode'] == 'holding', 7)
        self.assertFalse(old_process.is_alive())
        self.assertNotEqual(supervisor.process.pid, old_process.pid)

    def test_stop_cancels_recovery_without_reopening_hardware(self):
        supervisor = self.create(HangingCloseHardware)
        supervisor.recover(True)
        wait_for(lambda: supervisor.snapshot()['mode'] == 'holding', 5)
        supervisor.recover(True)
        started = time.monotonic()
        supervisor.stop()
        self.assertLess(time.monotonic() - started, .2)
        wait_for(lambda: not supervisor.snapshot()['recovering'], 7)
        self.assertEqual(supervisor.snapshot()['mode'], 'stopped')
        self.assertFalse(supervisor.snapshot()['connected'])


class StepTests(unittest.TestCase):
    def test_blend_crossfades_aligned_tail_and_preserves_bounds(self):
        old = np.full((40, 14), .2)
        new = np.full((50, 14), -.2)
        blended, overlap = dashboard.blend_predictions(new, old, 10)
        self.assertEqual(overlap, 10)
        self.assertTrue(np.all(np.diff(blended[:10, 0]) < 0))
        self.assertTrue(np.all(blended >= -.2))
        self.assertTrue(np.all(blended <= .2))
        np.testing.assert_array_equal(blended[9:], new[9:])
        np.testing.assert_array_equal(new, -.2)
        np.testing.assert_array_equal(old, .2)
        for tail, count in ((None, 10), (old, 0), (old[:0], 10)):
            result, overlap = dashboard.blend_predictions(new, tail, count)
            self.assertEqual(overlap, 0)
            np.testing.assert_array_equal(result, new)
        _, overlap = dashboard.blend_predictions(new, old[:3], 50)
        self.assertEqual(overlap, 3)
        for value in (-1, 51, True, 1.5, None):
            with self.assertRaises(ValueError):
                dashboard.blend_count(value)

    def test_count_rejects_fractional_boolean_and_out_of_range(self):
        for value in (0, 51, True, 1.5, '5', None):
            with self.assertRaises(ValueError):
                dashboard.action_count(value)

    def test_single_step_cannot_advance_into_the_next_action(self):
        actions = np.array([[.1] * 14, [-.1] * 14])
        step = dashboard.ActionStep(actions, 1, 10.)
        target = np.zeros(14)
        for tick in range(150):
            now = 10. + tick*.01
            previous = target.copy()
            target, done = step.tick(now, .01, target)
            self.assertTrue(np.all(target >= previous))
            self.assertTrue(np.all(target-previous <= step.speed*.01 + 1e-9))
            if done:
                break
        self.assertTrue(done)
        np.testing.assert_allclose(target, .1)
        final, done = step.tick(now + .01, .01, target)
        self.assertTrue(done)
        np.testing.assert_array_equal(final, target)

    def test_large_jump_is_smoothed_and_takes_longer_than_policy_time(self):
        step = dashboard.ActionStep(np.ones((50, 14)), 1, 0)
        target = np.zeros(14)
        for tick in range(6):
            previous = target.copy()
            target, done = step.tick(tick*.01, .01, target)
            self.assertFalse(done)
            self.assertTrue(np.all(np.abs(target-previous) <= step.speed*.01 + 1e-9))
        self.assertEqual(step.index, 0)


class CollectionResetTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((dashboard.ROOT / 'config/recording.json').read_text())['neutral_return']
        self.lower = np.full(14, -3.)
        self.upper = np.full(14, 3.)
        self.lower[[6, 13]] = 0
        self.upper[[6, 13]] = 2

    def test_reset_reaches_collection_pose_and_opening_with_speed_bounds(self):
        measured = np.zeros(14)
        measured[[6, 13]] = [1.94, .5]
        target = measured.copy()
        reset = dashboard.CollectionReset(self.settings, self.lower, self.upper, measured, 0)
        for tick in range(1, 600):
            previous = target.copy()
            target, done = reset.tick(tick*.01, .01, measured, target)
            speed = np.full(14, dashboard.CONTRACT['max_joint_speed_rad_s'])
            speed[[6, 13]] = dashboard.CONTRACT['gripper_speed_rad_s']
            self.assertTrue(np.all(np.abs(target-previous) <= speed*.01 + 1e-9))
            measured = target.copy()
            if done:
                break
        self.assertTrue(done)
        np.testing.assert_allclose(target, [0,.25,.25,0,0,0,1.2]*2)

    def test_gripper_obstruction_refuses_completion(self):
        measured = np.array([0,.25,.25,0,0,0,0.]*2)
        reset = dashboard.CollectionReset(self.settings, self.lower, self.upper, measured, 0)
        target = measured.copy()
        with self.assertRaisesRegex(RuntimeError, 'grippers'):
            for tick in range(1, 600):
                target, done = reset.tick(tick*.01, .01, measured, target)
                self.assertFalse(done)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.hardware = FakeHardware()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.controller = dashboard.Controller('http://unused', lambda: self.hardware,
                                               Path(directory.name) / 'steps.jsonl')
        self.addCleanup(self.stop)

    def stop(self):
        self.controller.shutdown()
        if self.controller.worker:
            self.controller.worker.join(2)

    def request(self, base, path, body=None, timeout=3):
        if path == '/health':
            return dict(status='ready', contract=dashboard.CONTRACT, checkpoint='saved')
        return dict(sequence=body['sequence'], fps=20, checkpoint='saved', mock=False,
                    actions=[[.01] * 14] * 50)

    def connect(self):
        self.controller.connect()
        wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')

    def test_play_repeats_fresh_steps_blends_and_applies_next_settings(self):
        observations = []
        def changing(base, path, body=None, timeout=3):
            result = self.request(base, path, body, timeout)
            if path == '/infer':
                observations.append(body['state'])
                result['actions'] = [[.001*(i+1)+body['state'][0]]*14 for i in range(50)]
            return result
        with patch.object(dashboard, 'request', changing):
            self.connect()
            self.controller.submit('play', 2, 10)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] >= 2)
            self.assertTrue(self.controller.snapshot()['playing'])
            self.controller.configure(3, 4)
            wait_for(lambda: self.controller.snapshot()['step_count'] == 3 and
                     self.controller.snapshot()['blended_actions'] == 4)
            self.controller.stop()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
            self.assertFalse(self.controller.snapshot()['playing'])
            held = self.hardware.position.copy()
            time.sleep(.08)
            np.testing.assert_array_equal(self.hardware.position, held)
            self.assertGreater(observations[1][0], observations[0][0])
            entries = [json.loads(line) for line in self.controller.log_path.read_text().splitlines()]
            blends = [row for row in entries if row.get('event') == 'play_blend']
            self.assertEqual(blends[0]['blended_actions'], 0)
            expected, _ = dashboard.blend_predictions(
                np.array([row for row in entries if 'response' in row][1]['response']['actions']),
                np.array(blends[0]['actions'])[2:], 10)
            np.testing.assert_allclose(blends[1]['actions'], expected)

    def test_play_pause_discards_late_response_and_resume_starts_fresh(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def delayed(base, path, body=None, timeout=3):
            result = self.request(base, path, body, timeout)
            if path == '/infer':
                calls.append(body['sequence'])
                if len(calls) == 1:
                    entered.set()
                    release.wait(2)
                    result['actions'] = [[.9]*14]*50
            return result
        with patch.object(dashboard, 'request', delayed):
            try:
                self.connect()
                self.controller.submit('play', 2)
                self.assertTrue(entered.wait(1))
                self.controller.stop()
                wait_for(lambda: self.controller.snapshot()['mode'] == 'holding', .3)
                release.set()
                time.sleep(.05)
                np.testing.assert_array_equal(self.hardware.position, 0)
                self.assertFalse(self.controller.snapshot()['playing'])
                self.controller.submit('play', 1)
                wait_for(lambda: self.controller.snapshot()['steps_completed'] >= 1)
                self.controller.stop()
                wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
                np.testing.assert_allclose(self.hardware.position, .01)
                blends = [json.loads(line) for line in self.controller.log_path.read_text().splitlines()
                          if 'play_blend' in line]
                self.assertEqual(blends[0]['blended_actions'], 0)
            finally:
                release.set()

    def test_play_error_disables_repetition_until_explicit_resume(self):
        calls = []
        def invalid(base, path, body=None, timeout=3):
            result = self.request(base, path, body, timeout)
            if path == '/infer':
                calls.append(body)
                result['actions'][-1][0] = 1.1
            return result
        with patch.object(dashboard, 'request', invalid):
            self.connect()
            self.controller.submit('play', 1)
            wait_for(lambda: self.controller.snapshot()['error'] is not None)
            self.assertFalse(self.controller.snapshot()['playing'])
            self.controller.heartbeat()
            time.sleep(.08)
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(self.hardware.position, 0)

    def test_expired_overlap_is_discarded_after_retimed_motion(self):
        with patch.object(dashboard, 'request', self.request), patch.dict(
                dashboard.CONTRACT, max_observation_age_s=.06):
            self.connect()
            self.controller.submit('play', 2, 10)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] >= 2)
            self.controller.stop()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
            blends = [json.loads(line) for line in self.controller.log_path.read_text().splitlines()
                      if 'play_blend' in line]
            self.assertEqual(blends[1]['blended_actions'], 0)

    def test_browser_loss_cancels_play_and_never_auto_resumes(self):
        with patch.object(dashboard, 'request', self.request), patch.object(dashboard, 'LEASE_SECONDS', .1):
            self.connect()
            self.controller.submit('play', 1)
            wait_for(lambda: not self.controller.snapshot()['playing'] and
                     self.controller.snapshot()['error'] is not None)
            held = self.hardware.position.copy()
            self.controller.heartbeat()
            time.sleep(.04)
            np.testing.assert_array_equal(self.hardware.position, held)
            self.assertFalse(self.hardware.closed.is_set())

    def test_camera_fault_clears_queued_play_before_camera_recovery(self):
        camera = {'ok': False, 'age_s': 0.}
        self.hardware.read = lambda: (self.hardware.position.copy(), {}, {'test': camera.copy()})
        with patch.object(dashboard, 'request', self.request):
            self.connect()
            self.controller.submit('play', 1)
            wait_for(lambda: self.controller.snapshot()['error'] is not None)
            self.assertFalse(self.controller.snapshot()['playing'])
            self.assertTrue(self.controller.commands.empty())
            camera['ok'] = True
            time.sleep(.08)
            self.assertEqual(self.controller.snapshot()['steps_completed'], 0)
            np.testing.assert_array_equal(self.hardware.position, 0)

    def test_preview_holds_and_one_action_step_pauses(self):
        with patch.object(dashboard, 'request', self.request):
            self.connect()
            self.controller.submit('predict')
            wait_for(lambda: self.controller.snapshot()['prediction'] is not None and
                     self.controller.snapshot()['mode'] == 'holding')
            self.assertTrue(all(np.all(q == 0) for q in self.hardware.commands))
            self.controller.submit('step', 1)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] == 1)
            state = self.controller.snapshot()
            self.assertEqual(state['mode'], 'holding')
            self.assertEqual(state['action_index'], 1)
            self.assertGreater(float(np.max(self.hardware.position)), 0)
            held = self.hardware.position.copy()
            time.sleep(.08)
            np.testing.assert_array_equal(self.hardware.position, held)

    def test_gripper_overshoots_execute_at_configured_limits(self):
        def overshoot(*args, **kwargs):
            result = self.request(*args, **kwargs)
            if 'actions' in result:
                action = [.01] * 14
                action[6], action[13] = -5., 8.
                result['actions'] = [action] * 50
            return result
        with patch.object(dashboard, 'request', overshoot):
            self.connect()
            self.controller.submit('step', 1)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] == 1)
            self.assertIsNone(self.controller.snapshot()['error'])
            np.testing.assert_allclose(self.hardware.position[[6, 13]], [-1., 1.])
            self.assertTrue(all(np.all(q >= self.hardware.lower) and
                                np.all(q <= self.hardware.upper) for q in self.hardware.commands))

    def test_bad_prediction_stops_before_any_nonzero_motion(self):
        def bad(*args, **kwargs):
            result = self.request(*args, **kwargs)
            if 'actions' in result:
                result['actions'] = [[1.1] * 14] * 50
            return result
        with patch.object(dashboard, 'request', bad):
            self.connect()
            self.controller.submit('step', 5)
            wait_for(lambda: self.controller.snapshot()['error'] is not None)
            self.assertEqual(self.controller.snapshot()['mode'], 'holding')
            self.assertTrue(self.controller.snapshot()['connected'])
            self.assertFalse(self.hardware.closed.is_set())
            self.assertTrue(all(np.all(q == 0) for q in self.hardware.commands))

    def test_one_action_executes_when_only_unselected_tail_is_outside_limits(self):
        def tail_invalid(*args, **kwargs):
            result = self.request(*args, **kwargs)
            if 'actions' in result:
                result['actions'][-1] = [1.1] * 14
            return result
        with patch.object(dashboard, 'request', tail_invalid):
            self.connect()
            self.controller.submit('step', 1)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] == 1)
            self.assertIsNone(self.controller.snapshot()['error'])
            np.testing.assert_allclose(self.hardware.position, .01)
            self.controller.submit('step', 50)
            wait_for(lambda: self.controller.snapshot()['error'] is not None)
            self.assertIn('action 50, left.joint1', self.controller.snapshot()['error'])
            np.testing.assert_allclose(self.hardware.position, .01)
            entries = [json.loads(line) for line in self.controller.log_path.read_text().splitlines()]
            self.assertEqual(sum('response' in row for row in entries), 2)

    def test_stop_interrupts_a_stretched_large_motion_and_retains_hold(self):
        def large(*args, **kwargs):
            result = self.request(*args, **kwargs)
            if 'actions' in result:
                result['actions'] = [[.9]*14] * 50
            return result
        with patch.object(dashboard, 'request', large):
            self.connect()
            self.controller.submit('step', 1)
            wait_for(lambda: self.hardware.position.max() > .01)
            self.assertEqual(self.controller.snapshot()['mode'], 'executing')
            self.controller.stop()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding', .3)
            held = self.hardware.position.copy()
            time.sleep(.08)
            np.testing.assert_array_equal(self.hardware.position, held)
            self.assertFalse(self.hardware.closed.is_set())
            self.assertLess(held.max(), .9)

    def test_browser_lease_expiry_keeps_hardware_connected_and_held(self):
        with patch.object(dashboard, 'request', self.request), patch.object(dashboard, 'LEASE_SECONDS', .05):
            self.connect()
            wait_for(lambda: self.controller.snapshot()['error'] is not None)
            self.assertIn('Browser connection lost', self.controller.snapshot()['error'])
            self.assertTrue(self.controller.snapshot()['connected'])
            self.assertFalse(self.hardware.closed.is_set())
            self.assertTrue(all(np.all(q == 0) for q in self.hardware.commands))

    def test_connect_resets_before_accepting_steps_and_stop_interrupts_reset(self):
        self.hardware.config['neutral_return'] = json.loads(
            (dashboard.ROOT / 'config/recording.json').read_text())['neutral_return']
        self.hardware.validate_reset_path = lambda *_: None
        with patch.object(dashboard, 'request', self.request):
            self.controller.connect()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'resetting')
            with self.assertRaisesRegex(RuntimeError, 'wait'):
                self.controller.submit('step')
            self.controller.stop()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding', .3)
            self.assertFalse(self.hardware.closed.is_set())

    def test_stop_does_not_wait_for_network_inference(self):
        entered, release = threading.Event(), threading.Event()
        def delayed(*args, **kwargs):
            if args[1] == '/infer':
                entered.set()
                release.wait(2)
            return self.request(*args, **kwargs)
        with patch.object(dashboard, 'request', delayed):
            try:
                self.connect()
                self.controller.submit('step', 50)
                self.assertTrue(entered.wait(1))
                self.controller.stop()
                wait_for(lambda: self.controller.snapshot()['mode'] == 'holding', .3)
                self.assertTrue(self.controller.snapshot()['connected'])
                self.assertFalse(self.hardware.closed.is_set())
                self.assertTrue(all(np.all(q == 0) for q in self.hardware.commands))
            finally:
                release.set()

    def test_inference_network_error_retains_hold(self):
        def broken(*args, **kwargs):
            if args[1] == '/infer':
                raise OSError('network offline')
            return self.request(*args, **kwargs)
        with patch.object(dashboard, 'request', broken):
            self.connect()
            self.controller.submit('step', 1)
            wait_for(lambda: self.controller.snapshot()['error'] is not None)
            self.assertEqual(self.controller.snapshot()['mode'], 'holding')
            self.assertTrue(self.controller.snapshot()['connected'])
            self.assertFalse(self.hardware.closed.is_set())
            self.assertTrue(all(np.all(q == 0) for q in self.hardware.commands))

    def test_motor_feedback_failure_retains_sdk_and_blocks_motion(self):
        with patch.object(dashboard, 'request', self.request):
            self.connect()
            def broken():
                raise RuntimeError('stale motor feedback')
            self.hardware.read = broken
            wait_for(lambda: self.controller.snapshot()['mode'] == 'fault_hold')
            self.assertTrue(self.controller.snapshot()['connected'])
            self.assertFalse(self.hardware.closed.is_set())
            with self.assertRaises(RuntimeError):
                self.controller.submit('step')

    def test_failed_hold_is_reported_as_failed_instead_of_retained_hold(self):
        with patch.object(dashboard, 'request', self.request):
            self.connect()
            def broken(*_):
                raise OSError('Bad file descriptor')
            self.hardware.command = broken
            wait_for(lambda: self.controller.snapshot()['hold_status'] == 'failed')
            self.assertIn('Powered hold is not reliable', self.controller.snapshot()['error'])
            self.assertEqual(self.controller.snapshot()['mode'], 'fault_hold')



class CollectionResetTimingTests(unittest.TestCase):
    def test_first_tick_before_reset_creation_retains_hold(self):
        settings = json.loads((dashboard.ROOT / 'config/recording.json').read_text())['neutral_return']
        lower, upper = np.full(14, -4.), np.full(14, 4.)
        measured = np.zeros(14)
        reset = dashboard.CollectionReset(settings, lower, upper, measured, 10.)
        target, done = reset.tick(9.999, .006, measured, measured)
        np.testing.assert_array_equal(target, measured)
        self.assertFalse(done)
        target, done = reset.tick(10.001, .006, measured, measured)
        self.assertTrue(np.isfinite(target).all())


class WebTests(unittest.TestCase):
    def test_mutations_require_token_and_trusted_origin(self):
        controller = dashboard.Controller('unused')
        server = dashboard.make_server(controller, 0)
        port = server.server_port
        # The handler validates the configured port, so use the actual bound port.
        server.server_close()
        server = dashboard.make_server(controller, port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{port}'
        with urlopen(base + '/api/status') as response:
            token = json.load(response)['token']
        for headers in ({}, {'X-Session-Token': token, 'Origin': 'https://untrusted.example'}):
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + '/api/stop', data=b'{}', headers=headers))
            self.assertEqual(error.exception.code, 403)
        self.assertFalse(controller.stop_event.is_set())
        with urlopen(Request(base + '/api/stop', data=b'{}', headers={'X-Session-Token': token})) as response:
            self.assertEqual(response.status, 202)
            self.assertTrue(controller.pause_event.is_set())
            self.assertFalse(controller.stop_event.is_set())
        headers = {'X-Session-Token': token, 'Content-Type': 'application/json'}
        with patch.object(controller, 'submit') as submit:
            with urlopen(Request(base + '/api/play', data=b'{"count":5,"blend":8}', headers=headers)) as response:
                self.assertEqual(response.status, 202)
            submit.assert_called_once_with('play', 5, 8)
        with patch.object(controller, 'configure') as configure:
            with urlopen(Request(base + '/api/configure', data=b'{"count":3,"blend":0}', headers=headers)) as response:
                self.assertEqual(response.status, 202)
            configure.assert_called_once_with(3, 0)
        with patch.object(controller, 'stop') as stop:
            with urlopen(Request(base + '/api/pause', data=b'{}', headers=headers)) as response:
                self.assertEqual(response.status, 202)
            stop.assert_called_once_with()
        for payload in (b'{"count":true}', b'{"count":51}', b'{"blend":-1}'):
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + '/api/play', data=payload, headers=headers))
            self.assertEqual(error.exception.code, 409)


if __name__ == '__main__':
    unittest.main()
