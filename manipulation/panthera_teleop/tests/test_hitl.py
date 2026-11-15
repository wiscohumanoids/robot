"""Exercise ownership handoffs, recording isolation, and training interval selection."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from hitl_demo import DemoHardware, demo_request
from inference_dashboard import Controller, LEASE_SECONDS, RobotHardware
from prepare_hitl import select_intervals


def wait_for(predicate, timeout=3):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError('Timed out')
        time.sleep(.005)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch('inference_dashboard.request', demo_request)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.controller = Controller('demo', DemoHardware,
            log_path=Path(self.tmp.name)/'steps.jsonl', data_dir=Path(self.tmp.name)/'raw')
        def cleanup():
            self.controller.shutdown()
            if self.controller.worker:
                self.controller.worker.join(3)
        self.addCleanup(cleanup)
        self.controller.connect(False)
        wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
        self.controller.hardware.auto_pull = False
        self.controller.hardware.config['takeover_hold'].update(baseline_seconds=.05, pull_confirm_seconds=.06)

    def pull(self, sides=('left', 'right')):
        wait_for(lambda: self.controller.snapshot()['handoff_baseline_ready'])
        for side in sides:
            self.controller.hardware.pull_torque[side] = [.9] + [0.]*5
        wait_for(lambda: all(self.controller.snapshot()['handoff_ready'].get(s) for s in sides))

    def begin(self):
        self.controller.collection('begin', dict(operator='tester', scene='test'))
        wait_for(lambda: self.controller.snapshot()['rollout_active'])
        wait_for(lambda: self.controller.recorder.snapshot()['active'] is not None)

    def test_finish_saves_milestones_before_reset_and_stop_interrupts_reset(self):
        self.begin()
        reset = Mock()
        reset.tick.side_effect = lambda now, dt, measured, target: (target, False)
        def start_reset(hardware, measured):
            self.assertFalse(self.controller.snapshot()['rollout_active'])
            self.controller.update(mode='resetting')
            return reset
        with patch.object(self.controller, 'collection_reset', side_effect=start_reset) as start:
            self.controller.collection('finish', dict(label='review', success=False,
                quality=2, notes='insertion failed', reset_after_finish=True,
                milestones={'gear_pickup': 'success', 'gear_seating': 'failure', 'pin_pushing': 'failure'}))
            wait_for(lambda: self.controller.snapshot()['mode'] == 'resetting')
            wait_for(lambda: self.controller.recorder.snapshot()['last_decision'] is not None)
            start.assert_called_once()
            directory = next((Path(self.tmp.name)/'raw/review').iterdir())
            outcome = json.loads((directory/'metadata.json').read_text())['outcome']
            self.assertEqual(outcome['milestones'],
                             {'gear_pickup': 'success', 'gear_seating': 'failure', 'pin_pushing': 'failure'})
            self.controller.stop()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
            self.assertFalse(self.controller.snapshot()['rollout_active'])

    def test_invalid_milestones_and_reset_flag_rejected(self):
        self.begin()
        payload = dict(label='kept', success=True, quality=5)
        for extra in ({'milestones': {'unknown': 'success'}},
                      {'milestones': {'gear_pickup': 'maybe'}},
                      {'reset_after_finish': 'yes'}):
            with self.assertRaises(ValueError):
                self.controller.collection('finish', dict(payload, **extra))
        self.assertTrue(self.controller.snapshot()['rollout_active'])

    def test_takeover_cancels_pending_inference_and_resume_uses_fresh_observation(self):
        self.begin()
        first = []
        def delayed(base, path, body=None, timeout=3):
            if path == '/infer' and not first:
                first.append(body['state'])
                time.sleep(.15)
            return demo_request(base, path, body, timeout)
        with patch('inference_dashboard.request', delayed):
            self.controller.submit('play', 5)
            wait_for(lambda: self.controller.snapshot()['mode'] == 'predicting')
            self.controller.collection('intervene')
            self.pull()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'human')
            time.sleep(.2)  # Old result arrives while human control owns both arms.
            self.assertEqual(self.controller.snapshot()['mode'], 'human')
            self.assertFalse(self.controller.snapshot()['playing'])
            self.assertIsNone(self.controller.snapshot()['prediction'])
            with self.assertRaises(RuntimeError):
                self.controller.submit('play')
            corrected = self.controller.hardware.position.copy()
            self.controller.collection('resume')
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
            self.assertFalse(self.controller.snapshot()['playing'])
            self.assertTrue((self.controller.hardware.position >= corrected).all())
            self.controller.submit('step', 5)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] == 1)
        self.controller.collection('finish', dict(label='kept', success=True, quality=4, notes='recovery'))
        wait_for(lambda: not self.controller.snapshot()['rollout_active'])
        wait_for(lambda: self.controller.recorder.snapshot()['last_decision'] is not None)
        directory = next((Path(self.tmp.name)/'raw/kept').iterdir())
        metadata = json.loads((directory/'metadata.json').read_text())
        states = [json.loads(s) for s in (directory/'states.jsonl').read_text().splitlines()]
        self.assertTrue(metadata['complete'])
        self.assertTrue(metadata['outcome']['success'])
        self.assertEqual(metadata['context']['rollout']['operator'], 'tester')
        self.assertEqual({'human', 'policy'}, {s['control_source'] for s in states})
        self.assertIn('hold', {t['source'] for t in metadata['control_transitions']})
        self.assertTrue(all(b['monotonic_ns'] > a['monotonic_ns'] for a,b in zip(states,states[1:])))
        self.assertTrue(all(s['command_target_rad'] is None for s in states if s['control_source']=='human'))
        self.assertGreater(sum(metadata['camera_frames'].values()), 0)
        self.assertEqual(metadata['context']['mode'], 'human_in_the_loop_simulation')
        self.assertEqual(select_intervals(metadata, states), [])  # Demo cannot enter training.

    def test_play_then_takeover_starts_recording_at_human_control_without_policy_prefix(self):
        self.controller.submit('play', 5)
        wait_for(lambda: self.controller.snapshot()['mode'] == 'executing')
        self.assertFalse(self.controller.snapshot()['rollout_active'])
        self.assertIsNone(self.controller.recorder.snapshot()['active'])
        self.controller.collection('intervene', dict(operator='tester', scene='correction-only'))
        self.pull()
        wait_for(lambda: self.controller.snapshot()['mode'] == 'human')
        wait_for(lambda: self.controller.recorder.snapshot()['active'] is not None)
        time.sleep(.04)
        self.controller.collection('resume')
        wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
        self.assertFalse(self.controller.snapshot()['playing'])
        self.controller.collection('finish', dict(label='kept', success=True, quality=5))
        wait_for(lambda: self.controller.recorder.snapshot()['last_decision'] is not None)
        directory = next((Path(self.tmp.name)/'raw/kept').iterdir())
        states = [json.loads(s) for s in (directory/'states.jsonl').read_text().splitlines()]
        metadata = json.loads((directory/'metadata.json').read_text())
        self.assertEqual(states[0]['control_source'], 'human')
        self.assertNotIn('policy', {s['control_source'] for s in states})
        self.assertTrue(all(s['monotonic_ns'] >= metadata['start_monotonic_ns'] for s in states))
        self.assertEqual(metadata['context']['rollout']['recording_start'], 'human_takeover')
        self.assertEqual(metadata['context']['rollout']['scene'], 'correction-only')

    def test_stop_from_human_establishes_measured_hold_without_finishing(self):
        self.begin()
        self.controller.collection('intervene')
        self.pull()
        wait_for(lambda: self.controller.snapshot()['mode'] == 'human')
        time.sleep(.03)
        self.controller.stop()
        wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
        pose = self.controller.hardware.position.copy()
        time.sleep(.03)
        self.assertTrue((self.controller.hardware.position == pose).all())
        self.assertTrue(self.controller.snapshot()['rollout_active'])
        self.assertEqual(self.controller.snapshot()['control_source'], 'hold')

    def test_idle_rollout_and_stopped_correction_write_no_hold_data(self):
        self.begin()
        time.sleep(.08)
        self.controller.recorder.events.join()
        self.assertEqual(self.controller.recorder.snapshot()['samples'], 0)
        self.assertFalse(self.controller.recorder.snapshot()['collecting'])
        self.controller.collection('intervene')
        self.pull(('left',))
        wait_for(lambda: self.controller.recorder.snapshot()['samples'] > 3)
        self.controller.stop()
        wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
        self.controller.recorder.events.join()
        before = self.controller.recorder.snapshot()
        frames = dict(self.controller.recorder.active['frames'])
        time.sleep(.1)
        self.controller.recorder.events.join()
        after = self.controller.recorder.snapshot()
        self.assertEqual(before['samples'], after['samples'])
        self.assertEqual(before['duration'], after['duration'])
        self.assertEqual(frames, self.controller.recorder.active['frames'])
        self.assertTrue(after['active'])

    def test_safety_slowed_policy_records_on_its_own_clock(self):
        self.begin()
        def large_move(base, path, body=None, timeout=3):
            result = demo_request(base, path, body, timeout)
            if path == '/infer':
                result['actions'] = [[.5]*14]*50
            return result
        with patch('inference_dashboard.request', large_move):
            self.controller.submit('step', 20)
            wait_for(lambda: self.controller.snapshot()['mode'] == 'executing')
            started = time.monotonic()
            while self.controller.snapshot()['steps_completed'] < 1:
                self.controller.heartbeat()
                self.assertLess(time.monotonic()-started, 8)
                time.sleep(.01)
            wait_for(lambda: not self.controller.recorder.snapshot()['collecting'])
        self.controller.collection('finish', dict(label='kept', success=True, quality=5))
        wait_for(lambda: self.controller.recorder.snapshot()['last_decision'] is not None)
        directory = next((Path(self.tmp.name)/'raw/kept').iterdir())
        states = [json.loads(s) for s in (directory/'states.jsonl').read_text().splitlines()]
        data_span = states[-1]['data_time_s'] - states[0]['data_time_s']
        wall_span = (states[-1]['monotonic_ns'] - states[0]['monotonic_ns'])/1e9
        self.assertAlmostEqual(data_span, 1., delta=.02)
        self.assertGreater(wall_span, data_span*1.5)
        self.assertLessEqual(len(states), 152)
        self.assertTrue(all(s['control_source'] == 'policy' for s in states))
        np.testing.assert_allclose(states[-1]['arms']['left']['position_rad'], .5, atol=.001)
        frames = [json.loads(s) for s in (directory/'frames.jsonl').read_text().splitlines()]
        self.assertLessEqual(len(frames), 63)
        self.assertTrue(all(f['data_time_s'] <= 1.01 for f in frames))

    def test_reset_blocked_while_recording_and_browser_loss_leaves_hold(self):
        self.begin()
        with self.assertRaises(RuntimeError):
            self.controller.submit('reset')
        self.controller.collection('intervene')
        self.pull()
        wait_for(lambda: self.controller.snapshot()['mode'] == 'human')
        self.controller.heartbeat_at = time.monotonic()-LEASE_SECONDS-1
        wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
        self.assertEqual(self.controller.snapshot()['hold_status'], 'commanded')
        self.assertFalse(self.controller.snapshot()['rollout_active'])
        wait_for(lambda: self.controller.recorder.snapshot()['last_decision'] is not None)
        self.assertEqual(self.controller.recorder.snapshot()['last_decision']['label'], 'review')
        with self.assertRaises(RuntimeError):
            self.controller.collection('intervene')

    def test_holds_until_pull_and_records_only_after_independent_release(self):
        self.controller.collection('intervene', dict(operator='tester', scene='pull'))
        wait_for(lambda: self.controller.snapshot()['mode'] == 'takeover_hold')
        wait_for(lambda: self.controller.snapshot()['handoff_baseline_ready'])
        pose = self.controller.hardware.position.copy()
        time.sleep(.15)
        self.assertFalse(self.controller.snapshot()['rollout_active'])
        self.assertIsNone(self.controller.recorder.snapshot()['active'])
        np.testing.assert_array_equal(self.controller.hardware.position, pose)
        self.pull(('left',))
        time.sleep(.1)
        self.assertEqual(self.controller.snapshot()['handoff_ready'], dict(left=True, right=False))
        self.assertTrue((self.controller.hardware.position[:7] > pose[:7]).all())
        np.testing.assert_array_equal(self.controller.hardware.position[7:], pose[7:])
        self.controller.hardware.pull_torque['left'] = [0.]*6
        time.sleep(.1)
        self.assertTrue(self.controller.snapshot()['handoff_ready']['left'])
        self.pull(('right',))
        self.controller.collection('finish', dict(label='kept', success=True, quality=5))
        wait_for(lambda: self.controller.recorder.snapshot()['last_decision'] is not None)
        directory = next((Path(self.tmp.name)/'raw/kept').iterdir())
        states = [json.loads(line) for line in (directory/'states.jsonl').read_text().splitlines()]
        metadata = json.loads((directory/'metadata.json').read_text())
        self.assertEqual(states[0]['control_source_by_arm'], dict(left='human', right='hold'))
        self.assertEqual(states[0]['control_source'], 'human')
        self.assertEqual([e['side'] for e in metadata['handoff_events']], ['left', 'right'])
        self.assertEqual(metadata['start_monotonic_ns'], metadata['handoff_events'][0]['monotonic_ns'])
        self.assertEqual(self.controller.snapshot()['handoff_ready'], {})

    def test_gripper_motion_starts_its_arm_without_handle_force(self):
        self.controller.collection('intervene', dict(operator='tester', scene='gripper'))
        wait_for(lambda: self.controller.snapshot()['handoff_baseline_ready'])
        pose = self.controller.hardware.position.copy()
        self.controller.hardware.position[13] += .1
        time.sleep(.03)
        self.assertFalse(self.controller.snapshot()['rollout_active'])
        np.testing.assert_array_equal(self.controller.hardware.position[:6], pose[:6])
        np.testing.assert_array_equal(self.controller.hardware.position[7:13], pose[7:13])
        wait_for(lambda: self.controller.snapshot()['mode'] == 'human')
        self.assertEqual(self.controller.snapshot()['handoff_ready'], dict(left=False, right=True))
        self.assertEqual(self.controller.hardware.pull_torque, dict(left=[0.]*6, right=[0.]*6))
        self.controller.collection('finish', dict(label='kept', success=True, quality=5))
        wait_for(lambda: self.controller.recorder.snapshot()['last_decision'] is not None)
        directory = next((Path(self.tmp.name)/'raw/kept').iterdir())
        metadata = json.loads((directory/'metadata.json').read_text())
        event = metadata['handoff_events'][0]
        self.assertEqual(event['event'], 'gripper_motion_release')
        self.assertEqual(event['side'], 'right')
        states = [json.loads(line) for line in (directory/'states.jsonl').read_text().splitlines()]
        self.assertEqual(states[0]['control_source_by_arm'], dict(left='hold', right='human'))
        self.assertEqual(metadata['start_monotonic_ns'], event['monotonic_ns'])
        self.assertGreater(states[0]['arms']['right']['gripper']['position'], pose[13])
        stopped = self.controller.hardware.position.copy()
        time.sleep(.03)
        np.testing.assert_array_equal(self.controller.hardware.position, stopped)

    def test_cancel_waiting_hold_creates_no_episode_and_force_cannot_release_after_stop(self):
        for action in ('resume', 'stop'):
            self.controller.collection('intervene')
            wait_for(lambda: self.controller.snapshot()['mode'] == 'takeover_hold')
            if action == 'stop':
                self.controller.stop()
            else:
                self.controller.collection('resume')
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
            self.controller.hardware.pull_torque['left'] = [2.]*6
            time.sleep(.15)
            self.assertEqual(self.controller.snapshot()['mode'], 'holding')
            self.assertFalse(self.controller.snapshot()['rollout_active'])
            self.assertIsNone(self.controller.recorder.snapshot()['active'])

    def test_held_arm_tracking_fault_cancels_takeover(self):
        self.controller.collection('intervene')
        wait_for(lambda: self.controller.snapshot()['mode'] == 'takeover_hold')
        self.controller.hardware.position[7] += .2
        wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
        self.assertIn('tracking deviation', self.controller.snapshot()['error'])
        self.assertEqual(self.controller.snapshot()['handoff_ready'], {})
        self.assertFalse(self.controller.snapshot()['rollout_active'])

    def test_validation_and_shutdown_quarantine(self):
        with self.assertRaises(ValueError):
            self.controller.collection('begin', dict(operator='', scene='x'))
        with self.assertRaises(RuntimeError):
            self.controller.collection('resume')
        self.begin()
        for value in (True, 0, 6, 1.5):
            with self.assertRaises(ValueError):
                self.controller.collection('finish', dict(label='kept', success=True, quality=value))
        self.controller.shutdown()
        self.controller.worker.join(3)
        metadata = json.loads(next((Path(self.tmp.name)/'raw/review').glob('*/metadata.json')).read_text())
        self.assertFalse(metadata['complete'])


class HardwareHandoffTests(unittest.TestCase):
    def test_held_joints_use_position_and_both_grippers_are_passive_when_armed(self):
        hardware = RobotHardware()
        hardware.robots = {side: Mock() for side in ('left', 'right')}
        target = np.arange(14, dtype=float)/10
        arms = {side: dict(gravity_model_torque_nm=[.1]*6, gripper={})
                for side in ('left', 'right')}
        hardware.guide(arms, target, dict(left=True, right=False))
        left, right = hardware.robots['left'], hardware.robots['right']
        left.Joint_Pos_Vel.assert_not_called()
        left.gripper_control.assert_not_called()
        left.pos_vel_tqe_kp_kd.assert_called_once()
        np.testing.assert_array_equal(left.pos_vel_tqe_kp_kd.call_args.args[2], [.1]*6)
        left.gripper_control_MIT.assert_called_once_with(0, 0, 0, 0, 0)
        right.pos_vel_tqe_kp_kd.assert_not_called()
        right.gripper_control_MIT.assert_called_once_with(0, 0, 0, 0, 0)
        right.Joint_Pos_Vel.assert_called_once()
        np.testing.assert_array_equal(right.Joint_Pos_Vel.call_args.args[0], target[7:13])
        right.gripper_control.assert_not_called()
        self.assertEqual(arms['left']['gripper']['control_mode'], 'passive')
        self.assertEqual(arms['right']['control_mode'], 'position_hold')


class SelectionTests(unittest.TestCase):
    def metadata(self, success=True):
        return dict(label='kept', complete=True, context=dict(mode='human_in_the_loop'),
                    outcome=dict(success=success))

    def states(self, sources):
        return [dict(monotonic_ns=(i+1)*1000000000, control_source=s) for i,s in enumerate(sources)]

    def test_only_corrections_from_assisted_rollouts_and_never_cross_handoffs(self):
        states=self.states(['policy','human','human','hold','policy','human','human','hold'])
        expected=[('human_correction',2000000000,3000000000),
                  ('human_correction',6000000000,7000000000)]
        self.assertEqual(select_intervals(self.metadata(), states), expected)
        self.assertEqual(select_intervals(self.metadata(False), states), expected)

    def test_unassisted_requires_success_and_policy_motion_and_trims_idle_ends(self):
        states=self.states(['hold','policy','hold','policy','hold'])
        self.assertEqual(select_intervals(self.metadata(),states),
                         [('successful_unassisted',2000000000,4000000000)])
        self.assertEqual(select_intervals(self.metadata(False),states), [])
        self.assertEqual(select_intervals(self.metadata(),self.states(['hold','hold'])), [])
        for update in (dict(label='review'),dict(label='discarded'),dict(complete=False)):
            self.assertEqual(select_intervals(dict(self.metadata(), **update),states), [])

    def test_missing_ownership_and_bad_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            select_intervals(self.metadata(),[dict(monotonic_ns=1)])
        with self.assertRaises(ValueError):
            select_intervals(self.metadata(),[dict(monotonic_ns=1,control_source='human')]*2)

    def test_corrections_split_across_omitted_hold_and_use_data_clock(self):
        states = self.states(['human']*4)
        for i, state in enumerate(states):
            state.update(data_monotonic_ns=(i+1)*10000000, recording_segment=i//2)
        self.assertEqual(select_intervals(self.metadata(), states), [
            ('human_correction', 10000000, 20000000),
            ('human_correction', 30000000, 40000000)])
