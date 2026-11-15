import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'app'))
from neutral_return import NeutralReturn
import test_recording as recording_tests


def arms(q, v=0):
    return {s: {'position_rad': list(q), 'velocity_rad_s': [v]*6,
                'gripper': {'position': 0, 'velocity': 0}}
            for s in ('left', 'right')}


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((ROOT/'config/recording.json').read_text())['neutral_return']
        self.limits = {s: (np.full(6, -1), np.full(6, 1)) for s in ('left', 'right')}

    def test_countdown_speed_limit_and_arrival(self):
        motion = NeutralReturn(self.settings, self.limits, 0)
        q = np.zeros(6)
        commanded, done, stage = motion.tick(self.settings['release_delay_seconds']/2, arms(q))
        self.assertFalse(done)
        if self.settings['release_delay_seconds'] > 0:
            self.assertIsNone(commanded)
            self.assertEqual(stage, 'countdown')
        else:
            self.assertIsNotNone(commanded)
            np.testing.assert_allclose(commanded['left'], q)
        previous = q.copy()
        max_speed = 0
        complete = False
        for now in np.arange(self.settings['release_delay_seconds'], 12, .01):
            target, complete, _ = motion.tick(float(now), arms(q))
            if target is not None:
                q = target['left'].copy()  # Fake motors track with one tick of latency.
                max_speed = max(max_speed, float(np.max(np.abs(q-previous)))/.01)
                previous = q
            if complete:
                break
        self.assertTrue(complete)
        self.assertLessEqual(max_speed, self.settings['max_joint_speed_rad_s']+1e-6)
        peak_acceleration = (10/np.sqrt(3))*float(np.max(np.abs(motion.targets['left']-motion.initial['left'])))/motion.duration**2
        self.assertLessEqual(peak_acceleration, self.settings['max_joint_acceleration_rad_s2']+1e-6)
        np.testing.assert_allclose(q, self.settings['position_rad']['left'])

    def test_stall_and_tracking_deviation_cancel(self):
        motion = NeutralReturn(self.settings, self.limits, 0)
        motion.tick(2, arms(np.zeros(6)))
        motion.tick(4.2, arms(np.zeros(6)))
        with self.assertRaisesRegex(RuntimeError, 'tracking deviation'):
            motion.tick(4.3, arms(np.zeros(6)))

    def test_moving_arms_start_return_on_first_tick(self):
        settings = copy.deepcopy(self.settings)
        settings['release_delay_seconds'] = 0
        motion = NeutralReturn(settings, self.limits, 0)
        q = np.zeros(6)
        target, complete, _ = motion.tick(0, arms(q, v=.2))
        self.assertIsNotNone(target)
        self.assertFalse(complete)
        self.assertEqual(motion.started, 0)
        for side in ('left', 'right'):
            np.testing.assert_allclose(target[side], q)

    def test_invalid_target_rejected_before_commands(self):
        settings = copy.deepcopy(self.settings)
        settings['position_rad']['left'][1] = 2
        with self.assertRaises(ValueError):
            NeutralReturn(settings, self.limits, 0)

    def test_noisy_reported_velocity_at_settled_pose_does_not_timeout(self):
        motion = NeutralReturn(self.settings, self.limits, 0)
        q = np.zeros(6)
        complete = False
        for i, now in enumerate(np.arange(2, 10, .01)):
            sample = arms(q, v=.15 if motion.started is not None else 0)
            target, complete, _ = motion.tick(float(now), sample)
            if target is not None:
                q = target['left'].copy()
                if now > motion.started+motion.duration:
                    q += .0008 if i % 2 else -.0008
            if complete:
                break
        self.assertTrue(complete)
        self.assertGreater(motion.diagnostics['reported_max_speed_rad_s']['left'], .04)

    def test_encoder_oscillation_cannot_be_mistaken_for_arrival(self):
        motion = NeutralReturn(self.settings, self.limits, 0)
        motion.tick(2, arms(np.zeros(6)))
        motion.tick(2.4, arms(np.zeros(6)))
        for i in range(80):
            now = motion.started+motion.duration+i*.01
            q = motion.targets['left']+(.006 if i % 2 else -.006)
            # Only the fake previous command is set to target to model the final stage.
            motion.last_targets = motion.targets
            _, complete, _ = motion.tick(now, arms(q))
            self.assertFalse(complete)

    def test_real_position_error_is_not_accepted_despite_low_velocity(self):
        motion = NeutralReturn(self.settings, self.limits, 0)
        motion.tick(2, arms(np.zeros(6)))
        motion.tick(2.4, arms(np.zeros(6)))
        motion.last_targets = motion.targets
        with self.assertRaisesRegex(RuntimeError, 'joint error 0.0200'):
            motion.tick(motion.started+motion.duration+5, arms(motion.targets['left']+.02))


class RecordingReturnTests(unittest.TestCase):
    setUp = recording_tests.EpisodeTests.setUp
    tearDown = recording_tests.EpisodeTests.tearDown
    sample = recording_tests.EpisodeTests.sample
    trigger = recording_tests.EpisodeTests.trigger
    decision = recording_tests.EpisodeTests.decision
    def test_all_labels_queue_exactly_one_return_and_exclude_reset_samples(self):
        for label in ('kept', 'review', 'discarded'):
            identifier = self.trigger()
            self.config['neutral_return']['enabled'] = True
            self.decision(label, identifier)
            self.assertEqual(self.store.snapshot()['phase'], 'returning_to_neutral')
            self.assertEqual(self.store.control_requests.get_nowait()['episode_id'], identifier)
            self.decision(label, identifier)
            self.assertTrue(self.store.control_requests.empty())
            for i in range(30):
                self.sample(.05+i*.01, .2)
            self.assertIsNone(self.store.snapshot()['active'])
            self.assertEqual(len(self.store.pre_roll), 0)
            self.assertEqual(self.store.snapshot()['phase'], 'returning_to_neutral')
            self.store.submit('return_complete', {'request':{'episode_id':identifier}})
            self.store.events.join()
            for _ in range(20):
                self.sample()
            self.assertEqual(self.store.snapshot()['phase'], 'armed')
            self.config['neutral_return']['enabled'] = False


if __name__ == '__main__':
    unittest.main()
