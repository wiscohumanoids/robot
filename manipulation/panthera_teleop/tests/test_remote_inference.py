"""Physical-boundary tests; never import/open the robot SDK."""
import importlib.util
import base64
import io
from concurrent.futures import Future
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('remote', ROOT / 'app/remote_inference.py')
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


class ServiceErrorTests(unittest.TestCase):
    def test_reports_backend_reason_for_503(self):
        error = HTTPError('http://127.0.0.1/health', 503, 'Unavailable', {},
                          io.BytesIO(b'{"error":"Backend checkpoint changed"}'))
        with patch.object(remote, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 503: Backend checkpoint changed'):
                remote.request('http://127.0.0.1', '/health')

    def test_non_json_error_preserves_http_error(self):
        error = HTTPError('http://127.0.0.1/health', 503, 'Unavailable', {},
                          io.BytesIO(b'<html>Unavailable</html>'))
        with patch.object(remote, 'urlopen', side_effect=error):
            with self.assertRaises(HTTPError):
                remote.request('http://127.0.0.1', '/health')


class ChunkSafetyTests(unittest.TestCase):
    def check(self, response, sequence=1):
        return remote.validate_chunk(response, sequence, np.zeros(14),
                                     np.full(14, -1.0), np.ones(14))

    def response(self):
        return dict(sequence=1, fps=20, mock=False, actions=[[0.01]*14]*50)

    def test_accepts_valid_chunk(self):
        self.assertEqual(self.check(self.response()).shape, (50, 14))

    def test_rejects_mock_wrong_sequence_and_rate(self):
        for key, value in [('mock', True), ('sequence', 2), ('fps', 30)]:
            r = self.response()
            r[key] = value
            with self.assertRaises(RuntimeError):
                self.check(r)

    def test_rejects_nonfinite_and_wrong_dimensions(self):
        for actions in ([[float('nan')]*14], [[0.0]*13], [], [[0.0]*14]*51):
            r = self.response()
            r['actions'] = actions
            with self.assertRaises(RuntimeError):
                self.check(r)

        for index in (6, 13):
            for value in (float('nan'), float('inf'), -float('inf')):
                r = self.response()
                r['actions'] = [[0.0] * 14]
                r['actions'][0][index] = value
                with self.assertRaisesRegex(RuntimeError, 'Malformed'):
                    self.check(r)

    def test_accepts_large_in_limit_changes_for_smoothing(self):
        r = self.response()
        r['actions'] = [[.8]*14, [-.8]*14]
        np.testing.assert_array_equal(self.check(r), r['actions'])

    def test_rejects_out_of_range_even_after_valid_actions(self):
        r = self.response()
        r['actions'] = [[.01]*14, [1.1]*14]
        with self.assertRaisesRegex(RuntimeError, 'limits'):
            self.check(r)

    def test_selected_prefix_does_not_validate_unexecuted_tail_limits(self):
        r = self.response()
        r['actions'][-1] = [1.1] * 14
        actions = remote.validate_chunk(r, 1, np.zeros(14), np.full(14, -1.),
                                        np.ones(14), action_count=1)
        self.assertEqual(actions.shape, (50, 14))
        np.testing.assert_array_equal(actions[0], [.01] * 14)
        with self.assertRaisesRegex(RuntimeError, 'action 50, left.joint1'):
            self.check(r)

    def test_selected_prefix_still_rejects_malformed_tail_and_invalid_count(self):
        for count in (0, 51, True, 1.5):
            with self.assertRaises(RuntimeError):
                remote.validate_chunk(self.response(), 1, np.zeros(14),
                                      np.full(14, -1.), np.ones(14), action_count=count)
        r = self.response()
        r['actions'][-1] = [float('nan')] * 14
        with self.assertRaisesRegex(RuntimeError, 'Malformed'):
            remote.validate_chunk(r, 1, np.zeros(14), np.full(14, -1.),
                                  np.ones(14), action_count=1)

    def test_gripper_overshoots_clamp_to_each_configured_stop_without_modifying_input(self):
        r = self.response()
        r['actions'] = [[.01] * 14, [.01] * 14]
        r['actions'][0][6], r['actions'][0][13] = -10., 10.
        r['actions'][1][6], r['actions'][1][13] = 10., -10.
        lower, upper = np.full(14, -1.), np.ones(14)
        lower[[6, 13]], upper[[6, 13]] = [0., .2], [2., 1.8]
        actions = remote.validate_chunk(r, 1, np.zeros(14), lower, upper)
        np.testing.assert_array_equal(actions[:, [6, 13]], [[0., 1.8], [2., .2]])
        np.testing.assert_array_equal(np.delete(actions, [6, 13], axis=1), .01)
        np.testing.assert_array_equal(np.asarray(r['actions'])[:, [6, 13]],
                                      [[-10., 10.], [10., -10.]])

    def test_any_joint_overshoot_still_rejects(self):
        for index in (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12):
            for value in (-1.001, 1.001):
                r = self.response()
                r['actions'] = [[.01] * 14]
                r['actions'][0][index] = value
                with self.assertRaisesRegex(RuntimeError, 'action 1'):
                    self.check(r)

    def test_limit_corrections_preserve_input_and_record_all_overshoots(self):
        raw = np.zeros((2, 14))
        raw[0, 6], raw[1, 13], raw[1, 6] = -.02, 2.02, -.5
        raw[0, 0] = 100.
        bounded, corrections = remote.clamp_gripper_actions(raw)
        self.assertEqual(bounded[0, 6], 0.)
        self.assertEqual(bounded[1, 13], 2.)
        self.assertEqual(bounded[1, 6], 0.)
        self.assertEqual(bounded[0, 0], 100.)
        self.assertEqual(raw[1, 6], -.5)
        self.assertEqual(len(corrections), 3)
        self.assertEqual(corrections[0]['raw_target_rad'], -.02)
        self.assertEqual(corrections[1], dict(action=2, index=6, raw_target_rad=-.5,
                                              command_target_rad=0.))

    def test_clamping_leaves_nonfinite_values_for_validation(self):
        raw = np.zeros((3, 14))
        raw[:, [6, 13]] = np.array([np.nan, np.inf, -np.inf])[:, None]
        bounded, corrections = remote.clamp_gripper_actions(raw)
        np.testing.assert_array_equal(bounded, raw)
        self.assertEqual(corrections, [])


class ServerClampingTests(unittest.TestCase):
    def test_server_clamps_grippers_and_reports_raw_corrections(self):
        spec = importlib.util.spec_from_file_location('inference_server',
                                                     ROOT / 'infra/inference/server.py')
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        buffer = io.BytesIO()
        Image.new('RGB', (256, 256)).save(buffer, format='JPEG')
        image = base64.b64encode(buffer.getvalue()).decode()
        state = [.01] * 14
        state[6], state[13] = -5., 8.
        payload = dict(sequence=1, state=state,
                       images={name: image for name in remote.CONTRACT['image_crops_xyxy']})
        result = server.Runtime('unused', mock=True).infer(payload)
        actions = np.asarray(result['actions'])
        np.testing.assert_array_equal(actions[:, [6, 13]], np.tile([0., 2.], (50, 1)))
        np.testing.assert_array_equal(np.delete(actions, [6, 13], axis=1),
                                      np.float32(.01))
        self.assertEqual(len(result['limit_corrections']), 100)
        self.assertEqual(result['limit_corrections'][0],
                         dict(action=1, index=6, raw_target_rad=-5., command_target_rad=0.))
        self.assertEqual(result['limit_corrections'][-1],
                         dict(action=50, index=13, raw_target_rad=8., command_target_rad=2.))


class SmoothTrajectoryTests(unittest.TestCase):
    def test_large_motion_reaches_each_waypoint_in_order_with_bounded_derivatives(self):
        actions = np.array([[.5]*14, [-.2]*14, [.1]*14])
        trajectory = remote.SmoothActionTrajectory(actions, 0.)
        target = np.zeros(14)
        previous_velocity = np.zeros(14)
        reached = []
        for tick in range(2000):
            previous = target.copy()
            index = trajectory.index
            target, done = trajectory.tick(tick * .005, .005, target)
            velocity = (target - previous) / .005
            self.assertTrue(np.all(np.abs(velocity) <= trajectory.speed + 1e-9))
            acceleration = (velocity - previous_velocity) / .005
            self.assertTrue(np.all(np.abs(acceleration) <= trajectory.acceleration + .01))
            previous_velocity = velocity
            self.assertTrue(np.all(target >= min(0., actions.min()) - 1e-9))
            self.assertTrue(np.all(target <= actions.max() + 1e-9))
            if trajectory.index != index:
                np.testing.assert_allclose(target, actions[index], atol=1e-10)
                reached.append(index)
            if done:
                break
        self.assertTrue(done)
        self.assertEqual(reached, [0, 1, 2])
        self.assertGreater(tick * .005, len(actions) / remote.CONTRACT['fps'])
        np.testing.assert_allclose(target, actions[-1])

    def test_slow_segment_does_not_expire_or_skip_to_later_waypoints(self):
        trajectory = remote.SmoothActionTrajectory(np.array([[1.]*14, [-1.]*14]), 0.)
        target = np.zeros(14)
        for tick in range(21):
            target, done = trajectory.tick(tick*.01, .01, target)
        self.assertFalse(done)
        self.assertEqual(trajectory.index, 0)
        self.assertTrue(np.all(target > 0))
        self.assertTrue(np.all(target < 1))

    def test_jitter_cannot_exceed_emitted_speed_or_jump_bound(self):
        trajectory = remote.SmoothActionTrajectory(np.ones((1, 14)), 0.)
        target = np.zeros(14)
        now = 0.
        for dt in [.002, .02, .07, .005, .09] * 40:
            now += dt
            previous = target.copy()
            target, done = trajectory.tick(now, dt, target)
            self.assertTrue(np.all(np.abs(target-previous) <= trajectory.speed*dt + 1e-9))
            self.assertLessEqual(np.max(np.abs(target-previous)), remote.CONTRACT['max_target_jump_rad'])
            if done:
                break
        self.assertTrue(done)
        np.testing.assert_allclose(target, 1.)

    def test_gripper_has_separate_speed_bound(self):
        goal = np.ones((1, 14))
        goal[0, [6, 13]] = 2.
        trajectory = remote.SmoothActionTrajectory(goal, 0.)
        trajectory.tick(0., .01, np.zeros(14))
        np.testing.assert_allclose(trajectory.speed[[6, 13]], 5.)
        np.testing.assert_allclose(trajectory.speed[[0, 7]], 1.)
        self.assertGreaterEqual(trajectory.duration, 1.875)

    def test_policy_clock_tracks_each_retimed_waypoint_and_caps_at_end(self):
        trajectory = remote.SmoothActionTrajectory(np.repeat([[.5]*14, [.51]*14], 1, axis=0), 10.)
        trajectory.tick(10., .005, np.zeros(14))
        first_duration = trajectory.durations[0]
        self.assertGreater(first_duration, 1/remote.CONTRACT['fps'])
        progress, rate = trajectory.policy_clock(10. + first_duration/2)
        self.assertAlmostEqual(progress, .025)
        self.assertAlmostEqual(rate, .05/first_duration)
        progress, rate = trajectory.policy_clock(10. + first_duration + trajectory.durations[1]/2)
        self.assertAlmostEqual(progress, .075)
        self.assertAlmostEqual(rate, .05/trajectory.durations[1])
        self.assertAlmostEqual(trajectory.policy_clock(10. + trajectory.duration + 1)[0], .1)

    def test_full_gripper_travel_finishes_in_under_one_second(self):
        goal = np.zeros((1, 14))
        goal[0, [6, 13]] = 2.
        trajectory = remote.SmoothActionTrajectory(goal, 0.)
        trajectory.tick(0., .005, np.zeros(14))
        self.assertLess(trajectory.duration, 1.)
        self.assertGreaterEqual(trajectory.duration, 1.875 * 2. / 5.)

    def test_same_direction_waypoints_have_continuous_nonzero_velocity(self):
        actions = np.repeat(np.linspace(.01, .5, 50)[:, None], 14, axis=1)
        trajectory = remote.SmoothActionTrajectory(actions, 0.)
        trajectory.tick(0., .005, np.zeros(14))
        controls = trajectory.controls
        duration = trajectory.durations[:, None]
        entering = 5 * (controls[1:, 1] - controls[1:, 0]) / duration[1:]
        leaving = 5 * (controls[:-1, 5] - controls[:-1, 4]) / duration[:-1]
        np.testing.assert_allclose(entering, leaving, atol=1e-12)
        self.assertTrue(np.all(entering > 0))
        np.testing.assert_allclose(controls[:, 5], actions)
        np.testing.assert_allclose(controls[1:, 0], actions[:-1])
        acceleration = 20 * np.diff(controls, n=2, axis=1) / duration[:, None, :]**2
        np.testing.assert_allclose(acceleration[:, [0, -1]], 0, atol=1e-12)
        # The old stop-at-every-action ramp took about 8.5 s for this path.
        self.assertLess(trajectory.duration, 8.5)

    def test_reversals_holds_and_endpoints_stop_without_overshoot(self):
        actions = np.repeat(np.array([.1, .2, .2, -.1, .05])[:, None], 14, axis=1)
        trajectory = remote.SmoothActionTrajectory(actions, 0.)
        trajectory.tick(0., .005, np.zeros(14))
        c = trajectory.controls
        np.testing.assert_allclose(c[0, :3], 0)
        np.testing.assert_allclose(c[-1, 3:], actions[-1][None, :].repeat(3, axis=0))
        for index in (1, 2, 3):
            np.testing.assert_allclose(c[index, 4], c[index, 5])
            np.testing.assert_allclose(c[index + 1, 0], c[index + 1, 1])
        low = np.minimum(c[:, 0], c[:, 5])[:, None]
        high = np.maximum(c[:, 0], c[:, 5])[:, None]
        self.assertTrue(np.all(c >= low - 1e-12))
        self.assertTrue(np.all(c <= high + 1e-12))

    def test_irregular_joint_paths_keep_emitted_speed_and_acceleration_bounds(self):
        actions = np.random.default_rng(4).uniform(-.15, .15, (8, 14))
        trajectory = remote.SmoothActionTrajectory(actions, 0.)
        target = np.zeros(14)
        previous_velocity = np.zeros(14)
        for tick in range(10000):
            previous = target.copy()
            target, done = trajectory.tick(tick * .005, .005, target)
            velocity = (target - previous) / .005
            self.assertTrue(np.all(np.abs(velocity) <= trajectory.speed + 1e-9))
            self.assertTrue(np.all(np.abs(velocity - previous_velocity) / .005 <=
                                   trajectory.acceleration + 1e-8))
            previous_velocity = velocity
            if done:
                break
        self.assertTrue(done)
        np.testing.assert_allclose(target, actions[-1])

    def test_invalid_tick_and_waypoints_rejected(self):
        for actions in ([], [[0.]*13], [[float('nan')]*14]):
            with self.assertRaises(ValueError):
                remote.SmoothActionTrajectory(actions, 0.)
        trajectory = remote.SmoothActionTrajectory(np.ones((1, 14)), 0.)
        for dt in (-.01, .101, float('nan')):
            with self.assertRaisesRegex(RuntimeError, 'deadline'):
                trajectory.tick(0., dt, np.zeros(14))


class ChunkStreamTests(unittest.TestCase):
    def setUp(self):
        self.stream = remote.InferenceChunkStream()
        self.target = np.zeros(14)
        self.lower, self.upper = np.full(14, -1.), np.ones(14)

    def response(self, **changes):
        result = dict(sequence=0, fps=20, mock=False, actions=[[.01] * 14] * 50)
        result.update(changes)
        return result

    def complete_request(self, response, observed_at=0.):
        self.stream.observed_at = observed_at
        self.stream.future = Future()
        self.stream.future.set_result(response)

    def test_prefetch_is_near_end_and_buffer_is_not_executed_early(self):
        trajectory = remote.SmoothActionTrajectory(np.full((1, 14), .5), 0.)
        trajectory.tick(0., .005, self.target)
        self.assertTrue(self.stream.needs_request(0., None))
        self.assertFalse(self.stream.needs_request(0., trajectory))
        now = trajectory.duration - .2
        self.assertTrue(self.stream.needs_request(now, trajectory))
        self.complete_request(self.response(), observed_at=now)
        self.stream.poll(now + .1, self.target, self.lower, self.upper)
        self.assertEqual(trajectory.index, 0)
        self.assertFalse(self.stream.needs_request(now + .1, trajectory))
        next_trajectory = self.stream.activate(trajectory.duration)
        self.assertEqual(next_trajectory.started, trajectory.duration)
        self.assertEqual(self.stream.sequence, 1)
        self.assertIsNone(self.stream.pending)

    def test_expired_requests_and_buffered_observations_cannot_execute(self):
        self.complete_request(self.response())
        with self.assertRaisesRegex(RuntimeError, 'expired'):
            self.stream.poll(2.01, self.target, self.lower, self.upper)
        self.stream.poll(.1, self.target, self.lower, self.upper)
        with self.assertRaisesRegex(RuntimeError, 'expired'):
            self.stream.activate(2.01)

    def test_invalid_prefetch_rejected_before_buffering(self):
        self.complete_request(self.response(actions=[[1.1] * 14]))
        with self.assertRaisesRegex(RuntimeError, 'outside'):
            self.stream.poll(.1, self.target, self.lower, self.upper)
        self.assertIsNone(self.stream.pending)

    def test_slow_request_does_not_replace_or_restart_current_trajectory(self):
        trajectory = remote.SmoothActionTrajectory(np.full((1, 14), .1), 0.)
        self.stream.observed_at = 0.
        self.stream.future = Future()
        self.assertFalse(self.stream.needs_request(.1, trajectory))
        self.assertIsNone(self.stream.poll(.1, self.target, self.lower, self.upper))
        self.assertIsNone(self.stream.activate(.1))
        self.assertEqual(trajectory.started, 0.)


if __name__ == '__main__':
    unittest.main()
