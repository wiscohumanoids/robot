"""Faithful opt-in decision logging with simulated cameras and arms only."""
import base64
import io
import json
import queue
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from expo_recording import ExplorationRolloutStore
from hitl_demo import DemoHardware, demo_request
from inference_dashboard import Controller
from remote_inference import image_payload


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('Timed out waiting for simulated collection')
        time.sleep(.005)


def jpeg_payload():
    stream = io.BytesIO()
    Image.new('RGB', (256, 256), '#176543').save(stream, format='JPEG')
    return base64.b64encode(stream.getvalue()).decode()


def rows(path):
    return [json.loads(row) for row in path.read_text().splitlines() if row]


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ExplorationRolloutStore(self.tmp.name, dict(checkpoint_id='weights', processor_id='stats'))
        self.addCleanup(self.store.close)
        self.store.begin(dict(operator='test', scene='sim'), 100)
        self.encoded = jpeg_payload()

    def request(self):
        self.store.requested(3, [0.] * 14, dict(overhead=self.encoded), 101,
                             dict(overhead=dict(sequence=4, monotonic_ns=99)))

    def complete(self, response=None):
        response = response or dict(actions=[[.01] * 14], checkpoint='saved')
        self.store.accepted(response, [[.01] * 14])
        self.store.command([.005] * 14, 102, .025)
        self.store.command([.01] * 14, 103, .05)
        self.store.completed([.009] * 14, dict(overhead=self.encoded), 104)

    def test_exact_observations_commands_and_missing_candidates_are_preserved(self):
        self.request()
        self.complete()
        self.store.end(dict(success=True, label='kept', complete=True))
        self.store.close()
        decision, = rows(self.store.path / 'decisions.jsonl')
        self.assertTrue(decision['completed'])
        self.assertEqual(decision['observation']['state'], [0.] * 14)
        self.assertEqual(decision['next_observation']['state'], [.009] * 14)
        self.assertEqual(decision['executed_actions'], [[.01] * 14])
        self.assertEqual(decision['emitted_command_count'], 2)
        self.assertNotIn('base_actions', decision)
        self.assertNotIn('next_base_actions', decision)
        self.assertIsNone(decision['rewards'])
        image = self.store.path / decision['observation']['images']['overhead']
        self.assertEqual(image.read_bytes(), base64.b64decode(self.encoded))
        commands = rows(self.store.path / 'commands.jsonl')
        self.assertEqual({c['decision_id'] for c in commands}, {decision['decision_id']})
        end = rows(self.store.path / 'episodes.jsonl')[-1]
        self.assertEqual(end['final_decision_id'], decision['decision_id'])
        self.assertTrue(end['success'])
        self.assertFalse(end['intervened'])
        self.assertEqual(end['human_interventions'], 0)

    def test_real_candidates_are_copied_without_fabricating_next_candidates(self):
        self.request()
        candidates = [[[.02] * 14]] * 8
        provenance = dict(kind='pi05_inference', sampling='independent')
        self.complete(dict(actions=[[.01] * 14], base_actions=candidates,
                           candidates_provenance=provenance, checkpoint_id='frozen', processor_id='exact'))
        self.store.end(dict(success=False, label='kept', complete=True))
        self.store.close()
        decision, = rows(self.store.path / 'decisions.jsonl')
        self.assertEqual(decision['base_actions'], candidates)
        self.assertEqual(decision['checkpoint_id'], 'frozen')
        self.assertEqual(decision['processor_id'], 'exact')
        self.assertNotIn('next_base_actions', decision)
        self.assertFalse(rows(self.store.path / 'episodes.jsonl')[-1]['success'])

    def test_intervention_makes_pending_decision_incomplete_and_truncated(self):
        self.request()
        self.store.accepted(dict(actions=[[.01] * 14]), [[.01] * 14])
        self.store.command([.002] * 14, 102, .01)
        self.store.boundary('intervene')
        self.store.end(dict(success=True, label='kept', complete=True))
        self.store.close()
        decision, = rows(self.store.path / 'decisions.jsonl')
        self.assertFalse(decision['completed'])
        self.assertTrue(decision['truncated'])
        self.assertFalse(decision['contiguous_policy'])
        self.assertIsNone(decision['next_observation'])
        self.assertTrue(rows(self.store.path / 'episodes.jsonl')[-1]['intervened'])

    def test_completion_requires_final_emitted_waypoint(self):
        self.request()
        self.store.accepted(dict(actions=[[.01] * 14]), [[.01] * 14])
        self.store.command([.005] * 14, 102, .01)
        self.store.completed([.005] * 14, dict(overhead=self.encoded), 104)
        wait_for(lambda: self.store.failure is not None)
        self.assertIn('final emitted waypoint', self.store.failure)
        self.store.close()
        self.assertEqual(rows(self.store.path / 'decisions.jsonl'), [])

    def test_queue_saturation_fails_collection_without_blocking_cleanup(self):
        with patch.object(self.store.events, 'put_nowait', side_effect=queue.Full):
            with self.assertRaisesRegex(RuntimeError, 'queue full'):
                self.store.boundary('operator_stop')
        self.store.close()
        self.assertTrue(self.store.closed)
        self.assertFalse(self.store.thread.is_alive())

    def test_finish_preserves_completed_prefix_and_cancels_unexecuted_request(self):
        self.request()
        self.complete()
        self.store.requested(4, [.009] * 14, dict(overhead=self.encoded), 105)
        self.store.end(dict(success=True, label='kept', complete=True))
        self.store.close()
        decision, = rows(self.store.path/'decisions.jsonl')
        events = rows(self.store.path/'episodes.jsonl')
        cancelled = next(event for event in events if event['event'] == 'unexecuted_request')
        self.assertTrue(cancelled['cancelled_before_execution'])
        self.assertEqual(cancelled['decision']['decision_index'], 4)
        self.assertEqual(cancelled['decision']['emitted_command_count'], 0)
        self.assertEqual(events[-1]['final_decision_id'], decision['decision_id'])
        self.assertFalse(events[-1]['interrupted'])
        self.assertTrue(events[-1]['success'])

    def test_finish_midway_through_emitted_prefix_stays_incomplete(self):
        self.request()
        self.complete()
        self.store.requested(4, [.009] * 14, dict(overhead=self.encoded), 105)
        self.store.accepted(dict(actions=[[.02] * 14]), [[.02] * 14])
        self.store.command([.015] * 14, 106, .025)
        self.store.end(dict(success=True, label='kept', complete=True))
        self.store.close()
        complete, incomplete = rows(self.store.path/'decisions.jsonl')
        self.assertTrue(complete['completed'])
        self.assertFalse(incomplete['completed'])
        self.assertTrue(incomplete['truncated'])
        self.assertTrue(rows(self.store.path/'episodes.jsonl')[-1]['interrupted'])
        self.assertFalse(any(event['event'] == 'unexecuted_request'
                             for event in rows(self.store.path/'episodes.jsonl')))


class ImageDemoHardware(DemoHardware):
    def expo_images(self):
        frames = self.cameras.recording_frames()
        return ({name: image_payload(frame['jpeg'], name) for name, frame in frames.items()},
                {name: {key: value for key, value in frame.items() if key != 'jpeg'}
                 for name, frame in frames.items()})


class MultiActionStoreTests(unittest.TestCase):
    def create(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = ExplorationRolloutStore(temporary.name, dict(fixed_prefix_length=17,
                                       training_objective='gear_insertion', contract=dict(fps=20)))
        self.addCleanup(store.close)
        store.begin(dict(operator='test'), 100)
        images = dict(overhead=jpeg_payload())
        store.requested(0, [0.] * 14, images, 101)
        actions = [[.002 * (index+1)] * 14 for index in range(17)]
        store.accepted(dict(actions=actions), actions)
        return store, actions, images

    def test_full_seventeen_action_prefix_has_one_following_observation(self):
        store, actions, images = self.create()
        for index, action in enumerate(actions):
            store.command(action, 102+index, (index+1)/20)
        store.completed(actions[-1], images, 120)
        store.end(dict(success=True, label='kept', complete=True))
        store.close()
        self.assertIsNone(store.failure)
        decision, = rows(store.path/'decisions.jsonl')
        self.assertTrue(decision['completed'])
        self.assertEqual(decision['actual_prefix_length'], 17)
        self.assertEqual(decision['executed_actions'], actions)
        self.assertEqual(decision['next_observation']['state'], actions[-1])
        self.assertEqual(len(rows(store.path/'commands.jsonl')), 17)
        self.assertEqual(json.loads((store.path/'manifest.json').read_text())['fixed_prefix_length'], 17)
        self.assertFalse(rows(store.path/'episodes.jsonl')[-1]['interrupted'])

    def test_finish_after_partial_seventeen_action_prefix_rejects_source(self):
        store, actions, images = self.create()
        for index, action in enumerate(actions[:8]):
            store.command(action, 102+index, (index+1)/20)
        store.end(dict(success=True, label='kept', complete=True))
        store.close()
        decision, = rows(store.path/'decisions.jsonl')
        self.assertFalse(decision['completed'])
        self.assertTrue(decision['truncated'])
        self.assertIsNone(decision['next_observation'])
        self.assertTrue(rows(store.path/'episodes.jsonl')[-1]['interrupted'])

    def test_final_target_without_full_policy_duration_cannot_complete(self):
        store, actions, images = self.create()
        store.command(actions[-1], 102, .05)
        store.completed(actions[-1], images, 103)
        wait_for(lambda: store.failure is not None)
        self.assertIn('full emitted policy duration', store.failure)
        store.close()
        self.assertEqual(rows(store.path/'decisions.jsonl'), [])


class ControllerTests(unittest.TestCase):
    def create(self, expo_objective='gear_and_pin'):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        controller = Controller('demo', ImageDemoHardware, log_path=root/'steps.jsonl',
                                data_dir=root/'hitl', expo_record_dir=root/'expo', expo_objective=expo_objective)
        def cleanup():
            controller.shutdown()
            if controller.worker:
                controller.worker.join(3)
        self.addCleanup(cleanup)
        controller.connect(False)
        wait_for(lambda: controller.snapshot()['mode'] == 'holding')
        controller.collection('begin', dict(operator='test', scene='sim'))
        wait_for(lambda: controller.snapshot()['rollout_active'])
        return controller

    def test_optional_mode_enforces_single_prefix_and_no_blending(self):
        normal = Controller('unused')
        self.assertEqual(normal.snapshot()['blend_actions'], 10)
        self.assertIsNone(normal.expo_recorder)
        optional = Controller('unused', expo_record_dir='unused')
        self.assertEqual(optional.snapshot()['blend_actions'], 0)
        for count, blend in ((5, 0), (1, 10)):
            with self.assertRaisesRegex(ValueError, 'exactly 1'):
                optional.configure(count, blend)
            with self.assertRaisesRegex(ValueError, 'exactly 1'):
                optional.submit('step', count, blend)

    def test_seventeen_prefix_is_fixed_for_the_session(self):
        controller = Controller('unused', expo_record_dir='unused', expo_prefix_length=17)
        self.assertEqual(controller.snapshot()['actions_per_step'], 17)
        self.assertEqual(controller.snapshot()['expo_prefix_length'], 17)
        for count, blend in ((1, 0), (16, 0), (18, 0), (17, 1)):
            with self.assertRaisesRegex(ValueError, 'exactly 17'):
                controller.configure(count, blend)
        for prefix in (0, 51):
            with self.assertRaises(ValueError):
                Controller('unused', expo_record_dir='unused', expo_prefix_length=prefix)

    def test_mismatched_server_prefix_rejected_before_hardware_open(self):
        def mismatched(base, path, body=None, timeout=3):
            return dict(demo_request(base, path, body, timeout), chunk_size=1)
        controller = Controller('demo', ImageDemoHardware, expo_record_dir='unused', expo_prefix_length=17)
        with patch('inference_dashboard.request', mismatched), patch.object(ImageDemoHardware, 'open') as opened:
            controller.connect(False)
            controller.worker.join(2)
        self.assertFalse(opened.called)
        self.assertIn('prefix length', controller.snapshot()['error'])

    def test_exact_request_and_following_tick_observation_are_logged(self):
        requested = []
        def delayed(base, path, body=None, timeout=3):
            if path == '/infer':
                requested.append(json.loads(json.dumps(body)))
                time.sleep(.025)
            return demo_request(base, path, body, timeout)
        with patch('inference_dashboard.request', delayed):
            controller = self.create()
            controller.submit('step', 1, 0)
            wait_for(lambda: controller.snapshot()['steps_completed'] == 1)
            wait_for(lambda: (controller.expo_recorder.path/'decisions.jsonl').exists() and
                     bool(rows(controller.expo_recorder.path/'decisions.jsonl')))
            controller.collection('finish', dict(label='kept', success=False, quality=2))
            wait_for(lambda: not controller.snapshot()['rollout_active'])
            controller.shutdown()
            controller.worker.join(3)
        decision, = rows(controller.expo_recorder.path / 'decisions.jsonl')
        self.assertEqual(decision['observation']['state'], requested[0]['state'])
        for camera, encoded in requested[0]['images'].items():
            image = controller.expo_recorder.path / decision['observation']['images'][camera]
            self.assertEqual(image.read_bytes(), base64.b64decode(encoded))
        self.assertEqual(decision['next_observation']['state'], decision['executed_actions'][0])
        self.assertGreater(decision['next_observation']['monotonic_ns'], decision['observation']['monotonic_ns'])
        self.assertTrue(decision['completed'])
        self.assertTrue(decision['simulated'])
        self.assertIsNone(decision['rewards'])
        self.assertFalse(rows(controller.expo_recorder.path/'episodes.jsonl')[-1]['success'])

    def test_stop_while_inference_pending_cannot_create_completed_transition(self):
        def delayed(base, path, body=None, timeout=3):
            if path == '/infer':
                time.sleep(.15)
            return demo_request(base, path, body, timeout)
        with patch('inference_dashboard.request', delayed):
            controller = self.create()
            controller.submit('step', 1, 0)
            wait_for(lambda: controller.snapshot()['mode'] == 'predicting')
            controller.stop()
            wait_for(lambda: controller.snapshot()['mode'] == 'holding')
            self.assertEqual(controller.snapshot()['hold_status'], 'commanded')
            controller.collection('finish', dict(label='review', success=False, quality=1))
            wait_for(lambda: not controller.snapshot()['rollout_active'])
            controller.shutdown()
            controller.worker.join(3)
        decision, = rows(controller.expo_recorder.path/'decisions.jsonl')
        self.assertTrue(decision['truncated'])
        self.assertFalse(decision['completed'])
        self.assertIsNone(decision['executed_actions'])

    def test_policy_identity_change_rejects_response_before_motion(self):
        def changed_identity(base, path, body=None, timeout=3):
            result = demo_request(base, path, body, timeout)
            result.update(checkpoint_id='base', processor_id='stats', normalizer_hash='normalizer',
                          rl_model_id='frozen' if path == '/health' else 'replacement',
                          training_objective='gear_and_pin')
            return result
        with patch('inference_dashboard.request', changed_identity):
            controller = self.create()
            controller.submit('step', 1, 0)
            wait_for(lambda: 'identity changed' in (controller.snapshot()['error'] or ''))
            self.assertEqual(controller.hardware.position.tolist(), [0.] * 14)
            self.assertEqual(controller.snapshot()['hold_status'], 'commanded')
            controller.shutdown()
            controller.worker.join(3)
        decision, = rows(controller.expo_recorder.path/'decisions.jsonl')
        self.assertEqual(decision['response']['rl_model_id'], 'replacement')
        self.assertIsNone(decision['executed_actions'])
        self.assertTrue(decision['truncated'])

    def test_play_finish_while_waiting_next_request_retains_terminal_completed_step(self):
        requests = []
        def delayed_next(base, path, body=None, timeout=3):
            if path == '/infer':
                requests.append(body)
                if len(requests) == 2:
                    time.sleep(.15)
            return demo_request(base, path, body, timeout)
        with patch('inference_dashboard.request', delayed_next):
            controller = self.create()
            controller.submit('play', 1, 0)
            wait_for(lambda: len(requests) == 2 and controller.snapshot()['mode'] == 'predicting')
            controller.collection('finish', dict(label='kept', success=True, quality=4))
            wait_for(lambda: not controller.snapshot()['rollout_active'])
            self.assertFalse(controller.snapshot()['playing'])
            controller.shutdown()
            controller.worker.join(3)
        decision, = rows(controller.expo_recorder.path/'decisions.jsonl')
        self.assertEqual(decision['next_observation']['observation_id'], requests[1]['observation_id'])
        self.assertTrue(decision['completed'])
        events = rows(controller.expo_recorder.path/'episodes.jsonl')
        self.assertTrue(any(event['event'] == 'unexecuted_request' for event in events))
        self.assertFalse(events[-1]['interrupted'])
        self.assertEqual(events[-1]['final_decision_id'], decision['decision_id'])

    def test_gear_objective_and_explicit_failed_trial_approval_are_preserved(self):
        with patch('inference_dashboard.request', demo_request):
            controller = self.create('gear_insertion')
            self.assertEqual(controller.snapshot()['expo_objective'], 'gear_insertion')
            with self.assertRaisesRegex(ValueError, 'objective'):
                controller.collection('finish', dict(label='review', success=False, quality=5,
                    review_approved=True, training_objective='gear_and_pin'))
            controller.submit('step', 1, 0)
            wait_for(lambda: controller.snapshot()['expo_recording']['completed_decisions'] == 1)
            controller.collection('finish', dict(label='review', success=False, quality=5,
                                                 review_approved=True))
            wait_for(lambda: not controller.snapshot()['rollout_active'])
            controller.shutdown();controller.worker.join(3)
        manifest=json.loads((controller.expo_recorder.path/'manifest.json').read_text())
        self.assertEqual(manifest['training_objective'], 'gear_insertion')
        self.assertTrue(rows(controller.expo_recorder.path/'episodes.jsonl')[-1]['review_approved'])
        for row in rows(controller.expo_recorder.path/'episodes.jsonl'):
            if row['event'] in ('begin','end'):
                self.assertEqual(row['training_objective'], 'gear_insertion')
        self.assertEqual(rows(controller.expo_recorder.path/'decisions.jsonl')[0]['training_objective'], 'gear_insertion')

    def test_failed_review_approval_restricted_to_expo_failure(self):
        for enabled,label,success in ((False,'review',False),(True,'kept',True),(True,'review',True)):
            controller=Controller('unused',expo_record_dir='unused' if enabled else None)
            with self.assertRaisesRegex(ValueError, 'explicitly failed EXPO'):
                controller.collection('finish',dict(label=label,success=success,quality=5,review_approved=True))

    def test_record_trial_required_before_expo_playback(self):
        controller=Controller('unused',expo_record_dir='unused')
        controller.update(mode='holding');controller.heartbeat()
        with self.assertRaisesRegex(RuntimeError,'Record an EXPO trial'):
            controller.submit('play',1,0)

    def test_mismatched_trained_objective_rejected_before_hardware_open(self):
        def mismatched(base,path,body=None,timeout=3):
            return dict(demo_request(base,path,body,timeout),rl_model_id='trained',training_objective='gear_and_pin')
        controller=Controller('demo',ImageDemoHardware,expo_record_dir='unused',expo_objective='gear_insertion')
        with patch('inference_dashboard.request',mismatched), patch.object(ImageDemoHardware,'open') as opened:
            controller.connect(False)
            controller.worker.join(2)
        self.assertFalse(opened.called)
        self.assertIn('training objective',controller.snapshot()['error'])


if __name__ == '__main__':
    unittest.main()
