"""C17 integration using the real controller and CPU-only in-memory fixtures.

The test-only hardware flag permits replay validation; these temporary files
are fabricated fixtures, never evidence of a real successful robot rollout.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'app'))
from hitl_demo import DemoHardware
from inference_dashboard import Controller
from remote_inference import image_payload
from infra.rl.remote_server import Runtime, CONTRACT, NAMES, canonical_hash
from infra.rl.replay import load_replay
from scripts.cache_expo_candidates import cache_candidates
from scripts.prepare_expo_replay import prepare


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class FixtureHardware(DemoHardware):
    # Only this temporary integration fixture bypasses the simulation gate.
    simulated = False

    def expo_images(self):
        frames = self.cameras.recording_frames()
        return ({name: image_payload(frame['jpeg'], name) for name, frame in frames.items()},
                {name: {key: value for key, value in frame.items() if key != 'jpeg'}
                 for name, frame in frames.items()})


class FixtureBackend:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.draws = 0
        self.calls = []

    def request(self, path, body=None):
        self.calls.append((path, copy.deepcopy(body)))
        if path == '/health':
            return dict(status='ready', checkpoint=self.checkpoint, contract=CONTRACT, names=NAMES)
        self.draws += 1
        actions = [(np.asarray(body['state']) + .001 * (i + 1) + .0001 * self.draws).tolist()
                   for i in range(50)]
        return dict(sequence=body['sequence'], checkpoint=self.checkpoint, fps=20,
                    actions=actions, mock=False, limit_corrections=[])


class Chunk17PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        files = {name: hashlib.sha256(name.encode()).hexdigest() for name in
                 ('config.json', 'deployment.json', 'model.safetensors',
                  'policy_preprocessor.json', 'policy_postprocessor.json')}
        checkpoint_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        processors = {name: digest for name, digest in files.items() if 'processor' in name}
        processor_id = hashlib.sha256(json.dumps(processors, sort_keys=True).encode()).hexdigest()
        self.normalization = dict(checkpoint_id=checkpoint_id, processor_id=processor_id,
            action=dict(mode='QUANTILES', q01=[-1.] * 14, q99=[1.] * 14),
            state=dict(mode='QUANTILES', q01=[-2.] * 14, q99=[2.] * 14))
        checkpoint = '/temporary-test-only/frozen-023500/pretrained_model'
        manifest = dict(schema_version=1, synthetic=False, verified=True,
            source_kind='verified_deployed_pi05_checkpoint', checkpoint=checkpoint,
            contract=CONTRACT, file_manifest=files, checkpoint_id=checkpoint_id,
            processor_id=processor_id, normalization=self.normalization,
            normalization_sha256=canonical_hash(self.normalization))
        self.manifest = self.root / 'fixture-manifest.json'
        self.manifest.write_text(json.dumps(manifest))
        self.backend = FixtureBackend(checkpoint)
        self.runtime = Runtime(self.manifest, client=self.backend, prefix_length=17)

    def wait(self, controller, predicate, timeout=4, allowed_error=None):
        deadline = time.monotonic() + timeout
        while not predicate():
            controller.heartbeat()
            state = controller.snapshot()
            if state.get('error') and state['error'] != allowed_error:
                raise AssertionError(state['error'])
            if time.monotonic() > deadline:
                raise AssertionError('C17 controller fixture timed out: ' + repr(state))
            time.sleep(.005)

    def connect_controller(self):
        controller = Controller('in-memory-only', FixtureHardware,
            log_path=self.root / 'steps.jsonl', data_dir=self.root / 'fixture-hitl',
            expo_record_dir=self.root / 'fixture-expo', expo_objective='gear_insertion',
            expo_prefix_length=17)
        def cleanup():
            controller.shutdown()
            if controller.worker:
                controller.worker.join(3)
        self.addCleanup(cleanup)
        controller.connect(False)
        self.wait(controller, lambda: controller.snapshot()['mode'] == 'holding')
        return controller

    def dispatch(self, base, path, body=None, timeout=3):
        return self.runtime.health() if path == '/health' else self.runtime.infer(body)

    def collect_complete(self):
        with patch('inference_dashboard.request', self.dispatch):
            controller = self.connect_controller()
            for success, count in ((True, 1), (False, 2)):
                controller.collection('begin', dict(operator='CPU-test-fixture', scene='no-hardware'))
                self.wait(controller, lambda: controller.snapshot()['rollout_active'])
                for _ in range(count):
                    completed = controller.snapshot()['steps_completed']
                    controller.submit('step', 17, 0)
                    self.wait(controller, lambda: controller.snapshot()['steps_completed'] == completed + 1)
                    self.wait(controller, lambda: controller.expo_recorder.snapshot()['completed_decisions'] >= _ + 1)
                controller.collection('finish', dict(label='kept' if success else 'review',
                    success=success, quality=3, **({} if success else {'review_approved': True})))
                self.wait(controller, lambda: not controller.snapshot()['rollout_active'])
            controller.shutdown()
            controller.worker.join(3)
        self.assertIsNone(controller.expo_recorder.failure)
        session = controller.expo_recorder.path
        decisions = read_rows(session / 'decisions.jsonl')
        self.assertEqual(len(decisions), 3)
        commands = read_rows(session / 'commands.jsonl')
        for decision in decisions:
            emitted = [row for row in commands if row['decision_id'] == decision['decision_id']]
            self.assertEqual(decision['actual_prefix_length'], 17)
            self.assertEqual(len(decision['executed_actions']), 17)
            self.assertGreaterEqual(len(emitted), 17)
            self.assertGreaterEqual(emitted[-1]['policy_time_s'], 17 / 20 - 1e-8)
            self.assertGreater(decision['next_observation']['monotonic_ns'], emitted[-1]['emitted_monotonic_ns'])
            self.assertGreaterEqual(decision['next_observation']['monotonic_ns'] -
                                    decision['observation']['monotonic_ns'], 17 / 20 * 1e9)
            np.testing.assert_allclose(decision['next_observation']['state'], decision['executed_actions'][-1], atol=1e-8)
            self.assertTrue(decision['completed'])
        cache = self.root / 'fixture-cache'
        draws_before = self.backend.draws
        summary = cache_candidates([session], None, cache, backend='http://127.0.0.1:8765',
            manifest=self.manifest, remote_runtime_factory=lambda manifest, backend:
                Runtime(manifest, backend=backend, client=self.backend, prefix_length=17))
        self.assertEqual(self.backend.draws - draws_before, summary['observations'] * 8)
        for entry in read_rows(cache / 'candidates.jsonl'):
            self.assertEqual(np.asarray(entry['base_actions']).shape, (8, 50, 14))
        output = self.root / 'fixture-c17-replay.npz'
        prepare(session, self.normalization, output, cache, C=17, training_objective='gear_insertion')
        return session, cache, load_replay(output)

    def test_actual_controller_seventeen_actions_cache_and_discounted_terminal_credit(self):
        _, _, (arrays, metadata) = self.collect_complete()
        self.assertEqual(arrays['actions'].shape, (3, 17, 14))
        self.assertEqual(arrays['base_actions'].shape, (3, 8, 17, 14))
        self.assertEqual(arrays['next_base_actions'].shape, (3, 8, 17, 14))
        expected = np.zeros((3, 17), np.float32)
        expected[0, -1] = 1.
        np.testing.assert_array_equal(arrays['rewards'], expected)
        np.testing.assert_allclose(arrays['reward_return'], [.99 ** 16, 0., 0.], atol=1e-7)
        np.testing.assert_allclose(arrays['bootstrap_discount'], [0., .99 ** 17, 0.], atol=1e-7)
        self.assertEqual(metadata['spec']['C'], 17)
        self.assertEqual(metadata['training_objective'], 'gear_insertion')

    def test_replay_feeds_one_small_cpu_learner_update_and_seventeen_action_selection(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional CPU learner integration requires PyTorch')
        from infra.rl.learner import EXPOConfig, EXPOLearner
        _, _, (arrays, _) = self.collect_complete()
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        learner = EXPOLearner(EXPOConfig(obs_dim=2318, action_dim=14, chunk_size=17,
            hidden_dim=16, num_critics=3, target_subset_size=2, utd_ratio=1), device='cpu')
        keys = ('obs', 'next_obs', 'actions', 'base_actions', 'next_base_actions',
                'reward_return', 'bootstrap_discount', 'editor_train_valid', 'transition_valid')
        batch = {key: torch.from_numpy(arrays[key]) for key in keys}
        before = [parameter.detach().clone() for parameter in learner.critic.parameters()]
        metrics = learner.update(batch)
        self.assertTrue(all(math.isfinite(float(value)) for value in metrics.values()))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, learner.critic.parameters())))
        result = learner.select_actions(batch['obs'], batch['base_actions'], deterministic=True)
        self.assertEqual(tuple(result['actions'].shape), (3, 17, 14))
        self.assertEqual(tuple(result['scores'].shape), (3, 16))

    def test_partial_control_execution_never_creates_completed_transition(self):
        with patch('inference_dashboard.request', self.dispatch):
            controller = self.connect_controller()
            controller.collection('begin', dict(operator='CPU-test-fixture', scene='no-hardware'))
            self.wait(controller, lambda: controller.snapshot()['rollout_active'])
            controller.submit('step', 17, 0)
            self.wait(controller, lambda: controller.snapshot()['mode'] == 'executing' and
                      controller.snapshot()['action_index'] >= 4)
            controller.stop()
            self.wait(controller, lambda: controller.snapshot()['mode'] == 'holding')
            controller.collection('finish', dict(label='review', success=False, quality=1, review_approved=True))
            self.wait(controller, lambda: not controller.snapshot()['rollout_active'],
                      allowed_error='Motion stopped; arms remain held')
            controller.shutdown()
            controller.worker.join(3)
        session = controller.expo_recorder.path
        decision, = read_rows(session / 'decisions.jsonl')
        self.assertFalse(decision['completed'])
        self.assertTrue(decision['truncated'])
        self.assertIsNone(decision['next_observation'])
        output = self.root / 'invalid-partial.npz'
        with self.assertRaisesRegex(ValueError, 'Zero faithful transitions'):
            prepare(session, self.normalization, output, C=17, training_objective='gear_insertion')
        self.assertFalse(output.exists())

    def test_mixed_session_prefix_contracts_reject_before_any_export(self):
        session, cache, _ = self.collect_complete()
        other = session.parent / 'different-prefix-session'
        other.mkdir()
        manifest = json.loads((session / 'manifest.json').read_text())
        manifest['fixed_prefix_length'] = 1
        (other / 'manifest.json').write_text(json.dumps(manifest))
        (other / 'decisions.jsonl').write_text('')
        output = self.root / 'invalid-mixed-prefix.npz'
        with self.assertRaisesRegex(ValueError, 'chunk lengths cannot mix'):
            prepare(session.parent, self.normalization, output, cache, C=17, training_objective='gear_insertion')
        self.assertFalse(output.exists())

    def test_completed_flag_cannot_hide_a_shortened_emitted_policy_duration(self):
        session, cache, _ = self.collect_complete()
        path = session / 'decisions.jsonl'
        decisions = read_rows(path)
        for decision in decisions:
            decision['last_emitted_policy_time_s'] = 16 / 20
        path.write_text(''.join(json.dumps(row) + '\n' for row in decisions))
        output = self.root / 'invalid-early-final-target.npz'
        with self.assertRaisesRegex(ValueError, 'Zero faithful transitions'):
            prepare(session, self.normalization, output, cache, C=17, training_objective='gear_insertion')
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
