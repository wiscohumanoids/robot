"""CPU-only serving and exact-ledger caching contract checks; no VLA is loaded."""
import base64
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infra.rl.contracts import CAMERAS, normalization_fingerprint
from infra.rl.server import Runtime
from infra.rl.replay import REWARD_SEMANTICS
from scripts.cache_expo_candidates import cache_candidates, read_observations


def jpeg():
    stream = io.BytesIO()
    Image.new('RGB', (256, 256), (12, 34, 56)).save(stream, format='JPEG')
    return stream.getvalue()


class FakeAdapter:
    """An explicit arithmetic fixture, never passed as a real policy cache."""
    checkpoint_id = 'test-checkpoint'
    processor_id = 'test-processor'
    constructed = 0

    def __init__(self, checkpoint, device):
        type(self).constructed += 1
        self.normalization = dict(checkpoint_id=self.checkpoint_id, processor_id=self.processor_id,
            action=dict(q01=[-1.] * 14, q99=[1.] * 14, mode='QUANTILES', eps=1e-8),
            state=dict(q01=[-2.] * 14, q99=[2.] * 14, mode='QUANTILES', eps=1e-8))

    def load(self):
        return self

    def infer_candidates(self, payload, n, seed):
        if set(payload['images']) != set(CAMERAS):
            raise ValueError('Missing camera')
        physical = np.zeros((n, 50, 14), dtype=np.float32)
        for index in range(n):
            physical[index, :, :6] = .01 * index
        return SimpleNamespace(physical=physical, normalized=physical.copy(),
            checkpoint_id=self.checkpoint_id, processor_id=self.processor_id,
            observation_id='fixture-observation-digest', seed=seed)

    def denormalize_actions(self, values):
        return np.asarray(values)


class ServerContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / 'base'
        self.checkpoint.mkdir()
        self.leases = []

    @contextmanager
    def guard(self, **kwargs):
        self.leases.append('enter')
        try:
            yield {'allowed': True, 'device': kwargs['device']}
        finally:
            self.leases.append('exit')

    def runtime(self, path=None, **kwargs):
        runtime = Runtime(path or self.checkpoint, device='cpu', guard=self.guard,
                          adapter_factory=FakeAdapter, **kwargs)
        self.addCleanup(runtime.close)
        return runtime

    def payload(self):
        return dict(sequence=7, observation_id='recorded-observation', state=[0.] * 14,
                    images={camera: base64.b64encode(jpeg()).decode() for camera in CAMERAS})

    def test_occupied_gpu_fails_before_model_construction(self):
        @contextmanager
        def occupied(**kwargs):
            raise RuntimeError('GPU has existing compute processes')
            yield
        before = FakeAdapter.constructed
        runtime = Runtime(self.checkpoint, guard=occupied, adapter_factory=FakeAdapter)
        with self.assertRaisesRegex(RuntimeError, 'existing compute'):
            runtime.load()
        self.assertEqual(FakeAdapter.constructed, before)
        self.assertEqual(runtime.status, 'failed')

    def test_base_service_keeps_lease_and_eight_full_chunks_but_commands_C1(self):
        runtime = self.runtime()
        runtime.load()
        self.assertEqual(self.leases, ['enter'])
        response = runtime.infer(self.payload())
        self.assertEqual(np.asarray(response['base_actions']).shape, (8, 50, 14))
        self.assertEqual(np.asarray(response['actions']).shape, (1, 14))
        self.assertEqual(response['observation_id'], 'recorded-observation')
        self.assertEqual(response['candidates_provenance']['observation_hash'], 'fixture-observation-digest')
        self.assertEqual(response['normalizer_hash'], normalization_fingerprint(runtime.normalization))
        self.assertIsNone(response['rl_model_id'])
        self.assertIsNone(response['training_objective'])
        self.assertIsNone(runtime.health()['training_objective'])
        runtime.close()
        self.assertEqual(self.leases, ['enter', 'exit'])

    def test_synthetic_editor_checkpoint_is_rejected_before_learner_load(self):
        editor = self.root / 'checkpoint-000001.pt'
        editor.write_bytes(b'fixture-only')
        (self.root / 'run.json').write_text(json.dumps(dict(
            mode='synthetic_bandit_algorithm_check_NOT_PANTHERA', synthetic=True)))
        called = []
        runtime = self.runtime(rl_checkpoint=editor, learner_factory=lambda *a, **k: called.append(True))
        with self.assertRaisesRegex(ValueError, 'synthetic checkpoints'):
            runtime.load()
        self.assertFalse(called)
        self.assertEqual(self.leases, ['enter', 'exit'])

    def test_rl_uses_normalized_C1_then_physical_selected_prefix(self):
        editor = self.root / 'checkpoint-000001.pt'
        editor.write_bytes(b'fixture-only')
        normalization = FakeAdapter(self.checkpoint, 'cpu').normalization
        lower, upper = [None] * 14, [None] * 14
        for index in (6, 13):
            lower[index], upper[index] = 0., 2.
        spec = dict(C=1, N=8, obs_dim=2318, checkpoint_id='test-checkpoint', processor_id='test-processor')
        report = dict(mode='frozen_base_expo_residual_rl', synthetic=False, completed_steps=1,
            training_objective='gear_insertion',
            reward_semantics=REWARD_SEMANTICS['gear_insertion'],
            config=dict(chunk_size=1, action_dim=14, obs_dim=2318,
                        action_lower=lower, action_upper=upper),
            replay_metadata=dict(kind='faithful_policy_replay', synthetic=False, training_eligible=True,
                training_objective='gear_insertion',
                spec=spec, normalization=normalization,
                normalization_sha256=normalization_fingerprint(normalization)))
        (self.root / 'run.json').write_text(json.dumps(report))
        class Learner:
            def __init__(inner):
                inner.config = SimpleNamespace(chunk_size=1, action_dim=14, obs_dim=2318,
                                               action_lower=lower, action_upper=upper)
                inner.checkpoint_metadata = dict(training_objective='gear_insertion',
                    reward_semantics=REWARD_SEMANTICS['gear_insertion'])
            def select_actions(inner, obs, bases, deterministic):
                self.assertEqual(obs.shape, (1, 2318))
                self.assertEqual(bases.shape, (1, 8, 1, 14))
                return dict(actions=bases[:, 3] + .02, selected=np.asarray([11]),
                            is_edited=np.asarray([True]), scores=np.zeros((1, 16)))
        runtime = self.runtime(rl_checkpoint=editor, learner_factory=lambda *a, **k: Learner())
        runtime.load()
        response = runtime.infer(self.payload())
        self.assertTrue(response['selected_is_edited'])
        self.assertEqual(response['selected_candidate'], 11)
        self.assertAlmostEqual(response['selected_physical_chunk'][0][0], .05, places=6)
        self.assertEqual(response['training_objective'], 'gear_insertion')
        self.assertEqual(runtime.health()['training_objective'], 'gear_insertion')
        self.assertEqual(runtime.health()['contract']['task'],
                         'put the gear in the carrier and put a pin in it')
        for metadata in ({}, dict(training_objective='gear_and_pin', reward_semantics=REWARD_SEMANTICS['gear_and_pin']),
                         dict(training_objective='gear_insertion', reward_semantics='different reward')):
            learner = Learner()
            learner.checkpoint_metadata = metadata
            invalid = self.runtime(rl_checkpoint=editor, learner_factory=lambda *a, **k: learner)
            with self.assertRaisesRegex(ValueError, 'objective/reward'):
                invalid.load()
        learner = Learner()
        learner.config.residual_bound = .5
        invalid = self.runtime(rl_checkpoint=editor, learner_factory=lambda *a, **k: learner)
        with self.assertRaisesRegex(ValueError, 'config differs'):
            invalid.load()

    def test_critic_objective_requires_explicit_matching_run_and_replay_stage(self):
        editor = self.root / 'checkpoint-000001.pt'
        editor.write_bytes(b'fixture-only')
        base = dict(mode='frozen_base_expo_residual_rl', synthetic=False, completed_steps=1,
                    replay_metadata=dict(kind='faithful_policy_replay', synthetic=False, training_eligible=True))
        runtime = self.runtime()
        runtime.load()
        runtime.rl_checkpoint = str(editor)
        for objective, replay_objective in ((None, None), ('unknown', 'unknown'),
                                            ('gear_insertion', None), ('gear_insertion', 'gear_and_pin')):
            report = dict(base, training_objective=objective,
                          replay_metadata=dict(base['replay_metadata'], training_objective=replay_objective))
            (self.root / 'run.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'training objective'):
                runtime._validate_rl_provenance()

    def test_successive_base_seeds_differ_and_explicit_cache_seed_is_preserved(self):
        runtime = self.runtime(seed=42)
        runtime.load()
        first = runtime.infer(self.payload())
        second = runtime.infer(self.payload())
        self.assertNotEqual(first['seed'], second['seed'])
        payload = self.payload()
        payload['seed'] = 91
        self.assertEqual(runtime.infer(payload)['candidates_provenance']['seed'], 91)

    def test_base_prefix_17_and_50_preserve_all_genuine_full_candidates(self):
        identities = []
        for length in (1, 17, 50):
            runtime = self.runtime(prefix_length=length)
            runtime.load()
            result = runtime.infer(self.payload())
            self.assertEqual(runtime.health()['chunk_size'], length)
            self.assertEqual(np.asarray(result['actions']).shape, (length, 14))
            self.assertEqual(np.asarray(result['base_actions']).shape, (8, 50, 14))
            identities.append(runtime.policy_bundle_id)
        self.assertEqual(len(set(identities)), 3)
        for length in (0, 51, True, 17.0):
            with self.assertRaisesRegex(ValueError, 'Prefix length'):
                self.runtime(prefix_length=length)

    def test_trained_17_step_chunk_is_complete_and_cannot_be_faked_from_C1(self):
        editor = self.root / 'checkpoint-000001.pt'
        editor.write_bytes(b'fixture-only')
        normalization = FakeAdapter(self.checkpoint, 'cpu').normalization
        lower, upper = [None] * 14, [None] * 14
        for index in (6, 13):
            lower[index], upper[index] = 0., 2.
        config = dict(chunk_size=17, action_dim=14, obs_dim=2318, action_lower=lower, action_upper=upper)
        metadata = dict(training_objective='gear_insertion', reward_semantics=REWARD_SEMANTICS['gear_insertion'])
        report = dict(mode='frozen_base_expo_residual_rl', synthetic=False, completed_steps=1,
            config=config, **metadata, replay_metadata=dict(kind='faithful_policy_replay', synthetic=False,
                training_eligible=True, training_objective='gear_insertion',
                spec=dict(C=17, N=8, obs_dim=2318, checkpoint_id='test-checkpoint', processor_id='test-processor'),
                normalization=normalization, normalization_sha256=normalization_fingerprint(normalization)))
        (self.root / 'run.json').write_text(json.dumps(report))
        class Learner:
            def __init__(inner):
                inner.config = SimpleNamespace(**config)
                inner.checkpoint_metadata = metadata
            def select_actions(inner, obs, bases, deterministic):
                self.assertEqual(bases.shape, (1, 8, 17, 14))
                return dict(actions=bases[:, 3] + .02, selected=np.asarray([11]),
                            is_edited=np.asarray([True]), scores=np.zeros((1, 16)))
        runtime = self.runtime(rl_checkpoint=editor, learner_factory=lambda *a, **k: Learner())
        runtime.load()
        result = runtime.infer(self.payload())
        self.assertEqual(runtime.health()['chunk_size'], 17)
        self.assertEqual(np.asarray(result['actions']).shape, (17, 14))
        self.assertEqual(np.asarray(result['base_actions']).shape, (8, 50, 14))
        invalid = self.runtime(rl_checkpoint=editor, prefix_length=1, learner_factory=lambda *a, **k: Learner())
        with self.assertRaisesRegex(ValueError, 'Requested prefix differs'):
            invalid.load()
        wrong = Learner()
        wrong.config.chunk_size = 1
        invalid = self.runtime(rl_checkpoint=editor, prefix_length=17, learner_factory=lambda *a, **k: wrong)
        with self.assertRaisesRegex(ValueError, 'config differs'):
            invalid.load()

    def make_session(self):
        session = self.root / 'session-uuid'
        session.mkdir()
        observations = []
        for identity in ('current-uuid', 'next-uuid'):
            images, hashes = {}, {}
            for camera in CAMERAS:
                relative = f'images/{identity}/{camera}.jpg'
                destination = session / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(jpeg())
                images[camera] = relative
                hashes[camera] = hashlib.sha256(jpeg()).hexdigest()
            observations.append(dict(observation_id=identity, state=[0.] * 14,
                                     images=images, image_sha256=hashes))
        (session / 'decisions.jsonl').write_text(json.dumps(dict(
            observation=observations[0], next_observation=observations[1])) + '\n')
        return session

    def test_cache_samples_exact_current_and_next_without_modifying_ledger(self):
        session = self.make_session()
        original = (session / 'decisions.jsonl').read_bytes()
        output = self.root / 'cache'
        result = cache_candidates([session], self.checkpoint, output,
                                  runtime_factory=lambda path: self.runtime(path))
        self.assertEqual(result['observations'], 2)
        rows = [json.loads(line) for line in (output / 'candidates.jsonl').read_text().splitlines()]
        self.assertEqual({row['observation_id'] for row in rows}, {'current-uuid', 'next-uuid'})
        self.assertTrue(all(np.asarray(row['base_actions']).shape == (8, 50, 14) for row in rows))
        self.assertEqual((session / 'decisions.jsonl').read_bytes(), original)
        self.assertEqual(self.leases, ['enter', 'exit'])

    def test_cache_refuses_missing_next_and_changed_exact_JPEG(self):
        session = self.make_session()
        image = session / 'images/current-uuid/overhead.jpg'
        image.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            read_observations([session])
        image.write_bytes(jpeg())
        row = json.loads((session / 'decisions.jsonl').read_text())
        del row['next_observation']
        (session / 'decisions.jsonl').write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'No exact recorded observations'):
            read_observations([session])

    def test_cache_skips_incomplete_and_nonpolicy_rows_in_mixed_session(self):
        session = self.make_session()
        ledger = session / 'decisions.jsonl'
        row = json.loads(ledger.read_text())
        with ledger.open('a') as stream:
            stream.write(json.dumps(dict(row, next_observation=None, completed=False)) + '\n')
            stream.write(json.dumps(dict(row, control_source='human')) + '\n')
            stream.write(json.dumps(dict(row, truncated=True)) + '\n')
        report = {}
        observations = read_observations([session], report)
        self.assertEqual(len(observations), 2)
        self.assertEqual(report['decisions_selected'], 1)
        self.assertEqual(report['skipped'], dict(incomplete=1, nonpolicy=1, truncated=1))


if __name__ == '__main__':
    unittest.main()
