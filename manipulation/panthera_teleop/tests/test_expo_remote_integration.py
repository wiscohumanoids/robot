"""Independent remote frozen-base contracts using an in-memory HTTP fixture.

No test contacts inference, allocates CUDA, loads a VLA, or opens hardware.
"""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infra.rl.contracts import ActionNormalizer
from infra.rl.remote_server import BackendChanged, CONTRACT, NAMES, Runtime, canonical_hash
from infra.rl.replay import load_replay
from scripts.cache_expo_candidates import cache_candidates
from scripts.prepare_expo_replay import prepare

sys.path.insert(0, str(ROOT / 'app'))
from expo_recording import ExplorationRolloutStore


class BackendFixture:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.calls = []
        self.draws = 0
        self.change_after = None

    def request(self, path, body=None):
        self.calls.append((path, copy.deepcopy(body)))
        if path == '/health':
            changed = self.change_after is not None and self.draws >= self.change_after
            return dict(status='ready', checkpoint='changed' if changed else self.checkpoint,
                        contract=CONTRACT, names=NAMES)
        self.draws += 1
        actions = [[.01 * self.draws] * 14 for _ in range(50)]
        for row in actions:
            row[6], row[13] = .8, 1.1
        actions[0][6], actions[2][6], actions[4][13] = 0., 0., 2.
        return dict(sequence=body['sequence'], fps=20, mock=False,
                    checkpoint=self.checkpoint, actions=actions,
                    limit_corrections=[dict(action=1, index=6, raw_target_rad=-.3,
                                            command_target_rad=0.),
                                       dict(action=3, index=6, raw_target_rad=-.2,
                                            command_target_rad=0.),
                                       dict(action=5, index=13, raw_target_rad=2.4,
                                            command_target_rad=2.)])


class RemoteFrozenBaseTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root / 'attestation.json'
        self.checkpoint = '/test-fixture/frozen-023500/pretrained_model'
        files = {key: hashlib.sha256(key.encode()).hexdigest() for key in
                 ('config.json', 'deployment.json', 'model.safetensors',
                  'policy_preprocessor.json', 'policy_postprocessor.json')}
        checkpoint_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        processors = {key: value for key, value in files.items() if 'processor' in key}
        processor_id = hashlib.sha256(json.dumps(processors, sort_keys=True).encode()).hexdigest()
        normalization = dict(checkpoint_id=checkpoint_id, processor_id=processor_id,
            action=dict(mode='QUANTILES', q01=[-.3] * 14, q99=[1.5] * 14, eps=1e-8),
            state=dict(mode='QUANTILES', q01=[-2.] * 14, q99=[2.] * 14, eps=1e-8))
        self.manifest = dict(schema_version=1, synthetic=False, verified=True,
                             source_kind='verified_deployed_pi05_checkpoint', checkpoint=self.checkpoint,
                             contract=CONTRACT, file_manifest=files, checkpoint_id=checkpoint_id,
                             processor_id=processor_id, normalization=normalization,
                             normalization_sha256=canonical_hash(normalization))
        self.path.write_text(json.dumps(self.manifest))
        self.client = BackendFixture(self.checkpoint)

    def payload(self):
        stream = io.BytesIO()
        Image.new('RGB', (256, 256), (12, 34, 56)).save(stream, format='JPEG')
        encoded = base64.b64encode(stream.getvalue()).decode()
        return dict(sequence=55, observation_id='exact-ledger-uuid', state=[.04] * 14,
                    images={camera: encoded for camera in CONTRACT['image_crops_xyxy']}, seed=999)

    def test_exactly_eight_genuine_draws_preserve_input_and_never_forward_fake_seed(self):
        runtime = Runtime(self.path, client=self.client)
        self.assertEqual(self.client.calls, [])
        payload = self.payload()
        untouched = copy.deepcopy(payload)
        result = runtime.sample_candidates(payload, num_candidates=8)
        requests = [body for path, body in self.client.calls if path == '/infer']
        self.assertEqual(len(requests), 8)
        self.assertEqual(sum(path == '/health' for path, _ in self.client.calls), 8)
        self.assertEqual(len(result['base_actions']), 8)
        self.assertEqual({round(chunk[0][0], 4) for chunk in result['base_actions']},
                         {.01, .02, .03, .04, .05, .06, .07, .08})
        for request in requests:
            self.assertEqual(request['state'], payload['state'])
            self.assertEqual(request['images'], payload['images'])
            self.assertNotIn('seed', request)
        self.assertEqual(payload, untouched)
        self.assertFalse(result['seed_controlled'])
        self.assertFalse(result['candidates_provenance']['seed_controlled'])

    def test_source_gripper_projection_provenance_and_normalization_roundtrip(self):
        result = Runtime(self.path, client=self.client).sample_candidates(self.payload(), 8)
        values = np.asarray(result['base_actions'], dtype=np.float32)
        normalizer = ActionNormalizer.from_stats(self.manifest['normalization']['action'])
        normalized = normalizer.normalize(values)
        np.testing.assert_allclose(normalizer.denormalize(normalized), values, atol=3e-7)
        # The physical 2-radian stop exceeds checkpoint q99=1.5. No generic
        # normalized [-1,1] clipping may remove that actual command target.
        self.assertGreater(float(normalized[:, 4, 13].min()), 1.)
        self.assertEqual(len(result['candidate_limit_corrections']), 8)
        for chunk, corrections in zip(values, result['candidate_limit_corrections']):
            for correction in corrections:
                self.assertEqual(float(chunk[correction['action'] - 1, correction['index']]),
                                 correction['command_target_rad'])
        self.assertEqual(result['candidates_provenance']['action_transform'],
                         'checkpoint_postprocess_and_source_gripper_clamp')

    def test_manifest_corruption_fails_before_any_backend_request(self):
        for mutate in ('weight-hash', 'normalization'):
            manifest = copy.deepcopy(self.manifest)
            if mutate == 'weight-hash':
                manifest['file_manifest']['model.safetensors'] = '0' * 64
            else:
                manifest['normalization']['action']['q01'][6] = -.2
            self.path.write_text(json.dumps(manifest))
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                Runtime(self.path, client=self.client)
        self.assertEqual(self.client.calls, [])

    def test_backend_identity_change_mid_pool_returns_no_partially_valid_candidate_set(self):
        self.client.change_after = 2
        runtime = Runtime(self.path, client=self.client)
        with self.assertRaises(BackendChanged):
            runtime.sample_candidates(self.payload(), 8)
        self.assertEqual(self.client.draws, 2)

    def test_attestation_mutation_after_construction_fails_before_inference(self):
        runtime = Runtime(self.path, client=self.client)
        self.path.write_text(self.path.read_text() + '\n')
        with self.assertRaises(BackendChanged):
            runtime.infer(self.payload())
        self.assertEqual(self.client.calls, [])

    def test_import_and_construction_need_no_model_framework_or_gpu_command(self):
        script = '''
import builtins, sys, subprocess
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'jax', 'tensorflow', 'lerobot'}:
        raise AssertionError('Unexpected model framework import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
subprocess.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Unexpected subprocess/GPU command'))
from infra.rl.remote_server import Runtime
class NoRequests:
    def request(self, *args, **kwargs):
        raise AssertionError('Construction must be lazy')
Runtime(sys.argv[1], client=NoRequests())
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.path)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def record_fixture_and_cache(self):
        """Temporary test-only fixture exercises real writer/cache/export APIs."""
        runtime = Runtime(self.path, client=self.client)
        store = ExplorationRolloutStore(self.root / 'test-ledgers',
            dict(simulated=False, checkpoint_id=runtime.checkpoint_id,
                 processor_id=runtime.processor_id, policy_bundle_id=runtime.policy_bundle_id,
                 training_objective='gear_insertion'))
        self.addCleanup(store.close)
        payload = self.payload()
        expected_commands = []
        for episode, success in enumerate((True, False)):
            store.begin(dict(operator='temporary-test-fixture', scene='no-hardware',
                             training_objective='gear_insertion'), 1000 + episode * 1000)
            for decision in range(2):
                now = 1010 + episode * 1000 + decision * 100
                response = runtime.infer(payload)
                store.requested(episode * 2 + decision, payload['state'], payload['images'], now)
                actions = response['actions']
                expected_commands.append(actions[0])
                store.accepted(response, actions)
                store.command(actions[0], now + 40, .05)
                store.completed(actions[0], payload['images'], now + 50)
            store.end(dict(label='kept' if success else 'review', success=success,
                           complete=True, review_approved=not success,
                           training_objective='gear_insertion',
                           reason='gear_seated' if success else 'gear_not_seated'))
        store.close()
        self.assertIsNone(store.failure)
        self.assertEqual(json.loads((store.path / 'manifest.json').read_text())['training_objective'],
                         'gear_insertion')
        for name in ('episodes.jsonl', 'decisions.jsonl'):
            for line in (store.path / name).read_text().splitlines():
                row = json.loads(line)
                if row.get('event') in (None, 'begin', 'end'):
                    self.assertEqual(row['training_objective'], 'gear_insertion')
        before = {path.relative_to(store.path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in store.path.rglob('*') if path.is_file()}
        draws_before = self.client.draws
        cache = self.root / 'test-candidate-cache'
        cache_manifest = cache_candidates([store.path], None, cache,
            backend='http://127.0.0.1:8765', manifest=self.path,
            remote_runtime_factory=lambda manifest, backend: Runtime(manifest, backend=backend, client=self.client))
        after = {path.relative_to(store.path): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in store.path.rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(cache_manifest['observations'], 8)
        self.assertEqual(self.client.draws - draws_before, 8 * 8)
        self.assertFalse(cache_manifest['seed_controlled'])
        self.assertIsNone(cache_manifest['seed'])
        self.assertFalse(cache_manifest['base_actions_normalized'])
        return store.path, cache, expected_commands

    def test_temporary_collector_remote_cache_and_replay_preserve_real_command_contracts(self):
        session, cache, commands = self.record_fixture_and_cache()
        output = self.root / 'test-only-replay.npz'
        normalization = self.manifest['normalization']
        prepare(session, normalization, output, cache, C=1, training_objective='gear_insertion')
        arrays, metadata = load_replay(output)
        self.assertEqual(len(arrays['actions']), 4)
        np.testing.assert_array_equal(arrays['rewards'][:, 0], [0., 1., 0., 0.])
        np.testing.assert_array_equal(arrays['terminated'], [False, True, False, True])
        np.testing.assert_allclose(arrays['bootstrap_discount'], [.99, 0., .99, 0.])
        normalizer = ActionNormalizer.from_stats(normalization['action'])
        np.testing.assert_allclose(normalizer.denormalize(arrays['actions'][:, 0]), commands, atol=3e-7)
        self.assertEqual(metadata['normalization'], normalization)
        self.assertEqual(metadata['training_objective'], 'gear_insertion')
        self.assertEqual(metadata['spec']['checkpoint_id'], self.manifest['checkpoint_id'])
        self.assertEqual(metadata['spec']['processor_id'], self.manifest['processor_id'])
        self.assertFalse(set(metadata['splits']['train_indices']) & set(metadata['splits']['eval_indices']))
        for row in metadata['transitions']:
            self.assertEqual(row['training_objective'], 'gear_insertion')
            self.assertEqual(row['action_provenance'], 'executed_policy_waypoints')
            self.assertEqual(row['candidate_provenance'], 'pi05_inference')
            self.assertFalse(row['intervened'])
            self.assertFalse(row['selected_is_edited'])
            self.assertEqual(row['selected_candidate'], 0)
            self.assertEqual(row['executed_limit_corrections'][0],
                             dict(action=1, index=6, raw_target_rad=-.3, command_target_rad=0.))
            for field in ('base_actions', 'next_base_actions'):
                evidence = row['candidate_evidence'][field]
                self.assertEqual(evidence['action_transform'],
                                 'checkpoint_postprocess_and_source_gripper_clamp')
                self.assertEqual(len(evidence['candidate_limit_corrections']), 8)
                self.assertEqual(evidence['candidates_provenance']['checkpoint_id'],
                                 self.manifest['checkpoint_id'])
                self.assertEqual(evidence['candidates_provenance']['processor_id'],
                                 self.manifest['processor_id'])
                self.assertFalse(evidence['candidates_provenance']['seed_controlled'])
                self.assertEqual(len(evidence['observation_payload_sha256']), 64)
        self.assertTrue(metadata['transitions'][2]['review_approved'])

    def test_remote_cache_corrupt_file_and_changed_processor_are_rejected(self):
        session, cache, _ = self.record_fixture_and_cache()
        manifest_path = cache / 'cache_manifest.json'
        original = json.loads(manifest_path.read_text())
        changed = copy.deepcopy(original)
        changed['processor_id'] = 'different'
        manifest_path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            prepare(session, self.manifest['normalization'], self.root / 'bad-processor.npz', cache, C=1)
        manifest_path.write_text(json.dumps(original))
        with (cache / 'candidates.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaises(ValueError):
            prepare(session, self.manifest['normalization'], self.root / 'bad-file.npz', cache, C=1)

    def test_cancelled_decision_excludes_entire_source_including_completed_prefix(self):
        session, cache, _ = self.record_fixture_and_cache()
        ledger = session / 'decisions.jsonl'
        decisions = [json.loads(line) for line in ledger.read_text().splitlines()]
        excluded_source = decisions[0]['source_episode_id']
        decisions[1].update(completed=False, truncated=True, contiguous_policy=False)
        ledger.write_text(''.join(json.dumps(row) + '\n' for row in decisions))
        output = self.root / 'cancelled-source-replay.npz'
        prepare(session, self.manifest['normalization'], output, cache, C=1)
        arrays, metadata = load_replay(output)
        self.assertEqual(len(arrays['actions']), 2)
        self.assertNotIn(excluded_source, metadata['sources'])
        np.testing.assert_array_equal(arrays['rewards'], 0.)
        self.assertEqual(len(metadata['audit']['rejected_sources']), 1)

    def test_intervened_episode_cannot_label_earlier_policy_prefix_successful(self):
        session, cache, _ = self.record_fixture_and_cache()
        path = session / 'episodes.jsonl'
        episodes = [json.loads(line) for line in path.read_text().splitlines()]
        excluded_source = None
        for row in episodes:
            if row.get('event') == 'end' and row.get('success') is True:
                row.update(intervened=True, human_interventions=1)
                excluded_source = row['source_episode_id']
        self.assertIsNotNone(excluded_source)
        path.write_text(''.join(json.dumps(row) + '\n' for row in episodes))
        output = self.root / 'assisted-source-replay.npz'
        prepare(session, self.manifest['normalization'], output, cache, C=1)
        arrays, metadata = load_replay(output)
        self.assertEqual(len(arrays['actions']), 2)
        self.assertNotIn(excluded_source, metadata['sources'])
        np.testing.assert_array_equal(arrays['rewards'], 0.)


if __name__ == '__main__':
    unittest.main()
