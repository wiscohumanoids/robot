"""No-network CPU fixtures for the lightweight existing-service EXPO proxy."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infra.rl.remote_server import (BackendChanged, BackendClient, CAMERAS, CONTRACT,
                                    NAMES, Runtime, canonical_hash, read_manifest)


def make_manifest():
    files = {name: hashlib.sha256(name.encode()).hexdigest() for name in (
        'config.json', 'deployment.json', 'model.safetensors', 'policy_preprocessor.json',
        'policy_postprocessor.json', 'policy_preprocessor_step_3_normalizer.safetensors')}
    checkpoint_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    processors = {name: digest for name, digest in files.items() if 'processor' in name or 'normalizer' in name}
    processor_id = hashlib.sha256(json.dumps(processors, sort_keys=True).encode()).hexdigest()
    normalization = dict(checkpoint_id=checkpoint_id, processor_id=processor_id,
        action=dict(q01=[-1.] * 14, q99=[1.] * 14, mode='QUANTILES', eps=1e-8),
        state=dict(q01=[-2.] * 14, q99=[2.] * 14, mode='QUANTILES', eps=1e-8))
    return dict(schema_version=1, synthetic=False, verified=True,
        source_kind='verified_deployed_pi05_checkpoint', checkpoint='/deployed/frozen-base',
        checkpoint_id=checkpoint_id, processor_id=processor_id, file_manifest=files,
        normalization=normalization, normalization_sha256=canonical_hash(normalization),
        contract=copy.deepcopy(CONTRACT), backend_service_identity={'MainPID': 100, 'InvocationID': 'test-only'})


class FakeClient:
    def __init__(self, manifest):
        self.calls = []
        self.health = dict(status='ready', checkpoint=manifest['checkpoint'], contract=copy.deepcopy(CONTRACT), names=NAMES)
        self.response_override = None
        self.health_override = None
        self.counter = 0

    def request(self, path, body=None):
        self.calls.append((path, copy.deepcopy(body)))
        if path == '/health':
            return copy.deepcopy(self.health_override or self.health)
        self.counter += 1
        response = dict(sequence=body['sequence'], checkpoint=self.health['checkpoint'],
            actions=[[self.counter / 100.] * 14 for _ in range(50)], fps=20,
            mock=False, limit_corrections=[])
        if self.response_override:
            response.update(self.response_override)
        return response


class RemoteProxyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'manifest.json'
        self.manifest = make_manifest()
        self.path.write_text(json.dumps(self.manifest))
        self.client = FakeClient(self.manifest)
        self.runtime = Runtime(self.path, client=self.client)
        stream = io.BytesIO()
        Image.new('RGB', (256, 256), (23, 45, 67)).save(stream, format='JPEG')
        encoded = base64.b64encode(stream.getvalue()).decode()
        self.payload = dict(sequence=7, observation_id='exact-recorded-uuid', state=[0.] * 14,
                            images={camera: encoded for camera in CAMERAS})

    def test_construction_has_no_network_requests_or_model_allocations(self):
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.runtime.checkpoint_id, self.manifest['checkpoint_id'])
        health = self.runtime.health()
        self.assertEqual(health['num_candidates'], 1)
        self.assertEqual(health['offline_num_candidates'], 8)
        self.assertEqual(health['normalization'], self.manifest['normalization'])
        self.assertIsNone(health['rl_model_id'])
        self.assertIsNone(health['training_objective'])
        self.assertFalse(health['live_weight_identity_verified'])
        self.assertEqual(health['live_identity_checks'], 'checkpoint_path_contract')

    def test_collection_uses_one_genuine_chunk_and_preserves_exact_input(self):
        self.payload['seed'] = 123
        result = self.runtime.infer(self.payload)
        self.assertEqual([path for path, _ in self.client.calls], ['/health', '/infer'])
        forwarded = self.client.calls[-1][1]
        self.assertEqual(forwarded['state'], self.payload['state'])
        self.assertEqual(forwarded['images'], self.payload['images'])
        self.assertNotIn('seed', forwarded)
        self.assertEqual(len(result['base_actions']), 1)
        self.assertEqual(len(result['base_actions'][0]), 50)
        self.assertEqual(result['actions'], result['base_actions'][0][:1])
        self.assertEqual(result['observation_id'], self.payload['observation_id'])
        self.assertFalse(result['candidates_provenance']['seed_controlled'])
        self.assertIsNone(result['training_objective'])
        self.assertNotIn('seed', result['candidates_provenance'])
        self.assertTrue(result['request_seed_ignored'])

    def test_offline_eight_candidates_are_eight_serial_backend_calls(self):
        result = self.runtime.sample_candidates(self.payload, num_candidates=8)
        self.assertEqual(len(self.client.calls), 16)
        self.assertEqual([path for path, _ in self.client.calls], ['/health', '/infer'] * 8)
        requests = [body for path, body in self.client.calls if path == '/infer']
        self.assertTrue(all(body == requests[0] for body in requests))
        self.assertEqual(len({chunk[0][0] for chunk in result['base_actions']}), 8)
        self.assertEqual(result['candidates_provenance']['sampling'], 'independent')
        self.assertEqual(result['candidates_provenance']['action_transform'],
                         'checkpoint_postprocess_and_source_gripper_clamp')

    def test_configured_17_step_prefix_preserves_full_candidate_and_binds_identity(self):
        actions = [[step / 100.] * 14 for step in range(50)]
        self.client.response_override = dict(actions=actions)
        runtime = Runtime(self.path, client=self.client, prefix_length=17)
        self.assertEqual(runtime.health()['chunk_size'], 17)
        result = runtime.infer(self.payload)
        self.assertEqual(result['chunk_size'], 17)
        self.assertEqual(result['actions'], actions[:17])
        self.assertEqual(result['selected_physical_chunk'], actions[:17])
        self.assertEqual(result['base_actions'], [actions])
        self.assertEqual(len(result['base_actions'][0]), 50)
        self.assertNotEqual(runtime.policy_bundle_id, self.runtime.policy_bundle_id)
        full = Runtime(self.path, client=self.client, prefix_length=50)
        self.assertEqual(full.infer(self.payload)['actions'], actions)

    def test_prefix_length_rejects_invalid_boundaries_before_any_backend_request(self):
        for length in (0, -1, 51, True, 17.0):
            with self.assertRaisesRegex(ValueError, 'Prefix length'):
                Runtime(self.path, client=self.client, prefix_length=length)
        self.assertEqual(self.client.calls, [])

    def test_changed_health_path_contract_or_mock_stops_before_inference(self):
        for changed in (dict(checkpoint='/changed/base'), dict(contract={}), dict(status='loading'),
                        dict(mock=True), dict(names=[])):
            with self.subTest(changed=changed):
                self.client.calls.clear()
                self.client.health_override = dict(self.client.health, **changed)
                with self.assertRaises(BackendChanged):
                    self.runtime.infer(self.payload)
                self.assertEqual([path for path, _ in self.client.calls], ['/health'])

    def test_health_is_rechecked_before_each_offline_candidate(self):
        original = self.client.request
        def request(path, body=None):
            result = original(path, body)
            if path == '/health' and self.client.counter == 1:
                result['checkpoint'] = '/restarted/other-model'
            return result
        self.client.request = request
        with self.assertRaises(BackendChanged):
            self.runtime.sample_candidates(self.payload)
        self.assertEqual(self.client.counter, 1)

    def test_malformed_response_and_source_limits_fail_closed(self):
        bad = [dict(checkpoint='/different'), dict(mock=True), dict(fps=10), dict(sequence=0),
               dict(actions=[[0.] * 14]), dict(actions=[[float('nan')] * 14 for _ in range(50)]),
               dict(actions=[[3.] * 14 for _ in range(50)])]
        for changed in bad:
            with self.subTest(changed=list(changed)):
                self.client.response_override = changed
                with self.assertRaises(BackendChanged):
                    self.runtime.infer(self.payload)

    def test_source_corrections_are_preserved_and_bound_to_returned_actions(self):
        actions = [[.1] * 14 for _ in range(50)]
        actions[2][6] = 0.
        correction = dict(action=3, index=6, raw_target_rad=-.2, command_target_rad=0.)
        self.client.response_override = dict(actions=actions, limit_corrections=[correction])
        result = self.runtime.infer(self.payload)
        self.assertEqual(result['limit_corrections'], [])
        self.assertEqual(result['candidate_limit_corrections'], [[correction]])
        longer = Runtime(self.path, client=self.client, prefix_length=17)
        self.assertEqual(longer.infer(self.payload)['limit_corrections'], [correction])
        for changed in (dict(command_target_rad=.1), dict(action=0), dict(index=0),
                        dict(raw_target_rad=float('nan')), dict(raw_target_rad=.5)):
            self.client.response_override['limit_corrections'] = [dict(correction, **changed)]
            with self.assertRaises(BackendChanged):
                self.runtime.infer(self.payload)

    def test_bad_observation_causes_no_backend_request(self):
        for changed in (dict(state=[0.] * 13), dict(state=[float('nan')] * 14),
                        dict(images={}), dict(task='another task')):
            with self.subTest(changed=list(changed)):
                with self.assertRaises(ValueError):
                    self.runtime.infer(dict(self.payload, **changed))
                self.assertEqual(self.client.calls, [])
        stream = io.BytesIO()
        Image.new('RGB', (256, 256)).save(stream, format='PNG')
        png = base64.b64encode(stream.getvalue()).decode()
        with self.assertRaisesRegex(ValueError, 'JPEG'):
            self.runtime.infer(dict(self.payload, images={camera: png for camera in CAMERAS}))
        self.assertEqual(self.client.calls, [])

    def test_manifest_change_and_synthetic_or_unverified_manifest_fail_closed(self):
        for changed in (dict(synthetic=True), dict(verified=False), dict(checkpoint_id='changed'),
                        dict(normalization_sha256='changed')):
            self.path.write_text(json.dumps(dict(self.manifest, **changed)))
            with self.assertRaises(ValueError):
                read_manifest(self.path)
        self.path.write_text(json.dumps(dict(self.manifest, checkpoint='/different')))
        with self.assertRaisesRegex(BackendChanged, 'manifest changed'):
            self.runtime.infer(self.payload)
        self.assertEqual(self.client.calls, [])

    def test_external_backends_and_busy_requests_are_rejected(self):
        for url in ('https://127.0.0.1:8765', 'http://example.com', 'http://127.0.0.1:8765/infer',
                    'http://user@127.0.0.1:8765'):
            with self.assertRaises(ValueError):
                BackendClient(url)
        self.runtime.lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'busy'):
                self.runtime.infer(self.payload)
        finally:
            self.runtime.lock.release()
        self.assertEqual(self.client.calls, [])


if __name__ == '__main__':
    unittest.main()
