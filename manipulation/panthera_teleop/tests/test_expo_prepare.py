"""Exporter fixtures exercise ledger joins; fixture candidates are never live data."""
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'app'))
from expo_recording import ExplorationRolloutStore
from infra.rl.contracts import CAMERAS, normalization_fingerprint
from infra.rl.replay import load_replay
from scripts.cache_expo_candidates import canonical_hash, exact_payload
from scripts.prepare_expo_replay import prepare, rows


class PrepareReplayTests(unittest.TestCase):
    def fixture(self, root, *, success=True, assisted=False, training_objective='gear_insertion'):
        normalization = dict(checkpoint_id='fixture-checkpoint', processor_id='fixture-processor',
            state={'q01': [-2.] * 14, 'q99': [2.] * 14},
            action={'q01': [-1.] * 14, 'q99': [1.] * 14})
        image = io.BytesIO()
        Image.fromarray(np.full((256, 256, 3), 128, np.uint8)).save(image, 'JPEG')
        images = {name: base64.b64encode(image.getvalue()).decode() for name in CAMERAS}
        store = ExplorationRolloutStore(root / 'ledger', dict(simulated=False,
            checkpoint_id=normalization['checkpoint_id'], processor_id=normalization['processor_id'],
            training_objective=training_objective))
        store.begin({'operator': 'fixture', 'scene': 'fixture'}, 1)
        store.requested(0, [0.] * 14, images, 10)
        response = dict(checkpoint_id=normalization['checkpoint_id'], processor_id=normalization['processor_id'])
        store.accepted(response, [[.1] * 14])
        store.command([.1] * 14, 15, .05)
        store.completed([.1] * 14, images, 20)
        if assisted:
            store.boundary('intervene')
        store.end(dict(success=success, complete=True, label='kept'))
        store.close()
        self.assertIsNone(store.failure)
        session = store.path
        decision = rows(session / 'decisions.jsonl')[0]
        cache = root / 'cache'
        cache.mkdir()
        entries = []
        for name in ('observation', 'next_observation'):
            obs = decision[name]
            entries.append(dict(session_id=session.name, observation_id=obs['observation_id'],
                observation_payload_sha256=canonical_hash(exact_payload(obs, session)),
                base_actions=np.full((8, 50, 14), .2).tolist(),
                candidates_provenance=dict(kind='pi05_inference', sampling='independent',
                    observation_id=obs['observation_id'], checkpoint_id=normalization['checkpoint_id'],
                    processor_id=normalization['processor_id']),
                checkpoint_id=normalization['checkpoint_id'], processor_id=normalization['processor_id'],
                normalizer_hash=normalization_fingerprint(normalization)))
        candidate_path = cache / 'candidates.jsonl'
        candidate_path.write_text(''.join(json.dumps(row) + '\n' for row in entries))
        (cache / 'cache_manifest.json').write_text(json.dumps(dict(kind='genuine_pi05_candidate_cache',
            synthetic=False, complete=True, checkpoint_id=normalization['checkpoint_id'],
            processor_id=normalization['processor_id'], normalization_sha256=normalization_fingerprint(normalization),
            candidates_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest())))
        return session, cache, normalization

    def test_complete_unassisted_terminal_reward_and_executed_action_export(self):
        for success in (True, False):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                session, cache, normalization = self.fixture(root, success=success)
                output = root / 'fixture-replay.npz'
                prepare(session, normalization, output, cache, C=1)
                arrays, metadata = load_replay(output)
                self.assertEqual(arrays['obs'].shape, (1, 2318))
                self.assertEqual(arrays['base_actions'].shape, (1, 8, 1, 14))
                np.testing.assert_allclose(arrays['actions'], .1, atol=1e-6)
                self.assertEqual(arrays['reward_return'][0], float(success))
                self.assertEqual(arrays['bootstrap_discount'][0], 0.)
                self.assertTrue(arrays['terminated'][0])
                self.assertEqual(len(metadata['sources']), 1)
                self.assertEqual(metadata['training_objective'], 'gear_insertion')
                self.assertEqual(metadata['transitions'][0]['training_objective'], 'gear_insertion')

    def test_assisted_episode_is_excluded_despite_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session, cache, normalization = self.fixture(root, assisted=True)
            with self.assertRaisesRegex(ValueError, 'Zero faithful transitions'):
                prepare(session, normalization, root / 'invalid.npz', cache, C=1)
            self.assertFalse((root / 'invalid.npz').exists())

    def test_missing_genuine_next_candidates_never_get_repeated_or_fabricated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session, _, normalization = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'Zero faithful transitions'):
                prepare(session, normalization, root / 'invalid.npz', C=1)

    def test_current_and_next_state_changes_under_same_id_reject_cached_candidates(self):
        for field in ('observation', 'next_observation'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                session, cache, normalization = self.fixture(root)
                decision = rows(session / 'decisions.jsonl')[0]
                decision[field]['state'][0] = .25
                (session / 'decisions.jsonl').write_text(json.dumps(decision) + '\n')
                with self.assertRaisesRegex(ValueError, 'Zero faithful transitions'):
                    prepare(session, normalization, root / 'invalid.npz', cache, C=1)

    def test_float64_emitted_proof_rejects_difference_hidden_by_float32_rounding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session, cache, normalization = self.fixture(root)
            decision = rows(session / 'decisions.jsonl')[0]
            decision['executed_actions'][0][6] = .8
            decision['last_emitted_target'][6] = .8 + 2e-8
            # The old float32-executed/float64-emitted comparison hid this
            # genuine mismatch below its tolerance after rounding 0.8 upward.
            self.assertLess(abs(float(np.float32(.8)) - (.8 + 2e-8)), 1e-8)
            (session / 'decisions.jsonl').write_text(json.dumps(decision) + '\n')
            with self.assertRaisesRegex(ValueError, 'Actual final emitted waypoint proof is missing'):
                prepare(session, normalization, root / 'invalid.npz', cache, C=1)

    def test_mixed_declared_session_objectives_are_rejected_before_any_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _, normalization = self.fixture(root / 'gear', training_objective='gear_insertion')
            second, _, _ = self.fixture(root / 'full', training_objective='gear_and_pin')
            mixed = root / 'mixed'
            mixed.mkdir()
            shutil.copytree(first, mixed / first.name)
            shutil.copytree(second, mixed / second.name)
            output = root / 'invalid.npz'
            with self.assertRaisesRegex(ValueError, 'Mixed training objectives'):
                prepare(mixed, normalization, output, C=1)
            self.assertFalse(output.exists())

    def test_episode_objective_must_match_session_and_requested_objective(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session, cache, normalization = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'Requested training objective differs'):
                prepare(session, normalization, root / 'invalid.npz', cache, C=1,
                        training_objective='gear_and_pin')
            endings = rows(session / 'episodes.jsonl')
            endings[-1]['training_objective'] = 'gear_and_pin'
            (session / 'episodes.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in endings))
            with self.assertRaisesRegex(ValueError, 'Episode ending training objective differs'):
                prepare(session, normalization, root / 'invalid.npz', cache, C=1)

    def test_legacy_missing_objective_requires_explicit_original_full_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session, cache, normalization = self.fixture(root)
            manifest_path = session / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            del manifest['training_objective']
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'explicit training_objective'):
                prepare(session, normalization, root / 'invalid.npz', cache, C=1)
            manifest['task'] = 'put the gear in the carrier and put a pin in it'
            manifest_path.write_text(json.dumps(manifest))
            for name in ('episodes.jsonl', 'decisions.jsonl'):
                values = rows(session / name)
                for value in values:
                    value.pop('training_objective', None)
                (session / name).write_text(''.join(json.dumps(e) + '\n' for e in values))
            output = root / 'legacy-test-only.npz'
            prepare(session, normalization, output, cache, C=1)
            _, metadata = load_replay(output)
            self.assertEqual(metadata['training_objective'], 'gear_and_pin')


if __name__ == '__main__':
    unittest.main()
