"""Native-action, image and source-boundary contracts without hardware."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infra.rl.contracts import ActionNormalizer, ReplaySpec, normalization_fingerprint
from infra.rl.observations import ObservationEncoder
from infra.rl.replay import (grouped_split, validate_replay, save_replay, load_replay,
                            training_objective_from_metadata)


class ActionContractTests(unittest.TestCase):
    def normalizer(self):
        low = np.arange(14, dtype=np.float32) - 20
        high = low + np.arange(1, 15, dtype=np.float32)
        return ActionNormalizer(low, high), low, high

    def test_checkpoint_quantiles_map_endpoints_and_roundtrip_without_clipping(self):
        normalizer, low, high = self.normalizer()
        # Deliberately asymmetric by dimension to detect swapped arms/grippers.
        fractions = np.linspace(-.4, 1.4, 14, dtype=np.float32)
        values = low + fractions * (high - low)
        batch = np.stack([low, high, values]).reshape(1, 3, 14)
        untouched = batch.copy()
        normalized = normalizer.normalize(batch)
        np.testing.assert_allclose(normalized[0, 0], -1, atol=1e-6)
        np.testing.assert_allclose(normalized[0, 1], 1, atol=1e-6)
        np.testing.assert_allclose(normalized[0, 2], 2 * fractions - 1, atol=3e-6)
        np.testing.assert_allclose(normalizer.denormalize(normalized), batch, atol=3e-6)
        np.testing.assert_array_equal(batch, untouched)

    def test_missing_invalid_or_degenerate_checkpoint_statistics_rejected(self):
        for low, high in [([0.] * 13, [1.] * 13), ([0.] * 14, [0.] * 14),
                          ([1.] * 14, [0.] * 14), ([float('nan')] * 14, [1.] * 14),
                          ([0.] * 14, [float('inf')] * 14)]:
            with self.subTest(low=low[0], dimensions=len(low)):
                with self.assertRaises((ValueError, TypeError)):
                    ActionNormalizer(low, high)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            ActionNormalizer.from_stats({})

    def test_nonfinite_or_wrong_dimension_actions_rejected(self):
        normalizer, _, _ = self.normalizer()
        for values in (np.zeros(13), np.full(14, np.nan), np.full(14, np.inf)):
            with self.assertRaises(ValueError):
                normalizer.normalize(values)
            with self.assertRaises(ValueError):
                normalizer.denormalize(values)


class ObservationContractTests(unittest.TestCase):
    def encoder(self):
        return ObservationEncoder(ActionNormalizer(np.full(14, -1.), np.full(14, 1.)))

    def observation(self):
        return {'state': np.linspace(-.8, .8, 14), 'images': {
            'wrist_port3': np.full((256, 256, 3), 255, dtype=np.uint8),
            'overhead': np.zeros((256, 256, 3), dtype=np.uint8),
            'wrist_port2': np.full((256, 256, 3), 128, dtype=np.uint8)}}

    def test_camera_order_is_canonical_and_independent_of_dictionary_order(self):
        obs = self.observation()
        features = self.encoder().encode(obs)
        self.assertEqual(features.shape, (2318,))
        np.testing.assert_allclose(features[:14], obs['state'], atol=1e-6)
        camera_features = features[14:].reshape(3, -1)
        np.testing.assert_allclose(camera_features[0], 0, atol=1e-6)
        np.testing.assert_allclose(camera_features[1], 128 / 255, atol=1e-6)
        np.testing.assert_allclose(camera_features[2], 1, atol=1e-6)
        self.assertTrue(np.isfinite(features).all())

    def test_missing_extra_wrong_size_and_non_rgb_views_rejected(self):
        encoder = self.encoder()
        for change in ('missing', 'extra', 'size', 'channels'):
            obs = self.observation()
            if change == 'missing':
                del obs['images']['overhead']
            elif change == 'extra':
                obs['images']['unverified_camera'] = obs['images']['overhead']
            elif change == 'size':
                obs['images']['overhead'] = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                obs['images']['overhead'] = np.zeros((256, 256), dtype=np.uint8)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                encoder.encode(obs)


class SourceSplitTests(unittest.TestCase):
    def test_corrections_from_same_original_episode_never_leak_between_splits(self):
        sources = ['original-a'] * 3 + ['original-b'] * 2 + ['original-c'] * 4 + ['original-d']
        train, evaluation = grouped_split(sources, eval_fraction=.4, seed=12)
        self.assertTrue(len(train))
        self.assertTrue(len(evaluation))
        self.assertFalse({sources[i] for i in train} & {sources[i] for i in evaluation})
        self.assertEqual(sorted(list(train) + list(evaluation)), list(range(len(sources))))
        train2, evaluation2 = grouped_split(sources, eval_fraction=.4, seed=12)
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(evaluation, evaluation2)


class ReplayProvenanceTests(unittest.TestCase):
    def fixture(self):
        spec = ReplaySpec(gamma=.5, C=3, N=2, obs_dim=8,
                          checkpoint_id='immutable-model-sha', processor_id='processor-sha')
        arrays = {'obs': np.zeros((3, 8), np.float32),
                  'next_obs': np.ones((3, 8), np.float32),
                  'actions': np.zeros((3, 3, 14), np.float32),
                  'base_actions': np.zeros((3, 2, 3, 14), np.float32),
                  'next_base_actions': np.ones((3, 2, 3, 14), np.float32),
                  'rewards': np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]], np.float32),
                  'reward_return': np.array([0., 0., .25], np.float32),
                  'bootstrap_discount': np.array([.125, 0., 0.], np.float32),
                  'terminated': np.array([False, True, True]),
                  'source_episode_ids': np.array(['failed-policy', 'failed-policy', 'successful-policy'])}
        normalization = {
            'action': {'q01': [-1.] * 14, 'q99': [1.] * 14},
            'state': {'q01': [-2.] * 14, 'q99': [2.] * 14},
            'checkpoint_id': spec.checkpoint_id, 'processor_id': spec.processor_id}
        transitions = [dict(source_episode_id=str(source), complete=True, simulation=False,
                            control_source='policy', contiguous_policy=True, task_success=i == 2,
                            intervened=False, human_interventions=0,
                            action_provenance='executed_policy_waypoints',
                            candidate_provenance='pi05_inference', checkpoint_id=spec.checkpoint_id,
                            processor_id=spec.processor_id, source_category='kept',
                            review_approved=False, training_objective='gear_insertion')
                       for i, source in enumerate(arrays['source_episode_ids'])]
        metadata = {'schema_version': 1, 'spec': spec.as_dict(), 'normalization': normalization,
                    'normalization_sha256': normalization_fingerprint(normalization),
                    'kind': 'faithful_policy_replay', 'training_eligible': True, 'synthetic': False,
                    'training_objective': 'gear_insertion', 'transitions': transitions}
        return spec, arrays, metadata

    def test_real_failures_remain_eligible_and_portable_replay_roundtrips(self):
        spec, arrays, metadata = self.fixture()
        self.assertEqual(validate_replay(arrays, metadata, spec), spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'faithful.npz'
            save_replay(path, arrays, metadata)
            restored, meta = load_replay(path, spec)
            self.assertEqual(meta, metadata)
            for key in arrays:
                np.testing.assert_array_equal(restored[key], arrays[key])
            with self.assertRaises(FileExistsError):
                save_replay(path, arrays, metadata)
            changed = ReplaySpec(gamma=.5, C=3, N=2, obs_dim=8,
                                 checkpoint_id='different-model', processor_id=spec.processor_id)
            with self.assertRaises(ValueError):
                load_replay(path, changed)

    def test_fake_candidates_measured_pose_proxy_and_untrusted_sources_rejected(self):
        _, arrays, original = self.fixture()
        changes = [('candidate_provenance', 'measured_action_repeat'),
                   ('action_provenance', 'next_measured_pose'), ('complete', False),
                   ('simulation', True), ('control_source', 'human'),
                   ('contiguous_policy', False), ('task_success', None),
                   ('intervened', True), ('human_interventions', 1),
                   ('checkpoint_id', 'different'), ('processor_id', 'different'),
                   ('source_category', 'discarded')]
        for field, value in changes:
            metadata = copy.deepcopy(original)
            metadata['transitions'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_replay(arrays, metadata)
        metadata = copy.deepcopy(original)
        metadata['transitions'][0]['source_category'] = 'review'
        with self.assertRaises(ValueError):
            validate_replay(arrays, metadata)

    def test_padding_and_terminal_bootstrap_cannot_change_validated_chunk_return(self):
        _, original, metadata = self.fixture()
        for field, index, value in [('reward_return', 2, 1.),
                                    ('bootstrap_discount', 1, .125),
                                    ('bootstrap_discount', 0, .5)]:
            arrays = copy.deepcopy(original)
            arrays[field][index] = value
            with self.subTest(field=field, index=index), self.assertRaises(ValueError):
                validate_replay(arrays, metadata)

    def test_sparse_success_credit_cannot_move_into_middle_or_failed_episode(self):
        _, original, metadata = self.fixture()
        for index, rewards in [(2, [0., 1., 0.]), (1, [0., 0., 1.])]:
            arrays = copy.deepcopy(original)
            arrays['rewards'][index] = rewards
            arrays['reward_return'][index] = np.dot(rewards, [1., .5, .25])
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_replay(arrays, metadata)

    def test_missing_or_duplicate_terminal_boundary_is_rejected(self):
        spec, original, metadata = self.fixture()
        for index, terminated in [(0, True), (1, False)]:
            arrays = copy.deepcopy(original)
            arrays['terminated'][index] = terminated
            arrays['bootstrap_discount'][index] = 0. if terminated else spec.gamma ** spec.C
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_replay(arrays, metadata)

    def test_synthetic_audit_artifact_cannot_be_loaded_as_training_replay(self):
        _, arrays, original = self.fixture()
        for key, value in [('kind', 'audit_only'), ('training_eligible', False), ('synthetic', True)]:
            metadata = copy.deepcopy(original)
            metadata[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_replay(arrays, metadata)

    def test_changed_quantiles_and_source_grouping_are_rejected(self):
        _, arrays, original = self.fixture()
        metadata = copy.deepcopy(original)
        metadata['normalization']['action']['q01'][6] = -.5
        with self.assertRaises(ValueError):
            validate_replay(arrays, metadata)
        metadata = copy.deepcopy(original)
        metadata['transitions'][0]['source_episode_id'] = 'wrong-source'
        with self.assertRaises(ValueError):
            validate_replay(arrays, metadata)

    def test_mixed_unknown_or_missing_objectives_cannot_enter_one_replay(self):
        _, arrays, original = self.fixture()
        changes = [('training_objective', 'gear_and_pin'),
                   ('training_objective', 'undefined_task'), ('training_objective', None)]
        for field, value in changes:
            metadata = copy.deepcopy(original)
            metadata['transitions'][0][field] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_replay(arrays, metadata)
        for value in ('undefined_task', None):
            metadata = copy.deepcopy(original)
            metadata['training_objective'] = value
            with self.subTest(metadata_value=value), self.assertRaises(ValueError):
                validate_replay(arrays, metadata)

    def test_only_explicit_old_full_task_can_resolve_missing_legacy_objective(self):
        spec, arrays, original = self.fixture()
        old_task = 'put the gear in the carrier and put a pin in it'
        for declaration in ({'task': old_task}, {'contract': {'task': old_task}}):
            metadata = copy.deepcopy(original)
            del metadata['training_objective']
            for row in metadata['transitions']:
                del row['training_objective']
            metadata.update(declaration)
            self.assertEqual(training_objective_from_metadata(metadata), 'gear_and_pin')
            self.assertEqual(validate_replay(arrays, metadata), spec)
        for task in (None, 'put the gear in the carrier', old_task + ' '):
            metadata = copy.deepcopy(original)
            del metadata['training_objective']
            metadata['task'] = task
            with self.subTest(task=task), self.assertRaises(ValueError):
                validate_replay(arrays, metadata)


if __name__ == '__main__':
    unittest.main()
