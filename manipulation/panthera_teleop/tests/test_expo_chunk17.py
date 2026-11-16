"""C=17 replay arithmetic and exporter boundaries; fixtures never contact hardware."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from infra.rl.contracts import ReplaySpec, normalization_fingerprint
from infra.rl.replay import validate_replay, load_replay
from scripts.prepare_expo_replay import prepare
from scripts.train_expo import parser
import test_expo_prepare as prepare_fixture


class Chunk17Tests(unittest.TestCase):
    def replay(self):
        spec = ReplaySpec(C=17, N=8, obs_dim=8, checkpoint_id='fixed', processor_id='saved')
        rewards = np.zeros((2, 17), np.float32)
        rewards[-1, -1] = 1.
        arrays = dict(obs=np.zeros((2, 8), np.float32), next_obs=np.ones((2, 8), np.float32),
            actions=np.zeros((2, 17, 14), np.float32),
            base_actions=np.zeros((2, 8, 17, 14), np.float32),
            next_base_actions=np.ones((2, 8, 17, 14), np.float32), rewards=rewards,
            reward_return=np.array([0., .99 ** 16], np.float32),
            bootstrap_discount=np.array([.99 ** 17, 0.], np.float32),
            terminated=np.array([False, True]), source_episode_ids=np.array(['trial', 'trial']))
        normalization = dict(checkpoint_id='fixed', processor_id='saved',
            action=dict(q01=[-1.] * 14, q99=[1.] * 14),
            state=dict(q01=[-1.] * 14, q99=[1.] * 14))
        row = dict(source_episode_id='trial', complete=True, simulation=False,
            control_source='policy', contiguous_policy=True, task_success=True,
            intervened=False, human_interventions=0, source_category='kept',
            action_provenance='executed_policy_waypoints', candidate_provenance='pi05_inference',
            checkpoint_id='fixed', processor_id='saved', training_objective='gear_insertion')
        metadata = dict(schema_version=1, kind='faithful_policy_replay', training_eligible=True,
            synthetic=False, training_objective='gear_insertion', spec=spec.as_dict(),
            normalization=normalization, normalization_sha256=normalization_fingerprint(normalization),
            transitions=[copy.deepcopy(row), copy.deepcopy(row)])
        return arrays, metadata

    def test_seventeen_waypoint_return_discount_and_terminal_credit(self):
        arrays, metadata = self.replay()
        self.assertEqual(validate_replay(arrays, metadata).C, 17)
        # This catches substituting C=1 or putting success before waypoint 17.
        for field, value in [('bootstrap_discount', .99), ('reward_return', 1.)]:
            wrong = copy.deepcopy(arrays)
            wrong[field][0 if field == 'bootstrap_discount' else 1] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_replay(wrong, metadata)
        wrong = copy.deepcopy(arrays)
        wrong['rewards'][-1, -1] = 0.
        wrong['rewards'][-1, 15] = 1.
        wrong['reward_return'][-1] = .99 ** 15
        with self.assertRaisesRegex(ValueError, 'final executed terminal waypoint'):
            validate_replay(wrong, metadata)

    def test_every_action_and_candidate_array_must_have_exact_same_chunk(self):
        arrays, metadata = self.replay()
        for field in ('actions', 'base_actions', 'next_base_actions'):
            wrong = copy.deepcopy(arrays)
            wrong[field] = wrong[field][..., :16, :]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_replay(wrong, metadata)

    def fixture17(self, root):
        session, cache, normalization = prepare_fixture.PrepareReplayTests().fixture(root)
        manifest_path = session / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['fixed_prefix_length'] = 17
        manifest_path.write_text(json.dumps(manifest))
        decision_path = session / 'decisions.jsonl'
        decision = json.loads(decision_path.read_text())
        executed = np.arange(17 * 14).reshape(17, 14) / 1000.
        decision.update(actual_prefix_length=17, executed_actions=executed.tolist(),
                        last_emitted_target=executed[-1].tolist(), emitted_command_count=17,
                        last_emitted_policy_time_s=17 / 20)
        decision_path.write_text(json.dumps(decision) + '\n')
        candidates = np.arange(8 * 50 * 14).reshape(8, 50, 14) / 10000.
        candidate_path = cache / 'candidates.jsonl'
        entries = [json.loads(line) for line in candidate_path.read_text().splitlines()]
        for row in entries:
            row['base_actions'] = candidates.tolist()
        candidate_path.write_text(''.join(json.dumps(row) + '\n' for row in entries))
        cache_manifest_path = cache / 'cache_manifest.json'
        cache_manifest = json.loads(cache_manifest_path.read_text())
        cache_manifest['candidates_sha256'] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        cache_manifest_path.write_text(json.dumps(cache_manifest))
        return session, cache, normalization, executed, candidates

    def test_default_export_slices_genuine_full50_proposals_to_first17(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session, cache, normalization, executed, candidates = self.fixture17(root)
            output = root / 'replay.npz'
            prepare(session, normalization, output, cache)
            arrays, metadata = load_replay(output)
            self.assertEqual(metadata['spec']['C'], 17)
            np.testing.assert_allclose(arrays['actions'][0], executed, atol=1e-6)
            for key in ('base_actions', 'next_base_actions'):
                self.assertEqual(arrays[key].shape, (1, 8, 17, 14))
                np.testing.assert_allclose(arrays[key][0], candidates[:, :17], atol=1e-6)
            self.assertAlmostEqual(float(arrays['reward_return'][0]), .99 ** 16, places=6)
            self.assertEqual(float(arrays['bootstrap_discount'][0]), 0.)

    def test_old_one_waypoint_and_mixed_session_chunks_need_explicit_separate_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, _, normalization = prepare_fixture.PrepareReplayTests().fixture(root / 'old')
            new, _, _, _, _ = self.fixture17(root / 'new')
            for ledger in (old, root / 'mixed'):
                if ledger.name == 'mixed':
                    ledger.mkdir()
                    shutil.copytree(old, ledger / old.name)
                    shutil.copytree(new, ledger / new.name)
                with self.subTest(ledger=ledger.name), self.assertRaisesRegex(ValueError, 'chunk lengths cannot mix'):
                    prepare(ledger, normalization, root / (ledger.name + '.npz'))
                self.assertFalse((root / (ledger.name + '.npz')).exists())

    def test_partial17_execution_cannot_be_rewarded_or_repeated_to_fill_chunk(self):
        for change in ('short_actions', 'short_command_count', 'short_actual_prefix',
                       'short_policy_duration', 'missing_policy_duration', 'forged_high_fps'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                session, cache, normalization, _, _ = self.fixture17(root)
                path = session / 'decisions.jsonl'
                row = json.loads(path.read_text())
                if change == 'short_actions':
                    row['executed_actions'] = row['executed_actions'][:16]
                elif change == 'short_command_count':
                    row['emitted_command_count'] = 16
                elif change == 'short_actual_prefix':
                    row['actual_prefix_length'] = 16
                elif change == 'short_policy_duration':
                    row['last_emitted_policy_time_s'] = 16 / 20
                elif change == 'missing_policy_duration':
                    row.pop('last_emitted_policy_time_s')
                else:
                    manifest_path = session / 'manifest.json'
                    manifest = json.loads(manifest_path.read_text())
                    manifest.setdefault('contract', {})['fps'] = 100
                    manifest_path.write_text(json.dumps(manifest))
                    row['last_emitted_policy_time_s'] = 16 / 20
                path.write_text(json.dumps(row) + '\n')
                output = root / 'invalid.npz'
                with self.assertRaisesRegex(ValueError, 'Zero faithful transitions'):
                    prepare(session, normalization, output, cache)
                self.assertFalse(output.exists())

    def test_trainer_defaults_to17_and_old_one_waypoint_requires_explicit_flag(self):
        common = ['--replay', 'saved.npz', '--output', 'unused']
        self.assertEqual(parser().parse_args(common).prefix_length, 17)
        self.assertEqual(parser().parse_args(common + ['--prefix-length', '1']).prefix_length, 1)


if __name__ == '__main__':
    unittest.main()
