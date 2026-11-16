"""Independent CPU arithmetic/gradient checks for the EXPO adaptation.

Fixtures deliberately use tiny features, never a VLA checkpoint or robot SDK.
"""
from pathlib import Path
from dataclasses import replace
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest('EXPO numerical tests require the PyTorch environment') from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infra.rl.learner import (EXPOConfig, EXPOLearner, chunk_bellman_target,
                             polyak_update, select_candidate_actions)
from infra.rl.models import scaled_tanh_log_prob, SquashedGaussianEditor


class NumericalAlgorithmTests(unittest.TestCase):
    def test_chunk_bootstrap_matches_hand_calculation_and_has_no_entropy_term(self):
        # gamma=.5, C=3, rewards 1,2,4 => return 1+1+1=3.
        target = chunk_bellman_target(torch.tensor([3., 3.]),
                                     torch.tensor([.5**3, 0.]), torch.tensor([8., 8.]))
        torch.testing.assert_close(target, torch.tensor([4., 3.]))

    def test_polyak_update_uses_source_fraction_and_leaves_online_weights_intact(self):
        source, target = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
        with torch.no_grad():
            source.weight.fill_(10.)
            source.bias.fill_(4.)
            target.weight.fill_(2.)
            target.bias.fill_(-4.)
        polyak_update(source, target, .25)
        torch.testing.assert_close(target.weight, torch.full_like(target.weight, 4.))
        torch.testing.assert_close(target.bias, torch.full_like(target.bias, -2.))
        torch.testing.assert_close(source.weight, torch.full_like(source.weight, 10.))
        polyak_update(source, target, 0.)
        torch.testing.assert_close(target.weight, torch.full_like(target.weight, 4.))
        polyak_update(source, target, 1.)
        torch.testing.assert_close(target.weight, source.weight)

    def test_candidate_argmax_is_per_observation_and_includes_unedited_candidates(self):
        candidates = torch.arange(2 * 4 * 2 * 14, dtype=torch.float32).reshape(2, 4, 2, 14)
        scores = torch.tensor([[9., 1., 2., 3.], [0., 1., 8., 2.]])
        result = select_candidate_actions(candidates, scores)
        torch.testing.assert_close(result['selected'], torch.tensor([0, 2]))
        torch.testing.assert_close(result['actions'], torch.stack([candidates[0, 0], candidates[1, 2]]))

    def test_nonfinite_candidate_scores_cannot_win_selection(self):
        candidates = torch.zeros(1, 2, 2, 14)
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_candidate_actions(candidates, torch.tensor([[0., value]]))

    def test_edit_density_matches_analytic_gaussian_and_scale_jacobian(self):
        # At zero, tanh derivative is one. This measures all C*14 coordinates.
        dims = 28
        zero = torch.zeros(2, dims, dtype=torch.float64)
        density = scaled_tanh_log_prob(zero, zero, zero, .05)
        expected = -dims * (.5 * math.log(2 * math.pi) + math.log(.05))
        torch.testing.assert_close(density, torch.full((2,), expected, dtype=torch.float64),
                                   atol=2e-5, rtol=1e-6)
        wider = scaled_tanh_log_prob(zero, zero, zero, .1)
        torch.testing.assert_close(density - wider,
                                   torch.full((2,), dims * math.log(2), dtype=torch.float64))

    def test_edit_density_is_stable_and_differentiable_at_tanh_saturation(self):
        pre_tanh = torch.tensor([[20., -20., 0.]], requires_grad=True)
        mean = torch.zeros_like(pre_tanh, requires_grad=True)
        log_std = torch.zeros_like(pre_tanh, requires_grad=True)
        density = scaled_tanh_log_prob(pre_tanh, mean, log_std, .05)
        self.assertTrue(torch.isfinite(density).all())
        density.sum().backward()
        for gradient in (pre_tanh.grad, mean.grad, log_std.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_editor_is_bounded_for_all_native_joint_coordinates_and_gets_gradient(self):
        torch.manual_seed(80)
        editor = SquashedGaussianEditor(8, 28, hidden_dim=16, residual_bound=.05)
        edit, density = editor.sample(torch.zeros(4, 8), torch.zeros(4, 28))
        self.assertEqual(tuple(edit.shape), (4, 28))
        self.assertEqual(tuple(density.shape), (4,))
        self.assertLessEqual(float(edit.detach().abs().max()), .050001)
        # DROID xyzg masks must not suppress Panthera joint4..6.
        self.assertTrue(torch.any(edit.detach()[:, [3, 4, 5, 10, 11, 12]] != 0))
        (-edit.sum() + .1 * density.sum()).backward()
        gradients = [p.grad for p in editor.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)


class LearnerIntegrationTests(unittest.TestCase):
    def make_learner(self):
        torch.manual_seed(901)
        return EXPOLearner(EXPOConfig(
            obs_dim=8, action_dim=14, chunk_size=2, hidden_dim=16,
            num_critics=3, target_subset_size=2, residual_bound=.05,
            utd_ratio=1), device='cpu')

    def batch(self):
        generator = torch.Generator().manual_seed(50)
        return {
            'obs': torch.randn(4, 8, generator=generator),
            'next_obs': torch.randn(4, 8, generator=generator),
            'actions': torch.randn(4, 2, 14, generator=generator) * .1,
            'base_actions': torch.randn(4, 2, 2, 14, generator=generator) * .1,
            'next_base_actions': torch.randn(4, 2, 2, 14, generator=generator) * .1,
            'reward_return': torch.tensor([0., 0., 1., 0.]),
            'bootstrap_discount': torch.tensor([.99**2, .99**2, 0., .99**2]),
            'transition_valid': torch.ones(4, dtype=torch.bool),
            'editor_train_valid': torch.ones(4, dtype=torch.bool),
        }

    def test_tiny_cpu_update_is_finite_and_changes_trainable_state(self):
        learner = self.make_learner()
        before = {key: value.detach().clone() for key, value in learner.critic.state_dict().items()}
        metrics = learner.update(self.batch())
        self.assertTrue(metrics)
        for key, value in metrics.items():
            if isinstance(value, (float, int, torch.Tensor)):
                self.assertTrue(torch.isfinite(torch.as_tensor(value)).all(), key)
        self.assertTrue(any(not torch.equal(before[key], value)
                            for key, value in learner.critic.state_dict().items()))

    def test_candidate_api_keeps_native_action_shape_and_edit_bound(self):
        learner = self.make_learner()
        base = torch.arange(4 * 3 * 2 * 14, dtype=torch.float32).reshape(4, 3, 2, 14) / 1000
        result = learner.select_actions(torch.zeros(4, 8), base, deterministic=True)
        self.assertEqual(tuple(result['actions'].shape), (4, 2, 14))
        self.assertEqual(tuple(result['scores'].shape), (4, 6))
        self.assertEqual(tuple(result['selected'].shape), (4,))
        for row, selected in enumerate(result['selected']):
            index = int(selected)
            self.assertGreaterEqual(index, 0)
            self.assertLess(index, 6)
            delta = result['actions'][row] - base[row, index % 3]
            if index < 3:
                torch.testing.assert_close(delta, torch.zeros_like(delta))
            else:
                self.assertLessEqual(float(delta.abs().max()), .050001)

    def test_incorrect_chunk_dimensions_are_rejected(self):
        learner = self.make_learner()
        with self.assertRaises((ValueError, RuntimeError)):
            learner.select_actions(torch.zeros(4, 8), torch.zeros(4, 3, 2, 13))

    def test_invalid_batch_skips_network_target_temperature_rng_and_counter_updates(self):
        learner = self.make_learner()
        before = {name: {key: value.detach().clone() for key, value in getattr(learner, name).state_dict().items()}
                  for name in ('critic', 'target_critic', 'editor')}
        temperature = learner.log_temperature.detach().clone()
        rng_state = learner.rng.get_state().clone()
        batch = self.batch()
        batch['transition_valid'].fill_(False)
        learner.update(batch)
        for name, state in before.items():
            for key, value in getattr(learner, name).state_dict().items():
                torch.testing.assert_close(value, state[key], atol=0, rtol=0)
        torch.testing.assert_close(learner.log_temperature, temperature, atol=0, rtol=0)
        torch.testing.assert_close(learner.rng.get_state(), rng_state, atol=0, rtol=0)
        self.assertEqual((learner.updates, learner.critic_updates, learner.editor_updates), (0, 0, 0))

    def test_failed_editor_eligibility_leaves_editor_and_temperature_unchanged(self):
        learner = self.make_learner()
        before = {key: value.detach().clone() for key, value in learner.editor.state_dict().items()}
        temperature = learner.log_temperature.detach().clone()
        batch = self.batch()
        batch['editor_train_valid'].fill_(False)
        learner.update(batch)
        for key, value in learner.editor.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        torch.testing.assert_close(learner.log_temperature, temperature, atol=0, rtol=0)
        self.assertEqual(learner.critic_updates, 1)
        self.assertEqual(learner.editor_updates, 0)

    def test_bellman_and_edit_updates_do_not_backpropagate_to_replay_or_target_parameters(self):
        learner = self.make_learner()
        batch = self.batch()
        for key in ('obs', 'next_obs', 'actions', 'base_actions', 'next_base_actions'):
            batch[key].requires_grad_(True)
        learner.update(batch)
        self.assertTrue(all(value.grad is None for value in batch.values() if value.requires_grad))
        self.assertTrue(all(p.grad is None for p in learner.target_critic.parameters()))
        self.assertTrue(all(p.grad is None for p in learner.critic.parameters()))
        self.assertTrue(all(p.requires_grad for p in learner.critic.parameters()))

    def test_resume_preserves_optimizer_targets_and_stochastic_continuation(self):
        learner = self.make_learner()
        batch = self.batch()
        learner.update(batch)
        learner.checkpoint_metadata = {
            'training_objective': 'gear_insertion',
            'reward_semantics': 'Fully seated gear = 1; failure = 0; terminal waypoint only',
            'checkpoint_id': 'frozen-base-fixture',
            'processor_id': 'frozen-processor-fixture',
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = learner.save(Path(tmp) / 'learner.pt')
            restored = EXPOLearner.load(path, device='cpu')
            self.assertEqual(restored.checkpoint_metadata, learner.checkpoint_metadata)
            first = learner.select_actions(batch['obs'], batch['base_actions'])
            second = restored.select_actions(batch['obs'], batch['base_actions'])
            torch.testing.assert_close(first['actions'], second['actions'], atol=0, rtol=0)
            torch.testing.assert_close(first['scores'], second['scores'], atol=0, rtol=0)
            metrics1, metrics2 = learner.update(batch), restored.update(batch)
            self.assertEqual(metrics1, metrics2)
            for name in ('critic', 'target_critic', 'editor'):
                for key, value in getattr(learner, name).state_dict().items():
                    torch.testing.assert_close(value, getattr(restored, name).state_dict()[key], atol=0, rtol=0)

    def test_validity_intersection_does_not_mutate_callers_replay_mask(self):
        learner = self.make_learner()
        batch = self.batch()
        batch['transition_valid'][0] = False
        untouched = batch['editor_train_valid'].clone()
        learner.update(batch)
        torch.testing.assert_close(batch['editor_train_valid'], untouched, atol=0, rtol=0)

    def bounded_learner(self):
        cfg = self.make_learner().config
        lower, upper = [None] * 14, [None] * 14
        for index in (6, 13):
            lower[index], upper[index] = .1, .3
        return EXPOLearner(replace(cfg, action_lower=tuple(lower), action_upper=tuple(upper)))

    def test_physical_projection_happens_before_ranking_and_survives_reload(self):
        learner = self.bounded_learner()
        base = torch.zeros(1, 2, 2, 14)
        base[0, 0, :, 6] = 2.
        base[0, 1, :, 6] = .2
        base[0, 1, :, 0] = .5

        def q(obs, actions, indices=None):
            score = actions[:, 0] + actions[:, 6]
            return score.unsqueeze(0).repeat(len(indices), 1)

        def zero_editor(obs, actions, **kwargs):
            return torch.zeros_like(actions), torch.zeros(len(obs))

        with patch.object(learner.target_critic, 'forward', side_effect=q), \
                patch.object(learner.editor, 'sample', side_effect=zero_editor):
            result = learner.select_actions(torch.zeros(1, 8), base, deterministic=True)
        self.assertEqual(int(result['selected'][0]), 1)
        torch.testing.assert_close(result['scores'], torch.tensor([[.3, .7, .3, .7]]))
        self.assertEqual(float(result['actions'][0, 0, 0]), .5)
        with tempfile.TemporaryDirectory() as tmp:
            restored = EXPOLearner.load(learner.save(Path(tmp) / 'bounded.pt'))
            self.assertEqual(restored.config.action_lower, learner.config.action_lower)
            self.assertEqual(restored.config.action_upper, learner.config.action_upper)
            actions = restored.select_actions(torch.zeros(1, 8), base, deterministic=True)['actions']
            self.assertTrue((actions[:, :, [6, 13]] >= .1).all())
            self.assertTrue((actions[:, :, [6, 13]] <= .3).all())

    def test_td_and_editor_critics_both_receive_projected_gripper_actions(self):
        learner = self.bounded_learner()
        batch = self.batch()
        batch['actions'][:, :, [6, 13]] = .2
        batch['base_actions'][:, :, :, [6, 13]] = 2.
        batch['next_base_actions'][:, :, :, [6, 13]] = -2.
        seen = []

        def check_projection(module, args):
            action = args[1].reshape(-1, 2, 14)
            grippers = action[:, :, [6, 13]]
            self.assertTrue((grippers >= .1).all())
            self.assertTrue((grippers <= .3).all())
            seen.append((module is learner.target_critic, args[1].requires_grad))

        handles = [module.register_forward_pre_hook(check_projection)
                   for module in (learner.critic, learner.target_critic)]
        try:
            learner.update(batch)
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(any(target for target, _ in seen))
        self.assertTrue(any(not target and gradients for target, gradients in seen))


if __name__ == '__main__':
    unittest.main()
