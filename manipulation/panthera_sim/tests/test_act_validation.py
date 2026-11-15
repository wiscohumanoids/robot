import unittest
from types import SimpleNamespace

import torch

from train_act import evaluate_inference, rollout_score, training_loss


class InferenceValidationTest(unittest.TestCase):
    def test_zero_objective_matches_deployment_and_backpropagates(self):
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        cfg = ACTConfig(
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (7,)),
                            "observation.environment_state": PolicyFeature(FeatureType.ENV, (3,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
            chunk_size=3, n_action_steps=3, dim_model=32, n_heads=4,
            dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
            n_vae_encoder_layers=1, dropout=0., device="cpu",
        )
        policy = ACTPolicy(cfg)
        batch = {"observation.state": torch.randn(2, 7),
                 "observation.environment_state": torch.randn(2, 3),
                 "action": torch.randn(2, 3, 7),
                 "action_is_pad": torch.tensor([[False, False, True], [False, True, True]])}
        prediction = policy.predict_action_chunk({key: batch[key] for key in cfg.input_features})
        expected = ((prediction - batch["action"]).abs() *
                    (~batch["action_is_pad"]).unsqueeze(-1)).mean()
        loss, _ = training_loss(policy, batch, "zero")
        torch.testing.assert_close(loss.detach(), expected)
        changed_padding = dict(batch, action=batch["action"].clone())
        changed_padding["action"][batch["action_is_pad"]] = 999.
        padded_loss, _ = training_loss(policy, changed_padding, "zero")
        torch.testing.assert_close(loss, padded_loss)
        loss.backward()
        self.assertGreater(float(policy.model.action_head.weight.grad.abs().sum()), 0.)
        self.assertIsNone(policy.model.vae_encoder_action_input_proj.weight.grad)
        vae_loss, _ = training_loss(policy, batch, "vae")
        vae_loss.backward()
        self.assertTrue(policy.training)
        self.assertIsNotNone(policy.model.vae_encoder_action_input_proj.weight.grad)

    def test_rollout_selection_prioritizes_completed_task(self):
        complete = dict(success=.125, two_stacked=.125, lifted=.125, grasped=.125)
        grasp_only = dict(success=0., two_stacked=0., lifted=1., grasped=1.)
        two_stack = dict(success=0., two_stacked=.25, lifted=.5, grasped=.5)
        self.assertGreater(rollout_score(complete), rollout_score(two_stack))
        self.assertGreater(rollout_score(two_stack), rollout_score(grasp_only))
        bumped = dict(success=0., two_stacked=0., lifted=1., grasped=0., grasp_and_lift=0.)
        self.assertGreater(rollout_score(grasp_only), rollout_score(bumped))

    def test_validation_omits_targets_and_excludes_padding(self):
        class Policy:
            config = SimpleNamespace(input_features={'observation.state': None})
            def eval(self):
                self.training = False
            def train(self):
                self.training = True
            def predict_action_chunk(self, batch):
                assert set(batch) == {'observation.state'}
                assert not self.training
                return torch.zeros(1, 3, 2)
        policy = Policy()
        batch = {'observation.state': torch.ones(1, 2),
                 'action': torch.tensor([[[1., 3.], [2., 6.], [999., 999.]]]),
                 'action_is_pad': torch.tensor([[False, False, True]])}
        metrics = evaluate_inference(policy, [batch], lambda x: x)
        self.assertEqual(metrics['validation_loss'], 3.)
        self.assertEqual(metrics['validation_first_action_l1'], 2.)
        self.assertEqual(metrics['validation_valid_action_elements'], 4)
        self.assertTrue(policy.training)


if __name__ == '__main__':
    unittest.main()
