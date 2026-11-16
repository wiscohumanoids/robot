"""CPU-only pi05 plumbing tests; no LeRobot weights or hardware are loaded."""
import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Optional EXPO tests require PyTorch")

from infra.rl.pi05_adapter import (PI05Adapter, IMAGE_KEYS, TASK, ACTION_NAMES,
                                   HORIZON, ACTION_DIM)
from infra.rl.base_training import success_only_flow_loss, success_only_flow_step, load_supervised_batch, main


class MockPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(max_action_dim=32, chunk_size=50, use_relative_actions=False)
        self.seen = None
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, batch, *, noise):
        self.seen = batch
        return noise[..., :14] + batch["observation.state"][:, None, :] * self.weight

    def forward(self, batch, reduction="mean"):
        self.seen = batch
        assert reduction == "mean"
        return (batch["action"] * self.weight).square().mean(), {"loss_per_dim": [1.0] * 14}


class MockPreprocessor:
    def __init__(self):
        self.calls = 0
        self.steps = [SimpleNamespace(_tensor_stats={
            "action": {"q01": torch.full((14,), -2.), "q99": torch.full((14,), 2.)},
            "observation.state": {"q01": torch.full((14,), -4.), "q99": torch.full((14,), 4.)}}, eps=1e-8)]

    def __call__(self, batch):
        self.calls += 1
        result = dict(batch)
        if result["observation.state"].ndim == 1:
            result = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
                      for key, value in result.items()}
        result["observation.state"] = result["observation.state"] / 4
        if "action" in result:
            result["action"] = result["action"] / 2
        return result


class PI05AdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
            (self.path / name).write_text("{}")
        (self.path / "model.safetensors").write_bytes(b"mock checkpoint fixture, never loaded")
        (self.path / "deployment.json").write_text(json.dumps({"fps": 20, "task": TASK,
                                                               "action_representation": "absolute"}))
        self.adapter = PI05Adapter(self.path, device="cpu")
        self.policy = MockPolicy().eval()
        self.pre = MockPreprocessor()
        self.adapter.policy = self.policy
        self.adapter.preprocessor = self.pre
        self.adapter.postprocessor = lambda actions: actions * 2
        self.adapter._parameter_versions = tuple((id(p), p._version) for p in self.policy.parameters())
        self.observation = {"observation.state": torch.arange(14).float(), "task": TASK}
        self.observation.update({key: torch.full((3, 256, 256), i / 4)
                                 for i, key in enumerate(IMAGE_KEYS)})

    def test_constructor_is_lazy_and_requires_complete_checkpoint(self):
        with patch.object(PI05Adapter, "_verify_runtime_revision", side_effect=AssertionError("loaded")):
            adapter = PI05Adapter(self.path, device="cpu")
            self.assertIsNone(adapter.policy)
        (self.path / "policy_postprocessor.json").unlink()
        with self.assertRaises(FileNotFoundError):
            PI05Adapter(self.path)

    def test_genuine_noise_sampling_and_preserved_processor_mapping(self):
        rng_before = torch.get_rng_state().clone()
        candidates = self.adapter.sample_candidates(self.observation, 3, seed=7)
        self.assertEqual(candidates.normalized.shape, (3, 50, 14))
        self.assertTrue(torch.equal(candidates.physical, candidates.normalized * 2))
        self.assertFalse(torch.equal(candidates.normalized[0], candidates.normalized[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertEqual(self.pre.calls, 1)
        self.assertEqual(self.policy.resets, 1)
        expected_state = self.observation["observation.state"] / 4
        self.assertTrue(torch.equal(self.policy.seen["observation.state"], expected_state.repeat(3, 1)))
        self.assertTrue(torch.equal(self.policy.seen[IMAGE_KEYS[1]][0], self.observation[IMAGE_KEYS[1]]))
        repeated = self.adapter.sample_candidates(self.observation, 3, seed=7)
        self.assertTrue(torch.equal(candidates.normalized, repeated.normalized))
        self.assertEqual(candidates.observation_id, repeated.observation_id)
        self.assertEqual(self.adapter.normalization["action"]["q99"], [2.0] * 14)

    def test_observation_hash_changes_and_camera_errors_never_sample(self):
        first = self.adapter.sample_candidates(self.observation, 2)
        changed = dict(self.observation)
        changed["observation.state"] = changed["observation.state"] + 1
        second = self.adapter.sample_candidates(changed, 2)
        self.assertNotEqual(first.observation_id, second.observation_id)
        missing = dict(self.observation)
        del missing[IMAGE_KEYS[-1]]
        with self.assertRaises(ValueError):
            self.adapter.sample_candidates(missing)
        invalid = dict(self.observation, task="other task")
        with self.assertRaises(ValueError):
            self.adapter.sample_candidates(invalid)
        self.assertEqual(self.policy.resets, 2)

    def test_file_and_weight_mutations_invalidate_identity(self):
        with torch.no_grad():
            self.policy.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, "parameters changed"):
            self.adapter.sample_candidates(self.observation)
        self.adapter._parameter_versions = tuple((id(p), p._version) for p in self.policy.parameters())
        (self.path / "config.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(RuntimeError, "Checkpoint changed"):
            self.adapter.sample_candidates(self.observation)

    def test_config_rejects_other_embodiments_and_normalization(self):
        config = SimpleNamespace(type="pi05", chunk_size=50, use_relative_actions=False,
                                 action_feature_names=list(ACTION_NAMES), image_features=dict.fromkeys(IMAGE_KEYS),
                                 input_features={"observation.state": SimpleNamespace(shape=(14,))},
                                 output_features={"action": SimpleNamespace(shape=(14,))},
                                 normalization_mapping={"STATE": "QUANTILES", "ACTION": "QUANTILES"})
        PI05Adapter._validate_config(config)
        config.normalization_mapping["ACTION"] = "MEAN_STD"
        with self.assertRaisesRegex(ValueError, "quantiles"):
            PI05Adapter._validate_config(config)


class BaseSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.policy = MockPolicy().eval()
        self.pre = MockPreprocessor()
        self.batch = {"observation.state": torch.zeros(3, 14),
                      "action": torch.stack([torch.ones(50, 14) * i for i in (2., 100., 500.)]),
                      "task": [TASK] * 3}
        self.batch.update({key: torch.zeros(3, 3, 256, 256) for key in IMAGE_KEYS})
        self.valid = torch.ones(3, 50, dtype=torch.bool)
        self.valid[2, -1] = False
        self.labels = dict(episode_success=torch.tensor([True, False, True]),
                           supervision_valid=self.valid,
                           action_provenance=["kinesthetic_demo", "executed_command", "human_intervention"])

    def test_success_and_provenance_filter_before_original_forward(self):
        loss, metrics = success_only_flow_loss(self.policy, self.pre, self.batch, **self.labels)
        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertEqual(metrics["base_examples_used"], 1)
        self.assertEqual(self.policy.seen["action"].shape, (1, 50, 14))
        self.assertFalse(self.policy.training)
        loss.backward()
        self.assertIsNotNone(self.policy.weight.grad)
        self.assertEqual(self.batch["action"][0, 0, 0].item(), 2.0)

    def test_empty_success_pool_skips_forward_and_unknown_provenance_rejects(self):
        labels = dict(self.labels, episode_success=torch.zeros(3, dtype=torch.bool))
        loss, metrics = success_only_flow_loss(self.policy, self.pre, self.batch, **labels)
        self.assertIsNone(loss)
        self.assertEqual(metrics["base_examples_used"], 0)
        self.assertIsNone(self.policy.seen)
        labels["action_provenance"] = ["fabricated"] * 3
        with self.assertRaisesRegex(ValueError, "provenance"):
            success_only_flow_loss(self.policy, self.pre, self.batch, **labels)

    def test_explicit_optimizer_step_updates_only_when_eligible(self):
        optimizer = torch.optim.SGD(self.policy.parameters(), lr=.1)
        before = self.policy.weight.detach().clone()
        result = success_only_flow_step(self.policy, self.pre, optimizer, self.batch, **self.labels)
        self.assertLess(self.policy.weight.item(), before.item())
        self.assertEqual(result["base_examples_used"], 1)
        with self.assertRaisesRegex(ValueError, "boolean"):
            success_only_flow_loss(self.policy, self.pre, self.batch,
                                   **dict(self.labels, episode_success=[1, 0, 1]))

    def test_supervised_npz_schema_requires_labels_and_source_identifiers(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "supervision.npz"
            values = {key: value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                      for key, value in self.batch.items()}
            values.update({key: value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                           for key, value in self.labels.items()})
            values["source_episode_id"] = np.asarray(["recorded-1", "recorded-2", "recorded-3"])
            np.savez(path, **values)
            batch, labels, sources = load_supervised_batch(path)
            self.assertEqual(batch["action"].shape, (3, 50, 14))
            self.assertEqual(sources[0], "recorded-1")
            self.assertTrue(labels["episode_success"][0])
            del values["episode_success"]
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "Missing explicit"):
                load_supervised_batch(path)

    def test_base_cli_runs_original_forward_with_injected_cpu_model(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "supervision.npz"
            values = {key: value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                      for key, value in self.batch.items()}
            values.update({key: value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                           for key, value in self.labels.items()})
            values["source_episode_id"] = np.asarray(["real-1", "real-2", "real-3"])
            np.savez(path, **values)
            output = Path(directory) / "new-policy"
            self.policy.save_pretrained = lambda target: (Path(target) / "model.safetensors").write_bytes(b"mock")
            self.pre.save_pretrained = lambda target, config_filename: (Path(target) / config_filename).write_text("{}")
            post = SimpleNamespace(save_pretrained=lambda target, config_filename:
                                   (Path(target) / config_filename).write_text("{}"))
            adapter = SimpleNamespace(policy=self.policy, preprocessor=self.pre, postprocessor=post,
                                      checkpoint_id="original-checkpoint", processor_id="original-processors",
                                      deployment={"fps": 20, "task": TASK, "action_representation": "absolute"})
            adapter.load = lambda: adapter
            with patch("infra.rl.pi05_adapter.PI05Adapter", return_value=adapter), \
                    patch("infra.rl.resources.require_idle_gpu", return_value=nullcontext({"device": "cpu"})):
                main(["--checkpoint", "unused-mock", "--batch", str(path), "--output", str(output),
                      "--steps", "2", "--device", "cpu", "--no-gradient-checkpointing"])
            metadata = json.loads((output / "base_training.json").read_text())
            self.assertEqual(metadata["source_processor_id"], "original-processors")
            self.assertEqual(len(metadata["history"]), 2)
            self.assertTrue((output / "PI05_COMPLETE").exists())
            self.assertFalse((output / "BASE_TRAINING_INCOMPLETE").exists())
            self.assertFalse(self.policy.training)

    def test_base_cli_resource_refusal_precedes_batch_and_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("infra.rl.resources.require_idle_gpu", side_effect=RuntimeError("existing compute process")), \
                    patch("infra.rl.pi05_adapter.PI05Adapter") as constructor, \
                    patch("infra.rl.base_training.load_supervised_batch") as loader:
                with self.assertRaisesRegex(RuntimeError, "existing compute"):
                    main(["--checkpoint", "not-loaded", "--batch", "not-opened.npz", "--output",
                          str(Path(directory) / "new-checkpoint")])
                constructor.assert_not_called()
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
