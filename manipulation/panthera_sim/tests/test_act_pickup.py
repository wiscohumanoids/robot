import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tools.act_pickup import crop_action_chunk, pickup_stats, RelativeChunkExecutor, SustainedPickup
from train_act import rollout_score


class PickupExperimentTest(unittest.TestCase):
    def test_uncached_initialization_defers_image_transform_until_sampling(self):
        from datasets import Dataset
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from tools.lerobot_image_cache import CachedLeRobotDataset
        data = Dataset.from_dict({"index": [0, 1], "camera": ["encoded", "encoded"]})
        calls = []
        def transform(batch):
            calls.append(batch["camera"])
            return batch
        data.set_transform(transform)
        def initialize(dataset):
            dataset.hf_dataset = dataset.load_hf_dataset()
            self.assertEqual(list(dataset.hf_dataset["index"]), [0, 1])
            self.assertEqual(calls, [])
        with patch.object(LeRobotDataset, "__init__", initialize), \
             patch.object(LeRobotDataset, "load_hf_dataset", return_value=data):
            dataset = CachedLeRobotDataset(image_cache=None)
        self.assertEqual(dataset.hf_dataset[0]["camera"], "encoded")
        self.assertEqual(len(calls), 1)

    def test_cached_index_initialization_does_not_decode_image_columns(self):
        from datasets import Dataset
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from tools.lerobot_image_cache import CachedLeRobotDataset
        data = Dataset.from_dict({"index": [0, 1], "camera": ["encoded", "encoded"]})
        def transform(batch):
            self.assertNotIn("camera", batch)
            return batch
        data.set_transform(transform)
        cached = object.__new__(CachedLeRobotDataset)
        cached._requested_image_cache = Path("cache.npy")
        cached.meta = SimpleNamespace(camera_keys=["camera"])
        with patch.object(LeRobotDataset, "load_hf_dataset", return_value=data):
            stripped = cached.load_hf_dataset()
        self.assertEqual(list(stripped["index"]), [0, 1])
        self.assertIn("camera", data.column_names)

    def test_action_statistics_exclude_validation_and_padded_targets(self):
        state = np.zeros((4, 7))
        state[:3, :6] = np.arange(3)[:, None]
        action = np.array([[2.]*6+[.04], [4.]*6+[.02], [8.]*6+[0.], [999.]*7])
        manifest = {"episodes": {"0": {"start": 0, "end_exclusive": 3}, "1": {"start": 3, "end_exclusive": 4}}}
        with patch("tools.act_pickup.scalar_data", return_value={"observation.state": state, "action": action}):
            stats = pickup_stats(None, manifest, [0], 2, "relative", {})["action"]
        expected_arm = np.array([2., 3., 6., 4., 7.])
        self.assertAlmostEqual(float(stats["mean"][0]), expected_arm.mean(), places=6)
        self.assertAlmostEqual(float(stats["std"][0]), expected_arm.std(), places=6)
        self.assertAlmostEqual(float(stats["mean"][6]), np.mean([.04, .02, 0., .02, 0.]), places=6)

    def test_crop_masks_future_stacking_and_preserves_absolute_gripper(self):
        action = torch.arange(28, dtype=torch.float32).reshape(4, 7)
        state = torch.tensor([1., 2., 3., 4., 5., 6., 99.])
        result, pad = crop_action_chunk(action, torch.zeros(4, dtype=torch.bool), state, 2, "relative")
        torch.testing.assert_close(result[:2, :6], action[:2, :6] - state[:6])
        torch.testing.assert_close(result[:2, 6], action[:2, 6])
        torch.testing.assert_close(result[2:], result[1:2].expand(2, 7))
        self.assertEqual(pad.tolist(), [False, False, True, True])
        self.assertEqual(action[0, 0], 0.)

    def test_queue_uses_query_time_anchor_for_every_action(self):
        class Policy:
            config = SimpleNamespace(temporal_ensemble_coeff=None, n_action_steps=3)
            calls = 0
            def predict_action_chunk(self, observation):
                self.calls += 1
                return torch.tensor([[[1.]*6+[.02], [2.]*6+[.01], [3.]*6+[0.]]])
        policy = Policy()
        executor = RelativeChunkExecutor(policy, lambda x: x)
        for step, current in enumerate((10., 90., 200.)):
            result = executor.select_action({}, np.array([current]*6+[.04]))
            self.assertAlmostEqual(float(result[0, 0]), 11.+step)
            self.assertAlmostEqual(float(result[0, 6]), [.02, .01, 0.][step])
        self.assertEqual(policy.calls, 1)
        self.assertAlmostEqual(float(executor.select_action({}, np.array([50.]*7))[0, 0]), 51.)
        executor.reset()
        self.assertAlmostEqual(float(executor.select_action({}, np.array([80.]*7))[0, 0]), 81.)

    def test_ensemble_averages_after_converting_each_chunk_to_absolute(self):
        class Policy:
            config = SimpleNamespace(temporal_ensemble_coeff=0., n_action_steps=1, chunk_size=2)
            def predict_action_chunk(self, observation):
                return torch.tensor([[[1.]*6+[.02], [2.]*6+[.01]]])
        executor = RelativeChunkExecutor(Policy(), lambda x: x)
        executor.select_action({}, np.array([10.]*7))
        result = executor.select_action({}, np.array([100.]*7))
        self.assertAlmostEqual(float(result[0, 0]), (12.+101.)/2)
        self.assertAlmostEqual(float(result[0, 6]), .015)

    def test_pickup_requires_same_cube_and_simultaneous_hold_and_height(self):
        tracker = SustainedPickup(30)
        for step in range(14):
            self.assertFalse(tracker.update([.13, .07, .07], [True, False, False], step))
        self.assertFalse(tracker.update([.07, .13, .07], [False, True, False], 14))
        self.assertFalse(tracker.update([.13, .07, .07], [False, True, False], 15))
        for step in range(16, 31):
            tracker.update([.13, .07, .07], [True, False, False], step)
        self.assertTrue(tracker.success)
        self.assertEqual(tracker.first_success_step, 30)

    def test_pickup_selection_ignores_transient_stack_and_prefers_faster_tie(self):
        failure = {"sustained_pickup": 0., "pickup_seconds_capped_mean": 8., "two_stacked": 1.}
        success = {"sustained_pickup": .5, "pickup_seconds_capped_mean": 6.}
        faster = {"sustained_pickup": .5, "pickup_seconds_capped_mean": 5.}
        self.assertGreater(rollout_score(success, "pickup"), rollout_score(failure, "pickup"))
        self.assertGreater(rollout_score(faster, "pickup"), rollout_score(success, "pickup"))


if __name__ == "__main__":
    unittest.main()
