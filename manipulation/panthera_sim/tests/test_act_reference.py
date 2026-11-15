import tempfile
import unittest
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

from tools.act_reference_data import ReferenceACTDataset, reference_split


class ReferenceDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        for episode, length in ((0, 4), (1, 3)):
            with h5py.File(self.root / f"episode_{episode}.hdf5", "w") as f:
                f.create_dataset("action", data=np.arange(length * 14).reshape(length, 14) + episode * 1000)
                f.create_dataset("observations/qpos", data=np.ones((length, 14)) * episode)
                f.create_dataset("observations/images/top", data=np.full((length, 8, 10, 3), 128, dtype=np.uint8))

    def tearDown(self):
        self.directory.cleanup()

    def test_same_timestep_alignment_zero_padding_and_image_layout(self):
        ds = ReferenceACTDataset(self.root, [0, 1], 3, training=False)
        self.assertEqual(len(ds), 7)
        sample = ds[2]
        torch.testing.assert_close(sample["action"][0], torch.arange(28, 42).float())
        self.assertEqual(sample["action_is_pad"].tolist(), [False, False, True])
        self.assertEqual(sample["action"][2].sum(), 0)
        self.assertEqual(sample["observation.images.top"].shape, (3, 8, 10))
        self.assertAlmostEqual(sample["observation.images.top"].mean().item(), 128 / 255, places=6)
        self.assertEqual(ds[4]["action"][0, 0].item(), 1000)
        self.assertEqual(ds[6]["action_is_pad"].tolist(), [False, True, True])

    def test_one_random_frame_per_episode(self):
        ds = ReferenceACTDataset(self.root, [0, 1], 3, training=True)
        self.assertEqual(len(ds), 2)
        with patch("numpy.random.randint", return_value=2):
            self.assertEqual(ds[1]["action"][0, 0].item(), 1028)

    def test_upstream_split_is_fixed_and_disjoint(self):
        train, val = reference_split()
        expected = np.random.RandomState(1).permutation(50).tolist()
        self.assertEqual(train, expected[:40])
        self.assertEqual(val, expected[40:])
        self.assertFalse(set(train) & set(val))

    def test_download_repacking_preserves_arrays_and_attributes(self):
        from tools.prepare_act_reference import download_episode
        source = self.root / "episode_0.hdf5"
        with h5py.File(source, "a") as f:
            f.attrs["sim"] = True
            f["observations"].attrs["note"] = "preserve groups too"
        destination = self.root / "downloaded"
        destination.mkdir()
        def fake_download(**kwargs):
            shutil.copyfile(source, kwargs["output"])
        with patch("gdown.download", side_effect=fake_download), patch(
                "shutil.disk_usage", return_value=SimpleNamespace(free=10 * 1024**3)):
            record = download_episode(SimpleNamespace(path=source.name, id="fixture"), destination)
        self.assertEqual(len(record["original_sha256"]), 64)
        self.assertFalse((destination / "episode_0.download").exists())
        with h5py.File(source) as a, h5py.File(destination / source.name) as b:
            self.assertEqual(dict(a.attrs), dict(b.attrs))
            self.assertEqual(dict(a["observations"].attrs), dict(b["observations"].attrs))
            for key in ("action", "observations/qpos", "observations/images/top"):
                np.testing.assert_array_equal(a[key][:], b[key][:])


if __name__ == "__main__":
    unittest.main()
