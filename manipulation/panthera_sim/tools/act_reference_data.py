"""Native tonyzhaozh/act HDF5 adapter for our LeRobot ACT trainer.

Training samples one random timestep per episode per epoch, exactly as the
upstream EpisodicDataset. Validation deterministically covers every held-out
frame and uses the deployed zero-latent path in train_act.py.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature


def reference_metadata(root: Path):
    manifest = root / "provenance.json"
    if not (root / "COMPLETE").is_file() or not manifest.is_file():
        raise ValueError("Run tools/prepare_act_reference.py to completion first")
    records = json.loads(manifest.read_text())["episodes"]
    ids = sorted(int(Path(row["file"]).stem.split("_")[-1]) for row in records)
    if ids != list(range(50)):
        raise ValueError("Reference benchmark requires all 50 published episodes")
    states, actions = [], []
    signature = hashlib.sha256(manifest.read_bytes())
    for episode in ids:
        path = root / f"episode_{episode}.hdf5"
        stat = path.stat()
        signature.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        with h5py.File(path, "r") as f:
            if not f.attrs["sim"]:
                raise ValueError("This benchmark only supports simulation demonstrations")
            if f["action"].shape != (400, 14) or f["observations/qpos"].shape != (400, 14):
                raise ValueError(f"Unexpected state/action shape in {path}")
            if f["observations/images/top"].shape != (400, 480, 640, 3):
                raise ValueError(f"Unexpected top-camera shape in {path}")
            states.append(torch.from_numpy(f["observations/qpos"][:]))
            actions.append(torch.from_numpy(f["action"][:]))
    stats = {}
    for key, values in (("observation.state", states), ("action", actions)):
        values = torch.stack(values)
        if not torch.isfinite(values).all():
            raise ValueError(f"Non-finite {key} in reference dataset")
        # Match torch.std's sample correction and upstream 1e-2 floor.
        stats[key] = {"mean": values.mean((0, 1)).float(),
                      "std": values.std((0, 1)).clamp(min=1e-2).float()}
    stats["observation.images.top"] = {
        "mean": torch.tensor([.485, .456, .406]).view(3, 1, 1),
        "std": torch.tensor([.229, .224, .225]).view(3, 1, 1),
    }
    features = {"observation.state": PolicyFeature(FeatureType.STATE, (14,)),
                "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                "action": PolicyFeature(FeatureType.ACTION, (14,))}
    return SimpleNamespace(fps=50, total_episodes=50, stats=stats, features=features,
                           signature=signature.hexdigest())


def reference_split():
    # Upstream main calls set_seed(1) before load_data, independently of model seed.
    ids = np.random.RandomState(1).permutation(50)
    return ids[:40].tolist(), ids[40:].tolist()


class ReferenceACTDataset(torch.utils.data.Dataset):
    def __init__(self, root, episodes, chunk_size, training):
        self.root = Path(root)
        self.episodes = list(episodes)
        self.chunk_size = chunk_size
        self.training = training
        self.lengths = []
        for episode in self.episodes:
            with h5py.File(self.root / f"episode_{episode}.hdf5", "r") as f:
                self.lengths.append(len(f["action"]))
        self.ends = np.cumsum(self.lengths)

    def __len__(self):
        return len(self.episodes) if self.training else int(self.ends[-1])

    def __getitem__(self, index):
        if self.training:
            episode_index = index
            timestep = int(np.random.randint(self.lengths[index]))
        else:
            episode_index = int(np.searchsorted(self.ends, index, side="right"))
            timestep = index - (int(self.ends[episode_index - 1]) if episode_index else 0)
        return self.frame(episode_index, timestep)

    def frame(self, episode_index, timestep):
        path = self.root / f"episode_{self.episodes[episode_index]}.hdf5"
        with h5py.File(path, "r") as f:
            state = torch.from_numpy(f["observations/qpos"][timestep]).float()
            image = torch.from_numpy(f["observations/images/top"][timestep]).permute(2, 0, 1).float() / 255
            # Simulation actions start at t, with no Panthera successor shift.
            actions = torch.from_numpy(f["action"][timestep:timestep + self.chunk_size]).float()
        padded = torch.zeros(self.chunk_size, actions.shape[-1])
        padded[:len(actions)] = actions
        return {"observation.state": state, "observation.images.top": image,
                "action": padded, "action_is_pad": torch.arange(self.chunk_size) >= len(actions)}
