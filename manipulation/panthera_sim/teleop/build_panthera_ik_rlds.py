"""Convert the published 30 Hz IK LeRobot Parquet dataset to VLA-Adapter RLDS.

Keep embedded JPEGs and already-shifted absolute joint/gripper actions unchanged.
Only proprio gains one zero padding element before the gripper for VLA's 8-D input.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tensorflow_datasets as tfds

CAMERAS = ("observation.images.shoulder", "observation.images.wrist")


def dataset_info(root, hz=30):
    root = Path(root)
    if not (root / "COMPLETE").is_file():
        raise ValueError("IK dataset publication is incomplete")
    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != hz or hz != 30:
        raise ValueError("The IK dataset must stay at its native 30 Hz; no resampling")
    if info["codebase_version"] != "v3.0" or info["total_episodes"] < 2:
        raise ValueError("Expected a LeRobot v3 dataset with at least two episodes")
    for key in ("action", "observation.state"):
        if info["features"][key]["shape"] != [7]:
            raise ValueError(f"Expected six joints and gripper for {key}")
    for key in CAMERAS:
        if info["features"][key]["shape"] != [256, 256, 3]:
            raise ValueError(f"Expected 256x256 RGB for {key}")
    return info


def iter_episodes(root, info):
    """Stream across shard/batch boundaries, buffering at most one episode."""
    columns = ["episode_index", "frame_index", "index", "timestamp",
               "observation.state", "action", *CAMERAS]
    paths = sorted((Path(root) / "data").rglob("*.parquet"))
    if not paths:
        raise ValueError("No IK Parquet shards found")

    def rows():
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=columns):
                yield from batch.to_pylist()

    total = 0
    episodes = 0
    for episode_id, group in itertools.groupby(rows(), key=lambda row: row["episode_index"]):
        episode = list(group)
        if episode_id != episodes:
            raise ValueError("Expected contiguous ordered episode indices")
        for frame, row in enumerate(episode):
            if row["frame_index"] != frame or row["index"] != total + frame:
                raise ValueError(f"Missing or reordered frames in episode {episode_id}")
            if not np.isclose(row["timestamp"], frame / 30, atol=1e-5, rtol=0):
                raise ValueError(f"Frame timing is not 30 Hz in episode {episode_id}")
        total += len(episode)
        episodes += 1
        yield episode_id, episode
    if total != info["total_frames"] or episodes != info["total_episodes"]:
        raise ValueError("IK frame/episode counts do not match dataset metadata")


def episode_steps(rows, instruction):
    states = np.asarray([row["observation.state"] for row in rows], dtype=np.float32)
    actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
    if (states.shape != (len(rows), 7) or actions.shape != (len(rows), 7)
            or not np.isfinite(states).all() or not np.isfinite(actions).all()):
        raise ValueError("Invalid joint state/action vectors")
    # Six measured joints, one padding joint, current opening command in metres.
    proprio = np.insert(states, 6, 0, axis=1)
    steps = []
    for index, row in enumerate(rows):
        images = [row[key]["bytes"] for key in CAMERAS]
        if not all(isinstance(value, bytes) and value for value in images):
            raise ValueError("Expected embedded JPEG bytes for both cameras")
        last = index == len(rows) - 1
        steps.append({
            "observation": {"image": images[0], "wrist_image": images[1],
                            "state": proprio[index]},
            # Labels already point to t+1/30 s. Never shift or delta-convert again.
            "action": actions[index],
            "language_instruction": instruction,
            "is_first": index == 0, "is_last": last, "is_terminal": last,
            "reward": np.float32(last), "discount": np.float32(not last),
        })
    return steps


class PantheraIkThreeBlock(tfds.core.GeneratorBasedBuilder):
    """Scripted three-block IK demonstrations, absolute joint control at 30 Hz."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Native 30 Hz joint actions and episode-level holdout."}
    # TFDS 4.9.3 cannot infer resource paths for the teleop namespace package.
    code_path = Path(__file__)

    def __init__(self, *args, dataset_dir, instruction="stack the three colored cubes",
                 hz=30, val_fraction=0.1, split_seed=20260920, **kwargs):
        self.dataset_dir = Path(dataset_dir)
        self.source_info = dataset_info(self.dataset_dir, hz)
        self.instruction = instruction
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")
        count = self.source_info["total_episodes"]
        val_count = min(count - 1, max(1, round(count * val_fraction)))
        shuffled = np.random.default_rng(split_seed).permutation(count)
        self.val_episodes = set(shuffled[:val_count].tolist())
        super().__init__(*args, **kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(builder=self, features=tfds.features.FeaturesDict({
            "steps": tfds.features.Dataset({
                "observation": {
                    "image": tfds.features.Image(shape=(256, 256, 3), encoding_format="jpeg"),
                    "wrist_image": tfds.features.Image(shape=(256, 256, 3), encoding_format="jpeg"),
                    "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                },
                "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                "language_instruction": tfds.features.Text(),
                "is_first": np.bool_, "is_last": np.bool_, "is_terminal": np.bool_,
                "reward": np.float32, "discount": np.float32,
            }),
            "episode_metadata": {"episode_index": np.int64},
        }), description=__doc__)

    def _split_generators(self, dl_manager):
        del dl_manager
        return {"train": self._generate_examples(False), "val": self._generate_examples(True)}

    def _generate_examples(self, validation):
        for episode_id, rows in iter_episodes(self.dataset_dir, self.source_info):
            if (episode_id in self.val_episodes) != validation:
                continue
            yield str(episode_id), {
                "steps": episode_steps(rows, self.instruction),
                "episode_metadata": {"episode_index": episode_id},
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--hz", type=float, default=30)
    parser.add_argument("--instruction", default="stack the three colored cubes")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20260920)
    args = parser.parse_args()
    builder = PantheraIkThreeBlock(
        data_dir=str(args.data_dir), dataset_dir=args.dataset_dir, hz=args.hz,
        instruction=args.instruction, val_fraction=args.val_fraction, split_seed=args.split_seed)
    # Refuse stale TFRecords with a different split/source, even for direct CLI use.
    contract = {"source": str(args.dataset_dir.resolve()), "hz": args.hz,
                "instruction": args.instruction, "val_fraction": args.val_fraction,
                "split_seed": args.split_seed, "version": str(builder.VERSION)}
    args.data_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.data_dir / "conversion.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("RLDS conversion settings changed; choose a fresh --data-dir")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    builder.download_and_prepare()
    print(f"Built {builder.info.full_name} at {builder.data_dir}")
    print(builder.info.splits)


if __name__ == "__main__":
    main()
