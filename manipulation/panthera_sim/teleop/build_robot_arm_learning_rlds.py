"""Build rendered RobotArmLearning demonstrations as an RLDS/TFDS dataset.

Run this after ``render_vla_dataset.py``.  It is kept in the RobotArmLearning
repository so remote training environments do not depend on an untracked
VLA-Adapter checkout containing a project-specific builder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
from scipy.spatial.transform import Rotation


class RobotArmLearningPanthera(tfds.core.GeneratorBasedBuilder):
    """Two-camera Panthera demonstrations recorded by RobotArmLearning."""

    VERSION = tfds.core.Version("1.1.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial shoulder+wrist RobotArmLearning dataset.",
        "1.0.1": "Correct the POS_EULER padding field in proprioceptive state.",
        "1.1.0": "Add a deterministic episode-level validation split.",
    }

    def __init__(
        self,
        *args,
        rendered_dir: Path,
        instruction: str,
        val_fraction: float = 0.1,
        split_seed: int = 20260920,
        **kwargs,
    ):
        self.rendered_dir = Path(rendered_dir).resolve()
        self.instruction = instruction
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be between 0 and 1")
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        step = tfds.features.FeaturesDict({
            "observation": tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(256, 256, 3), encoding_format="jpeg"),
                "wrist_image": tfds.features.Image(shape=(256, 256, 3), encoding_format="jpeg"),
                "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
            }),
            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
            "language_instruction": tfds.features.Text(),
            "is_first": np.bool_,
            "is_last": np.bool_,
            "is_terminal": np.bool_,
            "reward": np.float32,
            "discount": np.float32,
        })
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset(step),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                }),
            }),
            supervised_keys=None,
            description=__doc__,
        )

    def _split_generators(self, dl_manager):
        del dl_manager
        episodes = sorted(
            path for path in self.rendered_dir.glob("episode_*")
            if (path / "trajectory.npz").is_file()
        )
        if not episodes:
            raise FileNotFoundError(f"no rendered episodes in {self.rendered_dir}")
        if len(episodes) < 2:
            raise ValueError("at least two episodes are required for train/validation splits")

        # Split whole demonstrations, never frames. A frame-level split would put
        # near-identical neighboring observations on both sides and make validation
        # loss look much better than closed-loop generalization really is.
        rng = np.random.default_rng(self.split_seed)
        shuffled = [episodes[index] for index in rng.permutation(len(episodes))]
        val_count = min(len(episodes) - 1, max(1, round(len(episodes) * self.val_fraction)))
        val_names = {episode.name for episode in shuffled[:val_count]}
        train_episodes = [episode for episode in episodes if episode.name not in val_names]
        val_episodes = [episode for episode in episodes if episode.name in val_names]
        return {
            "train": self._generate_examples(train_episodes),
            "val": self._generate_examples(val_episodes),
        }

    def _generate_examples(self, episodes):
        for episode in episodes:
            with np.load(episode / "trajectory.npz") as data:
                ee_pos = np.asarray(data["ee_pos"], dtype=np.float32)
                quat_wxyz = np.asarray(data["ee_quat"], dtype=np.float32)
                gripper = np.asarray(data["gripper"], dtype=np.float32)

            # SciPy uses (x, y, z, w); recordings and MuJoCo use (w, x, y, z).
            rotations = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
            euler = rotations.as_euler("xyz").astype(np.float32)
            state = np.concatenate([
                ee_pos,
                euler,
                np.zeros((len(gripper), 1), dtype=np.float32),
                (0.04 * gripper)[:, None],
            ], axis=1).astype(np.float32)

            delta_pos = np.zeros_like(ee_pos)
            delta_pos[:-1] = ee_pos[1:] - ee_pos[:-1]
            delta_rot = np.zeros((len(ee_pos), 3), dtype=np.float32)
            relative = rotations[1:] * rotations[:-1].inv()
            delta_rot[:-1] = relative.as_rotvec().astype(np.float32)
            action = np.concatenate([
                delta_pos,
                delta_rot,
                np.concatenate([gripper[1:], gripper[-1:]])[:, None],
            ], axis=1).astype(np.float32)

            steps = []
            last = len(action) - 1
            for index in range(len(action)):
                steps.append({
                    "observation": {
                        "image": episode / "shoulder" / f"{index:05d}.jpg",
                        "wrist_image": episode / "wrist" / f"{index:05d}.jpg",
                        "state": state[index],
                    },
                    "action": action[index],
                    "language_instruction": self.instruction,
                    "is_first": index == 0,
                    "is_last": index == last,
                    "is_terminal": index == last,
                    "reward": np.float32(index == last),
                    "discount": np.float32(index != last),
                })
            yield episode.name, {
                "steps": steps,
                "episode_metadata": {"file_path": str(episode)},
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered-dir", type=Path,
                        default=Path("data/robot_arm_learning_rendered"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/robot_arm_learning"))
    parser.add_argument("--instruction", default="stack the three colored cubes")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="fraction of complete episodes reserved for validation")
    parser.add_argument("--split-seed", type=int, default=20260920,
                        help="seed for the deterministic episode-level split")
    parser.add_argument("--rebuild", action="store_true",
                        help="regenerate TFRecords after adding rendered episodes")
    args = parser.parse_args()

    manifest = json.loads((args.rendered_dir / "manifest.json").read_text())
    if manifest.get("image_size") != [256, 256]:
        raise SystemExit("the RLDS builder expects 256x256 renders")
    builder = RobotArmLearningPanthera(
        data_dir=str(args.data_dir),
        rendered_dir=args.rendered_dir,
        instruction=args.instruction,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
    download_config = tfds.download.DownloadConfig(
        download_mode=(
            tfds.download.GenerateMode.REUSE_CACHE_IF_EXISTS
            if args.rebuild
            else tfds.download.GenerateMode.REUSE_DATASET_IF_EXISTS
        )
    )
    builder.download_and_prepare(download_config=download_config)
    print(f"built {builder.info.full_name} at {builder.data_dir}")
    print(builder.info.splits)


if __name__ == "__main__":
    main()
