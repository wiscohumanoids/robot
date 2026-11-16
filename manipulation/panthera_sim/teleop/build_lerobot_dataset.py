"""Convert rendered Panthera demonstrations to a local LeRobot dataset.

The ACT policy observes shoulder and wrist RGB images plus the six arm joint
positions and gripper opening.  Its action is the *next* sampled joint target
and gripper command.  The one-sample shift matters because each source row was
recorded after applying that row's control; using the unshifted control would
teach a nearly identity state-to-action mapping.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from teleop.dataset_contract import DEFAULT_RENDERED, DEFAULT_DATASET, validate_rendered, file_hash

from sim.dynamics import provenance_dynamics

DEFAULT_INPUT = DEFAULT_RENDERED
DEFAULT_OUTPUT = DEFAULT_DATASET
JOINT_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper_open_m"]


def features(height: int, width: int) -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "observation.images.shoulder": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
    }


def episode_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.glob("episode_*")
        if (path / "trajectory.npz").is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default="local/panthera_stack")
    parser.add_argument("--task", default="stack the three colored cubes")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = validate_rendered(args.input)
    fps = int(round(float(manifest["sample_hz"])))
    if not np.isclose(fps, manifest["sample_hz"]):
        raise ValueError("LeRobot export requires an integer sample rate")
    height, width = map(int, manifest["image_size"])
    episodes = [args.input / name for name in manifest["episodes"]]
    if len(episodes) != int(manifest["num_episodes"]):
        raise SystemExit(
            f"manifest says {manifest['num_episodes']} episodes but found {len(episodes)}"
        )
    sources = {p.name: json.loads((p / "source.json").read_text()) for p in episodes}
    dynamics = provenance_dynamics({"sources": sources})
    if args.output.exists():
        if not args.force:
            raise SystemExit(f"output already exists: {args.output} (use --force to rebuild)")
        shutil.rmtree(args.output)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        fps=fps,
        robot_type="panthera_ht_sim",
        features=features(height, width),
        use_videos=False,
        image_writer_threads=8,
        metadata_buffer_size=20,
    )
    try:
        for ep_i, episode in enumerate(episodes):
            with np.load(episode / "trajectory.npz") as data:
                q = np.asarray(data["q"], dtype=np.float32)
                ctrl = np.asarray(data["ctrl"], dtype=np.float32)
                if not np.allclose(np.diff(data["t"]), 1 / fps, atol=1e-7):
                    raise ValueError(f"{episode}: nonuniform sample times")
            if len(q) < 2 or ctrl.shape != (len(q), 7):
                raise ValueError(f"{episode}: unexpected q/ctrl shapes {q.shape}/{ctrl.shape}")

            # The current gripper command is the best available proprioceptive
            # value; finger joint positions were not included in the recording.
            state = np.concatenate([q, ctrl[:, 6:7]], axis=1)
            # Drop the last observation: it has no demonstrated future action.
            action = ctrl[1:]
            for frame_i in range(len(q) - 1):
                shoulder_path = episode / "shoulder" / f"{frame_i:05d}.jpg"
                wrist_path = episode / "wrist" / f"{frame_i:05d}.jpg"
                if not shoulder_path.is_file() or not wrist_path.is_file():
                    raise FileNotFoundError(f"missing rendered images in {episode}")
                with Image.open(shoulder_path) as image:
                    shoulder = np.asarray(image.convert("RGB"))
                with Image.open(wrist_path) as image:
                    wrist = np.asarray(image.convert("RGB"))
                dataset.add_frame({
                    "observation.state": state[frame_i],
                    "observation.images.shoulder": shoulder,
                    "observation.images.wrist": wrist,
                    "action": action[frame_i],
                    "task": args.task,
                })
            dataset.save_episode()
            print(
                f"[{ep_i + 1:03d}/{len(episodes):03d}] {episode.name}: {len(q) - 1} frames",
                flush=True,
            )
    finally:
        dataset.finalize()
        dataset.stop_image_writer()

    provenance = {"simulation_dynamics": dynamics, "version": 2, "fps": fps, "rendered_root": str(args.input.resolve()),
                  "render_manifest_sha256": file_hash(args.input / "manifest.json"),
                  "converter_sha256": file_hash(Path(__file__)),
                  "state_gripper": "command", "action_alignment": "next_uniform_sample",
                  "sources": sources}
    (args.output / "meta/provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {len(episodes)} episodes / {manifest['num_frames'] - len(episodes)} frames to {args.output}")


if __name__ == "__main__":
    main()
