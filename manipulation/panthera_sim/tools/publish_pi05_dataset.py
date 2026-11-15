#!/usr/bin/env python3
"""Stage/publish only the validated scripted three-block LeRobot dataset.

Run with .venv-act/bin/python. Staging hard-links immutable Parquet files, so it
does not duplicate the 22 GiB export. --push explicitly enables Hub writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "FoxNerdSaysMoo/panthera-ik-three-block-stack-30hz"
CAMERAS = ("observation.images.shoulder", "observation.images.wrist")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def validate(root):
    import numpy as np
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text())
    audit = json.loads((root / "verification.json").read_text())
    if not (root / "COMPLETE").exists():
        raise ValueError("Export is incomplete")
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == audit["episodes"] == 1000
    assert info["total_frames"] == audit["frames"] == 862766
    assert info["fps"] == audit["fps"] == 30
    assert audit["all_actions_and_states_match"] and audit["provenance_valid"]
    for key in CAMERAS:
        assert info["features"][key]["shape"] == [256, 256, 3]
    for key in ("observation.state", "action"):
        assert info["features"][key]["shape"] == [7]
    # Re-read every numeric row without decoding/materializing all images.
    rows = 0
    counts = np.zeros(1000, dtype=np.int64)
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["index", "episode_index", "action", "observation.state"])
        idx = table["index"].to_numpy()
        assert np.array_equal(idx, np.arange(rows, rows + len(table)))
        eps = table["episode_index"].to_numpy()
        assert np.all((eps >= 0) & (eps < 1000))
        counts += np.bincount(eps, minlength=1000)
        for key in ("action", "observation.state"):
            values = np.asarray(table[key].to_pylist())
            assert values.shape == (len(table), 7) and np.isfinite(values).all()
        rows += len(table)
    assert rows == info["total_frames"] and np.all(counts > 0)
    return info, audit


def stage_dataset(source, stage, repo_id):
    info, audit = validate(source)
    stage.mkdir(parents=True, exist_ok=True)
    for relative in [Path("meta/info.json"), Path("meta/stats.json"), Path("meta/tasks.parquet"),
                     *sorted(p.relative_to(source) for p in (source / "meta/episodes").rglob("*.parquet")),
                     *sorted(p.relative_to(source) for p in (source / "data").rglob("*.parquet"))]:
        dest = stage / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(source / relative, dest)
        elif not os.path.samefile(source / relative, dest):
            raise ValueError(f"Refusing to replace unrelated staged file {dest}")
    # Remove machine-specific paths while preserving the source/code hashes.
    provenance = json.loads((source / "meta/provenance.json").read_text())
    provenance["rendered_root"] = "outputs/scripted_stack_rendered_30hz"
    for name, entry in provenance["sources"].items():
        entry["source"] = f"data/scripted_stack/{name}"
    (stage / "meta/provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (stage / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    (stage / "README.md").write_text(f"""---
tags:
- lerobot
- robotics
- mujoco
- imitation-learning
- pi05
pretty_name: Panthera IK three-block stacking (30 Hz)
size_categories:
- 100K<n<1M
---

# Panthera IK three-block stacking

{info['total_episodes']:,} successful scripted IK demonstrations; {info['total_frames']:,}
training frames at 30 Hz. This is the **three-block** Panthera MuJoCo task:
red onto green, then blue onto red. The stored task prompt is
`stack the three colored cubes`.

LeRobot v3.0 format, exported using LeRobot 0.4.4. Both 256×256 RGB cameras
(`observation.images.shoulder`, `observation.images.wrist`) are embedded JPEGs
in Parquet. No video decoder, raw-data conversion, or MuJoCo rendering is needed
for training. Download `data/**` and `meta/**` to use LeRobotDataset.

| Feature | Contract |
| --- | --- |
| observation.state | Six measured joint angles in radians, then current gripper opening **command** in metres |
| action | Six absolute joint position targets in radians, then absolute gripper opening command in metres |
| Timing | Observation at t predicts control at t + 1/30 s; actions already shifted once |
| Camera order | Shoulder, then wrist; preserve RGB and physical wrist roll |

Do not apply another action shift, delta conversion, gripper sign inversion,
or end-effector action transform. Short vectors are padded inside pi0.5, not in
the stored data. These are joint-space labels even though an IK planner produced them.

The collector accepted 1,000 of 1,402 randomized attempts. All 1,000 accepted
episodes passed a fresh physics replay of the resampled controls, including a
released three-stack held for the final second. `verification.json` records the
export audit: every action/state row matched its source, and 2,004 sampled camera
images decoded successfully. These are **demonstration** checks, not learned-policy
success rates. Starts are restricted to a reachable elevated region; rejected
attempts introduce selection bias. Test learned policies on held-out seeds and
report both matched starts and the broader teleoperation reset distribution.

`meta/provenance.json` retains source and renderer hashes. The training notebook
uses a deterministic episode holdout, and computes normalization statistics on
training episodes only. Dataset-wide statistics here are supplied for general
LeRobot compatibility. Full-model pi0.5 fine-tuning must explicitly unfreeze the
vision encoder and VLM and omit PEFT/LoRA.

Source project: https://github.com/zebulontaylor/RobotArmTraining
Dataset: https://huggingface.co/datasets/{repo_id}
""")
    paths = sorted(p for p in stage.rglob("*") if p.is_file() and ".cache" not in p.parts
                   and p.name not in {"upload_manifest.json", "COMPLETE"})
    manifest = {"repo_id": repo_id, "episodes": 1000, "frames": 862766,
                "files": {str(p.relative_to(stage)): {"size": p.stat().st_size, "sha256": digest(p)}
                          for p in paths}}
    (stage / "upload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/lerobot/panthera_scripted_stack_30hz")
    parser.add_argument("--stage", type=Path, default=ROOT / "outputs/hf/panthera-ik-three-block-stack-30hz")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    manifest = stage_dataset(args.source, args.stage, args.repo_id)
    print(f"Validated/staged {len(manifest['files'])} files, "
          f"{sum(v['size'] for v in manifest['files'].values()) / 2**30:.2f} GiB", flush=True)
    if not args.push:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    # Resumable uploader; explicit allowlist excludes caches and unrelated outputs.
    api.upload_large_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=args.stage,
                            allow_patterns=[*manifest["files"], "upload_manifest.json"],
                            num_workers=4, print_report_every=60)
    remote = api.dataset_info(args.repo_id, files_metadata=True)
    remote_files = {f.rfilename: f for f in remote.siblings}
    for path, record in manifest["files"].items():
        f = remote_files.get(path)
        if f is None or f.size != record["size"]:
            raise RuntimeError(f"Incomplete remote file: {path}")
        if f.lfs is not None and f.lfs.sha256 != record["sha256"]:
            raise RuntimeError(f"Remote hash mismatch: {path}")
    commit = api.upload_file(repo_id=args.repo_id, repo_type="dataset", path_in_repo="COMPLETE",
                             path_or_fileobj=b"1000 episodes, 862766 frames; remote file sizes and LFS hashes verified\n",
                             commit_message="Mark verified three-block dataset complete")
    result = {"repo_id": args.repo_id, "revision": commit.oid,
              "url": f"https://huggingface.co/datasets/{args.repo_id}",
              "episodes": 1000, "frames": 862766, "verified_files": len(manifest["files"])}
    (args.stage.parent / "pi05_dataset_upload.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
