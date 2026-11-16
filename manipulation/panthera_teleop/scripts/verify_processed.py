#!/usr/bin/env python3
"""Audit source identity, episode boundaries, actions, images, and native loading."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from postprocess_kept import (CAMERAS, prepare_image, read_rows, sha256, vector, write_json,
                              source_path, validate_hitl_selection, data_timestamp, data_origin)


def verify(root, source):
    manifest = json.loads((root / "processing/manifest.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["fps"] == manifest["fps"]
    assert info["total_frames"] == manifest["totals"]["observations"]
    assert info["total_episodes"] == len(manifest["episodes"])
    alignment = []
    for p in sorted((root / "processing/alignment").glob("*.parquet")):
        alignment.extend(pq.read_table(p).to_pylist())
    assert len(alignment) == info["total_frames"]
    state_map = {}
    origins = {}
    episode_lengths = {}
    for episode in manifest["episodes"]:
        ep = episode["episode_index"]
        p = source_path(source, episode)
        origins[ep] = data_origin(json.loads((p / "metadata.json").read_text()))
        for name in ("metadata.json", "states.jsonl", "frames.jsonl"):
            assert sha256(p / name) == episode["source_hashes"][name]
        states = read_rows(p / "states.jsonl")
        meta = json.loads((p / 'metadata.json').read_text())
        start = data_origin(meta) + round(episode['start_s'] * 1e9)
        end = data_origin(meta) + round(episode['end_s'] * 1e9)
        validate_hitl_selection(meta, episode, [s for s in states if start <= data_timestamp(s) <= end])
        state_map[ep] = {s["sequence"]: s for s in states}
        episode_lengths[ep] = episode["exported_observations"]
        archive = root / f"trimmed_full_rate/{episode['id']}.tar.gz"
        assert sha256(archive) == episode["trimmed_archive_sha256"]
    checked_rows = checked_images = checked_exact_images = 0
    previous_times = {}
    for file in sorted((root / "data").rglob("*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                a = alignment[checked_rows]
                ep = int(row["episode_index"])
                frame_index = int(row["frame_index"])
                episode = manifest["episodes"][ep]
                assert row["index"] == a["index"] == checked_rows
                assert ep == a["episode_index"] and frame_index == a["frame_index"]
                assert row["task_index"] == 0
                assert abs(row["timestamp"] - frame_index / info["fps"]) < 1e-4
                if ep in previous_times:
                    assert abs(row["timestamp"] - previous_times[ep] - 1 / info["fps"]) < 1e-4
                previous_times[ep] = row["timestamp"]
                state = state_map[ep][a["state_sequence"]]
                action = state_map[ep][a["action_sequence"]]
                assert state["monotonic_ns"] == a["state_monotonic_ns"]
                assert action["monotonic_ns"] == a["action_monotonic_ns"]
                assert data_timestamp(state) == a.get('state_data_monotonic_ns', a['state_monotonic_ns'])
                assert data_timestamp(action) == a.get('action_data_monotonic_ns', a['action_monotonic_ns'])
                assert state.get("data_time_s", state["episode_time_s"]) <= episode["end_s"]
                assert action.get("data_time_s", action["episode_time_s"]) <= episode["end_s"]
                assert data_timestamp(action) > data_timestamp(state)
                for key, expected in (
                    ("observation.state", vector(state)), ("action", vector(action)),
                    ("observation.velocity", vector(state, "velocity_rad_s", "velocity")),
                    ("observation.effort", vector(state, "measured_torque_nm", "torque"))):
                    np.testing.assert_array_equal(np.asarray(row[key], dtype=np.float32), expected)
                    assert np.isfinite(row[key]).all()
                assert abs(a["state_offset_s"]) <= manifest["max_state_alignment_s"]
                assert abs(a["action_offset_s"]) <= manifest["max_state_alignment_s"]
                for camera in CAMERAS:
                    encoded = row[f"observation.images.{camera}"]["bytes"]
                    assert encoded and encoded[:2] == b"\xff\xd8"
                    with Image.open(io.BytesIO(encoded)) as image:
                        image.load()
                        assert image.size == (256, 256) and image.mode == "RGB"
                    assert abs(a[f"{camera}.offset_s"]) <= manifest["max_camera_alignment_s"]
                    image_path = source_path(source, episode) / a[f"{camera}.source_path"]
                    assert sha256(image_path) == a[f"{camera}.source_sha256"]
                    assert a.get(f"{camera}.source_data_monotonic_ns", a[f"{camera}.source_monotonic_ns"]) <= origins[ep] + round(episode["end_s"] * 1e9)
                    if frame_index in (0, episode_lengths[ep] // 2, episode_lengths[ep] - 1):
                        transformed = prepare_image(image_path, manifest["image_crops_xyxy"][camera], 256)
                        buffer = io.BytesIO()
                        Image.fromarray(transformed).save(buffer, format="JPEG", quality=95, subsampling=0)
                        assert hashlib.sha256(buffer.getvalue()).digest() == hashlib.sha256(encoded).digest()
                        checked_exact_images += 1
                    checked_images += 1
                checked_rows += 1
    assert checked_rows == info["total_frames"]
    native = LeRobotDataset(manifest["repo_id"], root=root, video_backend="pyav",
        delta_timestamps={"action": [i / info["fps"] for i in range(10)]})
    assert native.num_episodes == info["total_episodes"] and len(native) == checked_rows
    native_checks = 0
    for ep in manifest["episodes"]:
        rows = [a for a in alignment if a["episode_index"] == ep["episode_index"]]
        for a in (rows[0], rows[len(rows) // 2], rows[-1]):
            loaded = native[a["index"]]
            assert loaded["task"] == manifest["task"]
            assert tuple(loaded["observation.state"].shape) == (14,)
            assert tuple(loaded["action"].shape) == (10, 14)
            for camera in CAMERAS:
                assert tuple(loaded[f"observation.images.{camera}"].shape) == (3, 256, 256)
            if a is rows[-1]:
                # LeRobot action chunks pad at the end of this episode; they do
                # not acquire the next episode's joint target or released tail.
                expected = vector(state_map[ep["episode_index"]][a["action_sequence"]])
                for action in loaded["action"].numpy():
                    np.testing.assert_array_equal(action, expected)
            native_checks += 1
    result = dict(passed=True, checked_observations=checked_rows, decoded_images=checked_images,
        exact_source_transform_image_checks=checked_exact_images,
        native_loader_items_with_10_step_action_chunks=native_checks,
        episode_boundary_padding_verified=True, source_logs_unchanged=True,
        action_state_velocity_effort_exact_source_match=True,
        all_source_image_hashes_match=True, archive_hashes_verified=True,
        counts=manifest["totals"], training_or_robot_execution_performed=False)
    write_json(root / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--source", type=Path, default=Path("data/kept"))
    args = p.parse_args()
    verify(args.root, args.source)
