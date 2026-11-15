#!/usr/bin/env python3
"""Non-destructive, annotated real demonstration export to LeRobot v3."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tarfile
import sys

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from hitl_recording import data_timestamp, data_origin

CAMERAS = ("overhead", "wrist_port2", "wrist_port3")
ARMS = ("left", "right")
NAMES = [f"{arm}.joint{i}" for arm in ARMS for i in range(1, 7)]
NAMES = NAMES[:6] + ["left.gripper_rad"] + NAMES[6:] + ["right.gripper_rad"]


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def vector(sample, field="position_rad", gripper_field="position"):
    result = np.asarray([value for arm in ARMS for value in
                       sample["arms"][arm][field] + [sample["arms"][arm]["gripper"][gripper_field]]],
                      dtype=np.float32)
    if field == 'velocity_rad_s':
        result = result / sample.get('data_clock_rate', 1.)
    return result


def nearest_index(times, timestamp):
    right = int(np.searchsorted(times, timestamp))
    choices = [max(0, right - 1), min(len(times) - 1, right)]
    return min(choices, key=lambda i: abs(int(times[i]) - timestamp))


def prepare_image(path, roi, size):
    with Image.open(path) as source:
        image = source.convert("RGB")
        assert image.size == (640, 480), (path, image.size)
        image = image.crop(tuple(roi))
        # Fixed square crops preserve geometry and fill every model pixel.
        assert image.width == image.height, (path, roi)
        image = image.resize((size, size), resample=Image.Resampling.LANCZOS)
        return np.asarray(image).copy()


class JpegLeRobotDataset(LeRobotDataset):
    def _save_image(self, image, fpath, compress_level=1):
        # Native LeRobot temporary paths end in .png; explicit format overrides
        # that suffix. The temporary files are deleted by save_episode. Parquet
        # embeds JPEG bytes, and statistics are computed from those same bytes.
        Image.fromarray(image).save(fpath, format="JPEG", quality=95, subsampling=0)


def source_path(root, annotation):
    category = annotation.get('source_category')
    if category is not None and category not in ('kept', 'review', 'discarded'):
        raise ValueError('Invalid source category')
    identifier = annotation.get('source_id', annotation['id'])
    for name in (identifier, annotation['id']):
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid episode identifier')
    if category and root.name in ('kept', 'review', 'discarded'):
        root = root.parent / category
    return root / identifier


def validate_hitl_selection(metadata, annotation, states):
    if metadata['context']['mode'] == 'independent_gravity_assist':
        return
    if metadata['context']['mode'] != 'human_in_the_loop':
        raise ValueError('Unsupported recording control mode')
    selection = annotation.get('selection')
    if selection == 'human_correction':
        if not states or any(s.get('control_source') != 'human' for s in states):
            raise ValueError('Correction interval crosses a human/policy boundary')
        if len({s.get('recording_segment') for s in states}) > 1:
            raise ValueError('Correction interval crosses a recording pause')
    elif selection == 'successful_unassisted':
        if metadata.get('outcome', {}).get('success') is not True or any(
                s.get('control_source') not in ('policy', 'hold') for s in states):
            raise ValueError('Unassisted interval is not a successful policy rollout')
    else:
        raise ValueError('HITL recordings need an explicit training selection')


def export(args):
    spec = json.loads(args.annotations.read_text())
    output = args.output.resolve()
    trimmed_root = args.trimmed_output.resolve()
    if output.exists() or trimmed_root.exists():
        raise FileExistsError("Choose new output directories; sources and previous exports are preserved")
    features = {
        key: {"dtype": "float32", "shape": (14,), "names": NAMES}
        for key in ("observation.state", "action", "observation.velocity", "observation.effort")
    }
    features.update({f"observation.images.{camera}": {
        "dtype": "image", "shape": (spec["image_size"], spec["image_size"], 3),
        "names": ["height", "width", "channels"]} for camera in CAMERAS})
    dataset = JpegLeRobotDataset.create(
        repo_id=args.repo_id, root=output, fps=spec["fps"],
        robot_type="panthera_ht_bimanual_real", features=features, use_videos=False,
        metadata_buffer_size=1)
    period_ns = round(1e9 / spec["fps"])
    manifest = dict(spec, repo_id=args.repo_id, source_kind=spec.get("source_kind", "operator_kept_real_recordings"),
                    lerobot_version="0.4.4", format="LeRobot v3.0", episodes=[])
    (output / "processing/alignment").mkdir(parents=True)
    (output / "trimmed_full_rate").mkdir()
    total_source_states = total_trimmed_states = total_source_images = total_trimmed_images = 0
    global_index = 0
    for episode_index, annotation in enumerate(spec["episodes"]):
        identifier = annotation["id"]
        source = source_path(args.source, annotation)
        metadata = json.loads((source / "metadata.json").read_text())
        assert metadata["label"] == "kept" or (metadata["label"] == "review" and
                annotation.get("review_approved") is True), "Review runs need explicit saved approval"
        assert metadata["complete"], "Incomplete recordings require separate source validation"
        states = read_rows(source / "states.jsonl")
        frames = read_rows(source / "frames.jsonl")
        origin = data_origin(metadata)
        start_ns = origin + round(annotation["start_s"] * 1e9)
        end_ns = origin + round(annotation["end_s"] * 1e9)
        state_times = np.asarray([data_timestamp(s) for s in states], dtype=np.int64)
        assert np.all(np.diff(state_times) > 0)
        kept_states = [s for s in states if start_ns <= data_timestamp(s) <= end_ns]
        validate_hitl_selection(metadata, annotation, kept_states)
        if annotation.get("selection") == "successful_unassisted" and any(
                s.get("control_source") == "human" for s in states):
            raise ValueError("A corrected rollout cannot export its policy prefix as unassisted")
        kept_frames = [f for f in frames if start_ns <= data_timestamp(f) <= end_ns]
        frame_groups = {camera: sorted([f for f in kept_frames if f["camera"] == camera],
                                     key=lambda f: data_timestamp(f)) for camera in CAMERAS}
        camera_times = {camera: np.asarray([data_timestamp(f) for f in group], dtype=np.int64)
                        for camera, group in frame_groups.items()}
        assert all(len(t) > 1 and np.all(np.diff(t) > 0) for t in camera_times.values())
        trim_dir = trimmed_root / identifier
        trim_dir.mkdir(parents=True)
        write_rows(trim_dir / "states.jsonl", kept_states)
        write_rows(trim_dir / "frames.jsonl", kept_frames)
        for frame in kept_frames:
            destination = trim_dir / frame["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / frame["path"], destination)
        hashes = {name: sha256(source / name) for name in ("metadata.json", "states.jsonl", "frames.jsonl")}
        images_digest = hashlib.sha256()
        for frame in frames:
            images_digest.update(frame["path"].encode())
            images_digest.update(bytes.fromhex(sha256(source / frame["path"])))
        hashes["ordered_original_images_sha256"] = images_digest.hexdigest()
        # Source metadata is preserved as provenance, with valid derived counts.
        derived_metadata = dict(metadata, samples=len(kept_states),
            camera_frames={c: len(g) for c, g in frame_groups.items()},
            end_monotonic_ns=kept_states[-1]["monotonic_ns"], end_data_monotonic_ns=end_ns, processing=dict(annotation, source_hashes=hashes,
            original_samples=metadata["samples"], original_camera_frames=metadata["camera_frames"],
            original_end_monotonic_ns=metadata["end_monotonic_ns"]))
        write_json(trim_dir / "metadata.json", derived_metadata)
        write_json(output / f"processing/source_metadata/{identifier}.json", metadata)
        # Start the regular grid only once every camera is available. A full next
        # state must exist inside the crop for every action, including the last.
        first_available = max([data_timestamp(kept_states[0])] +
                              [int(t[0]) for t in camera_times.values()])
        grid_start = origin + math.ceil((first_available - origin) / period_ns) * period_ns
        last_state_ns = data_timestamp(kept_states[-1])
        grid_stop = min(end_ns, last_state_ns) - period_ns
        ticks = list(range(grid_start, grid_stop + 1, period_ns))
        assert len(ticks) > 10
        local_state_times = np.asarray([data_timestamp(s) for s in kept_states], dtype=np.int64)
        alignments = []
        max_state_error = max_camera_error = max_action_error = 0.0
        selected_camera_ids = {c: [] for c in CAMERAS}
        for frame_index, tick in enumerate(ticks):
            state = kept_states[nearest_index(local_state_times, tick)]
            target = kept_states[nearest_index(local_state_times, tick + period_ns)]
            assert data_timestamp(target) > data_timestamp(state)
            row = dict(task=spec["task"], **{
                "observation.state": vector(state), "action": vector(target),
                "observation.velocity": vector(state, "velocity_rad_s", "velocity"),
                "observation.effort": vector(state, "measured_torque_nm", "torque")})
            alignment = dict(index=global_index, episode_index=episode_index, frame_index=frame_index,
                source_episode_id=identifier, source_grid_monotonic_ns=tick,
                source_episode_time_s=(tick - origin) / 1e9,
                state_sequence=state["sequence"], state_monotonic_ns=state["monotonic_ns"],
                action_sequence=target["sequence"], action_monotonic_ns=target["monotonic_ns"],
                state_data_monotonic_ns=data_timestamp(state), action_data_monotonic_ns=data_timestamp(target),
                state_offset_s=(data_timestamp(state) - tick) / 1e9,
                action_offset_s=(data_timestamp(target) - tick - period_ns) / 1e9)
            max_state_error = max(max_state_error, abs(alignment["state_offset_s"]))
            max_action_error = max(max_action_error, abs(alignment["action_offset_s"]))
            for camera in CAMERAS:
                frame = frame_groups[camera][nearest_index(camera_times[camera], tick)]
                image_path = source / frame["path"]
                row[f"observation.images.{camera}"] = prepare_image(
                    image_path, spec["image_crops_xyxy"][camera], spec["image_size"])
                offset = (data_timestamp(frame) - tick) / 1e9
                max_camera_error = max(max_camera_error, abs(offset))
                selected_camera_ids[camera].append(frame["frame_index"])
                alignment.update({f"{camera}.source_path": frame["path"],
                    f"{camera}.source_monotonic_ns": frame["monotonic_ns"],
                    f"{camera}.source_data_monotonic_ns": data_timestamp(frame),
                    f"{camera}.source_frame_index": frame["frame_index"],
                    f"{camera}.offset_s": offset,
                    f"{camera}.source_sha256": sha256(image_path)})
            assert max_state_error <= spec["max_state_alignment_s"]
            assert max_action_error <= spec["max_state_alignment_s"]
            assert max_camera_error <= spec["max_camera_alignment_s"]
            assert all(np.isfinite(value).all() for key, value in row.items() if key != "task")
            dataset.add_frame(row)
            alignments.append(alignment)
            global_index += 1
        dataset.save_episode()
        pq.write_table(pa.Table.from_pylist(alignments),
                       output / f"processing/alignment/episode_{episode_index:06d}.parquet", compression="zstd")
        # Publish the trimmed original-resolution/full-rate data as separate
        # archives so the main LeRobot download remains small and easy to train.
        archive_path = output / f"trimmed_full_rate/{identifier}.tar.gz"
        with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
            archive.add(trim_dir, arcname=identifier)
        actual_hz = (len(states) - 1) * 1e9 / (state_times[-1] - state_times[0])
        episode_result = dict(annotation, episode_index=episode_index,
            source_hashes=hashes, original_duration_s=states[-1].get("data_time_s", states[-1]["episode_time_s"]),
            original_wall_duration_s=states[-1]["episode_time_s"],
            removed_tail_s=states[-1].get("data_time_s", states[-1]["episode_time_s"]) - annotation["end_s"],
            original_states=len(states), trimmed_states=len(kept_states),
            original_camera_frames=metadata["camera_frames"], trimmed_camera_frames=derived_metadata["camera_frames"],
            measured_source_state_hz=actual_hz, exported_observations=len(ticks),
            first_source_grid_s=(ticks[0] - origin) / 1e9,
            last_source_observation_s=(ticks[-1] - origin) / 1e9,
            last_source_action_s=(ticks[-1] + period_ns - origin) / 1e9,
            max_state_alignment_s=max_state_error, max_action_alignment_s=max_action_error,
            max_camera_alignment_s=max_camera_error,
            repeated_camera_frames={c: len(ids) - len(set(ids)) for c, ids in selected_camera_ids.items()},
            trimmed_archive_sha256=sha256(archive_path))
        manifest["episodes"].append(episode_result)
        total_source_states += len(states)
        total_trimmed_states += len(kept_states)
        total_source_images += len(frames)
        total_trimmed_images += len(kept_frames)
        print(f"{episode_index + 1}/{len(spec['episodes'])} {identifier}: "
              f"trim at {annotation['end_s']:.2f}s, {len(ticks)} observations", flush=True)
    dataset.finalize()
    manifest["totals"] = dict(episodes=len(spec["episodes"]), observations=global_index,
        original_states=total_source_states, trimmed_states=total_trimmed_states,
        original_images=total_source_images, trimmed_images=total_trimmed_images,
        removed_states=total_source_states - total_trimmed_states,
        removed_images=total_source_images - total_trimmed_images,
        removed_tail_seconds=sum(e["removed_tail_s"] for e in manifest["episodes"]))
    write_json(output / "processing/manifest.json", manifest)
    shutil.copy2(args.annotations, output / "processing/annotations.json")
    print(json.dumps(manifest["totals"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/kept"))
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trimmed-output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    export(parser.parse_args())
