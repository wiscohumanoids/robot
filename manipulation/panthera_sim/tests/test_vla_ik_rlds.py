"""Round-trip IK frames through real Parquet and TFDS, including shard boundaries."""
import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

tfds = pytest.importorskip("tensorflow_datasets")
pytest.importorskip("tensorflow")
from teleop.build_panthera_ik_rlds import (  # noqa: E402
    CAMERAS, PantheraIkThreeBlock, dataset_info, iter_episodes,
)


@pytest.fixture
def ik_source(tmp_path):
    root = tmp_path / "source"
    (root / "data").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "COMPLETE").touch()
    lengths = [3, 70, 4]
    info = dict(codebase_version="v3.0", fps=30, total_episodes=3,
                total_frames=sum(lengths), features={
                    **{key: {"shape": [256, 256, 3]} for key in CAMERAS},
                    **{key: {"shape": [7]} for key in ("observation.state", "action")}})
    (root / "meta/info.json").write_text(json.dumps(info))
    rows = []
    for episode, length in enumerate(lengths):
        for frame in range(length):
            buf = io.BytesIO()
            Image.new("RGB", (256, 256), (episode * 80, frame, 127)).save(buf, format="JPEG")
            # Distinct labels catch a second shift, delta conversion or gripper scaling.
            rows.append(dict(episode_index=episode, frame_index=frame, index=len(rows),
                             timestamp=np.float32(frame / 30).item(), **{
                                 "observation.state": [float(episode + frame / 100)] * 6 + [0.012],
                                 "action": [float(episode - frame / 50)] * 6 + [0.031],
                                 **{key: {"bytes": buf.getvalue(), "path": None} for key in CAMERAS}}))
    # Both shard boundaries and PyArrow's 64-row batch boundary cut an episode.
    for shard, part in enumerate((rows[:5], rows[5:])):
        pq.write_table(pa.Table.from_pylist(part), root / "data" / f"file-{shard:03d}.parquet")
    return root, rows


def test_rlds_roundtrip_keeps_images_actions_timing_and_disjoint_splits(ik_source, tmp_path):
    root, rows = ik_source
    builder = PantheraIkThreeBlock(dataset_dir=root, data_dir=tmp_path / "rlds",
                                  val_fraction=1 / 3, split_seed=9)
    builder.download_and_prepare()
    assert builder.info.splits["train"].num_examples == 2
    assert builder.info.splits["val"].num_examples == 1
    seen = {}
    total = 0
    for split in ("train", "val"):
        dataset = builder.as_dataset(split=split, decoders={"steps": {"observation": {
            "image": tfds.decode.SkipDecoding(), "wrist_image": tfds.decode.SkipDecoding()}}})
        for episode in dataset:
            episode_id = int(episode["episode_metadata"]["episode_index"])
            assert episode_id not in seen
            seen[episode_id] = split
            expected = [row for row in rows if row["episode_index"] == episode_id]
            actual = list(tfds.as_numpy(episode["steps"]))
            assert len(actual) == len(expected)
            for index, (step, row) in enumerate(zip(actual, expected)):
                np.testing.assert_array_equal(step["action"], np.asarray(row["action"], dtype=np.float32))
                np.testing.assert_array_equal(step["observation"]["state"],
                                              np.insert(np.asarray(row["observation.state"], dtype=np.float32), 6, 0))
                assert step["observation"]["image"] == row[CAMERAS[0]]["bytes"]
                assert step["observation"]["wrist_image"] == row[CAMERAS[1]]["bytes"]
                assert step["is_first"] == (index == 0)
                assert step["is_last"] == step["is_terminal"] == (index == len(expected) - 1)
            total += len(actual)
    assert total == len(rows)
    assert {key for key, split in seen.items() if split == "val"} == builder.val_episodes


def test_rejects_resampling_or_missing_frames(ik_source):
    root, rows = ik_source
    with pytest.raises(ValueError, match="native 30 Hz"):
        dataset_info(root, 10)
    rows[7]["frame_index"] += 1
    for path in (root / "data").glob("*.parquet"):
        path.unlink()
    pq.write_table(pa.Table.from_pylist(rows), root / "data/file.parquet")
    with pytest.raises(ValueError, match="Missing or reordered frames"):
        list(iter_episodes(root, dataset_info(root)))
