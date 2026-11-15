#!/usr/bin/env python3
"""Build and consume a decoded image cache for a local LeRobot dataset.

LeRobot's embedded image columns are convenient, but random training samples
must extract and decode every image from Parquet.  This module stores the exact
decoded uint8 tensors in a memory-mapped NPY file.  The operating system pages
only the sampled frames into RAM, and DataLoader workers share the same file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torchvision.io import ImageReadMode, decode_image

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


CACHE_FILENAME = "decoded_images.uint8.npy"
MANIFEST_FILENAME = "decoded_images.uint8.json"


def default_cache_path(dataset_root: Path) -> Path:
    return dataset_root / "cache" / CACHE_FILENAME


def _manifest_path(cache_path: Path) -> Path:
    return cache_path.with_name(MANIFEST_FILENAME)


def _data_files(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "data").glob("**/*.parquet"))


def _dataset_signature(dataset_root: Path) -> str:
    """Fingerprint the metadata and identity of every Parquet shard."""
    digest = hashlib.sha256((dataset_root / "meta" / "info.json").read_bytes())
    for path in _data_files(dataset_root):
        stat = path.stat()
        digest.update(str(path.relative_to(dataset_root)).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _decode_parquet_file(
    parquet_path: str,
    cache_path: str,
    camera_keys: list[str],
) -> tuple[str, int]:
    """Decode one shard directly into its disjoint rows in the shared mmap."""
    torch.set_num_threads(1)
    warnings.filterwarnings(
        "ignore",
        message="The given buffer is not writable.*",
        category=UserWarning,
    )
    table = pq.read_table(parquet_path, columns=["index", *camera_keys])
    indices = table["index"].to_numpy(zero_copy_only=False)
    cache = np.lib.format.open_memmap(cache_path, mode="r+")

    for camera_index, key in enumerate(camera_keys):
        values = table[key].to_pylist()
        for absolute_index, value in zip(indices, values, strict=True):
            encoded = value["bytes"]
            if encoded is None:
                encoded = Path(value["path"]).read_bytes()
            # frombuffer is zero-copy.  decode_image only reads the encoded data.
            encoded_tensor = torch.frombuffer(encoded, dtype=torch.uint8)
            image = decode_image(encoded_tensor, mode=ImageReadMode.RGB)
            cache[int(absolute_index), camera_index] = image.numpy()

    cache.flush()
    return parquet_path, len(indices)


def build_image_cache(
    dataset_root: Path,
    cache_path: Path | None = None,
    workers: int | None = None,
) -> Path:
    dataset_root = dataset_root.resolve()
    cache_path = (cache_path or default_cache_path(dataset_root)).resolve()
    metadata = LeRobotDatasetMetadata("local/image_cache", root=dataset_root)
    camera_keys = list(metadata.camera_keys)
    if not camera_keys:
        raise ValueError(f"dataset has no image features: {dataset_root}")

    image_shapes = [tuple(metadata.features[key]["shape"]) for key in camera_keys]
    if any(shape != image_shapes[0] for shape in image_shapes):
        raise ValueError(f"all cameras must have the same shape, got {image_shapes}")
    height, width, channels = image_shapes[0]
    if channels != 3:
        raise ValueError(f"only RGB image caches are supported, got {image_shapes[0]}")

    files = _data_files(dataset_root)
    if not files:
        raise FileNotFoundError(f"no Parquet data files under {dataset_root / 'data'}")
    workers = workers or min(8, os.cpu_count() or 1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f"{cache_path.name}.tmp")
    temporary_manifest = _manifest_path(cache_path).with_suffix(".json.tmp")
    shape = (metadata.total_frames, len(camera_keys), channels, height, width)

    if temporary_path.exists():
        temporary_path.unlink()
    cache = np.lib.format.open_memmap(temporary_path, mode="w+", dtype=np.uint8, shape=shape)
    cache.flush()
    del cache

    print(
        f"building {cache_path} ({np.prod(shape) / 1024**3:.1f} GiB) "
        f"with {workers} workers",
        flush=True,
    )
    completed_rows = 0
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _decode_parquet_file,
                    str(path),
                    str(temporary_path),
                    camera_keys,
                )
                for path in files
            ]
            for completed_files, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                path, rows = future.result()
                completed_rows += rows
                print(
                    f"[{completed_files:03d}/{len(files):03d}] "
                    f"{Path(path).name}: {completed_rows}/{metadata.total_frames} frames",
                    flush=True,
                )
        if completed_rows != metadata.total_frames:
            raise RuntimeError(
                f"decoded {completed_rows} rows, expected {metadata.total_frames}"
            )
        manifest: dict[str, Any] = {
            "format_version": 1,
            "dataset_signature": _dataset_signature(dataset_root),
            "camera_keys": camera_keys,
            "shape": list(shape),
            "dtype": "uint8",
        }
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary_path.replace(cache_path)
        temporary_manifest.replace(_manifest_path(cache_path))
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise

    print(f"wrote {cache_path}", flush=True)
    return cache_path


def validate_image_cache(dataset_root: Path, cache_path: Path) -> dict[str, Any]:
    manifest_path = _manifest_path(cache_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing image-cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset_signature") != _dataset_signature(dataset_root):
        raise ValueError(
            f"image cache does not match the current dataset: {cache_path}; rebuild it"
        )
    cache = np.load(cache_path, mmap_mode="r")
    if list(cache.shape) != manifest.get("shape") or str(cache.dtype) != manifest.get("dtype"):
        raise ValueError(f"image cache shape or dtype does not match its manifest: {cache_path}")
    return manifest


class CachedLeRobotDataset(LeRobotDataset):
    """LeRobotDataset that serves image tensors from a decoded mmap cache."""

    def __init__(self, *args: Any, image_cache: Path | None = None, **kwargs: Any) -> None:
        # LeRobot iterates a formatted index column inside its constructor.
        # Hugging Face custom transforms receive every column, so leaving
        # images present there unnecessarily decodes the entire dataset.
        self._requested_image_cache = image_cache
        self._initializing = True
        try:
            super().__init__(*args, **kwargs)
        finally:
            self._initializing = False
        # Restore normal sample decoding after LeRobot builds its numeric index.
        self.hf_dataset.set_format(**self._sample_format)
        self._decoded_image_cache: np.memmap | None = None
        self._cached_camera_keys: list[str] = []
        if image_cache is None:
            return

        dataset_root = Path(self.root).resolve()
        image_cache = image_cache.resolve()
        manifest = validate_image_cache(dataset_root, image_cache)
        self._cached_camera_keys = manifest["camera_keys"]
        if self._cached_camera_keys != list(self.meta.camera_keys):
            raise ValueError("image cache camera keys do not match dataset metadata")
        # Copy-on-write makes NumPy expose writable views to torch without ever
        # modifying the on-disk cache.  Float conversion below creates the only
        # per-sample copy.
        self._decoded_image_cache = np.load(image_cache, mmap_mode="c")

    def load_hf_dataset(self):
        dataset = super().load_hf_dataset()
        if self._requested_image_cache is not None:
            dataset = dataset.remove_columns(list(self.meta.camera_keys))
        if getattr(self, "_initializing", False):
            self._sample_format = dataset.format
            dataset = dataset.with_format(None)
        return dataset

    @property
    def uses_image_cache(self) -> bool:
        return self._decoded_image_cache is not None

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        if self._decoded_image_cache is None:
            return item
        absolute_index = int(item["index"])
        for camera_index, key in enumerate(self._cached_camera_keys):
            image = torch.from_numpy(self._decoded_image_cache[absolute_index, camera_index])
            item[key] = image.to(dtype=torch.float32).div_(255)
        return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    build_image_cache(args.dataset, args.output, args.workers)


if __name__ == "__main__":
    main()
