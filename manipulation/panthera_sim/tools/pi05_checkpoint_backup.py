"""Keep one complete, resumable checkpoint on a mounted Google Drive.

No Drive API credentials are handled here. The notebook mounts Drive before
installation and passes this helper a directory on the mounted filesystem.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

FORMAT = "pi05-drive-checkpoint-v1"
MANIFEST = "BACKUP_MANIFEST.json"


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = hashlib.sha256()
    size = 0
    with source.open("rb") as reader, destination.open("wb") as writer:
        for block in iter(lambda: reader.read(8 * 1024**2), b""):
            writer.write(block)
            result.update(block)
            size += len(block)
    if size != source.stat().st_size or size != destination.stat().st_size:
        raise IOError(f"Incomplete checkpoint copy: {source.name}")
    return {"size": size, "sha256": result.hexdigest()}


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def read_manifest(directory):
    manifest = json.loads((directory / MANIFEST).read_text())
    if manifest["format"] != FORMAT:
        raise ValueError("Unrecognized checkpoint backup format")
    for name in manifest["files"]:
        p = Path(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"Invalid backup file path: {name}")
    return manifest


def backup_checkpoint(checkpoint, backup_root):
    checkpoint, backup_root = Path(checkpoint), Path(backup_root)
    if not checkpoint.name.isdecimal() or not (checkpoint / "PI05_COMPLETE").is_file():
        raise ValueError("Only a completed numbered pi05 checkpoint can be backed up")
    backup_root.mkdir(parents=True, exist_ok=True)
    destination = backup_root / checkpoint.name
    sources = sorted(p for p in checkpoint.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in checkpoint.rglob("*")):
        raise ValueError("Checkpoint backup does not follow symlinks")
    if destination.exists():
        manifest = read_manifest(destination)
        current = {str(p.relative_to(checkpoint)): {"size": p.stat().st_size, "sha256": sha256(p)}
                   for p in sources}
        if manifest["files"] != current:
            raise ValueError(f"A different checkpoint already exists at {destination}")
    else:
        staging = Path(tempfile.mkdtemp(prefix=f".upload-{checkpoint.name}-", dir=backup_root))
        try:
            files = {}
            for source in sources:
                name = str(source.relative_to(checkpoint))
                print(f"Drive backup: {name} ({source.stat().st_size / 2**30:.2f} GiB)", flush=True)
                files[name] = copy_file(source, staging / name)
            manifest = {"format": FORMAT, "step": int(checkpoint.name), "files": files,
                        "includes_optimizer": (staging / "training_state").is_dir()}
            write_json(staging / MANIFEST, manifest)
            # The previous backup and pointer remain untouched until every file
            # has copied and the new manifest has closed successfully.
            staging.rename(destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    write_json(backup_root / "latest.json", {"format": FORMAT, "checkpoint": destination.name})
    # Prune only backups made by this helper, after publishing the replacement.
    for old in backup_root.iterdir():
        if old == destination or old.is_symlink() or not old.is_dir() or not old.name.isdecimal():
            continue
        try:
            old_manifest = read_manifest(old)
        except (OSError, ValueError, KeyError):
            continue
        if old_manifest["format"] == FORMAT:
            shutil.rmtree(old)
    result = {"path": str(destination), "step": int(checkpoint.name),
              "gib": sum(v["size"] for v in manifest["files"].values()) / 2**30,
              "includes_optimizer": manifest["includes_optimizer"]}
    print("Drive backup complete:", json.dumps(result), flush=True)
    return result


def restore_latest_checkpoint(backup_root, restore_root):
    backup_root, restore_root = Path(backup_root), Path(restore_root)
    pointer = json.loads((backup_root / "latest.json").read_text())
    name = pointer["checkpoint"]
    if pointer["format"] != FORMAT or not isinstance(name, str) or not name.isdecimal():
        raise ValueError("Invalid latest checkpoint pointer")
    source = backup_root / name
    manifest = read_manifest(source)
    if not manifest["includes_optimizer"]:
        raise ValueError("This backup lacks the optimizer state needed to resume training")
    restore_root.mkdir(parents=True, exist_ok=True)
    destination = restore_root / name
    if destination.exists():
        # Idempotent reruns may reuse a verified local restore, never overwrite it.
        for filename, expected in manifest["files"].items():
            path = destination / filename
            if not path.is_file() or path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
                raise ValueError(f"Existing local restore differs: {path}")
    else:
        staging = Path(tempfile.mkdtemp(prefix=f".restore-{name}-", dir=restore_root))
        try:
            for filename, expected in manifest["files"].items():
                actual = copy_file(source / filename, staging / filename)
                if actual != expected:
                    raise ValueError(f"Drive checkpoint checksum mismatch: {filename}")
            staging.rename(destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return destination / "pretrained_model"
