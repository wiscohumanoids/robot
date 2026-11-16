"""Checkpoint backup replacement, failed-copy protection, and verified restore."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import pi05_checkpoint_backup as backup


def checkpoint(root, step):
    path = root / f"{step:06d}"
    (path / "pretrained_model").mkdir(parents=True)
    (path / "training_state").mkdir()
    (path / "pretrained_model/model.safetensors").write_bytes(f"model-{step}".encode())
    (path / "pretrained_model/train_config.json").write_text(json.dumps({"step": step}))
    (path / "training_state/optimizer_state.safetensors").write_bytes(f"optimizer-{step}".encode())
    (path / "PI05_COMPLETE").write_text("complete")
    return path


def test_drive_backup_keeps_latest_and_restores_optimizer(tmp_path):
    first = checkpoint(tmp_path / "local", 5000)
    second = checkpoint(tmp_path / "local", 10000)
    drive = tmp_path / "drive"
    backup.backup_checkpoint(first, drive)
    unrelated = drive / "keep_me"
    unrelated.mkdir()
    (drive / "004000").mkdir()  # An unmarked folder is not ours to prune.
    result = backup.backup_checkpoint(second, drive)
    assert result["includes_optimizer"] and result["step"] == 10000
    assert not (drive / first.name).exists()
    assert (drive / second.name).is_dir()
    assert unrelated.is_dir() and (drive / "004000").is_dir()
    assert json.loads((drive / "latest.json").read_text())["checkpoint"] == second.name
    restored = backup.restore_latest_checkpoint(drive, tmp_path / "restored")
    assert (restored / "model.safetensors").read_bytes() == b"model-10000"
    assert (restored.parent / "training_state/optimizer_state.safetensors").read_bytes() == b"optimizer-10000"
    assert backup.restore_latest_checkpoint(drive, tmp_path / "restored") == restored
    # Same source may be retried without making a second copy.
    assert backup.backup_checkpoint(second, drive)["step"] == 10000


def test_failed_drive_copy_preserves_previous_complete_backup(tmp_path):
    first = checkpoint(tmp_path / "local", 5000)
    second = checkpoint(tmp_path / "local", 10000)
    drive = tmp_path / "drive"
    backup.backup_checkpoint(first, drive)
    pointer = (drive / "latest.json").read_bytes()
    copy_file = backup.copy_file
    calls = 0

    def failed_copy(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("Drive disconnected")
        return copy_file(source, destination)

    with patch.object(backup, "copy_file", failed_copy), pytest.raises(OSError, match="disconnected"):
        backup.backup_checkpoint(second, drive)
    assert (drive / "latest.json").read_bytes() == pointer
    assert (drive / first.name / "pretrained_model/model.safetensors").read_bytes() == b"model-5000"
    assert not (drive / second.name).exists()
    assert not list(drive.glob(".upload-*"))
    assert (second / "pretrained_model/model.safetensors").exists()


def test_restore_rejects_corrupted_drive_copy(tmp_path):
    source = checkpoint(tmp_path / "local", 5000)
    drive = tmp_path / "drive"
    backup.backup_checkpoint(source, drive)
    (drive / source.name / "pretrained_model/model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        backup.restore_latest_checkpoint(drive, tmp_path / "restore")
    assert not (tmp_path / "restore" / source.name).exists()
