"""Deployment contracts for full-model joint VLA and legacy LoRA checkpoints."""
import io
import json
import tarfile
import zipfile

import numpy as np
import pytest

from rollout import (
    FULL_CHECKPOINT_FILES, JOINT_UNNORM_KEY, REQUIRED_CHECKPOINT_FILES,
    UNNORM_KEY, apply_joint_action, checkpoint_contract, current_state,
    materialize_checkpoint, normalize_proprio,
)


def archive_files(full=True):
    names = FULL_CHECKPOINT_FILES if full else REQUIRED_CHECKPOINT_FILES
    contents = {name: b"{}" for name in names}
    key = JOINT_UNNORM_KEY if full else UNNORM_KEY
    contents["dataset_statistics.json"] = json.dumps({key: {}}).encode()
    if full:
        contents["model.safetensors"] = b"model-weights"
        contents["tokenizer.json"] = b"{}"
        contents["validation_split.json"] = json.dumps({
            "dataset_name": key, "sample_hz": 30,
            "action_contract": "absolute_next_joint_targets_radians_gripper_metres",
        }).encode()
    contents["training_state--latest_checkpoint.pt"] = b"optimizer-not-needed"
    contents["processing_prismatic.py"] = b"downloaded-code-not-needed"
    return contents


@pytest.mark.parametrize("full", [True, False])
@pytest.mark.parametrize("kind", ["zip", "tar.gz"])
def test_extract_inference_only_and_cache(tmp_path, full, kind):
    source = tmp_path / f"checkpoint.{kind}"
    contents = archive_files(full)
    if kind == "zip":
        with zipfile.ZipFile(source, "w") as archive:
            for name, data in contents.items():
                archive.writestr("run/" + name, data)
    else:
        with tarfile.open(source, "w:gz") as archive:
            for name, data in contents.items():
                info = tarfile.TarInfo("run/" + name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    checkpoint = materialize_checkpoint(source, tmp_path / "cache")
    assert not (checkpoint / "training_state--latest_checkpoint.pt").exists()
    assert not (checkpoint / "processing_prismatic.py").exists()
    assert materialize_checkpoint(source, tmp_path / "cache") == checkpoint
    contract = checkpoint_contract(checkpoint)
    assert contract.joint_control == full
    assert contract.hz == (30 if full else 10)
    if full:
        assert (checkpoint / "model.safetensors").read_bytes() == b"model-weights"


def test_split_drive_download_requires_explicit_weights(tmp_path):
    source = tmp_path / "drive.zip"
    contents = archive_files()
    contents.pop("model.safetensors")
    with zipfile.ZipFile(source, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)  # Root-level archives work too.
    with pytest.raises(SystemExit, match="--model-weights"):
        materialize_checkpoint(source, tmp_path / "cache")
    weights = tmp_path / "model-001.safetensors"
    weights.write_bytes(b"separate-weights")
    checkpoint = materialize_checkpoint(source, tmp_path / "cache", weights)
    weights.unlink()
    assert (checkpoint / "model.safetensors").read_bytes() == b"separate-weights"
    assert materialize_checkpoint(checkpoint, tmp_path / "cache") == checkpoint


def test_nested_checkpoint_is_validated(tmp_path):
    nested = tmp_path / "run"
    nested.mkdir()
    (nested / "dataset_statistics.json").write_text("{}")
    with pytest.raises(SystemExit, match="missing"):
        materialize_checkpoint(tmp_path, tmp_path / "cache")


def test_reject_wrong_joint_contract(tmp_path):
    (tmp_path / "dataset_statistics.json").write_text(json.dumps({JOINT_UNNORM_KEY: {}}))
    (tmp_path / "validation_split.json").write_text(json.dumps({
        "dataset_name": JOINT_UNNORM_KEY, "action_contract": "eef_deltas",
    }))
    with pytest.raises(SystemExit, match="action contract"):
        checkpoint_contract(tmp_path)


def test_normalization_matches_rlds_constant_padding_and_masks():
    stats = {"q01": [0, 0, 0], "q99": [2, 0, 2],
             "min": [0, 0, 0], "max": [2, 0, 2], "mask": [True, True, False]}
    # Padded dimensions map to zero, masked-out values retain physical units.
    np.testing.assert_allclose(normalize_proprio(np.array([3, 0, 4]), stats), [1, 0, 4])


def test_absolute_joint_targets_and_gripper_metres_in_real_sim():
    pytest.importorskip("mujoco")
    from sim.panthera_env import PantheraSim
    from teleop.dataset_contract import PhysicsClock
    sim = PantheraSim()
    sim.reset(randomize=False)
    target = sim.q.copy()
    action = np.r_[target, .02]
    apply_joint_action(sim, action)
    # Applying twice must not integrate an absolute joint target into a delta.
    apply_joint_action(sim, action)
    np.testing.assert_allclose(sim.data.ctrl[:6], target)
    assert sim.data.ctrl[sim.grip_act] == pytest.approx(.02)
    state = current_state(sim, joint_control=True)
    np.testing.assert_allclose(state[:6], sim.q)
    np.testing.assert_allclose(state[6:], [0, .02])
    clock = PhysicsClock(30, sim.dt)
    for _ in range(30):
        sim.step(clock.next_steps())
    assert sim.data.time == pytest.approx(1.0, abs=sim.dt)
    with pytest.raises(RuntimeError, match="invalid action"):
        apply_joint_action(sim, np.full(7, np.nan))
