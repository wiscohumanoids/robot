"""pi05_full's checks against stand-in LeRobot trainer functions (the pinned trainer itself needs a GPU and the
pinned LeRobot commit): profiles, the episode split and train-only statistics, the export manifest, the resume step,
the model checks and the deployment contract written into checkpoints.

    python -m pytest manipulation/finetune -q
"""
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi05_full  # noqa: E402


# ---- stand-ins for the LeRobot pieces install_guards wraps -------------------------------------------------------

class Feature:
    """Stand-in for lerobot PolicyFeature."""
    def __init__(self, type=None, shape=None):
        """Store the fields."""
        self.type, self.shape = type, shape


class FakeDataset:
    """Stand-in for a LeRobot dataset: episodes, frames and one sample."""
    def __init__(self, episodes, frames, state_dim, task):
        """Store the fields."""
        self.episodes, self.num_episodes, self.num_frames, self.fps = episodes, len(episodes), frames, 30
        self.meta = types.SimpleNamespace(stats={})
        self._sample = {"observation.state": np.zeros(state_dim), "task": task}

    def __getitem__(self, i):
        """The one sample, whatever the index."""
        return self._sample


@pytest.fixture
def fake_lerobot(monkeypatch):
    """Install fake lerobot modules; returns the fake trainer module (its functions are set per test)."""
    trainer = types.ModuleType("lerobot.scripts.lerobot_train")
    trainer.saved = []
    trainer.make_optimizer_and_scheduler = lambda cfg, policy: (None, None)
    trainer.make_pre_post_processors = lambda *a, **k: k
    trainer.save_checkpoint = lambda *a, **k: trainer.saved.append(k["checkpoint_dir"])
    trainer.update_last_checkpoint = lambda c: None
    trainer.make_dataloaders = lambda cfg, ds, ev, step, pd: (None, None)
    pi05 = types.ModuleType("lerobot.policies.pi05.modeling_pi05")
    pi05.PI05Policy = type("PI05Policy", (), {})
    cfg_types = types.ModuleType("lerobot.configs.types")
    cfg_types.FeatureType = types.SimpleNamespace(VISUAL="VISUAL", STATE="STATE")
    cfg_types.PolicyFeature = Feature
    mods = {"lerobot": types.ModuleType("lerobot"), "lerobot.scripts": types.ModuleType("lerobot.scripts"),
            "lerobot.scripts.lerobot_train": trainer, "lerobot.policies": types.ModuleType("lerobot.policies"),
            "lerobot.policies.pi05": types.ModuleType("lerobot.policies.pi05"),
            "lerobot.policies.pi05.modeling_pi05": pi05, "lerobot.configs": types.ModuleType("lerobot.configs"),
            "lerobot.configs.types": cfg_types}
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        mods["torch"] = torch
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    return trainer


def make_cfg(root, output, resume=False, checkpoint=None, repo="local/test"):
    """A minimal training config namespace."""
    return types.SimpleNamespace(
        peft=None, policy=types.SimpleNamespace(type="pi05", input_features=None),
        accelerator=types.SimpleNamespace(gradient_accumulation=types.SimpleNamespace(steps=1)),
        dataset=types.SimpleNamespace(root=str(root), repo_id=repo, revision=None),
        output_dir=str(output), resume=resume, checkpoint_path=checkpoint, batch_size=8)


def write_parquet(root, episodes, frames_per_episode, dim):
    """data/chunk-000/file.parquet with episode_index, observation.state and action (value = episode index)."""
    pa = pytest.importorskip("pyarrow")          # here, not at import: a missing pyarrow skips only these tests
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [(e, [float(e)] * dim) for e in episodes for _ in range(frames_per_episode)]
    table = pa.table({"episode_index": [r[0] for r in rows], "observation.state": [r[1] for r in rows],
                      "action": [r[1] for r in rows]})
    (root / "data/chunk-000").mkdir(parents=True)
    pq.write_table(table, root / "data/chunk-000/file.parquet")


# ---- profiles ---------------------------------------------------------------------------------------------------

def test_profiles_load_and_reject_other_methods():
    """Profiles load by name; a profile without pi05, or a missing one, is refused."""
    sim = pi05_full.load_profile("panthera_sim")
    assert sim["cameras"] == ["observation.images.shoulder", "observation.images.wrist"] and sim["state_dim"] == 7
    real = pi05_full.load_profile("panthera_bimanual")
    assert real["action_dim"] == 14 and real["pi05"]["require_manifest"]
    with pytest.raises(ValueError, match="does not list pi05"):
        pi05_full.load_profile("g1_dex1")
    with pytest.raises(FileNotFoundError):
        pi05_full.load_profile("no_such_arm")


def test_real_profile_matches_the_inference_contract():
    """The real-arm profile agrees with the inference server's contract."""
    contract = json.loads((Path(pi05_full.HERE).parent / "panthera_teleop/infra/inference/contract.json").read_text())
    real = pi05_full.load_profile("panthera_bimanual")
    assert real["fps"] == contract["fps"] and real["task"] == contract["task"]
    assert [c.split(".")[-1] for c in real["cameras"]] == list(contract["image_crops_xyxy"])


def test_split_args():
    """This script's flags are taken out; LeRobot's pass through."""
    assert pi05_full.split_args(["--profile=x", "--expect_resume_step=5", "--resume=true"]) == ("x", 5, ["--resume=true"])
    with pytest.raises(SystemExit):
        pi05_full.split_args(["--resume=true"])


# ---- dataset checks ---------------------------------------------------------------------------------------------

def test_sim_split_and_train_only_stats(tmp_path, fake_lerobot):
    """The episode split is enforced and statistics come from training episodes only."""
    profile = pi05_full.load_profile("panthera_sim")
    profile["pi05"].update(train_episodes=[0, 3], val_episodes=[3, 4])
    root = tmp_path / "ds"
    write_parquet(root, range(4), 2, 7)
    train, val = FakeDataset([0, 1, 2], 6, 7, profile["task"]), FakeDataset([3], 2, 7, profile["task"])
    fake_lerobot.make_train_eval_datasets = lambda cfg: (train, val)
    trainer = pi05_full.install_guards(profile)
    cfg = make_cfg(root, tmp_path / "out")
    trainer.make_train_eval_datasets(cfg)
    assert train.meta.stats["action"]["max"].tolist() == [2.0] * 7      # episode 3 excluded
    assert val.meta.stats is not None and json.loads((tmp_path / "out/training_stats.json").read_text())
    assert list(cfg.policy.input_features) == [*profile["cameras"], "observation.state"]

    bad = FakeDataset([0, 1], 4, 7, profile["task"])
    fake_lerobot.make_train_eval_datasets = lambda cfg: (bad, val)
    with pytest.raises(ValueError, match="Training episodes"):
        pi05_full.install_guards(profile).make_train_eval_datasets(make_cfg(root, tmp_path / "out2"))


def test_wrong_task_and_state_size(tmp_path, fake_lerobot):
    """A different task prompt or state size stops the run."""
    profile = pi05_full.load_profile("panthera_sim")
    profile["pi05"] = {}
    fake_lerobot.make_train_eval_datasets = lambda cfg: (FakeDataset([0], 1, 7, "something else"), None)
    with pytest.raises(ValueError, match="task prompt"):
        pi05_full.install_guards(profile).make_train_eval_datasets(make_cfg(tmp_path, tmp_path / "o"))
    fake_lerobot.make_train_eval_datasets = lambda cfg: (FakeDataset([0], 1, 14, profile["task"]), None)
    with pytest.raises(ValueError, match="state"):
        pi05_full.install_guards(profile).make_train_eval_datasets(make_cfg(tmp_path, tmp_path / "o"))


def write_export(root, frames=10, passed=True, repo="local/export"):
    """A minimal panthera_teleop export: manifest and verification report; returns the manifest hash."""
    (root / "processing").mkdir(parents=True)
    manifest = {"repo_id": repo, "episodes": [{"id": "a"}, {"id": "b"}], "totals": {"observations": frames}}
    (root / "processing/manifest.json").write_text(json.dumps(manifest))
    (root / "verification.json").write_text(json.dumps({"passed": passed}))
    return hashlib.sha256((root / "processing/manifest.json").read_bytes()).hexdigest()


def test_export_manifest_checks(tmp_path, fake_lerobot, monkeypatch):
    """Manifest hash, repo id, frame count, sampler coverage and verification reports are enforced."""
    profile = pi05_full.load_profile("panthera_bimanual")
    root = tmp_path / "export"
    digest = write_export(root)
    fake_lerobot.make_train_eval_datasets = lambda cfg: (FakeDataset([0, 1], 10, 14, profile["task"]), None)
    monkeypatch.setenv("PI05_TRAINING_MANIFEST_SHA256", digest)
    loader = types.SimpleNamespace(sampler=types.SimpleNamespace(indices=list(range(9))))   # one frame short
    fake_lerobot.make_dataloaders = lambda *a: (loader, None)
    trainer = pi05_full.install_guards(profile)
    trainer.make_train_eval_datasets(make_cfg(root, tmp_path / "out", repo="local/export"))
    audit = json.loads((tmp_path / "out/dataset_audit.json").read_text())
    assert audit["dataset_manifest_sha256"] == digest and audit["source_episode_ids"] == ["a", "b"]
    # the sampler must cover exactly the export's frames
    with pytest.raises(ValueError, match="sampler"):
        trainer.make_dataloaders(None, None, None, 0, None)

    monkeypatch.setenv("PI05_TRAINING_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="manifest differs"):
        pi05_full.install_guards(profile).make_train_eval_datasets(make_cfg(root, tmp_path / "o", repo="local/export"))
    monkeypatch.delenv("PI05_TRAINING_MANIFEST_SHA256")
    with pytest.raises(ValueError, match="repo_id"):
        pi05_full.install_guards(profile).make_train_eval_datasets(make_cfg(root, tmp_path / "o", repo="local/other"))
    fake_lerobot.make_train_eval_datasets = lambda cfg: (FakeDataset([0, 1], 9, 14, profile["task"]), None)
    with pytest.raises(ValueError, match="frames"):
        pi05_full.install_guards(profile).make_train_eval_datasets(make_cfg(root, tmp_path / "o", repo="local/export"))
    failed = tmp_path / "failed"
    write_export(failed, passed=False)
    with pytest.raises(ValueError, match="did not pass"):
        pi05_full.install_guards(profile).make_train_eval_datasets(make_cfg(failed, tmp_path / "o"))


def test_expect_resume_step(tmp_path, fake_lerobot):
    """Resuming from a checkpoint at another step, or without --resume, is refused."""
    profile = pi05_full.load_profile("panthera_sim")
    profile["pi05"] = {}
    ckpt = tmp_path / "ckpt"
    (ckpt / "training_state").mkdir(parents=True)
    (ckpt / "PI05_COMPLETE").write_text("")
    (ckpt / "training_state/training_step.json").write_text(json.dumps({"step": 100}))
    fake_lerobot.make_train_eval_datasets = lambda cfg: (FakeDataset([0], 1, 7, profile["task"]), None)
    pi05_full.install_guards(profile, 100).make_train_eval_datasets(make_cfg(tmp_path, tmp_path / "o", True, str(ckpt)))
    with pytest.raises(ValueError, match="step 100, expected 200"):
        pi05_full.install_guards(profile, 200).make_train_eval_datasets(
            make_cfg(tmp_path, tmp_path / "o", True, str(ckpt)))
    with pytest.raises(ValueError, match="needs --resume"):
        pi05_full.install_guards(profile, 100).make_train_eval_datasets(make_cfg(tmp_path, tmp_path / "o"))


# ---- model checks, checkpoints ----------------------------------------------------------------------------------

def fake_policy(cameras, state_dim, action_dim, frozen=False):
    """A policy namespace with the given cameras and sizes."""
    param = types.SimpleNamespace(requires_grad=not frozen, numel=lambda: 4)
    config = types.SimpleNamespace(type="pi05", freeze_vision_encoder=False, train_expert_only=False,
                                   image_features=cameras, use_relative_actions=False,
                                   input_features={"observation.state": Feature(shape=(state_dim,))},
                                   output_features={"action": Feature(shape=(action_dim,))})
    return types.SimpleNamespace(config=config, named_parameters=lambda: [("w", param)], parameters=lambda: [param])


def test_check_full_model():
    """Camera, size and frozen-parameter mismatches are refused."""
    profile = pi05_full.load_profile("panthera_bimanual")
    assert pi05_full.check_full_model(fake_policy(profile["cameras"], 14, 14), profile)["frozen_parameters"] == 0
    with pytest.raises(ValueError, match="Camera"):
        pi05_full.check_full_model(fake_policy(profile["cameras"][:2], 14, 14), profile)
    with pytest.raises(ValueError, match="14-D"):
        pi05_full.check_full_model(fake_policy(profile["cameras"], 7, 7), profile)
    with pytest.raises(ValueError, match="Frozen"):
        pi05_full.check_full_model(fake_policy(profile["cameras"], 14, 14, frozen=True), profile)


def test_checkpoint_gets_the_profile_contract_and_old_ones_are_pruned(tmp_path, fake_lerobot, monkeypatch):
    """Checkpoints carry the profile contract and only the newest is kept."""
    profile = pi05_full.load_profile("panthera_sim")
    profile["pi05"] = {}
    monkeypatch.setenv("PI05_KEEP_CHECKPOINTS", "1")
    fake_lerobot.make_train_eval_datasets = lambda cfg: (FakeDataset([0, 1], 4, 7, profile["task"]), None)
    trainer = pi05_full.install_guards(profile)
    trainer.make_train_eval_datasets(make_cfg(tmp_path, tmp_path / "out"))
    empty = types.SimpleNamespace(state_dict=lambda: {})
    for step in ("000100", "000200"):
        ckpt = tmp_path / "out/checkpoints" / step
        (ckpt / "pretrained_model").mkdir(parents=True)
        trainer.save_checkpoint(checkpoint_dir=ckpt, policy=empty, optimizer=empty)
    contract = json.loads((ckpt / "pretrained_model/deployment.json").read_text())
    assert contract["fps"] == 30 and contract["action_representation"] == "absolute"
    assert contract["cameras"] == profile["cameras"] and contract["profile"] == "panthera_sim"
    assert "simulation_dynamics" in contract and contract["training_episode_indices"] == [0, 1]
    assert not (tmp_path / "out/checkpoints/000100").exists() and (ckpt / "PI05_COMPLETE").is_file()
