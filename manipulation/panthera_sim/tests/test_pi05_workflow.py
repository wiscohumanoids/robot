"""Task metric regressions and a reduced-size test of the real pi05 training code."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tools.evaluate_pi05 import StackScore, summarize, wilson_interval, wandb_metrics
from tools.train_pi05_full import prune_checkpoints, tensor_bytes

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_meshes_cover_xml_references(tmp_path):
    import xml.etree.ElementTree as ET
    from tools.bundle_pi05_runtime import simulator_meshes
    meshes = simulator_meshes(ROOT)
    referenced = [(ROOT / "sim/panthera" / node.attrib["file"]).resolve()
                  for node in ET.parse(ROOT / "sim/panthera/panthera.xml").iter("mesh")
                  if "file" in node.attrib]
    assert any("collision" in p.parts for p in referenced)
    assert set(referenced).issubset({p.resolve() for p in meshes})
    # Build only from the file list used by the bundler, in a fresh directory.
    # Loading from the original checkout previously concealed missing assets.
    import shutil
    for source in [ROOT / "sim/panthera/scene.xml", ROOT / "sim/panthera/panthera.xml", *meshes]:
        dest = tmp_path / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "sim/panthera/scene.xml"))
    assert model.nmesh == len(referenced)


def test_success_requires_order_support_release_and_continuous_hold():
    correct = np.array([[.4, 0, .1175], [.4, 0, .0725], [.4, 0, .1625]])
    for positions, holding in ((correct[[2, 1, 0]], [0, 0, 0]),
                               (correct + [0, 0, .1], [0, 0, 0]),
                               (correct, [0, 0, 1]),
                               (correct + [[.02, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0])):
        score = StackScore()
        for step in range(60):
            score.update(positions, holding, step)
        assert not score.success
    score = StackScore()
    for step in range(29):
        score.update(correct, [0, 0, 0], step)
    assert not score.success
    score.update(correct, [0, 0, 1], 29)
    assert score.stable_steps == 0
    for step in range(30, 60):
        score.update(correct, [0, 0, 0], step)
    assert score.success and score.first_success_step == 59


def test_failure_denominator_and_confidence_interval():
    records = [dict(success=True, grasped=True, lifted=True, two_stacked=True, error=None),
               dict(success=False, grasped=False, lifted=False, two_stacked=False, error="bad action")]
    result = summarize(records)
    assert result["success_rate"] == .5 and result["error_episodes"] == 1
    assert result["success_95pct_wilson_interval"][0] < .5 < result["success_95pct_wilson_interval"][1]
    assert wilson_interval(0, 50)[1] > 0 and wilson_interval(50, 50)[0] < 1
    metrics = wandb_metrics({"by_distribution": {"matched": result}})
    assert metrics["matched/success_rate"] == .5
    assert metrics["matched/error_episodes"] == 1
    assert metrics["matched/success_ci_low"] < .5 < metrics["matched/success_ci_high"]


def test_checkpoint_retention_preserves_latest_incomplete_and_unrelated(tmp_path):
    parent = tmp_path / "checkpoints"
    parent.mkdir()
    for name in ("000004", "005000", "010000"):
        path = parent / name
        path.mkdir()
        (path / "PI05_COMPLETE").write_text("complete")
    incomplete = parent / "015000"
    incomplete.mkdir()
    unrelated = parent / "best"
    unrelated.mkdir()
    last = parent / "last"
    last.symlink_to("010000")
    (parent / "external").symlink_to(unrelated, target_is_directory=True)
    assert prune_checkpoints(parent / "010000", 2) == ["000004"]
    assert (parent / "005000").exists() and last.resolve() == parent / "010000"
    assert prune_checkpoints(parent / "010000", 1) == ["005000"]
    assert incomplete.exists() and unrelated.exists() and (parent / "external").is_symlink()
    with pytest.raises(ValueError, match="complete"):
        prune_checkpoints(incomplete, 1)
    assert last.exists()
    with pytest.raises(ValueError, match="at least 1"):
        prune_checkpoints(parent / "010000", 0)


def test_checkpoint_size_estimator_handles_optimizer_state():
    torch = pytest.importorskip("torch")
    state = {"state": {0: {"exp_avg": torch.zeros(5), "exp_avg_sq": torch.zeros(5)}},
             "param_groups": [{"params": [0], "lr": .001}]}
    assert tensor_bytes(state) == 40


def test_notebook_python_and_no_saved_secrets_or_results():
    notebook = json.loads((ROOT / "notebooks/pi05_ik_three_block_full_finetune.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert not cell["outputs"] and cell["execution_count"] is None
        tree = ast.parse(cell["source"])
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "python":
                if node.args and isinstance(node.args[0], ast.Constant):
                    ast.parse(node.args[0].value)


def test_full_model_gradients_optimizer_and_strict_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    lerobot = pytest.importorskip("lerobot")
    if lerobot.__version__ != "0.6.2":
        pytest.skip("Run with the notebook's pinned LeRobot environment")
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import PolicyFeature, FeatureType
    from lerobot.policies.pi05 import modeling_pi05 as m
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.scripts import lerobot_train as trainer
    from tools.train_pi05_full import install_guards, strict_from_pretrained, check_full_model
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(2)
    original_vlm = m.PaliGemmaForConditionalGenerationWithPiGemma

    def small_vlm(config):
        config.vision_config.projection_dim = 32
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 4
        config.vision_config.patch_size = 14
        return original_vlm(config)

    cfg = PI05Config(device="cpu", dtype="float32", chunk_size=3, n_action_steps=2,
        image_resolution=(28, 28), gradient_checkpointing=True, push_to_hub=False,
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            **{f"observation.images.{camera}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28))
               for camera in ("shoulder", "wrist")}},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
    dataset_root = tmp_path / "dataset"
    (dataset_root / "data").mkdir(parents=True)
    # Held-out values must never enter the training normalization statistics.
    pq.write_table(pa.table({"episode_index": [0, 899, 900, 999],
        "action": [[1.] * 7, [2.] * 7, [1000.] * 7, [2000.] * 7],
        "observation.state": [[1.] * 7, [2.] * 7, [1000.] * 7, [2000.] * 7]}),
        dataset_root / "data/file.parquet")
    run_cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="test/ik3", root=str(dataset_root), eval_split=.1),
        policy=cfg, output_dir=tmp_path / "run", steps=4, eval_steps=4)
    run_cfg.validate()
    fake_datasets = lambda _: (SimpleNamespace(episodes=list(range(900)), meta=SimpleNamespace(stats={})),
                               SimpleNamespace(episodes=list(range(900, 1000)), meta=SimpleNamespace(stats={})))
    with patch.object(m, "get_gemma_config", lambda _: m.GemmaConfig(
            width=32, depth=2, mlp_dim=64, num_heads=8, num_kv_heads=1, head_dim=8)), \
         patch.object(m, "PaliGemmaForConditionalGenerationWithPiGemma", small_vlm), \
         patch.object(m.PI05Policy, "from_pretrained", m.PI05Policy.from_pretrained), \
         patch.object(trainer, "make_train_eval_datasets", fake_datasets), \
         patch.object(trainer, "make_optimizer_and_scheduler", trainer.make_optimizer_and_scheduler), \
         patch.object(trainer, "make_pre_post_processors", trainer.make_pre_post_processors), \
         patch.object(trainer, "save_checkpoint", trainer.save_checkpoint):
        guarded = install_guards()
        train, val = guarded.make_train_eval_datasets(run_cfg)
        np.testing.assert_allclose(train.meta.stats["action"]["mean"], 1.5)
        assert train.meta.stats["action"]["count"].tolist() == [2]
        policy = m.PI05Policy(cfg)
        report = check_full_model(policy)
        assert report["parameters"] == report["trainable_parameters"]
        next(policy.parameters()).requires_grad_(False)
        with pytest.raises(ValueError, match="Frozen"):
            check_full_model(policy)
        next(policy.parameters()).requires_grad_(True)
        optimizer, scheduler = guarded.make_optimizer_and_scheduler(run_cfg, policy)
        batch = {"observation.state": torch.rand(1, 7), "action": torch.rand(1, 3, 7),
            "observation.language.tokens": torch.tensor([[1, 3, 5, 7]]),
            "observation.language.attention_mask": torch.ones(1, 4, dtype=torch.bool),
            **{f"observation.images.{c}": torch.rand(1, 3, 28, 28) for c in ("shoulder", "wrist")}}
        for _ in range(2):
            policy.train()
            loss, _ = policy(batch)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        audit = json.loads((run_cfg.output_dir / "full_model_audit.json").read_text())
        assert len(audit["first_backward_gradient_norms"]) == 5
        checkpoint = tmp_path / "checkpoint"
        policy.save_pretrained(checkpoint)
        restored = strict_from_pretrained(m.PI05Policy, checkpoint, config=cfg)
        assert all(torch.equal(a, b) for a, b in zip(policy.parameters(), restored.parameters()))
        restored.eval()
        with torch.inference_mode():
            action = restored.select_action(batch)
        assert action.shape == (1, 7) and torch.isfinite(action).all()
        state = load_file(str(checkpoint / "model.safetensors"))
        del state["model.action_out_proj.weight"]
        save_file(state, str(checkpoint / "model.safetensors"))
        with pytest.raises(RuntimeError, match="Missing key"):
            strict_from_pretrained(m.PI05Policy, checkpoint, config=cfg)
