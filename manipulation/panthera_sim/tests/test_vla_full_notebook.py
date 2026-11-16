"""Exercise notebook patches against its pinned trainer without downloading models."""
import ast
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/robot_arm_learning_finetune_colab.ipynb"


def source(index):
    return "".join(json.loads(NOTEBOOK.read_text())["cells"][index]["source"])


def execute_nodes(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook trainer", "exec"), namespace)


@pytest.fixture
def patched_trainer(tmp_path):
    settings = {}
    exec(source(1), settings)
    checkout = ROOT / "VLA-Adapter"
    if not (checkout / ".git").exists():
        pytest.skip("Requires the notebook's local VLA-Adapter checkout")
    paths = ["vla-scripts/finetune.py", "prismatic/models/backbones/llm/base_llm.py",
             "prismatic/vla/datasets/rlds/dataset.py",
             *[f"prismatic/vla/datasets/rlds/oxe/{name}.py"
               for name in ("configs", "mixtures", "transforms", "materialize")]]
    for path in paths:
        result = subprocess.run(
            ["git", "-C", str(checkout), "show", f"{settings['VLA_COMMIT']}:{path}"],
            capture_output=True, text=True, check=True)
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.stdout)
    namespace = {"VLA_DIR": tmp_path}
    exec(source(7), namespace)
    first = {path: (tmp_path / path).read_text() for path in paths}
    exec(source(7), namespace)
    for path in paths:
        assert first[path] == (tmp_path / path).read_text(), path
        ast.parse(first[path])
    return ast.parse(first[paths[0]])


def test_notebook_code_and_training_command():
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert not cell.get("outputs")
    settings = {}
    exec(source(1), settings)
    settings.update({name: Path("/unused") for name in
                     ("VENV", "VLA_DIR", "MODEL_DIR", "RLDS_DIR", "OUTPUT_ROOT")})
    settings["RUN_ID"] = "test"
    assignment = next(node for node in ast.parse(source(17)).body
                      if isinstance(node, ast.Assign) and node.targets[0].id == "command")
    execute_nodes([assignment], settings)
    command = settings["command"]
    assert command[command.index("--use_lora") + 1] == "False"
    assert command[command.index("--use_fz") + 1] == "True"
    assert command[command.index("--dataset_name") + 1] == "panthera_ik_three_block"
    assert command[command.index("--balance_z_loss") + 1] == "False"
    assert settings["SAMPLE_HZ"] == 30
    assert settings["DATASET_REPO"] == "FoxNerdSaysMoo/panthera-ik-three-block-stack-30hz"
    assert "--lora_rank" not in command


def test_embedded_converter_matches_source():
    assignment = next(node for node in ast.parse(source(11)).body
                      if isinstance(node, ast.Assign) and node.targets[0].id == "IK_BUILDER_SOURCE")
    assert ast.literal_eval(assignment.value) == (ROOT / "teleop/build_panthera_ik_rlds.py").read_text()


@pytest.mark.parametrize("key,value", [
    ("dataset_repo", "FoxNerdSaysMoo/robot-arm-learning-data"),
    ("sample_hz", 10),
    ("action_contract", "relative_end_effector"),
])
def test_resume_rejects_changed_dataset_timing_or_actions(tmp_path, key, value):
    namespace = {"Path": Path, "json": json, "OUTPUT_ROOT": tmp_path,
                 "RUN_ID": "test", "RESUME_DIR": None}
    exec(source(1), namespace)
    nodes = ast.parse(source(17)).body
    start = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                 and node.targets[0].id == "validation_config")
    contract_nodes = nodes[start:start + 3]  # Config, path, and save/compare branch.
    execute_nodes(contract_nodes, namespace)
    namespace["RESUME_DIR"] = tmp_path / "test"
    execute_nodes(contract_nodes, namespace)  # Identical contract can resume.
    path = namespace["validation_config_path"]
    saved = json.loads(path.read_text())
    saved[key] = value
    path.write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match="Dataset/action contract"):
        execute_nodes(contract_nodes, namespace)


def test_ik_materialization_preserves_absolute_joint_contract(patched_trainer, tmp_path):
    from copy import deepcopy
    from enum import IntEnum
    from typing import Any, Dict, Tuple

    oxe = tmp_path / "prismatic/vla/datasets/rlds/oxe"
    config = ast.parse((oxe / "configs.py").read_text())
    namespace = dict(IntEnum=IntEnum)
    execute_nodes([node for node in config.body if isinstance(node, ast.ClassDef)], namespace)
    assignment = next(node for node in config.body if isinstance(node, ast.Assign)
                      and node.targets[0].id == "OXE_DATASET_CONFIGS")
    # Other datasets' configs reference unrelated imports; extract our entry only.
    index = next(i for i, key in enumerate(assignment.value.keys)
                 if key.value == "panthera_ik_three_block")
    our_config = eval(compile(ast.Expression(assignment.value.values[index]), "config", "eval"), namespace)
    namespace.update(deepcopy=deepcopy, Path=Path, Tuple=Tuple, Dict=Dict, Any=Any,
                     ACTION_PROPRIO_NORMALIZATION_TYPE="bounds_q99",
                     OXE_DATASET_CONFIGS={"panthera_ik_three_block": our_config},
                     OXE_STANDARDIZATION_TRANSFORMS={"panthera_ik_three_block": lambda x: x})
    function = next(node for node in ast.parse((oxe / "materialize.py").read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "make_oxe_dataset_kwargs")
    execute_nodes([function], namespace)
    kwargs = namespace["make_oxe_dataset_kwargs"](
        "panthera_ik_three_block", tmp_path, load_camera_views=("primary", "wrist"))
    assert kwargs["absolute_action_mask"] == [True] * 7
    assert kwargs["action_normalization_mask"] == [True] * 7
    assert kwargs["image_obs_keys"] == {"primary": "image", "wrist": "wrist_image"}
    assert kwargs["state_obs_keys"] == ["state"]
    assert our_config["state_encoding"] == namespace["StateEncoding"].JOINT
    assert our_config["action_encoding"] == namespace["ActionEncoding"].JOINT_POS
    dataset = (oxe.parent / "dataset.py").read_text()
    assert 'name == "panthera_ik_three_block" and "val" in builder.info.splits' in dataset
    assert "delta_scale" not in dataset


def test_all_backbones_unfreeze_and_full_checkpoint_roundtrip(patched_trainer, tmp_path):
    torch = pytest.importorskip("torch")

    class TinyVLA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("vision_backbone", "language_model", "projector", "action_queries"):
                setattr(self, name, torch.nn.Linear(2, 2))

        def forward(self, value):
            for module in self.children():
                value = module(value)
            return value

        def save_pretrained(self, directory):
            torch.save(self.state_dict(), directory / "pytorch_model.bin")
            (directory / "config.json").write_text("{}")

        @classmethod
        def from_pretrained(cls, directory, **kwargs):
            model = cls()
            model.load_state_dict(torch.load(Path(directory) / "pytorch_model.bin", weights_only=True))
            return model

    trainer = next(n for n in patched_trainer.body if isinstance(n, ast.FunctionDef) and n.name == "finetune")
    lora_branch = next(n for n in trainer.body if isinstance(n, ast.If)
                       and ast.unparse(n.test) == "cfg.use_lora")
    model = TinyVLA().requires_grad_(False)
    execute_nodes(lora_branch.orelse, {"vla": model})
    assert all(p.requires_grad for p in model.parameters())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model(torch.ones(1, 2)).square().sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    optimizer.step()
    assert all(not torch.equal(before[name], value) for name, value in model.state_dict().items())

    cfg = SimpleNamespace(save_latest_checkpoint_only=True, use_lora=False, use_fz=False,
                          use_proprio=True, use_diffusion=False, use_l1_regression=True,
                          use_film=False, resume_checkpoint=str(tmp_path))
    save_function = next(n for n in patched_trainer.body if isinstance(n, ast.FunctionDef)
                         and n.name == "save_training_checkpoint")
    namespace = dict(torch=torch, Path=Path, os=os, dist=SimpleNamespace(barrier=lambda: None),
                     save_dataset_statistics=Mock())
    execute_nodes([save_function], namespace)
    namespace["save_training_checkpoint"](
        cfg, tmp_path, 5, SimpleNamespace(module=model), Mock(),
        torch.nn.Linear(2, 2), None, torch.nn.Linear(2, 2),
        SimpleNamespace(dataset_statistics={}), SimpleNamespace(is_main_process=True), {})
    assert not (tmp_path / "lora_adapter").exists()
    assert (tmp_path / "action_head--latest_checkpoint.pt").exists()
    assert (tmp_path / "proprio_projector--latest_checkpoint.pt").exists()
    resume_branch = next(n for n in trainer.body if isinstance(n, ast.If)
                         and ast.unparse(n.test) == "cfg.resume_checkpoint and (not cfg.use_lora)")
    namespace.update(cfg=cfg, AutoModelForVision2Seq=TinyVLA, device_id="cpu")
    execute_nodes([resume_branch], namespace)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(namespace["vla"].state_dict()[name], value)


def test_export_rejects_adapter_only_checkpoint(tmp_path):
    namespace = {}
    helper = next(n for n in ast.parse(source(17)).body if isinstance(n, ast.FunctionDef)
                  and n.name == "require_full_checkpoint")
    execute_nodes([helper], namespace)
    (tmp_path / "lora_adapter").mkdir()
    with pytest.raises(RuntimeError, match="Old LoRA runs cannot be resumed"):
        namespace["require_full_checkpoint"](tmp_path)
    for name in ("config.json", "training_state--latest_checkpoint.pt", "model.safetensors",
                 "action_head--latest_checkpoint.pt", "proprio_projector--latest_checkpoint.pt"):
        (tmp_path / name).touch()
    namespace["require_full_checkpoint"](tmp_path)


def test_optimizer_resume_restores_state_and_rejects_lora(patched_trainer, tmp_path):
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    original = torch.optim.AdamW([parameter], lr=2e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(original, milestones=[3], gamma=.1)
    parameter.square().sum().backward()
    original.step()
    scheduler.step()
    checkpoint = {"optimizer": original.state_dict(), "scheduler": scheduler.state_dict(),
                  "step": 1, "finetuning_mode": "full"}
    path = tmp_path / "training_state--latest_checkpoint.pt"
    torch.save(checkpoint, path)
    trainer = next(n for n in patched_trainer.body if isinstance(n, ast.FunctionDef) and n.name == "finetune")
    resume = next(n for n in trainer.body if isinstance(n, ast.If)
                  and "training_state_path =" in ast.unparse(n))
    restored = torch.optim.AdamW([torch.nn.Parameter(torch.tensor([1.0]))], lr=.5)
    restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(restored, milestones=[3], gamma=.1)
    namespace = dict(torch=torch, Path=Path, optimizer=restored, scheduler=restored_scheduler,
                     cfg=SimpleNamespace(resume_checkpoint=str(tmp_path), use_lora=False,
                                         resume_learning_rate=-1))
    execute_nodes([resume], namespace)
    assert namespace["resume_step"] == 1
    assert restored.param_groups[0]["lr"] == original.param_groups[0]["lr"]
    torch.testing.assert_close(restored.state_dict()["state"][0]["exp_avg"],
                               original.state_dict()["state"][0]["exp_avg"])
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    checkpoint.pop("finetuning_mode")  # Legacy LoRA checkpoint.
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Cannot resume across LoRA/full"):
        execute_nodes([resume], namespace)
