#!/usr/bin/env python3
"""Run the pinned LeRobot trainer with strict weights and full-model checks.

The notebook downloads weights locally before calling this wrapper. No upstream
files are edited. The same wrapper handles fresh training and resumable runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LEROBOT_COMMIT = "e624f3f7f8411ec3a02635d06e79373341e5ef35"
CAMERAS = ["observation.images.shoulder", "observation.images.wrist"]


def prune_checkpoints(checkpoint, keep):
    """Prune only completed checkpoints created by this wrapper, within this run."""
    if keep < 1:
        raise ValueError("PI05_KEEP_CHECKPOINTS must be at least 1")
    checkpoint = Path(checkpoint)
    if not (checkpoint / "PI05_COMPLETE").is_file():
        raise ValueError("Cannot prune before the new checkpoint is complete")
    parent = checkpoint.parent
    complete = sorted((p for p in parent.iterdir() if p.is_dir() and not p.is_symlink()
                       and p.name.isdecimal() and (p / "PI05_COMPLETE").is_file()),
                      key=lambda p: int(p.name))
    last = (parent / "last").resolve()
    removed = []
    for old in complete[:-keep]:
        if old.resolve() in (checkpoint.resolve(), last):
            continue
        shutil.rmtree(old)
        removed.append(old.name)
    return removed


def tensor_bytes(value):
    """Conservative uncompressed checkpoint-size estimate (counts aliases twice)."""
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def strict_from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs):
    """The upstream pi05 loader catches errors; this local loader must propagate them."""
    from lerobot.configs.policies import PreTrainedConfig
    from safetensors.torch import load_file

    path = Path(pretrained_name_or_path)
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Download the complete checkpoint first: {path}")
    config = config or PreTrainedConfig.from_pretrained(path)
    model = cls(config)
    state = load_file(str(path / "model.safetensors"))
    state = model._fix_pytorch_state_dict_keys(state, config)
    state = {k if k.startswith("model.") else f"model.{k}": v for k, v in state.items()}
    state = model._prepare_pretrained_state_dict(state)
    # safetensors removes tied aliases when saving LeRobot checkpoints. Restore
    # ONLY aliases which this model actually ties, never missing distinct weights.
    aliases = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(param), []).append(name)
    for names in aliases.values():
        saved = next((state[n] for n in names if n in state), None)
        if saved is not None:
            for name in names:
                state.setdefault(name, saved)
    model.load_state_dict(state, strict=True)
    del state
    print(f"Strictly loaded pretrained pi0.5 weights from {path}", flush=True)
    return model


def check_full_model(policy):
    cfg = policy.config
    if cfg.type != "pi05" or cfg.freeze_vision_encoder or cfg.train_expert_only:
        raise ValueError("This run requires pi05 with vision and VLM fully unfrozen")
    if hasattr(policy, "peft_config") or any("lora_" in n.lower() for n, _ in policy.named_parameters()):
        raise ValueError("PEFT/LoRA is not permitted in this full-model run")
    frozen = [n for n, p in policy.named_parameters() if not p.requires_grad]
    if frozen:
        raise ValueError(f"Frozen parameters: {frozen[:20]}")
    if list(cfg.image_features) != CAMERAS:
        raise ValueError(f"Camera contract mismatch: {list(cfg.image_features)}")
    if cfg.output_features["action"].shape != (7,) or cfg.input_features["observation.state"].shape != (7,):
        raise ValueError("Expected seven-dimensional state and absolute joint/gripper actions")
    if cfg.use_relative_actions:
        raise ValueError("This dataset already contains the required absolute controls")
    return {"parameters": sum(p.numel() for p in policy.parameters()),
            "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
            "frozen_parameters": 0, "cameras": CAMERAS}


def install_guards():
    import numpy as np
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.scripts import lerobot_train as trainer

    PI05Policy.from_pretrained = classmethod(strict_from_pretrained)
    original_datasets = trainer.make_train_eval_datasets
    original_optimizer = trainer.make_optimizer_and_scheduler
    original_processors = trainer.make_pre_post_processors
    original_save = trainer.save_checkpoint
    run_state = {}
    keep_checkpoints = int(os.environ.get("PI05_KEEP_CHECKPOINTS", "2"))
    if keep_checkpoints < 1:
        raise ValueError("PI05_KEEP_CHECKPOINTS must be at least 1")

    def report_memory():
        if torch.cuda.is_available():
            memory = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
            (run_state["output"] / "peak_memory.json").write_text(json.dumps(memory, indent=2))
            print("GPU peak memory:", memory, flush=True)

    def datasets(cfg):
        if cfg.peft is not None or cfg.policy.type != "pi05":
            raise ValueError("Only full-model pi05 is supported")
        if cfg.accelerator.gradient_accumulation.steps != 1:
            raise ValueError("This notebook counts optimizer steps; keep accumulation at 1")
        train, val = original_datasets(cfg)
        expected_train = list(range(900))
        if train.episodes != expected_train or val is None or val.episodes != list(range(900, 1000)):
            raise ValueError("Expected episodes 0:900 train, 900:1000 validation")
        # Compute exact numerical quantiles ONLY on train episodes. Image stats
        # are unused because VISUAL uses IDENTITY normalization.
        import pyarrow.parquet as pq
        numeric = {key: [] for key in ("observation.state", "action")}
        for path in sorted((Path(cfg.dataset.root) / "data").rglob("*.parquet")):
            table = pq.read_table(path, columns=["episode_index", *numeric])
            mask = table["episode_index"].to_numpy() < 900
            if mask.any():
                for key in numeric:
                    numeric[key].append(np.asarray(table[key].to_pylist(), dtype=np.float32)[mask])
        stats = {}
        for key, chunks in numeric.items():
            values = np.concatenate(chunks)
            stats[key] = {"min": values.min(0), "max": values.max(0),
                          "mean": values.mean(0, dtype=np.float64),
                          "std": values.std(0, dtype=np.float64), "count": np.array([len(values)])}
            for q in (1, 10, 50, 90, 99):
                stats[key][f"q{q:02d}"] = np.quantile(values, q / 100, axis=0)
        for ds in (train, val):
            ds.meta.stats.update(stats)
        output = Path(cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        from sim.dynamics import provenance_dynamics
        provenance_path = Path(cfg.dataset.root) / "meta/provenance.json"
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        run_state.update(cfg=cfg, output=output, dynamics=provenance_dynamics(provenance))
        (output / "episode_split.json").write_text(json.dumps({
            "train": expected_train, "validation": list(range(900, 1000)),
            "normalization_episodes": expected_train}, indent=2))
        (output / "training_stats.json").write_text(json.dumps(
            {k: {s: v.tolist() for s, v in fields.items()} for k, fields in stats.items()}, indent=2))
        return train, val

    def processors(*args, **kwargs):
        if not run_state["cfg"].resume:
            # Rebuild for THIS embodiment and its two cameras; never reuse base
            # checkpoint transforms, DROID statistics, or camera names.
            kwargs["pretrained_path"] = None
            kwargs.pop("preprocessor_overrides", None)
            kwargs.pop("postprocessor_overrides", None)
        return original_processors(*args, **kwargs)

    def optimizer(cfg, policy):
        report = check_full_model(policy)
        opt, schedule = original_optimizer(cfg, policy)
        covered = {id(p) for g in opt.param_groups for p in g["params"]}
        if covered != {id(p) for p in policy.parameters()}:
            raise ValueError("Optimizer does not cover every model parameter")
        report["optimizer_covers_all_parameters"] = True
        output = run_state["output"]
        (output / "full_model_audit.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        # Check actual backward flow before the first optimizer update. We check
        # component gradients, not inactive language-generation heads that are
        # present in the architecture but unused by the action loss.
        selectors = {"vision_encoder": ".vision_tower.", "vision_projector": ".multi_modal_projector.",
                     "language_backbone": ".language_model.layers.", "action_expert": ".gemma_expert.model.layers.",
                     "action_projection": ".action_out_proj."}
        representatives = {group: next((p for n, p in policy.named_parameters()
                            if selector in n and p.ndim >= 2), None)
                           for group, selector in selectors.items()}
        if any(p is None for p in representatives.values()):
            raise ValueError("Could not identify every pi05 model component")

        def audit_gradients(optimizer, args, kwargs):
            gradients = {}
            for group, param in representatives.items():
                grad = param.grad
                if grad is None or not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
                    raise RuntimeError(f"Missing, nonfinite, or zero gradient in {group}")
                gradients[group] = float(grad.detach().float().norm().cpu())
            report["first_backward_gradient_norms"] = gradients
            (output / "full_model_audit.json").write_text(json.dumps(report, indent=2))
            print("Full-model backward check passed:", gradients, flush=True)
            handle.remove()
        handle = opt.register_step_pre_hook(audit_gradients)
        # Also report memory for the smoke run, which deliberately saves no weights.
        updates = 0
        def memory_after_step(optimizer, args, kwargs):
            nonlocal updates
            updates += 1
            if updates <= 4:
                report_memory()
        opt.register_step_post_hook(memory_after_step)
        return opt, schedule

    def save_checkpoint(*args, **kwargs):
        checkpoint = Path(kwargs["checkpoint_dir"])
        policy = kwargs["policy"]
        accelerator = kwargs.get("accelerator")
        if accelerator is not None:
            policy = accelerator.unwrap_model(policy)
        estimate = tensor_bytes(policy.state_dict()) + tensor_bytes(kwargs["optimizer"].state_dict())
        required = int(estimate * 1.1) + 2 * 2**30
        available = shutil.disk_usage(run_state["output"]).free
        if available < required:
            raise RuntimeError(f"Checkpoint needs approximately {required/2**30:.1f} GiB free; "
                               f"{available/2**30:.1f} GiB available. Existing checkpoints are preserved.")
        original_save(*args, **kwargs)
        (checkpoint / "pretrained_model/deployment.json").write_text(json.dumps({
            "simulation_dynamics": run_state["dynamics"],
            "fps": 30, "action_representation": "absolute", "action_alignment": "next_uniform_sample",
            "state_gripper": "command", "cameras": CAMERAS, "task": "stack the three colored cubes",
            "success_hold_seconds": 1.0, "dataset_repo": run_state["cfg"].dataset.repo_id,
            "dataset_revision": run_state["cfg"].dataset.revision, "lerobot_commit": LEROBOT_COMMIT}, indent=2))
        (checkpoint / "PI05_COMPLETE").write_text("Model, processors, optimizer and deployment contract saved.\n")
        # Point last at the successful replacement before removing old snapshots.
        trainer.update_last_checkpoint(checkpoint)
        drive_root = os.environ.get("PI05_DRIVE_BACKUP_ROOT")
        if drive_root:
            from tools.pi05_checkpoint_backup import backup_checkpoint
            result = backup_checkpoint(checkpoint, drive_root)
            (run_state["output"] / "drive_backup.json").write_text(json.dumps(result, indent=2))
        removed = prune_checkpoints(checkpoint, keep_checkpoints)
        saved_bytes = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
        storage = {"last_checkpoint_gib": saved_bytes / 2**30, "keep_checkpoints": keep_checkpoints,
                   "free_gib": shutil.disk_usage(run_state["output"]).free / 2**30,
                   "removed_checkpoint_steps": removed}
        (run_state["output"] / "storage.json").write_text(json.dumps(storage, indent=2))
        print("Checkpoint storage:", storage, flush=True)
        report_memory()
    trainer.make_train_eval_datasets = datasets
    trainer.make_pre_post_processors = processors
    trainer.make_optimizer_and_scheduler = optimizer
    trainer.save_checkpoint = save_checkpoint
    return trainer


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    install_guards().main()
