#!/usr/bin/env python3
"""Full pi0.5 fine-tune (no LoRA, nothing frozen) through the pinned LeRobot trainer, with strict checks, for any
arm profile (manipulation/profiles/<name>.yaml).

    python manipulation/finetune/pi05_full.py --profile=<name|path.yaml> [--expect_resume_step=N] <lerobot_train flags>

LeRobot's own command line (`--policy.*`, `--dataset.*`, `--output_dir`, `--resume`, ...) is passed through
unchanged; no upstream file is edited. What is added, around the trainer's own functions:

- weights: pi0.5 checkpoints load strictly (the upstream loader swallows errors); only weights the model actually
  ties are restored from their aliases;
- model: pi05 with the vision encoder and language model trainable, no PEFT/LoRA, no frozen parameter; cameras,
  state and action sizes equal the profile's; the optimizer covers every parameter, and the first backward pass must
  give finite, non-zero gradients in every component (vision, projector, language, action expert, action head);
- data: the episodes and task are the profile's. With `pi05.train_episodes` / `val_episodes` the split is checked and,
  with `normalization: train_episodes`, state/action statistics are recomputed on the training episodes only. With
  `require_manifest` (exports from panthera_teleop) the dataset's `processing/manifest.json` must match
  `PI05_TRAINING_MANIFEST_SHA256` when that is set, its verification reports must pass, and every exported frame must
  be in the training sampler;
- resume: `--expect_resume_step=N` refuses any checkpoint that did not stop at step N;
- checkpoints: written only with room on disk, completed with `deployment.json` (the profile's control contract plus
  the dataset's provenance) and a `PI05_COMPLETE` marker, pruned to `PI05_KEEP_CHECKPOINTS` (default 2), optionally
  copied to `PI05_DRIVE_BACKUP_ROOT`; peak GPU memory is reported.

Download the base weights locally first (the strict loader needs `model.safetensors`).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROFILES = HERE.parent / "profiles"
PANTHERA_SIM = HERE.parent / "panthera_sim"      # sim.dynamics (provenance) and tools.pi05_checkpoint_backup
LEROBOT_COMMIT = "e624f3f7f8411ec3a02635d06e79373341e5ef35"


def load_profile(name_or_path):
    """An arm profile by name (manipulation/profiles/<name>.yaml) or path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = PROFILES / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no arm profile {path} (profiles: {sorted(p.stem for p in PROFILES.glob('*.yaml'))})")
    profile = yaml.safe_load(path.read_text())
    if "pi05" not in profile.get("methods", []):
        raise ValueError(f"profile {profile['name']} does not list pi05 among its methods")
    profile.setdefault("pi05", {})
    return profile


def episode_range(span):
    """[start, stop) from a profile's two-element list, or None."""
    return list(range(*span)) if span else None


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


def check_full_model(policy, profile):
    """The policy is a fully trainable pi05 with the profile's cameras and state/action sizes."""
    cfg = policy.config
    if cfg.type != "pi05" or cfg.freeze_vision_encoder or cfg.train_expert_only:
        raise ValueError("This run requires pi05 with vision and VLM fully unfrozen")
    if hasattr(policy, "peft_config") or any("lora_" in n.lower() for n, _ in policy.named_parameters()):
        raise ValueError("PEFT/LoRA is not permitted in this full-model run")
    frozen = [n for n, p in policy.named_parameters() if not p.requires_grad]
    if frozen:
        raise ValueError(f"Frozen parameters: {frozen[:20]}")
    if list(cfg.image_features) != profile["cameras"]:
        raise ValueError(f"Camera contract mismatch: {list(cfg.image_features)} vs {profile['cameras']}")
    if (cfg.output_features["action"].shape != (profile["action_dim"],)
            or cfg.input_features["observation.state"].shape != (profile["state_dim"],)):
        raise ValueError(f"Expected {profile['state_dim']}-D state and {profile['action_dim']}-D absolute actions")
    if getattr(cfg, "use_relative_actions", False):
        raise ValueError("The profile's actions are absolute")
    return {"parameters": sum(p.numel() for p in policy.parameters()),
            "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
            "frozen_parameters": 0, "cameras": profile["cameras"]}


def train_only_stats(root, train_episodes):
    """Exact state/action statistics (min, max, mean, std, count, quantiles) over the training episodes only."""
    import numpy as np
    import pyarrow.parquet as pq
    keep = set(train_episodes)
    numeric = {key: [] for key in ("observation.state", "action")}
    for path in sorted((Path(root) / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", *numeric])
        mask = np.isin(table["episode_index"].to_numpy(), list(keep))
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
    return stats


def check_manifest(root):
    """The export's manifest (hash checked against PI05_TRAINING_MANIFEST_SHA256 when set) and passing reports."""
    manifest_path = Path(root) / "processing/manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{manifest_path} missing: the profile requires a verified panthera_teleop export")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    expected = os.environ.get("PI05_TRAINING_MANIFEST_SHA256")
    if expected and digest != expected:
        raise ValueError("Training dataset manifest differs from the verified export (PI05_TRAINING_MANIFEST_SHA256)")
    for audit_file in ("verification.json", "processing/full_rate_verification.json"):
        path = Path(root) / audit_file
        if path.is_file() and not json.loads(path.read_text())["passed"]:
            raise ValueError(f"{audit_file} did not pass")
    manifest = json.loads(manifest_path.read_text())
    return digest, manifest


def install_guards(profile, expect_resume_step=None):
    """Wrap the trainer's dataset, dataloader, processor, optimizer and checkpoint functions with the checks."""
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.scripts import lerobot_train as trainer

    PI05Policy.from_pretrained = classmethod(strict_from_pretrained)
    original_datasets = trainer.make_train_eval_datasets
    original_optimizer = trainer.make_optimizer_and_scheduler
    original_processors = trainer.make_pre_post_processors
    original_save = trainer.save_checkpoint
    original_dataloaders = trainer.make_dataloaders
    options = profile["pi05"]
    run_state = {}
    keep_checkpoints = int(os.environ.get("PI05_KEEP_CHECKPOINTS", "2"))
    if keep_checkpoints < 1:
        raise ValueError("PI05_KEEP_CHECKPOINTS must be at least 1")

    def report_memory():
        """Write and print peak GPU memory (no-op without CUDA)."""
        if torch.cuda.is_available():
            memory = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
            (run_state["output"] / "peak_memory.json").write_text(json.dumps(memory, indent=2))
            print("GPU peak memory:", memory, flush=True)

    def datasets(cfg):
        """Load the datasets through the trainer, then check them against the profile and write the audit."""
        if cfg.peft is not None or cfg.policy.type != "pi05":
            raise ValueError("Only full-model pi05 is supported")
        if cfg.accelerator.gradient_accumulation.steps != 1:
            raise ValueError("Steps count optimizer updates; keep gradient accumulation at 1")
        if expect_resume_step is not None:
            if not cfg.resume:
                raise ValueError("--expect_resume_step needs --resume=true")
            checkpoint = Path(cfg.checkpoint_path)
            if not (checkpoint / "PI05_COMPLETE").is_file():
                raise ValueError("Resume source is not a fully completed checkpoint")
            step = json.loads((checkpoint / "training_state/training_step.json").read_text())["step"]
            if step != expect_resume_step:
                raise ValueError(f"Resume checkpoint stopped at step {step}, expected {expect_resume_step}")
        root = Path(cfg.dataset.root).resolve()
        audit = {"profile": profile["name"], "dataset_root": str(root), "dataset_repo": cfg.dataset.repo_id,
                 "dataset_revision": cfg.dataset.revision}
        if options.get("require_manifest"):
            digest, manifest = check_manifest(root)
            if cfg.dataset.repo_id != manifest["repo_id"]:
                raise ValueError(f"--dataset.repo_id {cfg.dataset.repo_id} is not the export's {manifest['repo_id']}")
            run_state.update(manifest_hash=digest, source_ids=[e["id"] for e in manifest["episodes"]],
                             expected_frames=manifest["totals"]["observations"])
            audit.update(dataset_manifest_sha256=digest, source_episode_ids=run_state["source_ids"])
        train, val = original_datasets(cfg)
        train_eps, val_eps = episode_range(options.get("train_episodes")), episode_range(options.get("val_episodes"))
        if train_eps is not None and list(train.episodes) != train_eps:
            raise ValueError(f"Training episodes differ from the profile's {options['train_episodes']}")
        if val_eps is not None and (val is None or list(val.episodes) != val_eps):
            raise ValueError(f"Validation episodes differ from the profile's {options['val_episodes']}")
        if "expected_frames" in run_state and train.num_frames != run_state["expected_frames"]:
            raise ValueError(f"Training set has {train.num_frames} frames, the export {run_state['expected_frames']}")
        sample = train[0]
        if sample["observation.state"].shape != (profile["state_dim"],):
            raise ValueError(f"Dataset state is {tuple(sample['observation.state'].shape)}, "
                             f"the profile's {profile['state_dim']}")
        if profile.get("task") and sample["task"] != profile["task"]:
            raise ValueError(f"Unexpected task prompt {sample['task']!r}")
        from lerobot.configs.types import FeatureType, PolicyFeature
        size = profile.get("image_size", 256)
        cfg.policy.input_features = {key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, size, size))
                                     for key in profile["cameras"]}
        cfg.policy.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE,
                                                                       shape=(profile["state_dim"],))
        output = Path(cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if options.get("normalization") == "train_episodes":
            # Image statistics are unused: VISUAL features use IDENTITY normalization.
            stats = train_only_stats(root, train_eps if train_eps is not None else list(train.episodes))
            for ds in (train, val):
                if ds is not None:
                    ds.meta.stats.update(stats)
            (output / "training_stats.json").write_text(json.dumps(
                {k: {s: v.tolist() for s, v in fields.items()} for k, fields in stats.items()}, indent=2))
        dynamics = None
        if profile.get("simulation") and (PANTHERA_SIM / "sim" / "dynamics.py").is_file():
            sys.path.insert(0, str(PANTHERA_SIM))
            from sim.dynamics import provenance_dynamics
            provenance_path = root / "meta/provenance.json"
            dynamics = provenance_dynamics(json.loads(provenance_path.read_text()) if provenance_path.exists() else {})
        audit.update(episodes=train.num_episodes, frames=train.num_frames, fps=train.fps, task=sample["task"],
                     train_episodes=list(train.episodes),
                     validation_episodes=list(val.episodes) if val is not None else [],
                     normalization=options.get("normalization", "dataset"), simulation_dynamics=dynamics,
                     resume_checkpoint=str(cfg.checkpoint_path) if cfg.resume else None)
        (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2))
        run_state.update(cfg=cfg, output=output, dynamics=dynamics, train_episodes=list(train.episodes))
        print(f"Dataset check passed for {profile['name']}: {train.num_episodes} episodes, {train.num_frames} frames",
              flush=True)
        return train, val

    def dataloaders(cfg, dataset, eval_dataset, step, parallel_dims):
        """Build the loaders; for a verified export, check the sampler covers exactly its frames."""
        train_loader, eval_loader = original_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims)
        if "expected_frames" in run_state:
            indices = train_loader.sampler.indices
            if sorted(indices) != list(range(run_state["expected_frames"])):
                raise ValueError("Training sampler does not cover exactly the verified export's frames")
            (run_state["output"] / "sampler_audit.json").write_text(json.dumps({
                "resume_step": step, "sampleable_frames": len(indices), "batch_size": cfg.batch_size,
                "dataset_manifest_sha256": run_state["manifest_hash"]}, indent=2))
            print("Sampler check passed: every exported frame, nothing else", flush=True)
        return train_loader, eval_loader

    def processors(*args, **kwargs):
        """Fresh normalization processors for a new run; the checkpoint's own when resuming."""
        if not run_state["cfg"].resume:
            # Rebuild for THIS embodiment and its cameras; never reuse base
            # checkpoint transforms, DROID statistics, or camera names.
            kwargs["pretrained_path"] = None
            kwargs.pop("preprocessor_overrides", None)
            kwargs.pop("postprocessor_overrides", None)
        return original_processors(*args, **kwargs)

    def optimizer(cfg, policy):
        """Check the model, build the optimizer, and hook the first-step gradient check and memory report."""
        report = check_full_model(policy, profile)
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
            """Before the first update: every component must have a finite, non-zero gradient."""
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
            """Report peak memory after each of the first four updates."""
            nonlocal updates
            updates += 1
            if updates <= 4:
                report_memory()
        opt.register_step_post_hook(memory_after_step)
        return opt, schedule

    def deployment_contract():
        """The deployment.json for a checkpoint: the profile's control contract and the data's provenance."""
        cfg = run_state["cfg"]
        contract = {"robot_type": profile["robot_type"], "fps": profile["fps"], **profile.get("deployment", {}),
                    "cameras": profile["cameras"], "task": profile.get("task"), "profile": profile["name"],
                    "dataset_repo": cfg.dataset.repo_id, "dataset_revision": cfg.dataset.revision,
                    "lerobot_commit": LEROBOT_COMMIT, "training_episode_indices": run_state["train_episodes"]}
        if profile.get("simulation"):
            contract["simulation_dynamics"] = run_state["dynamics"]
        if "manifest_hash" in run_state:
            contract.update(dataset_manifest_sha256=run_state["manifest_hash"],
                            source_episode_ids=run_state["source_ids"])
        return contract

    def save_checkpoint(*args, **kwargs):
        """Save when the disk has room, then complete, mark, back up and prune checkpoints."""
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
        (checkpoint / "pretrained_model/deployment.json").write_text(json.dumps(deployment_contract(), indent=2))
        (checkpoint / "PI05_COMPLETE").write_text("Model, processors, optimizer and deployment contract saved.\n")
        # Point last at the successful replacement before removing old snapshots.
        trainer.update_last_checkpoint(checkpoint)
        drive_root = os.environ.get("PI05_DRIVE_BACKUP_ROOT")
        if drive_root:
            sys.path.insert(0, str(PANTHERA_SIM))
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
    trainer.make_dataloaders = dataloaders
    trainer.make_optimizer_and_scheduler = optimizer
    trainer.save_checkpoint = save_checkpoint
    return trainer


def split_args(argv):
    """(profile, expect_resume_step, remaining argv for LeRobot) from this script's own flags."""
    profile, step, rest = None, None, []
    for arg in argv:
        if arg.startswith("--profile="):
            profile = arg.split("=", 1)[1]
        elif arg.startswith("--expect_resume_step="):
            step = int(arg.split("=", 1)[1])
        else:
            rest.append(arg)
    if profile is None:
        raise SystemExit("usage: pi05_full.py --profile=<name|path.yaml> [--expect_resume_step=N] <lerobot_train flags>")
    return profile, step, rest


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    name, resume_step, rest = split_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]
    install_guards(load_profile(name), resume_step).main()
