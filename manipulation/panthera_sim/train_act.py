#!/usr/bin/env python3
"""Train a local LeRobot ACT policy and report held-out imitation loss."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES

from tools.lerobot_image_cache import CachedLeRobotDataset, default_cache_path, _dataset_signature
from teleop.dataset_contract import DEFAULT_DATASET, DEFAULT_CHECKPOINT, validate_rendered, file_hash


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DEFAULT_CHECKPOINT
TRAINING_STATE_NAME = "training_state.pt"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(
    destination: Path,
    policy,
    preprocessor,
    postprocessor,
    optimizer,
    step: int,
    initial_losses: list[float],
    recent_losses: list[float],
    deployment: dict,
    scaler=None,
) -> None:
    """Save inference files plus the state required to continue training."""
    destination.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(destination)
    preprocessor.save_pretrained(destination)
    postprocessor.save_pretrained(destination)
    (destination / "deployment.json").write_text(json.dumps(deployment, indent=2) + "\n")
    if getattr(policy, "_inference_config", None) is not None:
        (destination / "inference.json").write_text(json.dumps(policy._inference_config, indent=2) + "\n")
    state = {
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_validation_loss": getattr(policy, "_best_validation_loss", math.inf),
        "stale_validations": getattr(policy, "_stale_validations", 0),
        "version": 1,
        "training_objective": getattr(policy, "_training_objective", "vae"),
        "step": step,
        "optimizer": optimizer.state_dict(),
        "initial_losses": initial_losses,
        "recent_losses": recent_losses,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = destination / f".{TRAINING_STATE_NAME}.tmp"
    torch.save(state, temporary)
    temporary.replace(destination / TRAINING_STATE_NAME)


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    cuda_rng = state.get("torch_cuda_rng")
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng)


def training_loss(policy, batch, objective):
    """Train either the original VAE objective or the deployed zero-latent path."""
    if objective == "vae":
        policy.train()
        return policy.forward(batch)
    # predict_action_chunk disables gradients, so call the model directly.
    policy.eval()
    observations = {key: batch[key] for key in policy.config.input_features}
    if policy.config.image_features:
        observations[OBS_IMAGES] = [observations[key] for key in policy.config.image_features]
    prediction = policy.model(observations)[0]
    # Preserve stock ACT's padded L1 reduction, matching the controlled probe.
    loss = ((prediction - batch["action"]).abs() *
            (~batch["action_is_pad"]).unsqueeze(-1)).mean()
    return loss, {"l1_loss": float(loss.detach())}


@torch.inference_mode()
def evaluate_inference(policy, loader, preprocessor, postprocessor=None, action_representation="absolute") -> dict:
    """Evaluate the deployed zero-latent policy, excluding padded targets."""
    policy.eval()
    error_sum = first_sum = count = first_count = samples = 0
    per_joint = None
    physical = None
    total_batches = len(loader)
    started = time.monotonic()
    print(f"validation started: {total_batches} batches", flush=True)
    if postprocessor is not None:
        from tools.act_pickup import PhysicalErrors
        physical = PhysicalErrors(action_representation)
    for batch_index, batch in enumerate(loader, 1):
        physical_state = batch["observation.state"]
        batch = preprocessor(batch)
        # Explicitly omit targets: validation must never condition on them.
        observations = {key: batch[key] for key in policy.config.input_features}
        prediction = policy.predict_action_chunk(observations)
        if physical is not None:
            physical.add(postprocessor(prediction[:, 0]), postprocessor(batch["action"][:, 0]), physical_state)
        error = (prediction - batch["action"]).abs()
        valid = (~batch["action_is_pad"]).unsqueeze(-1).expand_as(error)
        error_sum += float(error.masked_select(valid).sum())
        count += int(valid.sum())
        first_sum += float(error[:, 0].sum())
        first_count += error[:, 0].numel()
        sums = error[:, 0].sum(0).cpu().numpy()
        per_joint = sums if per_joint is None else per_joint + sums
        samples += len(error)
        if batch_index == 1 or batch_index % 250 == 0 or batch_index == total_batches:
            print(f"validation progress: {batch_index}/{total_batches} batches "
                  f"({samples} frames, {time.monotonic() - started:.1f}s)", flush=True)
    policy.train()
    return {"validation_loss": error_sum / max(count, 1),
            "validation_first_action_l1": first_sum / max(first_count, 1),
            "validation_first_action_l1_per_joint": (per_joint / samples).tolist() if samples else [],
            "validation_samples": samples, "validation_mode": "inference_zero_latent",
            "validation_valid_action_elements": count,
            **(physical.summary() if physical is not None else {})}


def prune_checkpoints(root: Path, keep: int) -> None:
    checkpoints = sorted(root.glob("step_*"))
    for path in checkpoints[:-keep]:
        shutil.rmtree(path)


def rollout_score(rates: dict, objective="stack") -> tuple:
    """Rank real task milestones; never select by shaped return or imitation loss."""
    if objective == "pickup":
        return (rates["sustained_pickup"], -rates["pickup_seconds_capped_mean"])
    grasp_and_lift = rates.get("grasp_and_lift", min(rates["grasped"], rates["lifted"]))
    return (rates["success"], rates["two_stacked"], grasp_and_lift, rates["grasped"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-act", action="store_true", help="published ALOHA transfer-cube HDF5 benchmark with upstream defaults")
    parser.add_argument("--grad-clip", type=float, default=10., help="0 disables gradient clipping")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pickup-manifest", type=Path)
    parser.add_argument("--action-representation", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--selection-objective", choices=("stack", "pickup"), default="stack")
    parser.add_argument("--rollout-seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume",
        type=Path,
        help="checkpoint directory to resume (the target in --steps is the total step count)",
    )
    parser.add_argument("--steps", type=int, default=1000)
    # Keep the established 1.2M-sample schedule at 100k optimizer steps.
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, help="default: one second at dataset FPS")
    parser.add_argument("--action-steps", type=int, help="default: 0.1 second at dataset FPS")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--val-batches", type=int, default=0, help="0 evaluates all held-out frames; caps sample evenly across the split")
    parser.add_argument("--full-val-at-end", action="store_true",
                        help="add a full held-out evaluation after training when routine validation is capped")
    parser.add_argument("--reset-validation-best", action="store_true",
                        help="reset the validation selection baseline when resuming with a different validation sample")
    parser.add_argument("--eval-freq", type=int, default=2000)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--early-stop-patience", type=int, default=0, help="non-improving validations; 0 disables")
    parser.add_argument("--rollout-eval-freq", type=int, default=0)
    parser.add_argument("--rollout-eval-episodes", type=int, default=8)
    parser.add_argument("--rollout-eval-seconds", type=float, default=15.)
    parser.add_argument("--rollout-action-steps", type=int)
    parser.add_argument("--log-freq", type=int, default=25)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=250,
        help="save resumable checkpoints every N steps; 0 disables periodic checkpoints",
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--no-image-cache",
        action="store_true",
        help="ignore a decoded image cache even if one exists",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False,
                        help="autotune convolution kernels for fixed image shapes")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=False,
                        help="use channels-last storage for image tensors and the visual backbone")
    parser.add_argument("--lr", type=float, help="override learning rate, including on resume")
    parser.add_argument("--lr-backbone", type=float, help="override backbone learning rate")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--training-objective", choices=("vae", "zero"),
                        help="default: restore checkpoint objective, otherwise vae")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="resume model and step with fresh AdamW moments")
    preliminary, _ = parser.parse_known_args()
    if preliminary.reference_act:
        parser.set_defaults(dataset=Path("data/act_reference/sim_transfer_cube_scripted"),
                            output=Path("outputs/act/reference_transfer_cube"),
                            steps=10000, batch_size=8, chunk_size=100, action_steps=100,
                            seed=0, workers=2, prefetch_factor=1, amp=False, grad_clip=0.,
                            val_fraction=.2, eval_freq=1000, val_batches=128,
                            checkpoint_freq=1000, keep_checkpoints=2)
    args = parser.parse_args()
    if args.grad_clip < 0 or not math.isfinite(args.grad_clip):
        parser.error("grad-clip must be finite and nonnegative")
    if args.reference_act and (args.pickup_manifest or args.rollout_eval_freq or args.training_objective == "zero"):
        parser.error("reference ACT requires VAE training and tools/evaluate_act_reference.py for rollouts")
    if args.reference_act and args.resume is None and (args.output / "training_settings.json").exists():
        parser.error("output already contains a run; use --resume or a fresh --output")

    if any(value is not None and (not math.isfinite(value) or value <= 0)
           for value in (args.lr, args.lr_backbone)):
        parser.error("learning rates must be finite and positive")
    if args.dropout is not None and not 0 <= args.dropout < 1:
        parser.error("dropout must be in [0, 1)")

    if args.action_representation == "relative" and args.pickup_manifest is None:
        parser.error("relative targets require --pickup-manifest")

    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be between 0 and 1")
    if args.action_steps is not None and args.chunk_size is not None and args.action_steps > args.chunk_size:
        raise SystemExit("--action-steps cannot exceed --chunk-size")
    if args.steps < 0 or args.checkpoint_freq < 0:
        raise SystemExit("--steps and --checkpoint-freq cannot be negative")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers nonnegative")
    if args.log_freq <= 0:
        raise SystemExit("--log-freq must be positive")
    if min(args.val_batches, args.eval_freq, args.early_stop_patience, args.rollout_eval_freq) < 0 or args.keep_checkpoints < 1:
        parser.error("evaluation counts must be nonnegative; keep-checkpoints must be positive")
    if args.rollout_eval_episodes < 1 or args.rollout_eval_seconds <= 0:
        parser.error("rollout evaluation count and seconds must be positive")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    args.output.mkdir(parents=True, exist_ok=True)
    resume = args.resume.expanduser().resolve() if args.resume else None
    if resume is not None and not resume.is_dir():
        raise SystemExit(f"resume checkpoint is not a directory: {resume}")

    if args.reference_act:
        from tools.act_reference_data import reference_metadata, reference_split, ReferenceACTDataset
        metadata = reference_metadata(args.dataset)
        deployment = {"version": 1, "benchmark": "tonyzhaozh/act", "task": "sim_transfer_cube_scripted",
                      "fps": 50, "dataset": str(args.dataset.resolve()),
                      "dataset_signature": metadata.signature, "state_gripper": "measured_normalized",
                      "action_alignment": "same_timestep", "camera_names": ["top"]}
    else:
        metadata = LeRobotDatasetMetadata("local/panthera_stack", root=args.dataset)
        provenance_path = args.dataset / "meta/provenance.json"
        if not provenance_path.is_file():
            raise SystemExit("Dataset lacks provenance. Re-render and rebuild it with the corrected pipeline.")
        provenance = json.loads(provenance_path.read_text())
        if provenance["converter_sha256"] != file_hash(REPO_ROOT / "teleop/build_lerobot_dataset.py"):
            raise SystemExit("Dataset converter changed; rebuild the LeRobot dataset.")
        rendered_root = Path(provenance["rendered_root"])
        validate_rendered(rendered_root)
        if file_hash(rendered_root / "manifest.json") != provenance["render_manifest_sha256"]:
            raise SystemExit("Rendered manifest changed; rebuild the LeRobot dataset.")
        deployment = {"version": 2, "fps": metadata.fps, "dataset": str(args.dataset.resolve()),
                      "dataset_signature": _dataset_signature(args.dataset),
                      "state_gripper": "command", "max_joint_step": None}
        from sim.dynamics import provenance_dynamics
        deployment["simulation_dynamics"] = provenance_dynamics(provenance)
        from tools.act_scene import fixed_environment
        environment = fixed_environment(provenance)
        if environment is not None:
            deployment['environment'] = environment
    pickup_manifest = None
    if args.pickup_manifest is not None:
        from tools.act_pickup import load_manifest, PickupDataset, pickup_stats
        pickup_manifest = load_manifest(args.pickup_manifest, args.dataset)
        deployment.update(version=3, action_representation=args.action_representation,
                          task="first_sustained_pickup", pickup_manifest_sha256=file_hash(args.pickup_manifest),
                          pickup_manifest=str(args.pickup_manifest.resolve()),
                          selection_objective=args.selection_objective)
    if resume is not None:
        contract = resume / "deployment.json"
        if not contract.is_file() or json.loads(contract.read_text()) != deployment:
            raise SystemExit("Resume checkpoint uses a different dataset/control contract; start a fresh run.")
    if metadata.total_episodes < 2:
        raise SystemExit("At least two episodes are required for held-out validation")
    if args.chunk_size is None:
        args.chunk_size = round(metadata.fps)
    if args.action_steps is None:
        args.action_steps = max(1, round(metadata.fps * .1))
    if not 0 < args.action_steps <= args.chunk_size:
        parser.error("require 0 < action-steps <= chunk-size")
    all_episodes = np.arange(metadata.total_episodes)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_episodes)
    n_val = min(len(all_episodes) - 1, max(1, round(len(all_episodes) * args.val_fraction)))
    val_episodes = sorted(all_episodes[:n_val].tolist())
    train_episodes = sorted(all_episodes[n_val:].tolist())

    if args.reference_act:
        train_episodes, val_episodes = reference_split()
    (args.output / "split.json").write_text(json.dumps(
        {"train": train_episodes, "validation": val_episodes}, indent=2) + "\n")
    policy_features = metadata.features if args.reference_act else dataset_to_policy_features(metadata.features)
    output_features = {
        key: value for key, value in policy_features.items()
        if value.type is FeatureType.ACTION
    }
    input_features = {
        key: value for key, value in policy_features.items()
        if key not in output_features
    }
    if resume is None:
        cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=args.chunk_size,
            n_action_steps=args.action_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_amp=args.amp and torch.cuda.is_available(),
            push_to_hub=False,
        )
    else:
        cfg = PreTrainedConfig.from_pretrained(resume)
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.use_amp = args.amp and torch.cuda.is_available()
        cfg.push_to_hub = False
        print(f"loading checkpoint: {resume}", flush=True)
    if args.lr is not None:
        cfg.optimizer_lr = args.lr
    if args.lr_backbone is not None:
        cfg.optimizer_lr_backbone = args.lr_backbone
    if args.dropout is not None:
        cfg.dropout = args.dropout
    delta_timestamps = {
        "action": [index / metadata.fps for index in cfg.action_delta_indices],
    }
    if args.reference_act:
        train_dataset = ReferenceACTDataset(args.dataset, train_episodes, cfg.chunk_size, training=True)
        val_dataset = ReferenceACTDataset(args.dataset, val_episodes, cfg.chunk_size, training=False)
    else:
        image_cache = default_cache_path(args.dataset)
        if args.no_image_cache or not image_cache.is_file():
            image_cache = None
        else:
            print(f"using decoded image cache: {image_cache}", flush=True)
        train_dataset = CachedLeRobotDataset(
            "local/panthera_stack",
            root=args.dataset,
            episodes=train_episodes,
            delta_timestamps=delta_timestamps,
            image_cache=image_cache,
        )
        val_dataset = CachedLeRobotDataset(
            "local/panthera_stack",
            root=args.dataset,
            episodes=val_episodes,
            delta_timestamps=delta_timestamps,
            image_cache=image_cache,
        )
    dataset_stats = metadata.stats
    if pickup_manifest is not None:
        train_dataset = PickupDataset(train_dataset, pickup_manifest, args.action_representation)
        val_dataset = PickupDataset(val_dataset, pickup_manifest, args.action_representation)
        dataset_stats = pickup_stats(args.dataset, pickup_manifest, train_episodes, cfg.chunk_size,
                                     args.action_representation, metadata.stats)
        print(f"pickup windows: train={len(train_dataset)} val={len(val_dataset)} representation={args.action_representation}", flush=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=not args.reference_act,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )
    validation_data = val_dataset
    if args.val_batches and args.val_batches * args.batch_size < len(val_dataset):
        validation_data = torch.utils.data.Subset(val_dataset, np.linspace(
            0, len(val_dataset) - 1, args.val_batches * args.batch_size, dtype=int).tolist())
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )

    if resume is None:
        policy = ACTPolicy(cfg).to(cfg.device)
        preprocessor, postprocessor = make_pre_post_processors(
            cfg, dataset_stats=dataset_stats
        )
    else:
        policy = ACTPolicy.from_pretrained(resume, config=cfg).to(cfg.device)
        preprocessor, postprocessor = make_pre_post_processors(
            cfg, pretrained_path=str(resume)
        )
    if args.channels_last:
        if not cfg.image_features:
            parser.error("channels-last requires an image backbone")
        policy.model.backbone.to(memory_format=torch.channels_last)
    inference_path = resume / "inference.json" if resume else None
    policy._inference_config = json.loads(inference_path.read_text()) if inference_path and inference_path.is_file() else None
    if args.rollout_action_steps is not None:
        if not 1 <= args.rollout_action_steps <= cfg.chunk_size:
            parser.error("rollout-action-steps must be between 1 and chunk-size")
        policy._inference_config = {"temporal_ensemble": False, "action_steps": args.rollout_action_steps}
    selection_path = args.output / "best_rollout/selection.json"
    best_rollout_score = None
    if selection_path.is_file():
        best_rollout_score = rollout_score(json.loads(selection_path.read_text())["rates"], args.selection_objective)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in policy.named_parameters() if "backbone" not in n]},
            {
                "params": [p for n, p in policy.named_parameters() if "backbone" in n],
                "lr": cfg.optimizer_lr_backbone,
            },
        ],
        lr=cfg.optimizer_lr,
        weight_decay=cfg.optimizer_weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
    policy._best_validation_loss = math.inf
    policy._stale_validations = 0
    start_step = 0
    initial_losses: list[float] = []
    recent_losses: list[float] = []
    policy._training_objective = "vae"
    if resume is not None:
        state_path = resume / TRAINING_STATE_NAME
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if state.get("version") != 1:
                raise SystemExit(f"unsupported training state version in {state_path}")
            if not args.reset_optimizer:
                optimizer.load_state_dict(state["optimizer"])
            if state.get("scaler") and not args.reset_optimizer:
                scaler.load_state_dict(state["scaler"])
            policy._training_objective = state.get("training_objective", "vae")
            policy._best_validation_loss = state.get("best_validation_loss", math.inf)
            policy._stale_validations = state.get("stale_validations", 0)
            start_step = int(state["step"])
            initial_losses = list(state.get("initial_losses", []))
            recent_losses = list(state.get("recent_losses", []))
            restore_rng_state(state)
            print(f"resumed step={start_step} and RNG state; optimizer="
                  f"{'fresh AdamW' if args.reset_optimizer else 'restored'}", flush=True)
        else:
            print(
                f"warning: {resume} has no {TRAINING_STATE_NAME}; "
                "starting a new optimizer from its model weights",
                flush=True,
            )
    if args.training_objective is not None:
        policy._training_objective = args.training_objective
    if args.reference_act and policy._training_objective != "vae":
        raise SystemExit("Reference ACT requires a VAE-objective checkpoint")
    # Apply overrides after loading optimizer state, which carries the old LRs.
    if args.lr is not None:
        optimizer.param_groups[0]["lr"] = args.lr
    if args.lr_backbone is not None:
        optimizer.param_groups[1]["lr"] = args.lr_backbone
    cfg.optimizer_lr = optimizer.param_groups[0]["lr"]
    cfg.optimizer_lr_backbone = optimizer.param_groups[1]["lr"]
    training_settings = dict(objective=policy._training_objective, lr=cfg.optimizer_lr,
                             lr_backbone=cfg.optimizer_lr_backbone, dropout=cfg.dropout,
                             reference_act=args.reference_act, grad_clip=args.grad_clip,
                             workers=args.workers, prefetch_factor=args.prefetch_factor,
                             cudnn_benchmark=args.cudnn_benchmark,
                             channels_last=args.channels_last,
                             sampling="one_random_frame_per_episode" if args.reference_act else "all_frames",
                             amp=cfg.use_amp, weight_decay=optimizer.param_groups[0]["weight_decay"],
                             reset_optimizer=args.reset_optimizer, resumed_from_step=start_step)
    (args.output / "training_settings.json").write_text(json.dumps(training_settings, indent=2) + "\n")
    print(f"training settings: {json.dumps(training_settings)}", flush=True)
    if args.reset_validation_best:
        policy._best_validation_loss = math.inf
        policy._stale_validations = 0
        print("Reset validation selection baseline for the new validation sample", flush=True)
    if start_step > args.steps:
        raise SystemExit(
            f"checkpoint is already at step={start_step}, beyond --steps={args.steps}"
        )
    if start_step == args.steps:
        print(
            f"checkpoint is already at step={start_step}; no training needed for --steps={args.steps}",
            flush=True,
        )

    policy.train()
    iterator = iter(train_loader) if start_step < args.steps else None
    started = time.monotonic()
    step = start_step
    stale_validations = policy._stale_validations
    validation = None
    history_path = args.output / "validation.jsonl"
    for step in range(start_step + 1, args.steps + 1):
        try:
            assert iterator is not None
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = preprocessor(batch)
        if args.channels_last:
            for key in cfg.image_features:
                batch[key] = batch[key].contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=cfg.use_amp,
        ):
            loss, output = training_loss(policy, batch, policy._training_objective)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        loss_value = float(loss.detach())
        if len(initial_losses) < 25:
            initial_losses.append(loss_value)
        recent_losses.append(loss_value)
        recent_losses = recent_losses[-25:]
        completed_this_run = step - start_step
        if completed_this_run == 1 or step % args.log_freq == 0 or step == args.steps:
            window = recent_losses[-args.log_freq:]
            rate = completed_this_run / max(time.monotonic() - started, 1e-6)
            details = " ".join(
                f"{key}={value:.4f}" for key, value in (output or {}).items()
                if isinstance(value, (int, float))
            )
            print(
                f"step={step:05d} loss={np.mean(window):.4f} "
                f"rate={rate:.2f}step/s {details}",
                flush=True,
            )
        if completed_this_run == 1 or step % args.log_freq == 0 or step == args.steps:
            (args.output / "run_status.json").write_text(json.dumps({
                "status": "training", "step": step, "requested_steps": args.steps,
                "loss": loss_value, "elapsed_seconds": time.monotonic() - started,
                "updated_at_unix": time.time()}, indent=2) + "\n")
        if (args.eval_freq and step % args.eval_freq == 0) or step == args.steps:
            validation = evaluate_inference(policy, val_loader, preprocessor,
                                            postprocessor if pickup_manifest else None, args.action_representation)
            with history_path.open("a") as log:
                log.write(json.dumps({"step": step, **validation}) + "\n")
            print(f"validation step={step}: {json.dumps(validation)}", flush=True)
            if validation["validation_loss"] < policy._best_validation_loss:
                policy._best_validation_loss = validation["validation_loss"]
                stale_validations = 0
                policy._stale_validations = 0
                save_checkpoint(args.output / "best", policy, preprocessor, postprocessor,
                                optimizer, step, initial_losses, recent_losses, deployment, scaler)
            else:
                stale_validations += 1
                policy._stale_validations = stale_validations
            if args.early_stop_patience and stale_validations >= args.early_stop_patience:
                print("Early stop: held-out inference loss stopped improving", flush=True)
                break

        rollout_due = args.rollout_eval_freq and (step % args.rollout_eval_freq == 0 or step == args.steps)
        if (args.checkpoint_freq and step % args.checkpoint_freq == 0) or rollout_due:
            checkpoint = args.output / "checkpoints" / f"step_{step:08d}"
            save_checkpoint(
                checkpoint, policy, preprocessor, postprocessor, optimizer, step,
                initial_losses, recent_losses, deployment, scaler,
            )
            print(f"checkpoint={checkpoint}", flush=True)
            prune_checkpoints(args.output / "checkpoints", args.keep_checkpoints)
            if rollout_due:
                evaluation_dir = args.output / "rollouts" / f"step_{step:08d}"
                evaluation_dir.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(REPO_ROOT / "tools/evaluate_act.py"),
                           "--checkpoint", str(checkpoint), "--dataset", str(args.dataset),
                           "--output", str(evaluation_dir), "--episodes", str(args.rollout_eval_episodes),
                           "--seconds", str(args.rollout_eval_seconds), "--seed", str(args.rollout_seed)]
                if args.selection_objective == "pickup":
                    command += ["--save-failures"]
                with (evaluation_dir / "evaluation.log").open("w") as log:
                    subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                summary = json.loads((evaluation_dir / "summary.json").read_text())
                score = rollout_score(summary["rates"], args.selection_objective)
                print(f"rollout step={step}: {json.dumps(summary['rates'])}", flush=True)
                if best_rollout_score is None or score > best_rollout_score:
                    best_rollout_score = score
                    save_checkpoint(args.output / "best_rollout", policy, preprocessor, postprocessor,
                                    optimizer, step, initial_losses, recent_losses, deployment, scaler)
                    (args.output / "best_rollout/selection.json").write_text(
                        json.dumps({"step": step, "rates": summary["rates"], "seed": args.rollout_seed,
                                    "objective": args.selection_objective}, indent=2) + "\n")


    # Persist the last optimization state separately from the best validation checkpoint.
    save_checkpoint(
        args.output, policy, preprocessor, postprocessor, optimizer, step,
        initial_losses, recent_losses, deployment, scaler,
    )

    if validation is None:
        validation = evaluate_inference(policy, val_loader, preprocessor,
                                        postprocessor if pickup_manifest else None, args.action_representation)

    full_validation = None
    if args.full_val_at_end and len(validation_data) < len(val_dataset):
        full_val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=torch.cuda.is_available(),
            prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        )
        print(f"final full validation step={step}", flush=True)
        full_validation = evaluate_inference(
            policy, full_val_loader, preprocessor,
            postprocessor if pickup_manifest else None, args.action_representation,
        )
        (args.output / "full_validation.json").write_text(
            json.dumps({"step": step, **full_validation}, indent=2) + "\n")

    elapsed = time.monotonic() - started
    completed_this_run = step - start_step
    report = {
        "steps": step,
        "training_settings": training_settings,
        "action_representation": args.action_representation,
        "selection_objective": args.selection_objective,
        "pickup_manifest": str(args.pickup_manifest.resolve()) if args.pickup_manifest else None,
        "train_frames": len(train_dataset), "val_frames": len(val_dataset),
        "requested_steps": args.steps,
        "fps": metadata.fps,
        "chunk_seconds": cfg.chunk_size / metadata.fps,
        "resumed_from_step": start_step,
        "batch_size": args.batch_size,
        "chunk_size": cfg.chunk_size,
        "action_steps": cfg.n_action_steps,
        "train_episodes": len(train_episodes),
        "val_episodes": len(val_episodes),
        "held_out_episode_indices": val_episodes,
        "initial_train_loss": float(np.mean(initial_losses)) if initial_losses else math.nan,
        "final_train_loss": float(np.mean(recent_losses)) if recent_losses else math.nan,
        **validation,
        "best_validation_loss": policy._best_validation_loss,
        "validation_batch_cap": args.val_batches,
        "validation_frequency": args.eval_freq,
        "full_validation": full_validation,
        "elapsed_seconds": elapsed,
        "steps_per_second": completed_this_run / elapsed,
        "seed": args.seed,
    }
    (args.output / "experiment.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "run_status.json").write_text(json.dumps({
        "status": "complete", "step": step, "requested_steps": args.steps,
        "updated_at_unix": time.time()}, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
