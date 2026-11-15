"""Optional success-only pi0.5 supervision within EXPO-FT.

Uses the saved processor and original policy.forward flow-matching loss. This is
the supervised base-policy component, not the critic/editor RL update. No model
is imported, loaded, or trained merely by importing this module.
"""
from __future__ import annotations

from .pi05_adapter import ACTION_DIM, HORIZON, IMAGE_KEYS, TASK

SUPERVISED_PROVENANCE = frozenset({"executed_command", "kinesthetic_demo",
                                 "measured_next_pose", "human_intervention"})


def success_only_flow_loss(policy, preprocessor, batch, *, episode_success,
                           supervision_valid, action_provenance):
    """Original flow loss over trusted successful, complete 50-action sequences.

    ``episode_success`` is an explicit bool[B] outcome label;
    ``supervision_valid`` is bool[B,50] marking trusted recorded action targets;
    ``action_provenance`` names each row's source. Measured-next-pose demos are
    accepted for supervision, never represented here as executed RL commands.
    Partial or padded windows are excluded because stock forward has no
    per-timestep validity mask. If none remain, return (None, metrics).
    """
    import torch
    actions = torch.as_tensor(batch["action"])
    if actions.ndim != 3 or actions.shape[1:] != (HORIZON, ACTION_DIM):
        raise ValueError("Base supervision requires physical actions [B,50,14]")
    count = actions.shape[0]
    success = torch.as_tensor(episode_success)
    valid = torch.as_tensor(supervision_valid)
    if success.dtype != torch.bool or success.shape != (count,):
        raise ValueError("Explicit boolean episode_success [B] required")
    if valid.dtype != torch.bool or valid.shape != (count, HORIZON):
        raise ValueError("Explicit boolean supervision_valid [B,50] required")
    if len(action_provenance) != count or any(p not in SUPERVISED_PROVENANCE for p in action_provenance):
        raise ValueError("Every target sequence needs supported recorded action provenance")
    if batch["observation.state"].shape != (count, ACTION_DIM):
        raise ValueError("Expected physical state [B,14]")
    for key in IMAGE_KEYS:
        if key not in batch or batch[key].shape != (count, 3, 256, 256):
            raise ValueError("All three exact camera batches [B,3,256,256] are required")
    if (getattr(policy.config, "chunk_size", None) != HORIZON
            or getattr(policy.config, "use_relative_actions", True)):
        raise ValueError("Expected the matching absolute-action pi05 policy")
    eligible = success.cpu() & valid.cpu().all(dim=1)
    selected = eligible.nonzero().flatten().tolist()
    metrics = {"base_examples_seen": count, "base_examples_used": len(selected),
               "base_examples_excluded": count - len(selected),
               "base_objective": "success_only_original_flow_matching"}
    if not selected:
        return None, metrics
    filtered = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.shape[0] != count:
                raise ValueError(f"Batch size mismatch for {key}")
            filtered[key] = value[selected].clone()
            if value.is_floating_point() and not torch.isfinite(filtered[key]).all():
                raise ValueError(f"Nonfinite trusted supervision: {key}")
        elif isinstance(value, (list, tuple)):
            if len(value) != count:
                raise ValueError(f"Batch size mismatch for {key}")
            filtered[key] = [value[i] for i in selected]
    tasks = filtered.get("task", [TASK] * len(selected))
    if any(task != TASK for task in tasks):
        raise ValueError("Base supervision task differs from the checkpoint")
    filtered["task"] = tasks
    for key in IMAGE_KEYS:
        image = filtered[key]
        if image.dtype == torch.uint8:
            image = image.float() / 255
        elif not image.is_floating_point():
            raise ValueError("Training camera arrays must be uint8 or RGB floats")
        if image.min() < 0 or image.max() > 1:
            raise ValueError("Training RGB floats must be in [0,1]")
        filtered[key] = image
    if "action_is_pad" in filtered and filtered["action_is_pad"].any():
        raise ValueError("Padded action windows cannot be marked fully valid")
    prepared = preprocessor(filtered)
    was_training = policy.training
    try:
        policy.train()
        loss, info = policy.forward(prepared, reduction="mean")
    finally:
        policy.train(was_training)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("Original pi05 forward returned invalid flow loss")
    return loss, {**metrics, **info}


def success_only_flow_step(policy, preprocessor, optimizer, batch, *, max_grad_norm=1.0, **labels):
    """Explicit optimizer step; caller owns resources and checkpoint publication."""
    import torch
    import math
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = success_only_flow_loss(policy, preprocessor, batch, **labels)
    if loss is None:
        return metrics
    loss.backward()
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return {**metrics, "base_loss": float(loss.detach()), "base_grad_norm": float(norm)}


def load_supervised_batch(path):
    """Read an explicit, provenance-bearing full-horizon NPZ supervision batch.

    This validates the input contract, not correspondence to raw robot logs.
    Preparing and reviewing that correspondence remains a separate data step.
    """
    import numpy as np
    import torch
    with np.load(path, allow_pickle=False) as archive:
        required = {"action", "observation.state", "task", "episode_success", "supervision_valid",
                    "action_provenance", "source_episode_id", *IMAGE_KEYS}
        if missing := required - set(archive.files):
            raise ValueError(f"Missing explicit supervision fields: {sorted(missing)}")
        arrays = {key: archive[key].copy() for key in required}
        if "action_is_pad" in archive.files:
            arrays["action_is_pad"] = archive["action_is_pad"].copy()
    actions = arrays["action"]
    if actions.ndim != 3 or actions.shape[1:] != (HORIZON, ACTION_DIM) or actions.shape[0] < 1:
        raise ValueError("Supervised NPZ actions must be physical [B,50,14]")
    count = actions.shape[0]
    for key in ("task", "action_provenance", "source_episode_id"):
        if arrays[key].shape != (count,) or arrays[key].dtype.kind not in ("U", "S"):
            raise ValueError(f"Expected explicit string array [B]: {key}")
        arrays[key] = arrays[key].astype(str).tolist()
    if any(task != TASK for task in arrays["task"]):
        raise ValueError("Supervised NPZ task differs from the checkpoint")
    if any(not source.strip() for source in arrays["source_episode_id"]):
        raise ValueError("Every supervision row needs a real source episode identifier")
    if any(source not in SUPERVISED_PROVENANCE for source in arrays["action_provenance"]):
        raise ValueError("Unsupported recorded action provenance")
    if arrays["episode_success"].dtype != np.bool_ or arrays["episode_success"].shape != (count,):
        raise ValueError("Supervised NPZ needs explicit bool episode_success [B]")
    if arrays["supervision_valid"].dtype != np.bool_ or arrays["supervision_valid"].shape != (count, HORIZON):
        raise ValueError("Supervised NPZ needs explicit bool supervision_valid [B,50]")
    if arrays["observation.state"].shape != (count, ACTION_DIM):
        raise ValueError("Supervised NPZ physical state must be [B,14]")
    for key in IMAGE_KEYS:
        image = arrays[key]
        if image.shape != (count, 3, 256, 256):
            raise ValueError("Supervised NPZ needs all three [B,3,256,256] camera arrays")
        if image.dtype != np.uint8 and (image.dtype.kind != "f" or not np.isfinite(image).all()
                                       or image.min() < 0 or image.max() > 1):
            raise ValueError("Supervised NPZ camera data must be uint8 or RGB floats in [0,1]")
    for key in ("action", "observation.state"):
        if arrays[key].dtype.kind != "f" or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Expected finite physical floating point arrays: {key}")
    if "action_is_pad" in arrays:
        if arrays["action_is_pad"].dtype != np.bool_ or arrays["action_is_pad"].shape != (count, HORIZON):
            raise ValueError("action_is_pad must be bool[B,50]")
        if (arrays["supervision_valid"] & arrays["action_is_pad"]).any():
            raise ValueError("Padded actions cannot be marked valid supervision")
    eligible = arrays["episode_success"] & arrays["supervision_valid"].all(axis=1)
    if not eligible.any():
        raise ValueError("No complete, explicitly successful supervision sequences")
    batch = {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
             for key, value in arrays.items() if key in {"action", "observation.state", "task", "action_is_pad", *IMAGE_KEYS}}
    labels = {"episode_success": torch.from_numpy(arrays["episode_success"]),
              "supervision_valid": torch.from_numpy(arrays["supervision_valid"]),
              "action_provenance": arrays["action_provenance"]}
    return batch, labels, arrays["source_episode_id"]


def main(argv=None):
    """Explicit base fine-tuning executable; never run automatically by EXPO prep."""
    import argparse
    import json
    import math
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Optional original pi05 flow fine-tuning on supplied successful 50-step sequences")
    parser.add_argument("--checkpoint", required=True, help="Immutable saved Panthera pi05 checkpoint")
    parser.add_argument("--batch", required=True, help="Reviewed physical full-horizon supervision NPZ (not short critic replay)")
    parser.add_argument("--output", required=True, help="New checkpoint directory; must not exist")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("steps/batch-size and finite learning-rate must be positive")
    output = Path(args.output).expanduser().absolute()
    if output.exists():
        raise FileExistsError("Base training must save a new checkpoint directory")
    from .resources import require_idle_gpu
    # Hold the shared lease before any torch import/model initialization and
    # throughout batch validation, training, and checkpoint publication.
    with require_idle_gpu(args.device, disk_path=output, minimum_free_gib=32.0,
                          minimum_disk_gib=15.0) as resource_report:
        return _run_training(args, output, resource_report)


def _run_training(args, output, resource_report):
    import json
    from pathlib import Path
    batch, labels, source_ids = load_supervised_batch(args.batch)
    from .pi05_adapter import PI05Adapter, LEROBOT_COMMIT, _digest_file
    batch_hash = _digest_file(Path(args.batch))
    import torch
    adapter = PI05Adapter(args.checkpoint, device=args.device).load()
    policy = adapter.policy
    if args.gradient_checkpointing:
        policy.model.gradient_checkpointing_enable()
        policy.config.gradient_checkpointing = True
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Checkpoint exposes no trainable base parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    eligible = (labels["episode_success"] & labels["supervision_valid"].all(dim=1)).nonzero().flatten()
    torch.manual_seed(args.seed)
    history = []
    for step in range(args.steps):
        indexes = eligible[torch.randint(len(eligible), (args.batch_size,), generator=generator)].tolist()
        subset = {key: value[indexes] if isinstance(value, torch.Tensor) else [value[i] for i in indexes]
                  for key, value in batch.items()}
        row_labels = {"episode_success": labels["episode_success"][indexes],
                      "supervision_valid": labels["supervision_valid"][indexes],
                      "action_provenance": [labels["action_provenance"][i] for i in indexes]}
        metrics = success_only_flow_step(policy, adapter.preprocessor, optimizer, subset, **row_labels)
        history.append({"step": step + 1, "base_loss": metrics["base_loss"], "base_grad_norm": metrics["base_grad_norm"]})
        print(json.dumps(history[-1]), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / "BASE_TRAINING_INCOMPLETE"
    incomplete.touch()
    policy.save_pretrained(output)
    adapter.preprocessor.save_pretrained(output, config_filename="policy_preprocessor.json")
    adapter.postprocessor.save_pretrained(output, config_filename="policy_postprocessor.json")
    (output / "deployment.json").write_text(json.dumps(adapter.deployment, indent=2) + "\n")
    manifest = {"objective": "success_only_original_flow_matching", "source_checkpoint_id": adapter.checkpoint_id,
                "source_processor_id": adapter.processor_id, "lerobot_commit": LEROBOT_COMMIT,
                "supervised_batch_sha256": batch_hash, "source_episode_ids": sorted(set(source_ids)),
                "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "seed": args.seed, "normalization": "original saved processors retained",
                "resource_guard": resource_report,
                "provenance_validation": "NPZ schema checked; raw-recording correspondence requires prior review",
                "trainable_parameters": sum(parameter.numel() for parameter in trainable), "history": history}
    (output / "base_training.json").write_text(json.dumps(manifest, indent=2) + "\n")
    incomplete.unlink()
    (output / "PI05_COMPLETE").touch()
    print(json.dumps({"saved_checkpoint": str(output), "objective": manifest["objective"]}), flush=True)


if __name__ == "__main__":
    main()
