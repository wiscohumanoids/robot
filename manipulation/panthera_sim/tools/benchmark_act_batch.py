#!/usr/bin/env python3
"""Benchmark stable ACT training batch sizes on the local Panthera dataset."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

try:
    from tools.lerobot_image_cache import CachedLeRobotDataset, default_cache_path
except ModuleNotFoundError:  # Direct execution adds tools/, not the repo root, to sys.path.
    from lerobot_image_cache import CachedLeRobotDataset, default_cache_path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "outputs" / "lerobot" / "panthera_stack"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80]
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-image-cache", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires CUDA")

    metadata = LeRobotDatasetMetadata("local/panthera_stack", root=args.dataset)
    policy_features = dataset_to_policy_features(metadata.features)
    outputs = {k: v for k, v in policy_features.items() if v.type is FeatureType.ACTION}
    inputs = {k: v for k, v in policy_features.items() if k not in outputs}
    cfg = ACTConfig(
        input_features=inputs,
        output_features=outputs,
        chunk_size=args.chunk_size,
        n_action_steps=10,
        device="cuda",
        use_amp=True,
        push_to_hub=False,
    )
    image_cache = default_cache_path(args.dataset)
    if args.no_image_cache or not image_cache.is_file():
        image_cache = None
    else:
        print(f"using decoded image cache: {image_cache}", flush=True)
    dataset = CachedLeRobotDataset(
        "local/panthera_stack",
        root=args.dataset,
        delta_timestamps={
            "action": [index / metadata.fps for index in cfg.action_delta_indices],
        },
        image_cache=image_cache,
    )
    results = []
    for batch_size in args.batch_sizes:
        torch.manual_seed(20260922)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        )
        iterator = iter(loader)
        policy = None
        optimizer = None
        result = {"batch_size": batch_size, "status": "ok"}
        try:
            policy = ACTPolicy(cfg).cuda().train()
            preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=metadata.stats)
            optimizer = torch.optim.AdamW(
                policy.parameters(), lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay
            )
            compute_timings = []
            end_to_end_timings = []
            losses = []
            total = args.warmup + args.steps
            for step in range(total):
                end_to_end_started = time.perf_counter()
                batch = preprocessor(next(iterator))
                torch.cuda.synchronize()
                compute_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss, _ = policy.forward(batch)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                if step >= args.warmup:
                    finished = time.perf_counter()
                    compute_timings.append(finished - compute_started)
                    end_to_end_timings.append(finished - end_to_end_started)
                    losses.append(float(loss.detach()))
            compute_s = sum(compute_timings) / len(compute_timings)
            end_to_end_s = sum(end_to_end_timings) / len(end_to_end_timings)
            result.update({
                "compute_step_seconds": compute_s,
                "compute_samples_per_second": batch_size / compute_s,
                "end_to_end_step_seconds": end_to_end_s,
                "end_to_end_samples_per_second": batch_size / end_to_end_s,
                "peak_gpu_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "mean_loss": sum(losses) / len(losses),
            })
        except torch.OutOfMemoryError as exc:
            result.update({
                "status": "oom",
                "error": str(exc).split(". Tried to allocate")[0],
                "peak_gpu_gib": torch.cuda.max_memory_reserved() / 1024**3,
            })
        finally:
            del iterator, loader, optimizer, policy
            gc.collect()
            torch.cuda.empty_cache()
        results.append(result)
        print(json.dumps(result), flush=True)

    stable = [item for item in results if item["status"] == "ok"]
    best = max(stable, key=lambda item: item["end_to_end_samples_per_second"]) if stable else None
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "chunk_size": args.chunk_size,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "results": results,
        "best_batch_size": best["batch_size"] if best else None,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
