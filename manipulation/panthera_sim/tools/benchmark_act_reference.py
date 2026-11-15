#!/usr/bin/env python3
"""Measure reference ACT input-worker and precision settings from a checkpoint."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from tools.act_reference_data import ReferenceACTDataset, reference_split
from train_act import seed_everything, training_loss


def measure(checkpoint, warmup, steps, workers, amp, autotune, channels_last=False):
    seed_everything(0)
    torch.backends.cudnn.benchmark = autotune
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = "cuda"
    cfg.use_amp = amp
    policy = ACTPolicy.from_pretrained(checkpoint, config=cfg).cuda()
    if channels_last:
        policy.model.backbone.to(memory_format=torch.channels_last)
    pre, _ = make_pre_post_processors(cfg, pretrained_path=str(checkpoint))
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in policy.named_parameters() if "backbone" not in n]},
        {"params": [p for n, p in policy.named_parameters() if "backbone" in n]}])
    optimizer.load_state_dict(state["optimizer"])
    del state
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    contract = json.loads((checkpoint / "deployment.json").read_text())
    dataset = ReferenceACTDataset(contract["dataset"], reference_split()[0], cfg.chunk_size, training=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=workers,
                                       pin_memory=True, persistent_workers=True, prefetch_factor=2)
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    losses, scales = [], []
    start = None
    for step in range(warmup + steps):
        if step == warmup:
            torch.cuda.synchronize()
            start = time.monotonic()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = pre(batch)
        if channels_last:
            for key in cfg.image_features:
                batch[key] = batch[key].contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss, _ = training_loss(policy, batch, "vae")
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite benchmark loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        if step >= warmup:
            losses.append(float(loss.detach()))
            scales.append(scaler.get_scale())
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    result = dict(workers=workers, amp=amp, cudnn_benchmark=autotune,
                  channels_last=channels_last,
                  prefetch_factor=2, batch_size=8, measured_steps=steps,
                  steps_per_second=steps / elapsed, mean_loss=sum(losses) / len(losses),
                  scaler_values=sorted(set(scales)),
                  peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3)
    del iterator, loader, policy, optimizer, pre, batch, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    results = []
    for workers, amp, autotune, channels_last in ((2, False, False, False), (4, False, False, False),
                                    (4, True, False, False), (4, True, True, False), (4, True, True, True)):
        result = measure(args.checkpoint, args.warmup, args.steps, workers, amp, autotune, channels_last)
        results.append(result)
        print(json.dumps(result), flush=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
