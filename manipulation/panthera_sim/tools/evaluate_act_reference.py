#!/usr/bin/env python3
"""Evaluate our ACT checkpoints in the original ACT transfer-cube simulator."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=ROOT / "outputs/act_reference/upstream")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--video", action="store_true", help="save first rollout")
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        parser.error("episodes and steps must be positive")
    contract = json.loads((args.checkpoint / "deployment.json").read_text())
    if contract.get("benchmark") != "tonyzhaozh/act":
        parser.error("checkpoint is not an ACT reference benchmark model")
    sys.path.insert(0, str(args.upstream.resolve()))
    from sim_env import make_sim_env, BOX_POSE
    from utils import sample_box_pose
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ACTPolicy.from_pretrained(args.checkpoint, config=cfg).to(cfg.device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(args.checkpoint))
    args.output.mkdir(parents=True, exist_ok=True)
    env = make_sim_env(contract["task"])
    results = []
    with torch.inference_mode():
        for episode in range(args.episodes):
            np.random.seed(args.seed + episode)
            torch.manual_seed(args.seed + episode)
            BOX_POSE[0] = sample_box_pose()
            ts = env.reset()
            policy.reset()
            rewards = []
            writer = None
            if args.video and episode == 0:
                import cv2
                writer = cv2.VideoWriter(str(args.output / "rollout_0.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 50, (640, 480))
            try:
                for _ in range(args.steps):
                    obs = ts.observation
                    if writer is not None:
                        writer.write(obs["images"]["top"][..., ::-1])
                    batch = {"observation.state": torch.from_numpy(obs["qpos"]).float().unsqueeze(0),
                             "observation.images.top": torch.from_numpy(obs["images"]["top"].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255}
                    action = post(policy.select_action(pre(batch)))[0].cpu().numpy()
                    if not np.isfinite(action).all():
                        raise RuntimeError("non-finite predicted action")
                    ts = env.step(action)
                    rewards.append(float(ts.reward))
            finally:
                if writer is not None:
                    writer.release()
            result = {"seed": args.seed + episode, "return": sum(rewards),
                      "highest_reward": max(rewards), "success": max(rewards) == env.task.max_reward}
            results.append(result)
            summary = {"checkpoint": str(args.checkpoint.resolve()), "episodes": results,
                       "success_rate": float(np.mean([r["success"] for r in results])),
                       "steps": args.steps, "seed": args.seed}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    env.close()


if __name__ == "__main__":
    main()
