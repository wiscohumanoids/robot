"""Write synthetic episodes in the raw format F1's generator produces (docs/runs/2026-11-10_overnight_plan.md),
so the converter, fine-tune and server can be tested before real data exists. Not training data.

Each episode: a coloured "cup" square moves on a counter-coloured image while the arms follow smooth
random joint trajectories and the grippers close then open. 20 Hz.

  python make_synthetic_raw.py --out /tmp/synth_raw --episodes 6 --steps 120
"""
import argparse
import json
from pathlib import Path

import numpy as np


def smooth(rng, T, dims, scale):
    k = rng.normal(size=(T // 10 + 2, dims)) * scale
    t = np.linspace(0, len(k) - 1, T)
    return np.stack([np.interp(t, np.arange(len(k)), k[:, d]) for d in range(dims)], 1).astype(np.float32)


def episode(rng, T, H, W):
    q_l = smooth(rng, T, 7, 0.3); q_r = smooth(rng, T, 7, 0.3)
    act_l = np.vstack([q_l[1:], q_l[-1:]]); act_r = np.vstack([q_r[1:], q_r[-1:]])
    grip = np.ones(T, np.float32); grip[T // 3: 2 * T // 3] = 0.0
    img = np.zeros((T, H, W, 3), np.uint8); img[..., :] = (120, 100, 80)
    x = np.linspace(W * 0.2, W * 0.7, T).astype(int); y = int(H * 0.6)
    for t in range(T):
        img[t, y - 12:y + 12, x[t] - 12:x[t] + 12] = (200, 30, 30)
    obj = np.zeros((T, 7), np.float32); obj[:, 0] = x / W; obj[:, 3] = 1.0
    return dict(ego_rgb=img, q_left_arm=q_l, q_right_arm=q_r, q_waist=np.zeros((T, 3), np.float32),
                grip_left=np.ones(T, np.float32), grip_right=grip, act_left_arm=act_l, act_right_arm=act_r,
                act_grip_left=np.ones(T, np.float32), act_grip_right=grip.copy(), obj_pose=obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--h", type=int, default=240)
    ap.add_argument("--w", type=int, default=320)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(a.episodes):
        np.savez_compressed(out / f"episode_{i:06d}.npz", **episode(rng, a.steps, a.h, a.w))
        meta = {"instruction": "pick up the red cup and place it on the plate", "object_category": "cup",
                "object_name": "synthetic_cup", "success": True, "fps": 20,
                "camera_intrinsics": {"fx": 300.0, "fy": 300.0, "cx": a.w / 2, "cy": a.h / 2, "width": a.w, "height": a.h},
                "seed": i, "source": "synthetic"}
        (out / f"episode_{i:06d}.json").write_text(json.dumps(meta))
    print(f"wrote {a.episodes} synthetic episodes to {out}")


if __name__ == "__main__":
    main()
