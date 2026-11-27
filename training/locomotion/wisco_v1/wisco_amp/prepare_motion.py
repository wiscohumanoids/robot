#!/usr/bin/env python3
"""Convert a mjlab tracking motion .npz to an amp-rsl-rl .npy dataset.

  .venv-mjlab/bin/python training/locomotion/wisco_v1/wisco_amp/prepare_motion.py \
      --input /tmp/walk.npz --output ~/hsim_data/amp/walk.npy

amp-rsl-rl expects a dict with xyzw quaternions and one entry per frame. The
input .npz is the normal mjlab tracking format (wxyz body quaternions, root at
body index 0, 29 policy joints).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "robots" / "wisco_v1" / "model"
INFO = json.loads((MODEL_DIR / "train_model_info.json").read_text())
JOINTS = list(INFO["policy_joints_in_model_order"])
ROOT_BODY_INDEX = 0


def convert(input_path: str | Path, output_path: str | Path) -> dict:
  """Load one .npz and write the amp-rsl-rl .npy dict."""
  source = Path(input_path).expanduser()
  dest = Path(output_path).expanduser()
  with np.load(source, allow_pickle=False) as z:
    joint_pos = np.asarray(z["joint_pos"], dtype=np.float32)
    body_pos = np.asarray(z["body_pos_w"], dtype=np.float32)
    body_quat = np.asarray(z["body_quat_w"], dtype=np.float32)
    fps = float(np.asarray(z["fps"]).reshape(-1)[0]) if "fps" in z.files else 50.0

  if joint_pos.shape[1] != len(JOINTS):
    raise ValueError(f"{source}: {joint_pos.shape[1]} joints, expected {len(JOINTS)} from train_model_info")
  if body_pos.shape[1] <= ROOT_BODY_INDEX:
    raise ValueError(f"{source}: no body at index {ROOT_BODY_INDEX}")

  root_pos = body_pos[:, ROOT_BODY_INDEX, :]
  # mjlab/MuJoCo body quats are wxyz; amp-rsl-rl uses xyzw.
  w, x, y, zq = (body_quat[:, ROOT_BODY_INDEX, i] for i in range(4))
  root_quat = np.stack((x, y, zq, w), axis=-1).astype(np.float32)

  payload = {
    "joints_list": JOINTS,
    "joint_positions": [row.copy() for row in joint_pos],
    "root_position": [row.copy() for row in root_pos],
    "root_quaternion": [row.copy() for row in root_quat],
    "fps": fps,
  }
  dest.parent.mkdir(parents=True, exist_ok=True)
  np.save(dest, payload, allow_pickle=True)
  print(f"wrote {dest}: {len(joint_pos)} frames, {len(JOINTS)} joints, {fps:g} fps")
  return payload


def main() -> None:
  """Command line entry point."""
  ap = argparse.ArgumentParser()
  ap.add_argument("--input", required=True, help="mjlab tracking .npz")
  ap.add_argument("--output", required=True, help="output .npy for amp-rsl-rl")
  args = ap.parse_args()
  convert(args.input, args.output)


if __name__ == "__main__":
  main()
