"""Episode storage shared by keyboard teleoperation and replay tools."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np


FIELDS = (
    "t",
    "q",
    "dq",
    "ctrl",
    "ee_pos",
    "ee_quat",
    "obj_pos",
    "obj_quat",
    "target_pos",
    "target_quat",
    "gripper",
    "ik_pos_err",
    "ik_rot_err",
)


class Episode:
    def __init__(self, *, simulation_dynamics: str = "contact-v2") -> None:
        self.simulation_dynamics = simulation_dynamics
        self.rows: list[dict] = []
        self.sim_frames: list[np.ndarray] = []
        self.t0 = time.time()

    def add(self, row: dict, sim: np.ndarray | None = None) -> None:
        self.rows.append(row)
        if sim is not None:
            self.sim_frames.append(sim)

    def __len__(self) -> int:
        return len(self.rows)

    def save(self, out_dir: Path, meta: dict, fps: float,
             save_video: bool) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        optional = ("sim_time", "finger_q", "finger_dq", "physics_steps")
        fields = (*FIELDS, *(key for key in optional if self.rows and all(key in row for row in self.rows)))
        arrays = {key: np.asarray([row[key] for row in self.rows])
                  for key in fields}
        np.savez_compressed(out_dir / "data.npz", **arrays)

        saved_meta = dict(meta)
        saved_meta.setdefault("simulation_dynamics", self.simulation_dynamics)
        duration = float(self.rows[-1]["t"]) if self.rows else 0.0
        saved_meta.update({
            "n_steps": len(self.rows),
            "duration_s": duration,
            "fps_actual": len(self.rows) / max(duration, 1e-6),
            "fields": {key: list(np.shape(self.rows[0][key]))
                       for key in fields},
        })
        (out_dir / "meta.json").write_text(
            json.dumps(saved_meta, indent=2) + "\n")

        if save_video and self.sim_frames:
            write_video(out_dir / "sim.mp4", self.sim_frames, fps)
        print(f"  saved {len(self.rows)} steps -> {out_dir}")
        print(f"    {duration:.1f}s @ {saved_meta['fps_actual']:.1f} Hz")


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0),
        (width, height))
    if not writer.isOpened():
        raise OSError(f"could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def next_episode_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    number = 0
    while (root / f"episode_{number:03d}").exists():
        number += 1
    return root / f"episode_{number:03d}"
