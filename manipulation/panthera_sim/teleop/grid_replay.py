"""Render many recorded episodes into one presentation-ready grid video.

The default command finds the newest 36 episodes, stretches each replay to the
same 12 second running time, and writes a 6x6, 1920x1080 MP4.  Every tile uses
the same fixed camera in front of the table, looking back toward the robot.

    python teleop/grid_replay.py

Pass episode directories to choose them explicitly:

    python teleop/grid_replay.py data/episode_{000..035} --output replays.mp4

Use ``--timing realtime`` to preserve the recorded speed (short episodes hold
their final pose), or ``--no-labels`` for a clean wall of footage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sim"))

# Headless rendering must be selected before importing MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402

from panthera_env import PantheraSim  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_OUTPUT = REPO_ROOT / "replay_grid.mp4"


@dataclass
class Replay:
    path: Path
    label: str
    t: np.ndarray
    q: np.ndarray
    ctrl: np.ndarray | None
    obj_pos: np.ndarray | None
    obj_quat: np.ndarray | None
    finger_q: np.ndarray | None = None
    scene: str = 'sim/panthera/scene.xml'

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    def index_at(self, seconds: float) -> int:
        """Nearest recorded sample at or before ``seconds``."""
        return int(np.clip(np.searchsorted(self.t, seconds, side="right") - 1,
                           0, len(self.t) - 1))


def _episode_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def find_episodes(data_dir: Path, count: int) -> list[Path]:
    candidates = [p for p in data_dir.glob("episode_*")
                  if (p / "data.npz").is_file()]
    candidates.sort(key=lambda p: (_episode_number(p), p.name))
    if len(candidates) < count:
        raise SystemExit(f"found only {len(candidates)} episodes in {data_dir}; "
                         f"need {count}")
    return candidates[-count:]


def load_replay(path: Path) -> Replay:
    path = path.resolve()
    npz_path = path / "data.npz"
    if not npz_path.is_file():
        raise SystemExit(f"missing {npz_path}")
    with np.load(npz_path) as data:
        if "t" not in data or "q" not in data:
            raise SystemExit(f"{npz_path} does not contain t and q")
        t = np.asarray(data["t"], dtype=float).copy()
        q = np.asarray(data["q"], dtype=float).copy()
        ctrl = np.asarray(data["ctrl"], dtype=float).copy() if "ctrl" in data else None
        obj_pos = (np.asarray(data["obj_pos"], dtype=float).copy()
                   if "obj_pos" in data else None)
        obj_quat = (np.asarray(data["obj_quat"], dtype=float).copy()
                    if "obj_quat" in data else None)
        finger_q = (np.asarray(data['finger_q'], dtype=float).copy()
                    if 'finger_q' in data else None)
    if not len(t) or len(q) != len(t):
        raise SystemExit(f"{npz_path} is empty or has mismatched t/q arrays")
    # Old recordings normally start near zero, but make the compositor robust
    # to absolute or offset timestamps.
    t -= t[0]
    meta_path = path / 'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return Replay(path, path.name, t, q, ctrl, obj_pos, obj_quat, finger_q,
                  meta.get('scene', 'sim/panthera/scene.xml'))


def front_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    """Camera across the table, facing the robot and its work area."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    # The regular replay camera is behind the robot at ~14 degrees. Rotating
    # it half a turn puts this camera beyond the cubes, looking at the robot.
    cam.azimuth = 194.0
    cam.elevation = -27.0
    cam.distance = 1.12
    cam.lookat[:] = (0.38, 0.0, 0.18)
    return cam


def tile_rect(index: int, columns: int, rows: int,
              width: int, height: int) -> tuple[int, int, int, int]:
    """Return a gap-free top-origin tile rectangle."""
    row, col = divmod(index, columns)
    x0, x1 = width * col // columns, width * (col + 1) // columns
    y0, y1 = height * row // rows, height * (row + 1) // rows
    return x0, y0, x1, y1


def render_video(replays: list[Replay], output: Path, *, columns: int,
                 width: int, height: int, fps: float, duration: float,
                 timing: str, labels: bool, crf: int) -> None:
    rows = (len(replays) + columns - 1) // columns
    tile_w = max(width // columns, 1)
    tile_h = max(height // rows, 1)

    scenes = {replay.scene for replay in replays}
    if len(scenes) != 1:
        raise ValueError('Grid replay requires episodes from the same scene.')
    sim = PantheraSim(REPO_ROOT / scenes.pop())
    renderer = mujoco.Renderer(sim.model, height=tile_h, width=tile_w)
    cam = front_camera(sim.model)

    output.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV reliably produces the intermediate on machines where its H.264
    # encoder is unavailable. If ffmpeg is present, the final transcode is the
    # broadly slideshow-compatible H.264/yuv420p combination.
    tmp = tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}-", suffix=".mp4", dir=output.parent,
        delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(f"could not open video writer for {output}")

    total_frames = max(1, int(round(duration * fps)))
    print(f"rendering {len(replays)} episodes as {columns}x{rows} -> {output}")
    print(f"  {width}x{height}, {fps:g} fps, {duration:g}s, {timing} timing")
    try:
        for frame_i in range(total_frames):
            wall_t = frame_i / fps
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for tile_i, replay in enumerate(replays):
                if timing == "normalized":
                    replay_t = replay.duration * wall_t / max(duration, 1e-9)
                else:
                    replay_t = min(wall_t, replay.duration)
                sample_i = replay.index_at(replay_t)

                # Start each tile from the model state so an old episode with
                # no object arrays cannot inherit cubes from the previous tile.
                sim.data.qpos[:] = sim.model.qpos0
                sim.data.ctrl[:] = 0.0
                sim.data.qpos[sim.arm_qadr] = replay.q[sample_i]
                if replay.finger_q is not None:
                    sim.data.qpos[sim.finger_qadr] = replay.finger_q[sample_i]
                if replay.ctrl is not None:
                    nctrl = min(sim.model.nu, replay.ctrl.shape[1])
                    sim.data.ctrl[:nctrl] = replay.ctrl[sample_i, :nctrl]
                if (replay.obj_pos is not None and replay.obj_quat is not None
                        and replay.obj_pos.shape[1] == len(sim.object_names)):
                    sim.set_object_poses(replay.obj_pos[sample_i],
                                         replay.obj_quat[sample_i])
                mujoco.mj_forward(sim.model, sim.data)
                renderer.update_scene(sim.data, cam)
                tile = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

                x0, y0, x1, y1 = tile_rect(
                    tile_i, columns, rows, width, height)
                if tile.shape[1] != x1 - x0 or tile.shape[0] != y1 - y0:
                    tile = cv2.resize(tile, (x1 - x0, y1 - y0),
                                      interpolation=cv2.INTER_AREA)
                canvas[y0:y1, x0:x1] = tile
                if labels:
                    font_scale = max(0.32, min(x1 - x0, y1 - y0) / 430.0)
                    thickness = max(1, int(round(font_scale * 2)))
                    cv2.putText(canvas, replay.label, (x0 + 7, y0 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (15, 15, 15), thickness + 2, cv2.LINE_AA)
                    cv2.putText(canvas, replay.label, (x0 + 7, y0 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (245, 245, 245), thickness, cv2.LINE_AA)

            # Fine seams keep adjacent skyboxes from visually merging.
            for col in range(1, columns):
                x = width * col // columns
                cv2.line(canvas, (x, 0), (x, height), (8, 8, 8), 2)
            for row in range(1, rows):
                y = height * row // rows
                cv2.line(canvas, (0, y), (width, y), (8, 8, 8), 2)
            writer.write(canvas)
            if frame_i == 0 or (frame_i + 1) % max(int(fps), 1) == 0:
                print(f"\r  frame {frame_i + 1}/{total_frames}", end="", flush=True)
    finally:
        writer.release()
        renderer.close()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        encoded = tmp_path.with_name(tmp_path.stem + "-h264.mp4")
        command = [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp_path),
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded),
        ]
        try:
            subprocess.run(command, check=True)
            os.replace(encoded, output)
            tmp_path.unlink(missing_ok=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            encoded.unlink(missing_ok=True)
            print(f"\nwarning: H.264 encoding failed ({exc}); keeping mp4v")
            os.replace(tmp_path, output)
    else:
        print("\nwarning: ffmpeg not found; keeping mp4v instead of H.264")
        os.replace(tmp_path, output)
    print(f"\nfinished: {output} ({output.stat().st_size / 1_000_000:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="*", type=Path,
                    help="episode directories (default: newest --count in --data-dir)")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--count", type=int, default=36)
    ap.add_argument("--columns", type=int, default=6)
    ap.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=12.0,
                    help="output duration in seconds (default: 12)")
    ap.add_argument("--timing", choices=("normalized", "realtime"),
                    default="normalized",
                    help="stretch every replay to fill the video, or preserve speed")
    ap.add_argument("--no-labels", dest="labels", action="store_false")
    ap.add_argument("--crf", type=int, default=20,
                    help="H.264 quality, lower is better/larger (default: 20)")
    args = ap.parse_args()

    for name, value in (("count", args.count), ("columns", args.columns),
                        ("width", args.width), ("height", args.height)):
        if value <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if args.fps <= 0 or args.duration <= 0:
        ap.error("--fps and --duration must be positive")
    if not 0 <= args.crf <= 51:
        ap.error("--crf must be between 0 and 51")

    paths = args.episodes or find_episodes(args.data_dir, args.count)
    replays = [load_replay(path) for path in paths]
    render_video(replays, args.output.resolve(), columns=args.columns,
                 width=args.width, height=args.height, fps=args.fps,
                 duration=args.duration, timing=args.timing,
                 labels=args.labels, crf=args.crf)


if __name__ == "__main__":
    main()
