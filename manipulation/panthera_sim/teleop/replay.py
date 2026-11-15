"""Replay one recorded keyboard-teleoperation episode in MuJoCo.

    python teleop/replay.py data/episode_000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sim"))
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402

from panthera_env import PantheraSim  # noqa: E402


def replay(path: Path) -> None:
    with np.load(path / "data.npz") as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if "t" not in data or "q" not in data:
        raise SystemExit(f"{path / 'data.npz'} does not contain t and q")

    sim = PantheraSim(REPO_ROOT / meta.get('scene', 'sim/panthera/scene.xml'))
    renderer = mujoco.Renderer(sim.model, height=720, width=960)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(sim.model, camera)
    camera.azimuth, camera.elevation, camera.distance = 14.0, -34.0, 1.20
    camera.lookat[:] = (0.44, 0.0, 0.12)
    print(f"replaying {path.name}: {len(data['t'])} steps, "
          f"{meta.get('duration_s', float(data['t'][-1])):.1f}s")

    try:
        for index, seconds in enumerate(data["t"]):
            sim.data.qpos[sim.arm_qadr] = data["q"][index]
            if "finger_q" in data:
                sim.data.qpos[sim.finger_qadr] = data["finger_q"][index]
            if ("obj_pos" in data and "obj_quat" in data
                    and data["obj_pos"].shape[1] == len(sim.object_names)):
                sim.set_object_poses(data["obj_pos"][index],
                                     data["obj_quat"][index])
            if "ctrl" in data:
                sim.data.ctrl[:] = data["ctrl"][index]
            mujoco.mj_forward(sim.model, sim.data)
            renderer.update_scene(sim.data, camera)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f"{path.name}  t={seconds:5.2f}s", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.imshow("RobotArmLearning replay", frame)
            delay = (data["t"][index + 1] - seconds
                     if index + 1 < len(data["t"]) else 0.03)
            if cv2.waitKey(max(int(delay * 1000), 1)) & 0xFF == ord("q"):
                break
    finally:
        renderer.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    replay(args.episode)


if __name__ == "__main__":
    main()
