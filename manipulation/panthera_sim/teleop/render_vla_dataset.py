"""Render recorded RobotArmLearning episodes for two-camera VLA training.

The exported observation pair is deliberately the same pair used during
teleoperation: the fixed over-the-shoulder view (``shoulder``) and the camera
mounted on link6 (``wrist``).  Frames use a uniform simulation-time grid; legacy simulation clocks and
finger states are reconstructed and explicitly marked as estimates.

This is the rendering half of the pipeline.  The companion
``VLA-Adapter/scripts/build_robot_arm_learning_rlds.py`` turns this directory into a
TFDS/RLDS dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sim"))
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402

from panthera_env import PantheraSim  # noqa: E402
from sim.dynamics import recorded_dynamics
from teleop.dataset_contract import DEFAULT_HZ, DEFAULT_RENDERED, source_signature, render_signature

DEFAULT_INPUT = REPO_ROOT / "data"
DEFAULT_OUTPUT = DEFAULT_RENDERED
SHOULDER_AZIMUTH = 14.0
SHOULDER_ELEVATION = -34.0
SHOULDER_DISTANCE = 1.20
SHOULDER_LOOKAT = (0.44, 0.0, 0.12)


def shoulder_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth = SHOULDER_AZIMUTH
    cam.elevation = SHOULDER_ELEVATION
    cam.distance = SHOULDER_DISTANCE
    cam.lookat[:] = SHOULDER_LOOKAT
    return cam


def wrist_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    if camera_id < 0:
        raise SystemExit("scene has no 'wrist' camera; run `python sim/make_mjcf.py`")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = camera_id
    return cam


def recording_times(source, sim_dt: float) -> tuple[np.ndarray, str]:
    if "sim_time" in source:
        t = np.asarray(source["sim_time"], dtype=np.float64)
        method = "recorded_simulation_time"
    else:
        # Legacy teleop used int(wall_dt / sim.dt) physics ticks each frame.
        # Reconstruct that clock rather than labelling wall time as sim time.
        wall = np.asarray(source["t"], dtype=np.float64)
        ticks = np.maximum(np.floor(np.clip(np.diff(wall), .001, .1) / sim_dt), 1)
        t = np.r_[0., np.cumsum(ticks) * sim_dt]
        method = "reconstructed_legacy_simulation_time"
    t = t - t[0]
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Recording timestamps must be finite and strictly increasing")
    return t, method


def sample_indices(t: np.ndarray, hz: float) -> np.ndarray:
    """Source bracketing indices; never collapse the uniform time grid."""
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("sample rate must be positive")
    t = np.asarray(t) - t[0]
    return np.searchsorted(t, np.arange(0., t[-1] + 1e-9, 1 / hz), side="right") - 1


def interpolate(t, values, sample_t, quaternion=False):
    values = np.asarray(values)
    if quaternion:
        values = values / np.linalg.norm(values, axis=-1, keepdims=True)
        left = np.clip(np.searchsorted(t, sample_t, side="right") - 1, 0, len(t) - 2)
        fraction = (np.asarray(sample_t) - np.asarray(t)[left]) / (np.asarray(t)[left+1] - np.asarray(t)[left])
        fraction = fraction.reshape((-1,) + (1,) * (values.ndim - 1))
        q0, q1 = values[left], values[left+1]
        dot = np.sum(q0 * q1, axis=-1, keepdims=True)
        q1 = np.where(dot < 0, -q1, q1)
        dot = np.clip(np.abs(dot), 0., 1.)
        theta = np.arccos(dot)
        denominator = np.maximum(np.sin(theta), 1e-12)
        curved = (np.sin((1-fraction)*theta)*q0 + np.sin(fraction*theta)*q1) / denominator
        result = np.where(dot > .9995, (1-fraction)*q0 + fraction*q1, curved)
        return result / np.linalg.norm(result, axis=-1, keepdims=True)
    flat = values.reshape(len(t), -1)
    return np.stack([np.interp(sample_t, t, col) for col in flat.T], axis=1).reshape(
        (len(sample_t), *values.shape[1:]))


def reconstruct_fingers(source, t, sim):
    """Estimate missing legacy fingers by replaying their actuator/contact dynamics.

    Arm/cube poses are restored each raw tick to constrain drift. This cannot
    recover exact historical contacts; new recordings store measured fingers.
    """
    if "finger_q" in source:
        return np.asarray(source["finger_q"]).copy(), "recorded"
    sim.reset(randomize=False)
    sim.data.qpos[sim.finger_qadr] = [.04, -.04]
    fingers = []
    for i in range(len(t)):
        sim.data.qpos[sim.arm_qadr] = source["q"][i]
        sim.data.qvel[sim.arm_dofadr] = source["dq"][i] if "dq" in source else 0
        sim.set_object_poses(source["obj_pos"][i], source["obj_quat"][i])
        for adr in sim.object_dofadr:
            sim.data.qvel[adr:adr + 6] = 0
        sim.data.ctrl[:] = source["ctrl"][i]
        mujoco.mj_forward(sim.model, sim.data)
        if i:
            mujoco.mj_step(sim.model, sim.data, nstep=max(1, round((t[i] - t[i-1]) / sim.dt)))
        fingers.append(sim.data.qpos[sim.finger_qadr].copy())
    return np.asarray(fingers), "reconstructed_dynamics"


def cache_valid(output: Path, signature: dict, settings: dict) -> bool:
    try:
        saved = json.loads((output / "source.json").read_text())
        if saved["source_signature"] != signature or saved["render_signature"] != settings:
            return False
        with np.load(output / "trajectory.npz") as data:
            count = len(data["q"])
        return count == saved["frames"] and all(
            (output / camera / f"{i:05d}.jpg").is_file()
            for camera in ("shoulder", "wrist") for i in range(count))
    except (OSError, ValueError, KeyError):
        return False


def render_episode(path: Path, output: Path, sim: PantheraSim,
                   renderers, cameras, hz: float, quality: int, settings=None) -> int:
    signature = source_signature(path)
    mode = recorded_dynamics(json.loads((path / "meta.json").read_text()))
    # Only legacy finger reconstruction simulates here; recorded poses render
    # directly. Match the source's contact impedance for that reconstruction.
    sim.model.opt.impratio = 100 if mode == "contact-v2" else 10
    settings = settings or render_signature(hz, renderers[0].height, quality)
    with np.load(path / "data.npz") as source:
        required = {"q", "ctrl", "ee_pos", "ee_quat", "obj_pos", "obj_quat", "gripper"}
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        if source['obj_pos'].shape[1] != len(sim.object_names):
            raise ValueError(f"{path}: episode object count does not match rendering scene")
        t, clock_method = recording_times(source, sim.dt)
        native_hz = 1 / np.median(np.diff(t))
        if hz > native_hz * 1.05:
            raise ValueError(f"{path}: {hz:g} Hz exceeds recorded rate {native_hz:.1f} Hz")
        sample_t = np.arange(0., t[-1] + 1e-9, 1 / hz)
        indices = sample_indices(t, hz)
        arrays = {key: interpolate(t, source[key], sample_t, key.endswith("quat")) for key in required}
        finger_q, finger_method = reconstruct_fingers(source, t, sim)
        arrays["finger_q"] = interpolate(t, finger_q, sample_t)
    if len(indices) < 2:
        raise ValueError(f"{path}: fewer than two samples at {hz:g} Hz")
    temp = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=output.parent))
    try:
        (temp / "shoulder").mkdir()
        (temp / "wrist").mkdir()
        sim.reset(randomize=False)
        for frame_i in range(len(indices)):
            sim.data.qpos[:] = sim.model.qpos0
            sim.data.qvel[:] = 0.
            sim.data.qpos[sim.arm_qadr] = arrays["q"][frame_i]
            sim.data.qpos[sim.finger_qadr] = arrays["finger_q"][frame_i]
            sim.data.ctrl[:] = arrays["ctrl"][frame_i]
            sim.set_object_poses(arrays["obj_pos"][frame_i], arrays["obj_quat"][frame_i])
            mujoco.mj_forward(sim.model, sim.data)
            for name, renderer, camera in zip(("shoulder", "wrist"), renderers, cameras):
                renderer.update_scene(sim.data, camera)
                bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(temp / name / f"{frame_i:05d}.jpg"), bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, quality]):
                    raise OSError("could not write rendered image")
        np.savez_compressed(temp / "trajectory.npz", source_indices=indices,
                            t=sample_t, sample_hz=np.float32(hz),
                            **{key: value.astype(np.float32) for key, value in arrays.items()})
        if source_signature(path) != signature:
            raise ValueError(f"{path}: recording changed during rendering")
        (temp / "source.json").write_text(json.dumps({
            "episode": path.name, "source": str(path.resolve()), "frames": len(indices),
            "sample_hz": hz, "cameras": ["shoulder", "wrist"],
            "source_signature": signature, "render_signature": settings,
            "clock_method": clock_method, "finger_method": finger_method,
            "simulation_dynamics": mode,
            "native_hz": native_hz,
        }, indent=2) + "\n")
        if output.exists():
            shutil.rmtree(output)
        temp.rename(output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return len(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not np.isfinite(args.hz) or args.hz <= 0 or args.size <= 0 or not 1 <= args.jpeg_quality <= 100:
        parser.error("invalid rate, image size, or JPEG quality")
    settings = render_signature(args.hz, args.size, args.jpeg_quality)
    episodes = sorted(p for p in args.input.glob("episode_*")
                      if (p / "data.npz").is_file())
    if not episodes:
        raise SystemExit(f"no episodes found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    scenes = {json.loads((p / 'meta.json').read_text()).get('scene', 'sim/panthera/scene.xml')
              for p in episodes}
    if len(scenes) != 1:
        raise SystemExit('Render one scene per dataset.')
    sim = PantheraSim(Path(__file__).resolve().parents[1] / scenes.pop())
    renderers = (
        mujoco.Renderer(sim.model, height=args.size, width=args.size),
        mujoco.Renderer(sim.model, height=args.size, width=args.size),
    )
    cameras = (shoulder_camera(sim.model), wrist_camera(sim.model))
    total = 0
    try:
        for number, episode in enumerate(episodes, 1):
            destination = args.output / episode.name
            complete = cache_valid(destination, source_signature(episode), settings)
            if complete and not args.force:
                with np.load(destination / "trajectory.npz") as cached:
                    count = len(cached["q"])
                print(f"[{number:03d}/{len(episodes):03d}] {episode.name}: cached ({count} frames)", flush=True)
            else:
                count = render_episode(episode, destination, sim, renderers,
                                       cameras, args.hz, args.jpeg_quality, settings)
                print(f"[{number:03d}/{len(episodes):03d}] {episode.name}: {count} frames", flush=True)
            total += count
    finally:
        for renderer in renderers:
            renderer.close()

    manifest = {
        "format": "robot-arm-learning-vla-render-v2",
        "render_signature": settings,
        "episodes": [p.name for p in episodes],
        "num_episodes": len(episodes),
        "num_frames": total,
        "sample_hz": args.hz,
        "image_size": [args.size, args.size],
        "cameras": {"primary": "shoulder", "wrist": "wrist"},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"rendered {len(episodes)} episodes / {total} observations to {args.output}")


if __name__ == "__main__":
    main()
