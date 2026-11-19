"""The Panthera MuJoCo sim (../panthera_sim) as a Robot: rendered cameras, straight-line IK moves, the gripper.

Moves are the scripted collector's (`panthera_sim/tools/collect_scripted.py`, `Planner.move`): a smoothstep along
the line at `motion.control_hz`, bounded IK per tick warm-started from the last solution, then a short hold so the
position servos finish tracking before the next contact transition. Physics runs at the sim's own step.

Ground truth (`truth_objects`) is for evaluation only: the skills never read it.
"""
from __future__ import annotations

import math
import os
import pathlib
import sys

os.environ.setdefault('MUJOCO_GL', 'cgl' if sys.platform == 'darwin' else 'egl')   # offscreen GL: macOS / Linux

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from core.camera import Camera  # noqa: E402
from core.geometry import gripper_rotation  # noqa: E402
from robots.base import MoveResult, Observation  # noqa: E402

SIM_ROOT = pathlib.Path(__file__).resolve().parents[2] / 'panthera_sim'
sys.path.insert(0, str(SIM_ROOT))
from sim.panthera_env import PantheraSim, mat_to_quat  # noqa: E402

# Start and home pose: high and near the base, so the arm hides none of the table from the overhead camera (from the
# collector's start, over the middle of the table, it covered part of a block in 7 of 36 placements).
START_POSITION = np.array([0.20, 0.0, 0.30])


class SimRobot:
    """One simulated Panthera arm on the table, with the scene's overhead and wrist cameras."""

    def __init__(self, config: dict, seed: int | None = 0):
        self.config = config
        self.sim = PantheraSim(SIM_ROOT / 'sim' / 'panthera' / config['scene'])
        self.table_z = float(config['table']['z'])
        self.camera_sizes = {name: (int(c['width']), int(c['height'])) for name, c in config['cameras'].items()}
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.camera_errors: dict[str, np.ndarray] = {}     # S3: name -> 4x4 error in the camera's frame
        m = config['motion']
        self.speed, self.hz, self.max_error = float(m['speed']), float(m['control_hz']), float(m['max_tracking_error'])
        self.close_s = float(config['grasp']['close_s'])
        self.steps_per_tick = max(1, round(1.0 / self.hz / self.sim.dt))
        self.start_rotation = gripper_rotation(0.0, np.deg2rad(config['grasp']['pitch']))
        self.reset(seed)

    # ---------- scene ----------
    def reset(self, seed: int | None = 0) -> None:
        """New random block layout (seeded), arm at the start pose with the jaws open."""
        sim = self.sim
        sim.reset(rng=np.random.default_rng(seed))
        quat = mat_to_quat(self.start_rotation)
        q, pe, re = sim.ik(START_POSITION, quat, max_joint_step=None, iters=150)
        if pe > 1e-3 or re > 1e-2:
            raise RuntimeError(f'start pose unreachable: {pe:.4f} m, {re:.4f} rad')
        sim.data.qpos[sim.arm_qadr] = q
        sim.data.qvel[:] = 0
        sim.data.qpos[sim.finger_qadr] = [0.04, -0.04]
        sim.set_arm_ctrl(q, immediate=True)
        sim.set_gripper(1.0)
        mujoco.mj_forward(sim.model, sim.data)
        self._q = q
        self._target = START_POSITION.copy()
        self._quat = quat
        self._grip = 1.0
        self._hold(0.3)

    def settle(self, seconds: float = 0.5) -> None:
        """Let the scene come to rest (after placing objects by hand), holding the arm where it is."""
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self._hold(seconds)

    def truth_objects(self) -> dict[str, np.ndarray]:
        """{body name: (position (3,), rotation (3, 3))} of the free blocks: evaluation only."""
        pos, quat = self.sim.object_poses()
        out = {}
        for name, p, q in zip(self.sim.object_names, pos, quat):
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, q)
            out[name] = (p, R.reshape(3, 3))
        return out

    # ---------- cameras ----------
    def camera_names(self) -> list[str]:
        """The rendered cameras."""
        return list(self.camera_sizes)

    def _renderer(self, size) -> mujoco.Renderer:
        if size not in self._renderers:
            self._renderers[size] = mujoco.Renderer(self.sim.model, height=size[1], width=size[0])
        return self._renderers[size]

    def camera(self, name: str) -> Camera:
        """The camera's model at the current state (a wrist camera moves with the arm): the ideal one, or with the
        calibration error set in `camera_errors` (the image is still rendered from the true pose)."""
        w, h = self.camera_sizes[name]
        cam = Camera.from_mujoco(self.sim.model, self.sim.data, name, w, h)
        if name in self.camera_errors:
            cam.T_base_cam = cam.T_base_cam @ self.camera_errors[name]
        return cam

    def observe(self, camera: str) -> Observation:
        """RGB plus, for the ground-truth segmenter, the body under each pixel (-1 for none)."""
        model, data = self.sim.model, self.sim.data
        r = self._renderer(self.camera_sizes[camera])
        r.update_scene(data, camera=camera)
        rgb = r.render().copy()
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=camera)
        seg = r.render()
        r.disable_segmentation_rendering()
        ids, types = seg[..., 0], seg[..., 1]
        body = np.full(ids.shape, -1, int)
        is_geom = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        body[is_geom] = model.geom_bodyid[ids[is_geom]]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or '' for b in range(model.nbody)]
        return Observation(self.camera(camera), rgb, float(data.time),
                           extra={'segmentation': body, 'body_names': names})

    # ---------- motion ----------
    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP (grip_site) position and rotation."""
        pos, quat = self.sim.ee_pose()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        return pos, R.reshape(3, 3)

    def reachable(self, position, rotation, tolerance: float = 2e-3) -> bool:
        """IK from the current command reaches the pose within `tolerance` (and 0.02 rad)."""
        _, pe, re = self.sim.ik(np.asarray(position, float), mat_to_quat(rotation), q_init=self._q, iters=200,
                                max_joint_step=None)
        return pe < tolerance and re < 0.02

    def _tick(self, target, quat, grip):
        sim = self.sim
        q, _, _ = sim.ik(target, quat, q_init=self._q, iters=35, max_joint_step=0.08, posture_gain=0.0)
        self._q = q
        sim.set_arm_ctrl(q)
        sim.set_gripper(grip)
        sim.step(self.steps_per_tick)
        if not np.isfinite(sim.data.qpos).all():
            raise RuntimeError('simulator state is not finite')

    def _hold(self, seconds: float) -> None:
        for _ in range(max(1, math.ceil(seconds * self.hz))):
            self._tick(self._target, self._quat, self._grip)

    def move_linear(self, position, rotation, speed=None) -> MoveResult:
        """Smoothstep along the line (orientation by normalised lerp), then hold for the servos to finish."""
        target = np.asarray(position, float)
        quat = mat_to_quat(rotation)
        start, qstart = self._target.copy(), self._quat.copy()
        if np.dot(quat, qstart) < 0:
            quat = -quat
        angle = 2 * np.arccos(np.clip(abs(np.dot(quat, qstart)), -1, 1))
        speed = speed or self.speed
        duration = max(0.4, 1.5 * np.linalg.norm(target - start) / speed, angle / 0.7)
        n = math.ceil(duration * self.hz)
        for i in range(1, n + 1):
            u = i / n
            u = u * u * (3 - 2 * u)
            q = qstart * (1 - u) + quat * u
            self._tick(start * (1 - u) + target * u, q / np.linalg.norm(q), self._grip)
        self._target, self._quat = target, quat
        self._hold(0.3)
        err = float(np.linalg.norm(self.sim.ee_pos() - target))
        if err > self.max_error:
            return MoveResult(False, err, f'tracking error {err * 1000:.1f} mm (unreachable or blocked)')
        return MoveResult(True, err)

    def set_gripper(self, opening: float) -> None:
        """Command the jaws and hold the arm still while they move."""
        self._grip = float(np.clip(opening, 0.0, 1.0))
        self._hold(self.close_s)

    def gripper_width(self) -> float:
        """Jaw separation from the two finger joints (each 0 to 0.04 m)."""
        qL, qR = self.sim.data.qpos[self.sim.finger_qadr]
        return float(qL - qR)

    def home(self) -> MoveResult:
        """Up to travel height, over to the start pose, jaws open."""
        self.set_gripper(1.0)
        pos, _ = self.ee_pose()
        up = pos.copy()
        up[2] = max(up[2], START_POSITION[2])
        r = self.move_linear(up, self.start_rotation)
        return self.move_linear(START_POSITION, self.start_rotation) if r.ok else r
