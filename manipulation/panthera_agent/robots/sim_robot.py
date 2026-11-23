"""The Panthera MuJoCo sim (../panthera_sim) as a Robot: rendered cameras, straight-line IK moves, the gripper.

Moves follow a straight line at `motion.control_hz`, with bounded IK per tick warm-started from the last solution,
then a short hold so the position servos finish tracking before the next contact transition. Physics runs at the
sim's own step. The profile along the line is `motion.profile`: `quintic` (zero speed and acceleration at both ends,
duration from speed and jerk limits; from the Mega Bloks motion on the codex/dino-ik branch) or `smoothstep` (the
scripted collector's, `panthera_sim/tools/collect_scripted.py`).

Scenes: panthera_sim's own (scene.xml, scene_two_blocks.xml) or the agent's (`agent_<name>.xml`, scenes/catalog.py),
which this class lays out on reset from the catalog: random positions and yaws for the free objects, beads inside
their cup, the cloth flat on the table.

Two arms: when the scene and panthera_sim provide them (`PantheraSim.select_arm`, the bimanual scene on the
codex/dino-ik branch), `arm_names()` lists them and `select_arm` switches IK, commands and the wrist camera.

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
from robots.base import MoveResult, Observation, go_home  # noqa: E402
from scenes.catalog import SCENES, TABLE_Z, layout  # noqa: E402

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
        name = config['scene']
        self.scene = SCENES.get(name[len('agent_'):-len('.xml')]) if name.startswith('agent_') else None
        self.table_z = float(config['table']['z'])
        self.camera_sizes = {name: (int(c['width']), int(c['height'])) for name, c in config['cameras'].items()}
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        # Camera images show what a real camera would: no sites (grip_site is drawn as a red ball at the TCP, which
        # the segmenters took for a red object in the wrist view).
        self._camera_option = mujoco.MjvOption()
        self._camera_option.sitegroup[:] = 0
        self.camera_errors: dict[str, np.ndarray] = {}     # S3: name -> 4x4 error in the camera's frame
        self.on_tick = None                                # () -> None after every control tick (tools/viewer.py)
        m = config['motion']
        self.speed, self.hz, self.max_error = float(m['speed']), float(m['control_hz']), float(m['max_tracking_error'])
        self.profile = m.get('profile', 'smoothstep')
        self.max_jerk = float(m.get('max_jerk', 8.0))
        self.on_frame = None                               # (dict) -> None after every control tick (recorders)
        self.close_s = float(config['grasp']['close_s'])
        self.steps_per_tick = max(1, round(1.0 / self.hz / self.sim.dt))
        self.start_rotation = gripper_rotation(0.0, np.deg2rad(config['grasp']['pitch']))
        self.reset(seed)

    # ---------- scene ----------
    def reset(self, seed: int | None = 0) -> None:
        """New random block layout (seeded), arm at the start pose with the jaws open."""
        sim = self.sim
        rng = np.random.default_rng(seed)
        sim.reset(rng=rng)
        if self.scene is not None:
            self._lay_out(rng)
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

    def _lay_out(self, rng) -> None:
        """The agent scene's seeded layout (scenes/catalog.py), written over panthera_sim's cube scatter."""
        import mujoco as mj
        sim, m, d = self.sim, self.sim.model, self.sim.data
        spots = layout(self.scene, rng)
        cups = {}
        for adr, body in zip(sim.object_qadr, sim.object_bodies):
            name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, body)
            if name in spots:
                (x, y), yaw = spots[name]
                d.qpos[adr:adr + 3] = (x, y, TABLE_Z + 0.0005)
                d.qpos[adr + 3:adr + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
                cups[name] = (x, y)
        if self.scene.beads:
            b = self.scene.beads
            cx, cy = cups[b['into']]
            k = 0
            for adr, body in zip(sim.object_qadr, sim.object_bodies):
                if mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, body).startswith('bead'):
                    a = 2.4 * k
                    r = 0.006 + 0.0022 * (k % 4)
                    d.qpos[adr:adr + 3] = (cx + r * np.cos(a), cy + r * np.sin(a),
                                           TABLE_Z + 0.012 + 0.014 * (k // 4))
                    d.qpos[adr + 3:adr + 7] = (1, 0, 0, 0)
                    k += 1
        for b in self.scene.bodies:                  # fixtures placed at random: move the body itself
            if not b.free and b.name in spots:
                bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, b.name)
                (x, y), yaw = spots[b.name]
                m.body_pos[bid] = (x, y, TABLE_Z + 0.0005)
                m.body_quat[bid] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        mj.mj_forward(m, d)

    def flex_names(self) -> list[str]:
        """The deformable objects (the towel)."""
        import mujoco as mj
        return [mj.mj_id2name(self.sim.model, mj.mjtObj.mjOBJ_FLEX, i) or f'flex{i}'
                for i in range(self.sim.model.nflex)]

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

    def truth_flex(self) -> dict[str, np.ndarray]:
        """{flex name: (N, 3) vertex positions}: the cloth's shape, evaluation only."""
        m, d = self.sim.model, self.sim.data
        out = {}
        for i, name in enumerate(self.flex_names()):
            a, n = m.flex_vertadr[i], m.flex_vertnum[i]
            out[name] = d.flexvert_xpos[a:a + n].copy()
        return out

    def close(self) -> None:
        """Free the GL renderers (several sims in one process: close one before starting the next)."""
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

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
        r.update_scene(data, camera=camera, scene_option=self._camera_option)
        rgb = r.render().copy()
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=camera, scene_option=self._camera_option)
        seg = r.render()
        r.disable_segmentation_rendering()
        ids, types = seg[..., 0], seg[..., 1]
        body = np.full(ids.shape, -1, int)
        is_geom = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        body[is_geom] = model.geom_bodyid[ids[is_geom]]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or '' for b in range(model.nbody)]
        is_flex = (types == int(mujoco.mjtObj.mjOBJ_FLEX)) & (ids >= 0)
        if is_flex.any():                                  # the cloth: one pseudo-body per flex, after the bodies
            body[is_flex] = model.nbody + ids[is_flex]
            names = names + self.flex_names()
        return Observation(self.camera(camera), rgb, float(data.time),
                           extra={'segmentation': body, 'body_names': names})

    def empty_table_image(self, camera: str = 'overhead') -> np.ndarray:
        """The camera's image of the bare table, the arm where it is (the sim's version of photographing the empty
        table at calibration, which the candidate points use as background): every free object and cloth hidden for
        one render, then put back. Nothing is stepped."""
        m, d = self.sim.model, self.sim.data
        qpos = d.qpos.copy()
        alpha = m.flex_rgba[:, 3].copy() if m.nflex else None
        geom_alpha = m.geom_rgba[:, 3].copy()
        if self.scene is not None:                 # fixtures too: they are things on the table, not the table
            for b in self.scene.bodies:
                if not b.free:
                    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b.name)
                    m.geom_rgba[m.geom_bodyid == bid, 3] = 0.0
        for adr in self.sim.object_qadr:
            d.qpos[adr:adr + 3] = (5.0, 5.0, -5.0)
        if m.nflex:
            m.flex_rgba[:, 3] = 0.0
        mujoco.mj_forward(m, d)
        try:
            r = self._renderer(self.camera_sizes[camera])
            r.update_scene(d, camera=camera, scene_option=self._camera_option)
            return r.render().copy()
        finally:
            d.qpos[:] = qpos
            m.geom_rgba[:, 3] = geom_alpha
            if m.nflex:
                m.flex_rgba[:, 3] = alpha
            mujoco.mj_forward(m, d)

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
        if self.on_frame is not None:
            self.on_frame(dict(t=float(sim.data.time), q=q.copy(), target=np.asarray(target, float).copy(),
                               quat=np.asarray(quat, float).copy(), grip=float(grip)))
        if self.on_tick is not None:
            self.on_tick()

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
        dist = float(np.linalg.norm(target - start))
        if self.profile == 'quintic':        # peak speed 1.875x the mean, peak jerk 60 d / T^3
            duration = max(0.4, 1.875 * dist / speed, 1.875 * angle / 0.9, (60 * dist / self.max_jerk) ** (1 / 3))
        else:
            duration = max(0.4, 1.5 * dist / speed, angle / 0.7)
        n = math.ceil(duration * self.hz)
        for i in range(1, n + 1):
            u = i / n
            u = u ** 3 * (10 + u * (-15 + 6 * u)) if self.profile == 'quintic' else u * u * (3 - 2 * u)
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
        """Up as far as reachable, over to the start pose, jaws open."""
        return go_home(self, START_POSITION, self.start_rotation)

    # ---------- arms ----------
    def base_xy(self) -> np.ndarray:
        """Where the active arm's base stands (the world origin for the single-arm scenes)."""
        if hasattr(self.sim, 'model') and self.active_arm == 'right':
            bid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, 'right_base_link')
            if bid >= 0:
                return self.sim.data.xpos[bid][:2].copy()
        return np.zeros(2)

    def arm_names(self) -> list[str]:
        """The arms this scene has (one, unless panthera_sim and the scene provide two)."""
        return list(getattr(self.sim, 'available_arms', None) or ['left'])

    @property
    def active_arm(self) -> str:
        return getattr(self.sim, 'active_arm', 'left')

    def select_arm(self, arm: str) -> None:
        """Drive `arm` from now on (IK, commands, the wrist camera)."""
        if arm == self.active_arm:
            return
        if not hasattr(self.sim, 'select_arm'):
            raise ValueError(f'this sim has one arm, not {arm!r}')
        self.sim.select_arm(arm)
        self._q = self.sim.q.copy()
        pos, quat = self.sim.ee_pose()
        self._target, self._quat = pos.copy(), quat.copy()
