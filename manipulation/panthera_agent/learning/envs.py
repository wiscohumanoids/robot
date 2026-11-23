"""Training environments for the local (finishing) skills, in the sim.

A local skill starts where the geometry hands over (the held object a few millimetres above the target, off by
whatever the perception and calibration got wrong) and ends in contact: inserted, seated, caught. The policy sees
only what it would on the real arm, in the frame of the handoff:

    observation = {'wrist': (64, 64, 3) uint8 wrist image, 'state': (5,) float32}
    state       = TCP position minus the handoff position (m, x y z), yaw turned since the handoff (rad),
                  jaw width (m)
    action      = (5,) in [-1, 1]: dx, dy, dz (each times 3 mm), dyaw (times 3 deg), done (> 0.5 stops)

so a policy trained here does not depend on where on the table the task is (ManipGen, SPHINX: wrist view plus a
relative frame transfers across poses). `local_observation` builds the same thing on any Robot, which is what the
`insert` skill runs a trained policy on.

`InsertEnv`: the tight peg scene (scenes/catalog.py `peg_tight`, 1 mm clearance a side). Reset grasps the peg with
privileged knowledge once per seed (it is setup, not the skill being learned), snapshots the whole physics state
above the hole, and every episode restores it and shifts the hand by a random handoff error. `expert` is a
privileged controller (it reads the true peg and hole poses): the sim's stand-in for a human demonstrator and
supervisor. Success is read from the simulator: the peg's bottom 15 mm below the hole's top.
"""
from __future__ import annotations

import cv2
import mujoco
import numpy as np

from core.geometry import gripper_rotation, rot_z

STEP_M = 0.003
STEP_YAW = np.deg2rad(3.0)
IMAGE = 64


def local_observation(robot, start: np.ndarray, yaw_turned: float) -> dict:
    """The policy's observation on any Robot: the wrist image (64 x 64) and the state relative to `start`."""
    obs = robot.observe('wrist')
    img = cv2.resize(obs.rgb, (IMAGE, IMAGE), interpolation=cv2.INTER_AREA)
    pos, _ = robot.ee_pose()
    state = np.concatenate([pos - start, [yaw_turned, robot.gripper_width()]]).astype(np.float32)
    return {'wrist': img, 'state': state}


def _yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


class InsertEnv:
    """Peg into a tight hole, from a perturbed handoff pose."""

    skill = 'insert'
    max_steps = 40

    def __init__(self, config: dict, seed: int = 0, offset_m: float = 0.006, offset_deg: float = 8.0,
                 depth: float = 0.015, scene: str = 'agent_peg_tight.xml', layouts: list[int] | None = None,
                 spec: dict | None = None):
        """`layouts`: the scene seeds episodes are drawn from (the fixture and the peg are placed differently in
        each); default just `seed`. Train on some, evaluate on others. `spec`: an insertion task from the
        curriculum (scenes/catalog.py insert_scene: peg shape, size, colours, clearance, handoff error); without
        it, the tight square peg scene."""
        from core.build import config_for_scene
        from robots.sim_robot import SimRobot
        self.spec = spec
        if spec is not None:
            from scenes.catalog import insert_scene, register
            scene = register(insert_scene(spec))
            offset_m = float(spec.get('offset_mm', offset_m * 1000)) / 1000
            offset_deg = float(spec.get('offset_deg', offset_deg))
        self.task = spec['id'] if spec else 'peg_tight'
        self.cfg = config_for_scene(config, scene)
        self.layouts = list(layouts) if layouts else [seed]
        self.robot = SimRobot(self.cfg, seed=self.layouts[0])
        self.rng = np.random.default_rng(seed)
        self._snapshots = {}
        self.offset_m, self.offset_deg, self.depth = offset_m, np.deg2rad(offset_deg), depth
        self.pitch = np.deg2rad(self.cfg['grasp']['pitch'])
        m = self.robot.sim.model
        pc, fc = (spec['peg_colour'], spec['fixture_colour']) if spec else ('red', 'orange')
        self._peg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'peg_{pc}')
        self._hole = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'fixture_{fc}')
        self.peg_h = float(spec['peg_height']) if spec else 0.06
        self.round = bool(spec and spec['peg_shape'] == 'cylinder')
        self.fixture_h = 0.035
        self._use(self.layouts[0])

    def _use(self, layout: int) -> None:
        """Switch to a scene layout, setting it up (once) the privileged way."""
        m = self.robot.sim.model
        if layout not in self._snapshots:
            self.robot.reset(layout)
            self.hole_top = float(self.robot.sim.data.xpos[self._hole][2]) + self.fixture_h
            self.feature = self._object_feature()
            self._setup()
            # The fixture is a fixed body placed per layout in the model, not the state: keep its place too.
            self._snapshots[layout] = (self._snapshot, self.R0.copy(), self.handoff.copy(), self.in_hand.copy(),
                                       m.body_pos[self._hole].copy(), m.body_quat[self._hole].copy(),
                                       self.hole_top, self.feature)
        (self._snapshot, self.R0, self.handoff, self.in_hand, pos, quat, self.hole_top,
         self.feature) = self._snapshots[layout]
        m.body_pos[self._hole], m.body_quat[self._hole] = pos, quat
        self.layout = layout

    def close(self) -> None:
        self.robot.close()

    def _object_feature(self):
        """The peg's DINO feature from the overhead image (what the agent's memory would hold for it)."""
        from learning.cond import object_feature
        obs = self.robot.observe('overhead')
        return object_feature(obs.rgb, obs.extra['segmentation'] == self._peg)

    # ---------- privileged set-up ----------
    def _setup(self) -> None:
        """Grasp the peg near its top (true pose) and carry it above the hole; snapshot that state."""
        r, d = self.robot, self.robot.sim.data
        peg = d.xpos[self._peg].copy()
        yaw_peg = _yaw_of(d.xmat[self._peg].reshape(3, 3))
        bearing = float(np.arctan2(peg[1], peg[0]))
        yaw = bearing if self.round else yaw_peg + np.round((bearing - yaw_peg) / (np.pi / 2)) * (np.pi / 2)
        R = gripper_rotation(yaw, self.pitch)
        tcp_off = float(self.cfg['grasp']['tcp_offset'])
        grasp = np.array([peg[0], peg[1], peg[2] + self.peg_h - 0.014]) + tcp_off * R[:, 0]
        r.set_gripper(1.0)
        for p in (grasp + [0, 0, 0.10], grasp + [0, 0, 0.03], grasp):
            r.move_linear(p, R)
        r.set_gripper(0.0)
        r.move_linear(grasp + [0, 0, 0.12], R)
        tcp, Rl = r.ee_pose()
        self.in_hand = Rl.T @ (d.xpos[self._peg].copy() - tcp)               # peg base - TCP, gripper frame
        hole = d.xpos[self._hole].copy()
        hyaw = _yaw_of(d.xmat[self._hole].reshape(3, 3))
        hb = float(np.arctan2(hole[1], hole[0]))
        # The peg turns with the hand: a hand yaw that turns it face-on to the hole, as radial as possible.
        turn = 0.0 if self.round else (hyaw - yaw_peg + np.pi / 4) % (np.pi / 2) - np.pi / 4
        yaw_g = yaw + turn
        yaw_g = hb if self.round else yaw_g + np.round((hb - yaw_g) / (np.pi / 2)) * (np.pi / 2)
        self.R0 = gripper_rotation(yaw_g, self.pitch)
        tcp, _ = r.ee_pose()
        target = np.array([hole[0], hole[1], self.hole_top + 0.006]) - self.R0 @ self.in_hand
        high = max(tcp[2], target[2] + 0.06)
        r.move_linear(np.array([hole[0], hole[1], high]) - (R @ self.in_hand) * [1, 1, 0], R)
        r.move_linear(target + [0, 0, high - target[2]], self.R0)
        r.move_linear(target + [0, 0, 0.03], self.R0)
        r.move_linear(target, self.R0)
        self.handoff = r.ee_pose()[0].copy()
        self._snapshot = self._save()

    def _save(self):
        r = self.robot
        size = mujoco.mj_stateSize(r.sim.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(size)
        mujoco.mj_getState(r.sim.model, r.sim.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        return state, r._q.copy(), r._target.copy(), r._quat.copy(), r._grip, r.sim._applied_arm_ctrl.copy()

    def _restore(self, snap) -> None:
        r = self.robot
        state, q, target, quat, grip, applied = snap
        mujoco.mj_setState(r.sim.model, r.sim.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(r.sim.model, r.sim.data)
        r._q, r._target, r._quat, r._grip = q.copy(), target.copy(), quat.copy(), grip
        r.sim._applied_arm_ctrl = applied.copy()

    # ---------- episodes ----------
    def reset(self, offset: np.ndarray | None = None) -> dict:
        """Back to the snapshot, then the hand shifted by a random (or given) handoff error: (dx, dy, dyaw)."""
        self._use(int(self.rng.choice(self.layouts)))
        self._restore(self._snapshot)
        if offset is None:
            a = self.rng.uniform(0, 2 * np.pi)
            rad = self.offset_m * np.sqrt(self.rng.uniform(0.0, 1.0))
            offset = np.array([rad * np.cos(a), rad * np.sin(a), self.rng.uniform(-1, 1) * self.offset_deg])
        self.offset = np.asarray(offset, float)
        R = rot_z(self.offset[2]) @ self.R0
        self.robot.move_linear(self.handoff + np.array([self.offset[0], self.offset[1], 0.0]), R, speed=0.05)
        # The policy's frame starts where it is handed over, as on the rig: nothing in its state says where the
        # hole is; the wrist image has to.
        self.start = self.robot.ee_pose()[0].copy()
        self.yaw = 0.0
        self.steps = 0
        return self.observe()

    def observe(self) -> dict:
        from learning.cond import condition
        obs = local_observation(self.robot, self.start, self.yaw)
        obs['cond'] = condition(self.skill, self.feature)
        return obs

    def final_views(self) -> dict:
        """Full-resolution images of how the attempt ended, for labelling."""
        return {cam: self.robot.observe(cam).rgb for cam in ('wrist', 'overhead')}

    def peg_state(self):
        d = self.robot.sim.data
        return d.xpos[self._peg].copy(), _yaw_of(d.xmat[self._peg].reshape(3, 3))

    def errors(self):
        """(xy error of the peg from the hole (2,), yaw error folded to +-45 deg, depth below the hole's top)."""
        d = self.robot.sim.data
        peg, pyaw = self.peg_state()
        hole = d.xpos[self._hole]
        hyaw = _yaw_of(d.xmat[self._hole].reshape(3, 3))
        dyaw = 0.0 if self.round else (pyaw - hyaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        return peg[:2] - hole[:2], float(dyaw), float(self.hole_top - peg[2])

    def success(self) -> bool:
        xy, _, depth = self.errors()
        return depth >= self.depth and np.linalg.norm(xy) < 0.004

    def dropped(self) -> bool:
        peg, _ = self.peg_state()
        tcp, R = self.robot.ee_pose()
        return np.linalg.norm(peg - (tcp + R @ self.in_hand)) > 0.02

    def step(self, action: np.ndarray):
        """One 10 Hz step: a small relative move. Returns (observation, reward, done, info)."""
        a = np.clip(np.asarray(action, float), -1, 1)
        self.steps += 1
        if a[4] > 0.5:
            ok = self.success()
            return self.observe(), float(ok), True, dict(success=ok, stopped=True)
        pos, R = self.robot.ee_pose()
        self.yaw += a[3] * STEP_YAW
        self.robot.move_linear(pos + a[:3] * STEP_M, rot_z(a[3] * STEP_YAW) @ R, speed=0.04)
        ok = self.success()
        done = ok or self.steps >= self.max_steps or self.dropped()
        return self.observe(), float(ok), done, dict(success=ok, dropped=self.dropped())

    # ---------- the privileged expert (sim stand-in for a human) ----------
    def expert(self) -> np.ndarray:
        """Centre the peg over the hole and match its yaw; descend once within 0.6 mm; when blocked high, rise."""
        xy, dyaw, depth = self.errors()
        a = np.zeros(5)
        a[:2] = np.clip(-xy / STEP_M, -1, 1)
        a[3] = np.clip(-dyaw / STEP_YAW, -1, 1)
        aligned = np.linalg.norm(xy) < 0.0006 and abs(dyaw) < np.deg2rad(1.0)
        if aligned or depth > 0.002:
            a[2] = -1.0
        elif depth > -0.003:
            a[2] = 0.3                                  # touching the rim while misaligned: back off a little
        if self.success():
            a[:4] = 0.0
            a[4] = 1.0
        return a

    def scripted_search(self) -> np.ndarray:
        """The non-learned baseline the `insert` skill uses: press down; when blocked, spiral (no privileged
        information): an action sequence keyed on the step count."""
        k = self.steps
        if k % 3 != 2:
            return np.array([0, 0, -1.0, 0, 0])
        ang, rad = 2.4 * (k // 3), min(1.0, 0.25 * np.sqrt(k // 3 + 1))
        return np.array([rad * np.cos(ang), rad * np.sin(ang), 0.5, 0, 0])


class SkillEnv:
    """The insertion skill as an environment on any Robot, through the skills (no privileged information): the rig's
    practice environment, and the sim's when the privileged one is not wanted.

    Reset: the supervisor's reset leaves the peg held (or `prepare` picks it: find, pick by its top); then the
    insert skill's own handoff (line up over the hole, lower to just above it), plus an optional extra random error
    so practice covers worse handoffs than the current calibration gives. Success: the geometric check (the held
    object's bottom `depth` below the hole's top, from the arm's pose)."""

    skill = 'insert'
    max_steps = 40

    def __init__(self, skills, peg_query: str, hole_query: str, extra_m: float = 0.0, extra_deg: float = 0.0,
                 depth: float = 0.015, seed: int = 0):
        self.s, self.peg_query, self.hole_query = skills, peg_query, hole_query
        self.extra_m, self.extra_deg, self.depth = extra_m, np.deg2rad(extra_deg), depth
        self.rng = np.random.default_rng(seed)
        self.robot = skills.robot

    def prepare(self) -> None:
        """Hold the peg: find and pick it by its top, if not holding it already."""
        s = self.s
        if s.held is None:
            s.home()
            s.find(self.peg_query)
            peg = max((o for o in s.world.objects.values() if o.step == s.world.step),
                      key=lambda o: (o.trusted, o.score))
            r = s.pick(peg.number, 'top')
            if not r.ok:
                raise RuntimeError(f'could not pick the peg: {r.message}')

    def reset(self) -> dict:
        s = self.s
        self.prepare()
        # The hole is found once (a fixture does not move; with the peg held over it, the arm hides it from above).
        if getattr(self, 'hole_number', None) not in s.world.objects:
            if s.held is None:
                s.home()
            s.find(self.hole_query)
            found = [o for o in s.world.objects.values() if o.step == s.world.step]
            if not found:
                raise RuntimeError(f'no "{self.hole_query}" in view')
            self.hole_number = max(found, key=lambda o: (o.trusted, o.score)).number
        h = s.insert_handoff(self.hole_number)
        if not isinstance(h, tuple):
            raise RuntimeError(f'no handoff: {h.message}')
        start, R, self.hole, self.pre, _ = h
        if self.extra_m or self.extra_deg:
            a = self.rng.uniform(0, 2 * np.pi)
            rad = self.extra_m * np.sqrt(self.rng.uniform())
            R = rot_z(self.rng.uniform(-1, 1) * self.extra_deg) @ R
            self.robot.move_linear(start + [rad * np.cos(a), rad * np.sin(a), 0.0], R, speed=0.03)
        self.start = self.robot.ee_pose()[0].copy()
        self.yaw, self.steps = 0.0, 0
        return self.observe()

    def observe(self) -> dict:
        return local_observation(self.robot, self.start, self.yaw)

    def success(self) -> bool:
        return self.s.held is not None and self.s.insert_depth(self.hole) >= self.depth

    def errors(self):
        return np.zeros(2), 0.0, float(self.s.insert_depth(self.hole))

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        self.steps += 1
        if a[4] > 0.5:
            return self.observe(), float(self.success()), True, dict(success=self.success(), stopped=True)
        pos, R = self.robot.ee_pose()
        self.yaw += a[3] * STEP_YAW
        r = self.robot.move_linear(pos + a[:3] * STEP_M, rot_z(a[3] * STEP_YAW) @ R, speed=0.03)
        ok = self.success()
        done = ok or self.steps >= self.max_steps or not r.ok
        return self.observe(), float(ok), done, dict(success=ok, blocked=not r.ok)

    def scripted_search(self) -> np.ndarray:
        k = self.steps
        if k % 3 != 2:
            return np.array([0, 0, -1.0, 0, 0])
        ang, rad = 2.4 * (k // 3), min(1.0, 0.25 * np.sqrt(k // 3 + 1))
        return np.array([rad * np.cos(ang), rad * np.sin(ang), 0.5, 0, 0])

    def finish(self, success: bool) -> None:
        """After an episode: release into the hole when it went in and lift the peg back out (practice continues),
        or back up."""
        R = self.robot.ee_pose()[1]
        self.robot.move_linear(self.pre, R, speed=0.03)

    def final_views(self) -> dict:
        return {cam: self.robot.observe(cam).rgb for cam in ('wrist', 'overhead')}
