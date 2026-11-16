"""MuJoCo environment and IK for the Panthera-HT arm.

Two pieces:

`PantheraSim` wraps the model: reset, step, read the end-effector, and a damped
least-squares IK that maps a Cartesian target onto the six joints.

IK notes. The Jacobian is taken at `grip_site` and only the six arm DoF are
solved; the fingers are commanded directly. Damped least squares, with the
damping raised as the smallest singular value collapses, so the solve stays
stable through the wrist singularity -- where the arm loses a DoF and an
undamped pseudo-inverse would demand enormous joint velocities -- without
paying for that stability in accuracy everywhere else.

The hard part is not reaching a reachable target; warm-started, that converges
to 0.02 mm in a fifth of a millisecond. It is what happens when the operator
asks for something the arm cannot do, which in teleoperation is constant: you
drive past the edge of the workspace. Three things handle that, and all three were
missing:

- Joints driven into a limit are frozen and the system re-solved on the ones
  still free, rather than being silently clipped. Clipping loses the motion
  that joint was supposed to supply instead of redistributing it, and the
  solve walks into a corner of the joint box -- five joints pinned at once --
  that damped least squares cannot climb back out of. Reaching out of the box
  and coming back used to strand the arm about a quarter of the time, needing
  a manual reset.
- A nullspace bias toward the home posture, which by construction cannot move
  the end-effector, keeps the arm away from limits and singularities in the
  first place and makes the chosen configuration repeatable frame to frame.
- A real rate limit on how far any joint may move in one call. `max_step`
  bounds one interior iteration, not the frame: a hundred iterations could
  still slew a joint 72 degrees in a single tick, which is exactly the snap a
  marker dropout produces.
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np

try:  # Also imported as panthera_env by standalone teleop scripts.
    from .dynamics import CONTACT_DYNAMICS, LEGACY_DYNAMICS, DYNAMICS_MODES
except ImportError:
    from dynamics import CONTACT_DYNAMICS, LEGACY_DYNAMICS, DYNAMICS_MODES

HERE = pathlib.Path(__file__).parent
DEFAULT_SCENE = HERE / "panthera" / "scene.xml"

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_OPEN = 0.04          # metres, per finger
# Legacy weld-v1 command-space hysteresis, retained for historical replay only.
GRASP_LOCK = 0.25
GRASP_UNLOCK = 0.70
GRASP_ALIGN_COS = float(np.cos(np.deg2rad(25.0)))

# Tabletop spawn box for the free cubes. Inset from both the table edge
# (a yawed 45 mm cube's diagonal is 32 mm) and the default control region
# so every reset is reachable without touching --region. Must stay in
# step with TABLE_* / CUBE_HALF in make_mjcf.py.
_CUBE_HALF = 0.0225
_TABLE_TOP = 0.05
_OBJECT_XY_LO = np.array([0.28, -0.22])
_OBJECT_XY_HI = np.array([0.52, 0.22])
_OBJECT_MIN_SEP = 0.09


class PantheraSim:
    def __init__(self, scene: pathlib.Path = DEFAULT_SCENE, ee_site: str = "grip_site",
                 *, dynamics: str = CONTACT_DYNAMICS):
        if dynamics not in DYNAMICS_MODES:
            raise ValueError(f"Unknown simulation dynamics: {dynamics!r}")
        self.dynamics = dynamics
        self.scene_path = pathlib.Path(scene)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.impratio = 100 if dynamics == CONTACT_DYNAMICS else 10
        self.data = mujoco.MjData(self.model)
        self.ee_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self.ee_site < 0:
            raise SystemExit(f"no site named {ee_site!r} in {scene}")

        self.arm_qadr = np.array([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ARM_JOINTS])
        self.arm_dofadr = np.array([
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ARM_JOINTS])
        self.arm_range = np.array([
            self.model.jnt_range[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ARM_JOINTS])
        self.finger_qadr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in ("L_finger_joint", "R_finger_joint")])
        self.finger_dofadr = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in ("L_finger_joint", "R_finger_joint")])
        self.grip_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")

        # Free bodies in the scene -- the cubes on the table. Found by joint
        # type rather than by name so any scene works, and kept in model order
        # so the logged columns line up with `object_names`.
        self.object_qadr, self.object_dofadr, self.object_bodies = [], [], []
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.object_qadr.append(self.model.jnt_qposadr[j])
                self.object_dofadr.append(self.model.jnt_dofadr[j])
                self.object_bodies.append(self.model.jnt_bodyid[j])
        self.object_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in self.object_bodies]
        self._link6 = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        self._pad_L = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "L_finger_pad")
        self._pad_R = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "R_finger_pad")
        self._cube_geoms = {}
        self._grasp_eq = []
        for name in self.object_names:
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._cube_geoms[gid] = name
            self._grasp_eq.append(mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"pad_grasp_{name}"))
        # Scratch state for IK. Allocated once: MjData is expensive to create
        # and the solver is called every control tick.
        self._ik_data = mujoco.MjData(self.model)
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self._grasp_contact_time = np.zeros(len(self.object_names))
        self._grasp_flags = np.zeros(len(self.object_names), dtype=bool)
        self.reset()
        # Posture the nullspace bias pulls toward. The home keyframe rather
        # than the midpoint of each range: joints 2 and 3 are one-sided, so
        # their midpoints are a folded-up pose, and home is a known-good,
        # well-conditioned configuration away from every limit.
        self._q_nominal = self.q

    # ---------- state ----------
    def reset(self, *, randomize: bool = True, rng=None) -> None:
        """Arm back to the home keyframe; cubes re-racked on the table.

        The keyframe is defined in panthera.xml, which knows nothing about the
        scene it gets included into, so it is shorter than `nq` as soon as the
        scene adds free bodies. MuJoCo pads the missing tail with the *identity*
        pose rather than with `qpos0` -- so a plain keyframe reset would teleport
        every cube to the origin, where they land in a heap inside the arm's
        base. The free joints are written afterwards: a fresh random layout
        by default (so `r` is a new grasp scene, not the same three spots),
        or `qpos0` when `randomize` is off.
        """
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        if randomize and self.object_qadr:
            self._scatter_objects(np.random.default_rng() if rng is None else rng)
        else:
            for adr in self.object_qadr:
                self.data.qpos[adr:adr + 7] = self.model.qpos0[adr:adr + 7]
        for adr in self.object_dofadr:
            self.data.qvel[adr:adr + 6] = 0.0
        self._release_grasps()
        self.sync_control_state()
        mujoco.mj_forward(self.model, self.data)

    def _scatter_objects(self, rng) -> None:
        """Sit each free body on the table at a non-overlapping random pose.

        Position is uniform in the spawn box; orientation is a random yaw so
        the cube stays face-down (a free SO(3) sample would leave it on an
        edge or a corner, and it would tip the moment the first step ran).
        Rejection sampling keeps centres at least `_OBJECT_MIN_SEP` apart;
        if a cube cannot be placed it falls back to its XML pose.
        """
        z = _TABLE_TOP + _CUBE_HALF
        placed = []
        axis_z = np.array([0.0, 0.0, 1.0])
        for adr in self.object_qadr:
            xy = None
            for _ in range(80):
                cand = rng.uniform(_OBJECT_XY_LO, _OBJECT_XY_HI)
                if all(np.linalg.norm(cand - p) >= _OBJECT_MIN_SEP for p in placed):
                    xy = cand
                    break
            if xy is None:
                xy = self.model.qpos0[adr:adr + 2].copy()
            placed.append(xy)
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, axis_z, rng.uniform(-np.pi, np.pi))
            self.data.qpos[adr:adr + 3] = (xy[0], xy[1], z)
            self.data.qpos[adr + 3:adr + 7] = quat

    def object_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """(n, 3) positions and (n, 4) quaternions of the free bodies."""
        if not self.object_bodies:
            return np.zeros((0, 3)), np.zeros((0, 4))
        return (self.data.xpos[self.object_bodies].copy(),
                self.data.xquat[self.object_bodies].copy())

    def set_object_poses(self, pos, quat) -> None:
        """Inverse of `object_poses`, for replaying a recorded episode."""
        for i, adr in enumerate(self.object_qadr):
            self.data.qpos[adr:adr + 3] = pos[i]
            self.data.qpos[adr + 3:adr + 7] = quat[i]

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.data.qvel[self.arm_dofadr].copy()

    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site].copy()

    def ee_quat(self) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self.ee_site])
        return q

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.ee_pos(), self.ee_quat()

    # ---------- IK ----------
    def _ee_error(self, target_pos, target_quat, d=None) -> np.ndarray:
        """6-vector [dx, dy, dz, rx, ry, rz] from `d`'s pose to the target.

        The rotation part is the world-frame axis-angle of
        `target * current^-1`, which is the frame `mj_jacSite`'s rotational
        block is also expressed in. `mju_quat2Vel` already takes the short way
        round, so there is no double-cover sign to fix up here.
        """
        d = self.data if d is None else d
        err = np.zeros(6)
        err[:3] = np.asarray(target_pos) - d.site_xpos[self.ee_site]
        if target_quat is not None:
            q_cur = np.zeros(4)
            mujoco.mju_mat2Quat(q_cur, d.site_xmat[self.ee_site])
            q_inv = np.zeros(4)
            mujoco.mju_negQuat(q_inv, q_cur)
            q_err = np.zeros(4)
            mujoco.mju_mulQuat(q_err, np.asarray(target_quat, dtype=float), q_inv)
            mujoco.mju_quat2Vel(err[3:], q_err, 1.0)
        return err

    def _dls(self, J, e, damping, min_damping):
        """Damped least squares with damping raised near a singularity.

        Fixed damping cannot serve both jobs. Small enough to track accurately
        in a well-conditioned pose is too small to stay stable when the arm
        stretches out and a singular value collapses; large enough to be stable
        there costs accuracy everywhere else. So the damping floor is `damping`
        and it is raised only as sigma_min falls below it, in the usual
        Nakamura/Chiaverini way. `J` is 6x6 at most, so the SVD is a few
        microseconds and the solve stays well inside a control tick.
        """
        s = np.linalg.svd(J, compute_uv=False)
        smin = s[-1] if s.size else 0.0
        lam = min_damping
        if smin < damping:
            # ramps from min_damping at sigma=damping to `damping` at sigma=0
            lam = max(min_damping, damping * (1.0 - (smin / damping) ** 2))
        JJt = J @ J.T
        return J.T @ np.linalg.solve(
            JJt + (lam ** 2) * np.eye(JJt.shape[0]), e)

    def ik(self, target_pos, target_quat=None, *, iters: int = 100,
           damping: float = 0.05, min_damping: float = 0.01,
           pos_tol: float = 1e-4, rot_tol: float = 1e-3,
           rot_weight: float = 0.35, max_step: float = 0.25,
           max_joint_step: float | None = 0.15,
           posture_gain: float = 0.05, q_nominal: np.ndarray | None = None,
           limit_margin: float = 0.02,
           q_init: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
        """Solve for arm joints reaching the target. Returns (q, pos_err, rot_err).

        Runs on a scratch copy so a failed solve never disturbs the live sim.
        `rot_weight` below 1 tells the solver to give up orientation before
        position when it cannot have both -- for teleoperated grasping, being in
        the right place matters more than being at exactly the right angle.

        Always warm-start (`q_init`) when you can. Damped least squares is a
        local method: from the previous frame's solution it converges to well
        under 0.1 mm every time, but cold-started across the whole workspace it
        falls into a local minimum on roughly a third of targets. In teleoperation
        the previous solution is always available, so this is a non-issue there --
        but it is why you should not cold-call this for a far-away pose and
        believe the answer without checking the returned error.

        Three things keep an *unreachable* target from wrecking the next solve,
        which is the case teleoperation actually lives in -- see the module
        docstring:

        `limit_margin` drives the clamping loop. A joint pushed into its limit
        used to be silently clipped, which left the motion it was supposed to
        supply simply undelivered: the other five never picked up the slack, and
        the solve walked into a corner of the joint box it could not climb out
        of. Saturated joints are now frozen and the system re-solved on the
        joints that remain free.

        `posture_gain` pulls the arm toward `q_nominal` (the home keyframe by
        default) through the nullspace of the task, so it never affects the
        tracked pose. With rotation tracked the nullspace is usually empty and
        this does nothing; with position only it picks a consistent, away-from-
        the-limits elbow instead of wherever the last frame happened to drift.

        `max_joint_step` caps how far any joint may move from `q_init` in one
        call -- the actual rate limit. `max_step` only ever bounded a single
        interior iteration, so a hundred of them could still slew a joint most
        of its range in one frame; a target moved 400 mm sideways moved a joint
        72 degrees in a single tick. Pass None to
        disable (the self-test does, to measure raw convergence).
        """
        d = self._ik_data
        d.qpos[:] = self.data.qpos
        if q_init is not None:
            d.qpos[self.arm_qadr] = q_init
        q_start = d.qpos[self.arm_qadr].copy()
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)

        jacp, jacr = self._jacp, self._jacr
        lo, hi = self.arm_range[:, 0], self.arm_range[:, 1]
        if q_nominal is None:
            q_nominal = self._q_nominal
        n = len(self.arm_qadr)

        perr = rerr = float("inf")
        for _ in range(iters):
            err = self._ee_error(target_pos, target_quat, d)
            perr = float(np.linalg.norm(err[:3]))
            rerr = float(np.linalg.norm(err[3:]))
            if perr < pos_tol and (target_quat is None or rerr < rot_tol):
                break

            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.ee_site)
            if target_quat is None:
                J_full = jacp[:, self.arm_dofadr]
                e = err[:3]
            else:
                J_full = np.vstack([jacp[:, self.arm_dofadr],
                                    rot_weight * jacr[:, self.arm_dofadr]])
                e = np.concatenate([err[:3], rot_weight * err[3:]])

            q_cur = d.qpos[self.arm_qadr]
            # Clamping loop: solve, find joints the step drives past a limit,
            # freeze them, solve again on the rest. At most `n` passes, since
            # each one freezes at least one more joint.
            frozen = np.zeros(n, dtype=bool)
            dq = np.zeros(n)
            for _ in range(n + 1):
                J = J_full.copy()
                J[:, frozen] = 0.0
                dq = self._dls(J, e, damping, min_damping)
                dq[frozen] = 0.0

                if posture_gain > 0.0:
                    # Nullspace posture bias: (I - J^+ J) k (q_nom - q).
                    # By construction this cannot move the end-effector.
                    Jp = np.linalg.pinv(J, rcond=1e-4)
                    dq += (np.eye(n) - Jp @ J) @ (
                        posture_gain * (q_nominal - q_cur))
                    dq[frozen] = 0.0

                # A joint is newly saturated only if it is already at the edge
                # and this step pushes it further out -- being at a limit while
                # moving back inside is fine and must not be frozen.
                out = (((q_cur <= lo + limit_margin) & (dq < 0)) |
                       ((q_cur >= hi - limit_margin) & (dq > 0))) & ~frozen
                if not out.any():
                    break
                frozen |= out
                if frozen.all():
                    break

            sn = float(np.linalg.norm(dq))
            if sn > max_step:
                dq *= max_step / sn
            d.qpos[self.arm_qadr] = np.clip(q_cur + dq, lo, hi)
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)

        q_out = d.qpos[self.arm_qadr].copy()
        if max_joint_step is not None:
            # Rate limit against the frame we started from. Scaled as a whole
            # rather than clipped per joint, so the arm slews toward the same
            # configuration instead of down a different, distorted path.
            move = q_out - q_start
            big = float(np.abs(move).max())
            if big > max_joint_step:
                q_out = q_start + move * (max_joint_step / big)

        # The last iteration can change q even when the rate limiter does not.
        # Always report the residual of the returned solution, not the pose
        # before that iteration (also covers iters=0).
        d.qpos[self.arm_qadr] = q_out
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        err = self._ee_error(target_pos, target_quat, d)
        perr = float(np.linalg.norm(err[:3]))
        rerr = float(np.linalg.norm(err[3:]))

        return q_out, perr, rerr

    # ---------- actuation ----------
    def set_arm_ctrl(self, q: np.ndarray, *, immediate: bool = False) -> None:
        """Set an interval endpoint; step(n) interpolates at physics rate.

        immediate is for initialization after explicitly restoring qpos.
        Outside step(), data.ctrl holds the endpoint for action logging.
        """
        self.data.ctrl[:6] = np.clip(q, self.arm_range[:, 0], self.arm_range[:, 1])
        if immediate:
            self._applied_arm_ctrl = self.data.ctrl[:6].copy()

    def sync_control_state(self) -> None:
        """Call after restoring qpos/ctrl from a recording or reset snapshot."""
        self._applied_arm_ctrl = self.data.ctrl[:6].copy()
        self._grasp_contact_time[:] = 0
        self._grasp_flags[:] = False

    def pad_normal_forces(self) -> np.ndarray:
        """Per-object left/right compressive loads in newtons; read-only."""
        forces = np.zeros((len(self.object_names), 2))
        indices = {name: i for i, name in enumerate(self.object_names)}
        wrench = np.zeros(6)
        for cindex in range(self.data.ncon):
            contact = self.data.contact[cindex]
            if contact.efc_address < 0:
                continue
            for cube, pad in ((contact.geom1, contact.geom2),
                              (contact.geom2, contact.geom1)):
                name = self._cube_geoms.get(cube)
                if name is not None and pad in (self._pad_L, self._pad_R):
                    mujoco.mj_contactForce(self.model, self.data, cindex, wrench)
                    forces[indices[name], int(pad == self._pad_R)] += max(0., wrench[0])
        return forces

    def grasp_flags(self) -> np.ndarray:
        """Loaded bilateral contact for 20 ms; legacy mode exposes its latch.

        Physical flags require >0.01 N on both pads. Contact loss clears them
        immediately, even at partial opening. Lift/success needs the caller's
        sustained elevation/placement check. Flags never apply object forces.
        """
        if self.dynamics == LEGACY_DYNAMICS:
            return np.array([e >= 0 and bool(self.data.eq_active[e]) for e in self._grasp_eq], dtype=bool)
        return self._grasp_flags.copy()

    @property
    def grasped(self) -> bool:
        return bool(self.grasp_flags().any())

    def set_gripper(self, opening: float) -> None:
        """`opening` in [0, 1]: 0 closed, 1 fully open."""
        self.data.ctrl[self.grip_act] = float(np.clip(opening, 0, 1)) * GRIPPER_OPEN

    def _release_grasps(self) -> None:
        for eid in self._grasp_eq:
            if eid >= 0:
                self.data.eq_active[eid] = False

    def _pinched(self) -> set[str]:
        """Object names currently touching both compliant pad surfaces."""
        if self._pad_L < 0 or self._pad_R < 0:
            return set()
        left, right = set(), set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for cube_geom, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                name = self._cube_geoms.get(cube_geom)
                if name is None:
                    continue
                if other == self._pad_L:
                    left.add(name)
                elif other == self._pad_R:
                    right.add(name)
        return left & right

    def _face_aligned(self, obj_index: int) -> bool:
        """Whether the jaw closing axis is within 25 degrees of a cube axis."""
        grip_R = self.data.xmat[self._link6].reshape(3, 3)
        cube_R = self.data.xmat[self.object_bodies[obj_index]].reshape(3, 3)
        closing_axis = grip_R[:, 1]
        return float(np.max(np.abs(cube_R.T @ closing_axis))) >= GRASP_ALIGN_COS

    def _capture_grasp(self, eid: int, obj_index: int) -> None:
        """Capture the seated pose used by the compliant-pad grasp assist."""
        bid2 = self.object_bodies[obj_index]
        dp = self.data.xpos[bid2] - self.data.xpos[self._link6]
        q1inv = np.zeros(4)
        mujoco.mju_negQuat(q1inv, self.data.xquat[self._link6])
        rpos = np.zeros(3)
        mujoco.mju_rotVecQuat(rpos, dp, q1inv)
        rquat = np.zeros(4)
        mujoco.mju_mulQuat(rquat, q1inv, self.data.xquat[bid2])
        self.model.eq_data[eid, 0:3] = 0.0
        self.model.eq_data[eid, 3:6] = rpos
        self.model.eq_data[eid, 6:10] = rquat
        self.model.eq_data[eid, 10] = 1.0

    def _update_grasp(self) -> None:
        """Latch only a two-pad, face-aligned grasp; leave corner grasps free."""
        if not self._grasp_eq or self._link6 < 0:
            return
        opening = float(self.data.ctrl[self.grip_act]) / GRIPPER_OPEN
        pinched = self._pinched() if opening < GRASP_LOCK else set()
        for i, name in enumerate(self.object_names):
            eid = self._grasp_eq[i]
            if eid < 0:
                continue
            if self.data.eq_active[eid]:
                if opening > GRASP_UNLOCK:
                    self.data.eq_active[eid] = False
            elif name in pinched and self._face_aligned(i):
                self._capture_grasp(eid, i)
                self.data.eq_active[eid] = True

    def step(self, n: int = 1) -> None:
        if n < 0 or int(n) != n:
            raise ValueError("Physics step count must be a nonnegative integer")
        if not n:
            return  # Preserve pending targets and the last applied command.
        n = int(n)
        target = self.data.ctrl[:6].copy()
        start = self._applied_arm_ctrl.copy()
        physical = self.dynamics == CONTACT_DYNAMICS
        if physical:
            self._release_grasps()
        for j in range(1, n + 1):
            if physical:
                self.data.ctrl[:6] = start + (target - start) * (j / n)
            else:
                self._update_grasp()
            mujoco.mj_step(self.model, self.data)
            if physical:
                loaded = np.all(self.pad_normal_forces() > .01, axis=1)
                self._grasp_contact_time = np.where(loaded, self._grasp_contact_time + self.dt, 0.)
                self._grasp_flags = self._grasp_contact_time >= .020 - 1e-12
        self.data.ctrl[:6] = target
        self._applied_arm_ctrl = target

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).reshape(9))
    return q


if __name__ == "__main__":
    # Self-test: IK accuracy over reachable targets.
    sim = PantheraSim()
    print(f"\nmodel: nq={sim.model.nq} nu={sim.model.nu} dt={sim.dt}")
    print("home EE:", sim.ee_pos().round(4))

    rng = np.random.default_rng(0)
    pe, re_, n = [], [], 0
    for _ in range(200):
        q = rng.uniform(sim.arm_range[:, 0], sim.arm_range[:, 1])
        sim.data.qpos[sim.arm_qadr] = q
        mujoco.mj_kinematics(sim.model, sim.data)
        tp, tq = sim.ee_pose()
        sim.reset()
        # Raw convergence, so the rate limit is off: this is one cold call,
        # not a teleoperation frame, and 0.15 rad of slew would dominate it.
        qs, p, r = sim.ik(tp, tq, max_joint_step=None)
        pe.append(p); re_.append(r); n += p < 2e-3
    print(f"\nIK from home to 200 random reachable poses (cold, unlimited):")
    print(f"  position error  median {np.median(pe)*1000:6.2f} mm   "
          f"mean {np.mean(pe)*1000:6.2f}")
    print(f"  rotation error  median {np.degrees(np.median(re_)):6.2f} deg")
    print(f"  converged <2mm: {n}/200")

    # The case teleoperation is actually in: warm-started, frame after frame,
    # including reaching past what the arm can do and coming back. Before the
    # clamping loop and the nullspace posture this stranded a quarter of the
    # time in a corner of the joint box that damped least squares could not
    # climb out of, and only a manual reset recovered it.
    box_lo, box_hi = np.array([0.25, -0.25, 0.06]), np.array([0.55, 0.25, 0.30])
    # The nominal grasp orientation: 30 deg nose-down from forward, which is
    # the rest pose the workspace test is sized around.
    a = np.radians(30.0)
    q_grasp = mat_to_quat(np.array([[np.cos(a), 0, np.sin(a)],
                                    [0, 1, 0],
                                    [-np.sin(a), 0, np.cos(a)]]))
    t = np.linspace(0, 6 * np.pi, 600)
    traj = np.stack([0.40 + 0.13 * np.sin(t), 0.22 * np.sin(1.3 * t),
                     0.18 + 0.11 * np.sin(0.7 * t)], 1)
    q, errs, jumps = sim.q.copy(), [], []
    for pt in traj:
        prev = q
        q, p, r = sim.ik(pt, q_grasp, q_init=q)
        errs.append(p); jumps.append(np.abs(q - prev).max())
    errs, jumps = np.array(errs[5:]), np.array(jumps[5:])
    print(f"\nwarm-started tracking, 600 frames inside the workspace box:")
    print(f"  position error   median {np.median(errs)*1000:.4f} mm  "
          f"max {errs.max()*1000:.4f} mm   over 5 mm: {(errs > 5e-3).sum()}")
    print(f"  joint step/frame max {jumps.max():.3f} rad "
          f"({np.degrees(jumps.max()):.1f} deg)")

    stranded, recov = 0, []
    for _ in range(60):
        q = sim.q.copy()
        home_t = rng.uniform(box_lo, box_hi)
        for _ in range(8):
            q, _, _ = sim.ik(home_t, q_grasp, q_init=q)
        for _ in range(6):      # operator reaches well past the arm's reach
            q, _, _ = sim.ik(home_t + rng.uniform(0.25, 0.5) *
                             rng.choice([-1, 1], 3), q_grasp, q_init=q)
        back, k = rng.uniform(box_lo, box_hi), None
        for i in range(60):     # 2 s at 30 fps to come back
            q, p, _ = sim.ik(back, q_grasp, q_init=q)
            if p < 5e-3:
                k = i + 1
                break
        stranded += k is None
        if k is not None:
            recov.append(k)
    print(f"  reach out of the box and come back, 60 trials:")
    print(f"    never recovers within 2 s: {stranded}/60")
    if recov:
        print(f"    recovery median {np.median(recov):.0f} frames "
              f"({np.median(recov)/30:.2f} s), p95 {np.percentile(recov, 95):.0f}")

    # Scripted grasp, lift, then a shake. The force-limited pad contacts should
    # hold a well-centred lift, permit a few millimetres of in-hand motion under
    # the shake, and release immediately when the fingers open.
    if sim.object_names:
        print("\nscripted grasp, 120 mm lift, then a 0.4 m/s shake:")
        cubes = {
            "cube_red": np.array([0.38, 0.12, 0.0725]),
            "cube_green": np.array([0.45, 0.00, 0.0725]),
            "cube_blue": np.array([0.36, -0.13, 0.0725]),
        }
        lift_steps = 333
        for name, target in cubes.items():
            if name not in sim.object_names:
                continue
            sim.reset(randomize=False)
            q = sim.q.copy()
            hover = target + np.array([0.0, 0.0, 0.08])
            for pos, opening, n in ((hover, 1.0, 250), (target, 1.0, 200)):
                for _ in range(n):
                    q, _, _ = sim.ik(pos, q_grasp, q_init=q)
                    sim.set_arm_ctrl(q)
                    sim.set_gripper(opening)
                    sim.step(1)
            for _ in range(200):
                q, _, _ = sim.ik(target, q_grasp, q_init=q)
                sim.set_arm_ctrl(q)
                sim.set_gripper(0.0)
                sim.step(1)
            idx = sim.object_names.index(name)
            close_rel = (sim.data.xpos[sim.object_bodies[idx]] - sim.ee_pos()).copy()
            lift = target + np.array([0.0, 0.0, 0.12])
            for i in range(lift_steps):
                a = i / (lift_steps - 1)
                q, _, _ = sim.ik(target + a * (lift - target),
                                 q_grasp, q_init=q)
                sim.set_arm_ctrl(q)
                sim.set_gripper(0.0)
                sim.step(1)
            held = sim.data.xpos[sim.object_bodies[idx]].copy()
            slip_lift = np.linalg.norm((held - sim.ee_pos()) - close_rel)
            side = lift + np.array([0.0, 0.12, 0.0])
            for dest in (side, lift + np.array([0.06, -0.04, 0.03]), lift):
                start = sim.ee_pos().copy()
                for i in range(150):
                    a = i / 149
                    q, _, _ = sim.ik(start + a * (dest - start),
                                     q_grasp, q_init=q)
                    sim.set_arm_ctrl(q)
                    sim.set_gripper(0.0)
                    sim.step(1)
            shaken = sim.data.xpos[sim.object_bodies[idx]].copy()
            slip_move = np.linalg.norm((shaken - sim.ee_pos()) - close_rel)
            for _ in range(250):
                sim.set_gripper(1.0)
                sim.step(1)
            dropped = sim.data.xpos[sim.object_bodies[idx]].copy()
            print(f"  {name:12s}  z={shaken[2]:.3f}  lift {slip_lift*1000:.2f} mm  "
                  f"move {slip_move*1000:.2f} mm  after open z={dropped[2]:.3f}")
