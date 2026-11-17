"""Standing tabletop manipulation env: G1 + Dex1-1 at a RoboCasa kitchen counter (fixed pelvis).

    env = ManipEnv("build/kitchen_l1_s1_v1.mjb")
    obs = env.reset(seed)                  # random object, target plate, poses, robot jitter, lighting
    obs = env.step(q_left7, q_right7, grip_left, grip_right)   # 20 Hz; grip 0 = closed, 1 = open

Control: arm joint position targets (14) + gripper opening (2), 20 Hz, physics 2 ms (25 substeps).
Observation: ego RGB (head D435 camera), arm joint angles, waist (welded: zeros), gripper openings,
object / target poses (privileged; used by the scripted expert and the success check only).
Set MUJOCO_GL=egl for headless rendering (VM).
"""
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from ik import ARM, ArmIK, mat_to_quat, quat_to_mat

GRIP_OPEN, GRIP_CLOSED = 0.0245, -0.020
CTRL_DT = 0.05
EGO_W, EGO_H = 320, 240
WRIST_W, WRIST_H = 160, 120


def yaw_quat(yaw):
    """Quaternion (w, x, y, z) of a yaw."""
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


class ManipEnv:
    """The tabletop pick-and-place environment: G1 + Dex1-1 at a kitchen counter."""
    def __init__(self, mjb, wrist_cams=False, render=True, prefix="g1/"):
        """Load a compiled scene and its metadata, resolve joints, actuators and cameras."""
        self.mjb = Path(mjb)
        self.m = mujoco.MjModel.from_binary_path(str(self.mjb))
        self.d = mujoco.MjData(self.m)
        self.info = json.loads(self.mjb.with_suffix(".json").read_text())
        self.p = prefix
        self.substeps = int(round(CTRL_DT / self.m.opt.timestep))
        m = self.m
        self.arm_q = {s: np.array([m.jnt_qposadr[m.joint(f"{prefix}{s}_{j}_joint").id] for j in ARM]) for s in ("left", "right")}
        self.arm_act = {s: np.array([m.actuator(f"{prefix}{s}_{j}").id for j in ARM]) for s in ("left", "right")}
        self.grip_act = {s: m.actuator(f"{prefix}{s}_gripper").id for s in ("left", "right")}
        self.grip_q = {s: m.jnt_qposadr[m.joint(f"{prefix}{s}_dex1_finger_joint_1").id] for s in ("left", "right")}
        self.grip_q2 = {s: m.jnt_qposadr[m.joint(f"{prefix}{s}_dex1_finger_joint_2").id] for s in ("left", "right")}
        self.tcp = {s: m.site(f"{prefix}tcp_{s}").id for s in ("left", "right")}
        self.base_body = m.body("robot_base").id
        self.base_pos0 = m.body_pos[self.base_body].copy()
        self.base_quat0 = m.body_quat[self.base_body].copy()
        self.ik = {s: ArmIK(m, s, prefix) for s in ("left", "right")}
        self.obj_joint = {o["name"]: m.joint(f"{o['name']}_free").id for o in self.info["objects"] + self.info["targets"]}
        self.light_diffuse0 = m.light_diffuse.copy()
        self.render_on = render
        self.wrist_cams = wrist_cams
        self.renderer = mujoco.Renderer(m, EGO_H, EGO_W) if render else None
        self.wrenderer = mujoco.Renderer(m, WRIST_H, WRIST_W) if (render and wrist_cams) else None
        self.ego_cam = m.camera(f"{prefix}ego").id
        # draw visuals only: RoboCasa/object colliders are group 0, robot colliders group 3
        self.vopt = mujoco.MjvOption()
        self.vopt.geomgroup[:] = [0, 1, 1, 0, 0, 0]
        self.ready_q = None
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ geometry helpers
    @property
    def base_frame(self):
        """Robot base position and rotation."""
        return self.d.xpos[self.base_body].copy(), self.d.xmat[self.base_body].reshape(3, 3).copy()

    def to_world(self, p_local):
        """Base-frame point to world."""
        o, R = self.base_frame
        return o + R @ np.asarray(p_local)

    def obj_pose(self, name):
        """Pose of a free object."""
        a = self.m.jnt_qposadr[self.obj_joint[name]]
        return self.d.qpos[a:a + 7].copy()

    def set_obj_pose(self, name, pos, quat):
        """Place a free object."""
        a = self.m.jnt_qposadr[self.obj_joint[name]]
        self.d.qpos[a:a + 3] = pos
        self.d.qpos[a + 3:a + 7] = quat
        self.d.qvel[self.m.jnt_dofadr[self.obj_joint[name]]:self.m.jnt_dofadr[self.obj_joint[name]] + 6] = 0

    def meta(self, name):
        """Metadata of an object or target."""
        for o in self.info["objects"] + self.info["targets"]:
            if o["name"] == name:
                return o
        raise KeyError(name)

    def bottom_offset(self, name):
        """Distance from the object frame origin down to its bottom (from the bbox)."""
        o = self.meta(name)
        return o.get("half_extents", [0.04, 0.04, 0.04])[2] - o.get("bbox_offset", [0, 0, 0])[2]

    def tcp_pose(self, side):
        """Tool pose on one side."""
        return self.d.site_xpos[self.tcp[side]].copy(), self.d.site_xmat[self.tcp[side]].reshape(3, 3).copy()

    # ------------------------------------------------------------------ reset
    def ready_tcp(self, side):
        """Hands in front of the chest, fingers forward and horizontal (robot frame)."""
        o, R = self.base_frame
        y = 0.26 if side == "left" else -0.26
        pos = o + R @ np.array([0.30, y, 0.22])
        Rt = R.copy()                                  # tcp x = robot forward, z = up
        return pos, Rt

    def reset(self, seed, obj=None, target=None, randomize=True):
        """New episode: pick object and target, randomize layout and arm start."""
        rng = np.random.default_rng(seed)
        self.rng = rng
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        info = self.info
        # robot pose jitter (pelvis is welded to robot_base; base has no joint, so set the model pose)
        if randomize:
            dx, dy = rng.uniform(-1, 1, 2) * 0.03
            dz = rng.uniform(-0.02, 0.0)
            yaw0 = math.radians(info["robot_yaw_deg"]) + math.radians(rng.uniform(-5, 5))
            m.body_pos[self.base_body] = self.base_pos0 + np.array([dx, dy, dz])
            m.body_quat[self.base_body] = yaw_quat(yaw0)
            m.light_diffuse[:] = self.light_diffuse0 * rng.uniform(0.7, 1.2)
        else:
            m.body_pos[self.base_body] = self.base_pos0
            m.body_quat[self.base_body] = self.base_quat0
            m.light_diffuse[:] = self.light_diffuse0
        mujoco.mj_forward(m, d)
        # arms to the ready pose (IK, cached per base pose)
        for s in ("left", "right"):
            pos, R = self.ready_tcp(s)
            if self.ready_q is None:
                self.ik[s].rest = None
            q, _, _ = self.ik[s].solve(d.qpos.copy(), pos, R, q_init=self.ik[s].rest if self.ik[s].rest is not None else None, iters=200)
            if self.ik[s].rest is None:
                self.ik[s].rest = q.copy()
            d.qpos[self.arm_q[s]] = q
            d.ctrl[self.arm_act[s]] = q
            d.qpos[self.grip_q[s]] = GRIP_OPEN
            d.qpos[self.grip_q2[s]] = GRIP_OPEN
            d.ctrl[self.grip_act[s]] = GRIP_OPEN
        self.ready_q = {s: d.qpos[self.arm_q[s]].copy() for s in ("left", "right")}
        # choose object + target
        objs, tgts = info["objects"], info["targets"]
        self.obj = obj or objs[rng.integers(len(objs))]["name"]
        self.target = target or tgts[rng.integers(len(tgts))]["name"]
        wa, ra = info["work_area"], info["reach_area_robot"]
        top = info["counter_top_z"]
        o_b, R_b = self.base_frame
        r_o = max(self.meta(self.obj).get("half_extents", [0.04] * 3)[:2])
        r_t = max(self.meta(self.target).get("half_extents", [0.09] * 3)[:2])

        # active arm first; object on that arm's side, target on its side or the middle (never far across
        # the body: the crossing arm would hit the idle hand)
        self.active_side = "left" if rng.random() < 0.5 else "right"
        sgn = 1.0 if self.active_side == "left" else -1.0
        obj_y = sorted([sgn * ra["y_abs"][0], sgn * ra["y_abs"][1]])
        tgt_y = obj_y

        def sample(margin, yr):
            for _ in range(100):
                pl = np.array([rng.uniform(*ra["x"]), rng.uniform(*yr), 0.0])
                pw = o_b + R_b @ pl
                if wa["x"][0] + margin < pw[0] < wa["x"][1] - margin and wa["y"][0] + margin < pw[1] < wa["y"][1]:
                    return pw[:2]
            return pw[:2]
        q_settle = d.qpos.copy()
        for attempt in range(10):
            d.qpos[:] = q_settle
            d.qvel[:] = 0
            for _ in range(200):
                po, pt = sample(r_o, obj_y), sample(r_t * 0.5, tgt_y)
                if np.linalg.norm(po - pt) > r_o + r_t + 0.03:
                    break
            want_o = np.array([*po, top + self.bottom_offset(self.obj) + 0.003])
            want_t = np.array([*pt, top + self.bottom_offset(self.target) + 0.003])
            self.set_obj_pose(self.obj, want_o, yaw_quat(rng.uniform(-math.pi, math.pi)))
            self.set_obj_pose(self.target, want_t, yaw_quat(rng.uniform(-math.pi, math.pi)))
            mujoco.mj_forward(m, d)
            for _ in range(int(0.3 / m.opt.timestep)):          # settle
                mujoco.mj_step(m, d)
            ok = True
            for n, want in ((self.obj, want_o), (self.target, want_t)):
                p = self.obj_pose(n)
                if np.linalg.norm(p[:2] - want[:2]) > 0.01 or quat_to_mat(p[3:])[2, 2] < math.cos(math.radians(8)):
                    ok = False
            if ok:
                break
        self.reset_ok = ok
        self.t = 0
        names = self.meta(self.obj)
        tmpl = info.get("instruction_templates") or ["pick up the {obj} and put it on the plate"]
        self.instruction = tmpl[rng.integers(len(tmpl))].format(obj=names["label"])
        return self.observe()

    # ------------------------------------------------------------------ step / observe
    def grip_open_frac(self, side):
        """Gripper opening, 0 closed to 1 open."""
        return float((self.d.qpos[self.grip_q[side]] - GRIP_CLOSED) / (GRIP_OPEN - GRIP_CLOSED))

    def step(self, q_left, q_right, g_left, g_right):
        """Apply arm and gripper targets and step one control period."""
        d = self.d
        d.ctrl[self.arm_act["left"]] = q_left
        d.ctrl[self.arm_act["right"]] = q_right
        d.ctrl[self.grip_act["left"]] = GRIP_CLOSED + float(np.clip(g_left, 0, 1)) * (GRIP_OPEN - GRIP_CLOSED)
        d.ctrl[self.grip_act["right"]] = GRIP_CLOSED + float(np.clip(g_right, 0, 1)) * (GRIP_OPEN - GRIP_CLOSED)
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, d)
        self.t += 1
        return self.observe()

    def render_cam(self, renderer, cam):
        """Render one camera."""
        renderer.update_scene(self.d, camera=cam, scene_option=self.vopt)
        return renderer.render().copy()

    def observe(self):
        """Joint state, gripper state and camera images."""
        d = self.d
        obs = {
            "q_left_arm": d.qpos[self.arm_q["left"]].copy(),
            "q_right_arm": d.qpos[self.arm_q["right"]].copy(),
            "q_waist": np.zeros(3),
            "grip_left": self.grip_open_frac("left"),
            "grip_right": self.grip_open_frac("right"),
            "obj_pose": self.obj_pose(self.obj),
            "target_pose": self.obj_pose(self.target),
        }
        if self.render_on:
            obs["ego_rgb"] = self.render_cam(self.renderer, self.ego_cam)
            if self.wrist_cams:
                obs["wrist_left_rgb"] = self.render_cam(self.wrenderer, self.m.camera(f"{self.p}wrist_left").id)
                obs["wrist_right_rgb"] = self.render_cam(self.wrenderer, self.m.camera(f"{self.p}wrist_right").id)
        return obs

    # ------------------------------------------------------------------ planning checks
    def plan_pose_ok(self, ik_data, side, allow_body_prefix):
        """Collision check of a planned arm configuration (ik_data.qpos already set): no arm link may touch
        the environment; fingers may touch only the object being picked (allow_body_prefix, or None).
        Elbow kept >= 10 cm from the body centre line (the torso <-> elbow contact is excluded in the model)."""
        m, d = self.m, ik_data
        mujoco.mj_fwdPosition(m, d)
        for c in d.contact[:d.ncon]:
            b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
            r1, r2 = b1.startswith(self.p), b2.startswith(self.p)
            if r1 == r2:
                continue                                  # robot-robot (allowed pairs) or env-env
            rb, ob = (b1, b2) if r1 else (b2, b1)
            if side not in rb or not any(k in rb for k in ("shoulder", "elbow", "wrist", "dex1")):
                continue                                  # idle arm, legs/feet on the floor: not planned motion
            if "finger" in rb and allow_body_prefix and ob.startswith(allow_body_prefix):
                continue
            return False
        o, R = self.base_frame
        e = R.T @ (d.xpos[m.body(f"{self.p}{side}_elbow_link").id] - o)
        return abs(e[1]) >= 0.10

    # ------------------------------------------------------------------ success
    def success(self):
        """Object resting upright on the target, gripper released and away."""
        po, pt = self.obj_pose(self.obj), self.obj_pose(self.target)
        r_t = max(self.meta(self.target).get("half_extents", [0.09] * 3)[:2])
        on_xy = np.linalg.norm(po[:2] - pt[:2]) < 0.75 * r_t
        tgt_top = pt[2] - self.bottom_offset(self.target) + 2 * self.meta(self.target).get("half_extents", [0, 0, 0.01])[2]
        resting = (po[2] - self.bottom_offset(self.obj)) > tgt_top - 0.02 and (po[2] - self.bottom_offset(self.obj)) < tgt_top + 0.03
        up = quat_to_mat(po[3:])[2, 2] > math.cos(math.radians(25))
        released = all(np.linalg.norm(self.tcp_pose(s)[0] - po[:3]) > 0.06 for s in ("left", "right"))
        still = np.linalg.norm(self.d.qvel[self.m.jnt_dofadr[self.obj_joint[self.obj]]:][:3]) < 0.05
        return bool(on_xy and resting and up and released and still)

    def camera_intrinsics(self):
        """Pinhole intrinsics of the ego camera."""
        fovy = self.m.cam_fovy[self.ego_cam]
        f = 0.5 * EGO_H / math.tan(math.radians(fovy) / 2)
        return {"width": EGO_W, "height": EGO_H, "fx": f, "fy": f, "cx": EGO_W / 2, "cy": EGO_H / 2, "fovy_deg": float(fovy)}
