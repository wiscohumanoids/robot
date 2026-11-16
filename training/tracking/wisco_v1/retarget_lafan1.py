"""Retarget a LAFAN1 human mocap clip (Ubisoft BVH, CC BY-NC-ND 4.0) to the Wisco V1 humanoid.

Own implementation (numpy + MuJoCo only; GMR could not be installed here). Method:
  1. Parse the BVH, forward kinematics -> every human joint's world position/orientation per frame
     (BVH is Y-up centimetres; converted to Z-up metres).
  2. "Bone-direction" retargeting: walk each limb chain keeping the human bone DIRECTIONS but Wisco's own
     segment LENGTHS, starting from the robot's own (current) hip/shoulder pivots. Targets are therefore
     always reachable and proportions are Wisco's, not the human's.
       legs: hip pivot -> knee -> ankle -> toe;  arms: shoulder pivot -> elbow -> wrist
     Pelvis position = human hips scaled by (robot leg length / human leg length); pelvis and torso
     orientation from the human hip / shoulder lines; foot orientation = human foot yaw+pitch, zero roll.
  3. Per-frame damped-least-squares IK on the Wisco training model (29 joints, head welded), warm-started
     from the previous frame (smooth), joint limits enforced (incl. ankle pitch >= training limit), with a
     small pull toward the home pose for joints the targets don't pin down (forearm yaw, wrists).
  4. Ground: shift the pelvis height so the lowest sole point is on the floor whenever the human has a foot
     on the floor, and never below it otherwise.
  5. Quality report: joint-limit margins, foot penetration, foot skating, self-collision, target errors.
Output CSV rows (30 fps, same layout as Unitree's G1 LAFAN1 CSVs, read by training/tracking/motion_to_npz.py):
  pelvis x y z, pelvis quat x y z w, then the 29 joints in the training model's joint order.

  .venv-mjlab/bin/python training/tracking/wisco_v1/retarget_lafan1.py --bvh dance1_subject2.bvh \
      --out training/tracking/wisco_v1/motions/dance1_subject2.csv [--video preview.mp4] [--frames 0:600]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parents[2] / "robots" / "wisco_v1" / "model"
TRAIN_XML = MODEL_DIR / "wisco_v1_train.xml"
INFO = json.loads((MODEL_DIR / "train_model_info.json").read_text())
JOINTS = list(INFO["policy_joints_in_model_order"])
ANKLE_PITCH_LOWER = INFO["ankle_pitch_lower_training"]

IK_BOUNDS = {"hip_yaw": (-0.9, 0.9), "shoulder_yaw": (-1.6, 1.6), "forearm_yaw": (-1.6, 1.6)}
# relative pull toward the home pose (x W_REG): twist joints more, the rest lightly
REG_WEIGHTS = {"hip_yaw": 5.0, "shoulder_yaw": 3.0, "forearm_yaw": 3.0, "torso_yaw": 2.0,
               "wrist_pitch": 3.0, "wrist_roll": 3.0}
RESET_RESIDUAL = 0.25          # re-solve the legs from the home pose if a frame's residual is above this
W_PREV = 0.05                  # continuity: pull toward the previous frame's solution (stops arm-solution flips)
SMOOTH_SIGMA_FRAMES = 1.5      # final Gaussian smoothing of joint trajectories (~50 ms at 30 fps)

# Y-up (BVH) -> Z-up (robot world): x stays, y -> z, z -> -y (a proper rotation)
R_CONV = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


# ----------------------------------------------------------------------------------------------- BVH
def _rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load_bvh(path):
    """-> names, parents, offsets (J,3), channels per joint, frames (F, C), frame_time."""
    names, parents, offsets, channels = [], [], [], []
    stack = []
    lines = Path(path).read_text().split("\n")
    i = 0
    while not lines[i].strip().startswith("MOTION"):
        t = lines[i].split()
        if t and t[0] in ("ROOT", "JOINT"):
            names.append(t[1]); parents.append(stack[-1] if stack else -1); offsets.append(None); channels.append([])
            cur = len(names) - 1
        elif t and t[0] == "End":
            names.append(None); parents.append(stack[-1]); offsets.append(None); channels.append([])
            cur = len(names) - 1
        elif t and t[0] == "{":
            stack.append(cur)
        elif t and t[0] == "}":
            stack.pop()
        elif t and t[0] == "OFFSET":
            offsets[cur] = np.array([float(x) for x in t[1:4]])
        elif t and t[0] == "CHANNELS":
            channels[cur] = t[2:2 + int(t[1])]
        i += 1
    n_frames = int(lines[i + 1].split()[-1]); frame_time = float(lines[i + 2].split()[-1])
    data = np.array([[float(x) for x in l.split()] for l in lines[i + 3:i + 3 + n_frames] if l.strip()])
    return names, parents, np.array(offsets), channels, data, frame_time


def bvh_fk(names, parents, offsets, channels, data):
    """World positions (F,J,3) [m, Z-up] and rotations (F,J,3,3) for named joints (end sites dropped)."""
    F, J = len(data), len(names)
    pos = np.zeros((F, J, 3)); rot = np.zeros((F, J, 3, 3))
    for f in range(F):
        col = 0
        for j in range(J):
            local_p = offsets[j].copy(); R = np.eye(3)
            for ch in channels[j]:
                v = data[f, col]; col += 1
                if ch.endswith("position"):
                    local_p["XYZ".index(ch[0])] = v
                else:
                    R = R @ _rot(ch[0], v)
            if parents[j] < 0:
                rot[f, j] = R; pos[f, j] = local_p
            else:
                p = parents[j]
                rot[f, j] = rot[f, p] @ R; pos[f, j] = pos[f, p] + rot[f, p] @ local_p
    keep = [j for j, n in enumerate(names) if n is not None]
    pos = (pos[:, keep] / 100.0) @ R_CONV.T
    rot = np.einsum("ab,fjbc->fjac", R_CONV, rot[:, keep])
    return [names[j] for j in keep], pos, rot


# ----------------------------------------------------------------------------------------------- robot
class Robot:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(TRAIN_XML))
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINTS])
        self.vadr = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in JOINTS])
        lo = self.m.jnt_range[[self.m.joint(n).id for n in JOINTS], 0].copy()
        hi = self.m.jnt_range[[self.m.joint(n).id for n in JOINTS], 1].copy()
        margin = 0.02
        self.lo, self.hi = lo + margin, hi - margin
        # Retargeting-only bounds (not training limits): human hips and arms don't twist ~180 deg. Without these the
        # IK found "spun hip yaw" solutions for a backward heel kick and got stuck at the limits (dance1 32-41 s).
        for n, (a, b) in IK_BOUNDS.items():
            for side in ("left", "right"):
                i = JOINTS.index(f"{side}_{n}")
                self.lo[i], self.hi[i] = max(self.lo[i], a), min(self.hi[i], b)
        self.leg_idx = np.array([i for i, n in enumerate(JOINTS) if any(k in n for k in ("hip", "knee", "ankle"))])
        self.reg = np.array([REG_WEIGHTS.get(n.split("_", 1)[1] if n.startswith(("left", "right")) else n, 1.0)
                             for n in JOINTS])
        self.home = np.array([INFO["default_pose"].get(n, 0.0) for n in JOINTS])
        self.bid = lambda n: self.m.body(n).id
        # zero-pose segment lengths (pivot to pivot)
        self.d.qpos[:] = 0; self.d.qpos[3] = 1; mujoco.mj_kinematics(self.m, self.d)
        P = lambda n: self.d.xpos[self.bid(n)].copy()
        self.len = {}
        for s in ("left", "right"):
            self.len[f"{s}_thigh"] = np.linalg.norm(P(f"{s}_knee_link") - P(f"{s}_hip_pitch_link"))
            self.len[f"{s}_shank"] = np.linalg.norm(P(f"{s}_ankle_pitch_link") - P(f"{s}_knee_link"))
            self.len[f"{s}_foot"] = np.linalg.norm(self.toe_point(s) - P(f"{s}_ankle_pitch_link"))
            self.len[f"{s}_upper_arm"] = np.linalg.norm(P(f"{s}_elbow_link") - P(f"{s}_shoulder_pitch_link"))
            self.len[f"{s}_forearm"] = np.linalg.norm(P(f"{s}_wrist_pitch_link") - P(f"{s}_elbow_link"))
        self.hip_height_zero = P("left_hip_pitch_link")[2] - self.sole_lowest()
        self.foot_geoms = [self.m.geom(f"{s}_foot_collision").id for s in ("left", "right")]

    def toe_point(self, side):
        """Front-centre of the sole box (foot frame)."""
        g = self.m.geom(f"{side}_foot_collision").id
        R = self.d.geom_xmat[g].reshape(3, 3); h = self.m.geom_size[g]
        return self.d.geom_xpos[g] + R @ np.array([h[0], 0, -h[2]])

    def sole_corners(self, side):
        g = self.m.geom(f"{side}_foot_collision").id
        R = self.d.geom_xmat[g].reshape(3, 3); h = self.m.geom_size[g]
        c = np.array([[x, y, -1] for x in (-1, 1) for y in (-1, 1)]) * h
        return c @ R.T + self.d.geom_xpos[g]

    def sole_lowest(self, side=None):
        sides = (side,) if side else ("left", "right")
        return min(self.sole_corners(s)[:, 2].min() for s in sides)

    def set(self, root_pos, root_quat, q):
        self.d.qpos[0:3] = root_pos; self.d.qpos[3:7] = root_quat; self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)


def rot_from_left_up(left, up):
    """Rotation with y = left, z = up (Gram-Schmidt on up), x = y cross z."""
    up = up / np.linalg.norm(up)
    left = left - up * (left @ up); left /= np.linalg.norm(left)
    fwd = np.cross(left, up)
    return np.stack([fwd, left, up], axis=1)


def rot_from_fwd_flat(fwd):
    """Foot frame: x = fwd (incl. pitch), y horizontal (zero roll), z completes."""
    fwd = fwd / np.linalg.norm(fwd)
    left = np.cross(np.array([0, 0, 1.0]), fwd); left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    return np.stack([fwd, left, up], axis=1)


def mat2quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(9)); return q   # wxyz


def rot_err(R_target, R_cur):
    """Rotation vector taking R_cur to R_target (world frame)."""
    q = np.zeros(4); mujoco.mju_mat2Quat(q, (R_target @ R_cur.T).reshape(9))
    v = np.zeros(3); mujoco.mju_quat2Vel(v, q, 1.0); return v


# ----------------------------------------------------------------------------------------------- main
def retarget(bvh_path, frames=None, iters=12, verbose=True, calib_frame=0, debug_frames=()):
    names, parents, offsets, channels, data, ft = load_bvh(bvh_path)
    _, HP_all, _ = bvh_fk(names, parents, offsets, channels, data[calib_frame:calib_frame + 1])
    if frames is not None:
        data = data[frames[0]:frames[1]]
    jn, HP, HR = bvh_fk(names, parents, offsets, channels, data)
    # the calibration frame is taken from the full clip (frame index in the original take)
    HP = np.concatenate([HP_all, HP], axis=0); HP_offset = 1   # row 0 = calibration frame
    H = {n: i for i, n in enumerate(jn)}
    hp = lambda n, f: HP[f + HP_offset, H[n]] if f >= 0 else HP[0, H[n]]
    F = len(data)
    rb = Robot()

    # scale for the pelvis trajectory: robot leg length / human leg length (hip -> ankle + ankle height)
    def human_leg(f):
        return np.mean([np.linalg.norm(hp(f"{s}UpLeg", f) - hp(f"{s}Leg", f)) + np.linalg.norm(hp(f"{s}Leg", f) - hp(f"{s}Foot", f))
                        for s in ("Left", "Right")])
    human_leg_len = np.median([human_leg(f) for f in range(0, F, max(1, F // 50))])
    robot_leg_len = rb.len["left_thigh"] + rb.len["left_shank"]
    s_root = robot_leg_len / human_leg_len
    # human toe height when standing (for foot pitch calibration and contact detection)
    toe_z = np.minimum(HP[HP_offset:, H["LeftToe"], 2], HP[HP_offset:, H["RightToe"], 2])
    foot_z = np.minimum(HP[HP_offset:, H["LeftFoot"], 2], HP[HP_offset:, H["RightFoot"], 2])
    ground_h = np.percentile(np.minimum(toe_z, foot_z), 2)
    # human foot pitch when flat: Foot->Toe vector slope in frames where both foot and toe are low
    pitches = []
    for s in ("Left", "Right"):
        v = HP[HP_offset:, H[f"{s}Toe"]] - HP[HP_offset:, H[f"{s}Foot"]]
        low = (HP[HP_offset:, H[f"{s}Toe"], 2] < ground_h + 0.03) & (HP[HP_offset:, H[f"{s}Foot"], 2] < ground_h + 0.10)
        if low.any():
            pitches += list(np.arctan2(v[low, 2], np.linalg.norm(v[low, :2], axis=1)))
    flat_pitch = float(np.median(pitches)) if pitches else 0.0

    # Orientation calibration at the clip's neutral standing frame (LAFAN1 takes start in a T-pose): the human's
    # pelvis / chest frames built from joint positions are not upright (spine curvature), so measure their offset
    # from "upright, same heading" there and remove it everywhere: R_target(t) = R_human(t) @ R_off.
    def yaw_only(R):
        f = R[:, 0].copy(); f[2] = 0; f /= np.linalg.norm(f)
        return rot_from_left_up(np.cross(np.array([0, 0, 1.0]), f), np.array([0, 0, 1.0]))
    c = -1                                   # -1 = the calibration frame row
    Rp0 = rot_from_left_up(hp("LeftUpLeg", c) - hp("RightUpLeg", c), hp("Spine", c) - hp("Hips", c))
    Rt0 = rot_from_left_up(hp("LeftArm", c) - hp("RightArm", c), hp("Neck", c) - hp("Spine1", c))
    R_off_pel = Rp0.T @ yaw_only(Rp0)
    R_off_torso = Rt0.T @ yaw_only(Rt0)
    span = np.linalg.norm(hp("LeftHand", c) - hp("RightHand", c))
    height = hp("Head", c)[2] - ground_h
    calib = dict(calib_frame=calib_frame, calib_arm_span_over_height=float(span / height),
                 pelvis_offset_deg=float(np.degrees(np.arccos(np.clip((np.trace(R_off_pel) - 1) / 2, -1, 1)))),
                 torso_offset_deg=float(np.degrees(np.arccos(np.clip((np.trace(R_off_torso) - 1) / 2, -1, 1)))))
    if verbose:
        print("  calibration:", calib)

    q = rb.home.copy()
    out_root_pos = np.zeros((F, 3)); out_root_quat = np.zeros((F, 4)); out_q = np.zeros((F, len(JOINTS)))
    err_log = []
    W_POS, W_ROT, W_REG, LAMBDA = 1.0, 0.3, 0.02, 1e-3
    for f in range(F):
        # pelvis pose from the human hips (heading, lean) and scaled position
        R_pel = rot_from_left_up(hp("LeftUpLeg", f) - hp("RightUpLeg", f), hp("Spine", f) - hp("Hips", f)) @ R_off_pel
        root_p = hp("Hips", f) * s_root
        root_p[2] = (hp("Hips", f)[2] - ground_h) * s_root
        R_torso = rot_from_left_up(hp("LeftArm", f) - hp("RightArm", f), hp("Neck", f) - hp("Spine1", f)) @ R_off_torso
        foot_R = {}
        for s, S in (("left", "Left"), ("right", "Right")):
            v = hp(f"{S}Toe", f) - hp(f"{S}Foot", f)
            pitch = np.arctan2(v[2], np.linalg.norm(v[:2])) - flat_pitch
            yaw_dir = v[:2] / (np.linalg.norm(v[:2]) + 1e-9)
            fwd = np.array([yaw_dir[0] * np.cos(pitch), yaw_dir[1] * np.cos(pitch), np.sin(pitch)])
            foot_R[s] = rot_from_fwd_flat(fwd)
        rq = mat2quat(R_pel)

        q_prev = q.copy()

        def solve(q):
          for it in range(iters):
              rb.set(root_p, rq, q)
              res, jac, labels = [], [], []
              P = lambda n: rb.d.xpos[rb.bid(n)]

              def add_pos(body, target, w=W_POS, point=None):
                  jp = np.zeros((3, rb.m.nv))
                  pt = P(body) if point is None else point
                  mujoco.mj_jac(rb.m, rb.d, jp, None, pt, rb.bid(body))
                  res.append(w * (target - pt)); jac.append(w * jp[:, rb.vadr]); labels.append("pos " + body)

              def add_rot(body, R_t, w=W_ROT):
                  jr = np.zeros((3, rb.m.nv)); mujoco.mj_jacBody(rb.m, rb.d, None, jr, rb.bid(body))
                  res.append(w * rot_err(R_t, rb.d.xmat[rb.bid(body)].reshape(3, 3))); jac.append(w * jr[:, rb.vadr]); labels.append("rot " + body)

              for s, S in (("left", "Left"), ("right", "Right")):
                  hip = P(f"{s}_hip_pitch_link").copy()
                  d_th = hp(f"{S}Leg", f) - hp(f"{S}UpLeg", f); d_th /= np.linalg.norm(d_th)
                  d_sh = hp(f"{S}Foot", f) - hp(f"{S}Leg", f); d_sh /= np.linalg.norm(d_sh)
                  knee_t = hip + d_th * rb.len[f"{s}_thigh"]
                  ankle_t = knee_t + d_sh * rb.len[f"{s}_shank"]
                  add_pos(f"{s}_knee_link", knee_t)
                  add_pos(f"{s}_ankle_pitch_link", ankle_t, w=2.0 * W_POS)
                  add_rot(f"{s}_ankle_roll_link", foot_R[s], w=W_ROT)
                  sh = P(f"{s}_shoulder_pitch_link").copy()
                  d_ua = hp(f"{S}ForeArm", f) - hp(f"{S}Arm", f); d_ua /= np.linalg.norm(d_ua)
                  d_fa = hp(f"{S}Hand", f) - hp(f"{S}ForeArm", f); d_fa /= np.linalg.norm(d_fa)
                  elbow_t = sh + d_ua * rb.len[f"{s}_upper_arm"]
                  wrist_t = elbow_t + d_fa * rb.len[f"{s}_forearm"]
                  add_pos(f"{s}_elbow_link", elbow_t, w=0.7 * W_POS)
                  add_pos(f"{s}_wrist_pitch_link", wrist_t, w=0.7 * W_POS)
              add_rot("torso", R_torso, w=2.0 * W_ROT)
              res.append(W_REG * rb.reg * (rb.home - q)); jac.append(W_REG * np.diag(rb.reg))
              if f > 0:
                  res.append(W_PREV * (q_prev - q)); jac.append(W_PREV * np.eye(len(JOINTS)))
              r = np.concatenate(res); Jm = np.concatenate(jac)
              dq = np.linalg.solve(Jm.T @ Jm + LAMBDA * np.eye(len(JOINTS)), Jm.T @ r)
              q = np.clip(q + dq, rb.lo, rb.hi)
              if np.linalg.norm(dq) < 1e-4:
                  break
          return q, r, res, labels

        q_new, r, res, labels = solve(q.copy())
        n_reg = len(JOINTS) * (2 if f > 0 else 1)              # regularization rows at the end of r
        if np.linalg.norm(r[:-n_reg]) > RESET_RESIDUAL:
            q0 = q_prev.copy(); q0[rb.leg_idx] = rb.home[rb.leg_idx]   # legs only: arms keep continuity
            q_alt, r_alt, res_alt, labels_alt = solve(q0)
            if np.linalg.norm(r_alt[:-n_reg]) < np.linalg.norm(r[:-n_reg]):
                q_new, r, res, labels = q_alt, r_alt, res_alt, labels_alt
        q = q_new
        rb.set(root_p, rq, q)
        if f in debug_frames:
            print(f"  frame {f}: " + ", ".join(f"{l} {np.linalg.norm(v):.2f}" for l, v in zip(labels, res) if np.linalg.norm(v) > 0.02))
        out_root_pos[f] = root_p; out_root_quat[f] = rq; out_q[f] = q
        err_log.append(np.linalg.norm(r[:-n_reg]))
        if verbose and f % 500 == 0:
            print(f"  frame {f}/{F}  residual {err_log[-1]:.3f}")

    # light temporal smoothing of joint angles (removes single-frame flips), re-clipped to the bounds
    if SMOOTH_SIGMA_FRAMES > 0:
        k = int(3 * SMOOTH_SIGMA_FRAMES); x = np.arange(-k, k + 1)
        g = np.exp(-0.5 * (x / SMOOTH_SIGMA_FRAMES) ** 2); g /= g.sum()
        padded = np.pad(out_q, ((k, k), (0, 0)), mode="edge")
        out_q = np.stack([np.convolve(padded[:, j], g, mode="valid") for j in range(out_q.shape[1])], axis=1)
        out_q = np.clip(out_q, rb.lo, rb.hi)

    # ground contact: lowest sole on the floor while the human has a foot down; never below it
    human_down = (np.minimum(toe_z, foot_z) - ground_h) < 0.04
    shift = np.zeros(F)
    for f in range(F):
        rb.set(out_root_pos[f], out_root_quat[f], out_q[f])
        low = rb.sole_lowest()
        shift[f] = -low if human_down[f] else max(-low, 0.0)
    k = 5                                                   # light smoothing (0.33 s) of the height shift
    shift_s = np.convolve(np.pad(shift, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    out_root_pos[:, 2] += shift_s
    meta = dict(_err=np.array(err_log), _shift=shift_s, frames=F, fps=1.0 / ft, s_root=float(s_root), human_leg_len=float(human_leg_len),
                robot_leg_len=float(robot_leg_len), flat_foot_pitch_deg=float(np.degrees(flat_pitch)),
                mean_ik_residual=float(np.mean(err_log)), p95_ik_residual=float(np.percentile(err_log, 95)), **calib)
    return out_root_pos, out_root_quat, out_q, meta, rb


def quality(rb, root_pos, root_quat, qs, fps):
    """Joint-limit margins, penetration, skating, self-collision over the clip."""
    F = len(qs)
    lo = rb.m.jnt_range[[rb.m.joint(n).id for n in JOINTS], 0]; hi = rb.m.jnt_range[[rb.m.joint(n).id for n in JOINTS], 1]
    viol = np.maximum(lo - qs, 0) + np.maximum(qs - hi, 0)
    ap = [JOINTS.index("left_ankle_pitch"), JOINTS.index("right_ankle_pitch")]
    pen, skate, selfc = [], {"left": [], "right": []}, []
    prev = None
    md = mujoco.MjData(rb.m)
    for f in range(F):
        md.qpos[0:3] = root_pos[f]; md.qpos[3:7] = root_quat[f]; md.qpos[rb.qadr] = qs[f]
        mujoco.mj_forward(rb.m, md)
        rb.d.qpos[:] = md.qpos; mujoco.mj_kinematics(rb.m, rb.d)
        pen.append(min(rb.sole_lowest(), 0.0))
        cur = {s: rb.sole_corners(s) for s in ("left", "right")}
        if prev is not None:
            for s in ("left", "right"):
                if cur[s][:, 2].min() < 0.015:
                    skate[s].append(np.linalg.norm((cur[s][:, :2] - prev[s][:, :2]).mean(0)) * fps)
        prev = cur
        worst = 0.0
        for c in md.contact[: md.ncon]:
            b1, b2 = rb.m.geom_bodyid[c.geom1], rb.m.geom_bodyid[c.geom2]
            if b1 and b2:
                worst = min(worst, c.dist)
        selfc.append(-worst)
    sk = np.concatenate([skate["left"], skate["right"]]) if (skate["left"] or skate["right"]) else np.zeros(1)
    selfc = np.array(selfc)
    per_frame = dict(_penetration=np.array(pen), _selfc=selfc)
    return dict(
        **per_frame,
        joint_limit_violation_max_rad=float(viol.max()),
        ankle_pitch_min=float(qs[:, ap].min()), ankle_pitch_limit=float(ANKLE_PITCH_LOWER),
        foot_penetration_max_mm=float(-min(pen) * 1000),
        skating_speed_mean_mps=float(sk.mean()), skating_speed_p95_mps=float(np.percentile(sk, 95)),
        self_collision_frames_over_1cm=int((selfc > 0.01).sum()), self_collision_max_mm=float(selfc.max() * 1000),
        self_collision_fraction_over_1cm=float((selfc > 0.01).mean()),
    )


def render_video(rb, root_pos, root_quat, qs, path, fps=30, every=1, size=(640, 480)):
    import mediapy as media
    spec = mujoco.MjSpec.from_file(str(TRAIN_XML))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.55, 0.6, 0.65, 1])
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    m = spec.compile(); d = mujoco.MjData(m)
    qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINTS])
    r = mujoco.Renderer(m, size[1], size[0])
    cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 3.0, -12, 135
    frames = []
    for f in range(0, len(qs), every):
        d.qpos[0:3] = root_pos[f]; d.qpos[3:7] = root_quat[f]; d.qpos[qadr] = qs[f]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = [root_pos[f][0], root_pos[f][1], 0.6]
        r.update_scene(d, cam); frames.append(r.render())
    media.write_video(path, frames, fps=fps / every)
    r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh", required=True)
    ap.add_argument("--out", required=True, help="output CSV (30 fps)")
    ap.add_argument("--frames", default=None, help="start:end subset, e.g. 0:600")
    ap.add_argument("--video", default=None)
    ap.add_argument("--calib-frame", type=int, default=0, help="frame where the human stands upright (LAFAN1: 0 = T-pose)")
    a = ap.parse_args()
    fr = tuple(int(x) for x in a.frames.split(":")) if a.frames else None
    root_pos, root_quat, qs, meta, rb = retarget(a.bvh, fr, calib_frame=a.calib_frame)
    qual = quality(rb, root_pos, root_quat, qs, meta["fps"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    quat_xyzw = root_quat[:, [1, 2, 3, 0]]
    np.savetxt(a.out, np.concatenate([root_pos, quat_xyzw, qs], axis=1), delimiter=",", fmt="%.6f")
    debug = {k[1:]: v for k, v in {**meta, **qual}.items() if k.startswith("_")}
    np.savez(Path(a.out).with_suffix(".debug.npz"), **debug)
    report = {"bvh": str(a.bvh), "frames_range": a.frames, "joints": JOINTS,
              **{k: v for k, v in {**meta, **qual}.items() if not k.startswith("_")}}
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "joints"}, indent=1))
    if a.video:
        render_video(rb, root_pos, root_quat, qs, a.video)
        print("wrote", a.video)


if __name__ == "__main__":
    main()
