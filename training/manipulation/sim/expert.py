"""Scripted pick-and-place expert (privileged: knows true object/target poses).

The motion is planned as TCP waypoints in two object-centric segments (what MimicGen needs):
  segment 0  (relative to the OBJECT pose at segment start): pre-grasp -> grasp -> close -> lift
  segment 1  (relative to the TARGET pose at segment start): above target -> lower -> open -> retreat
Execution: min-jerk interpolation of the TCP pose at 20 Hz, damped-least-squares IK per step for the
active arm (the other arm holds its ready pose), joint position targets sent to the env.

v1 grasp (cups): side grasp, gripper horizontal (fingers close horizontally, TCP z axis up), TCP at the
cup axis, low-mid height (bottom + 45 % of height, clamped 25-60 mm), approach along the horizontal line
from the active shoulder to the cup.
"""
import math

import mujoco
import numpy as np

from ik import mat_to_quat, quat_to_mat, solve_feasible

OPEN, CLOSED = 1.0, 0.0
PITCH_DEG = 15.0


def min_jerk(s):
    """Minimum-jerk profile at phase `s`."""
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def slerp(q0, q1, s):
    """Spherical interpolation of quaternions."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(dot)
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


def pose_mat(pos, quat):
    """Transform from position and quaternion."""
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(quat)
    T[:3, 3] = pos
    return T


def mat_pose(T):
    """Position and quaternion of a transform."""
    return T[:3, 3].copy(), mat_to_quat(T[:3, :3])


def choose_side(env):
    """Arm on the object's side of the robot (env.reset already sampled it)."""
    if getattr(env, "active_side", None):
        return env.active_side
    o, R = env.base_frame
    p = R.T @ (env.obj_pose(env.obj)[:3] - o)
    return "left" if p[1] >= 0 else "right"


def grasp_frame(env, side, center, yaw_off_deg=0.0, pitch_deg=None):
    """TCP frame for a side grasp: x = approach (horizontal toward the object, from the active shoulder,
    clamped to +/-40 deg of robot forward, plus yaw_off), y = closing axis (horizontal), fingers pitched down."""
    sh = env.d.xpos[env.m.body(f"{env.p}{side}_shoulder_pitch_link").id]
    a = center - sh
    a[2] = 0
    a /= np.linalg.norm(a)
    fwd = env.base_frame[1][:, 0]
    ang = math.atan2(fwd[0] * a[1] - fwd[1] * a[0], float(fwd @ a))
    ang = float(np.clip(ang, -math.radians(40), math.radians(40))) + math.radians(yaw_off_deg)
    c_, s_ = math.cos(ang), math.sin(ang)
    a = np.array([c_ * fwd[0] - s_ * fwd[1], s_ * fwd[0] + c_ * fwd[1], 0.0])
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, a)
    th = math.radians(PITCH_DEG if pitch_deg is None else pitch_deg)
    a2 = math.cos(th) * a - math.sin(th) * z
    z2 = math.sin(th) * a + math.cos(th) * z
    return np.column_stack([a2, y, z2])


YAW_OFFSETS = (0.0, 20.0, -20.0, 40.0, -40.0)
PITCHES = (15.0, 0.0, 30.0)


def _feasible(env, side, poses, allow):
    """Whether every pose has a collision-free IK solution."""
    ik = env.ik[side]
    base_q = env.d.qpos.copy()
    q = env.ready_q[side]
    for pos, R in poses:
        sol = solve_feasible(ik, base_q, np.asarray(pos), R, lambda dd: env.plan_pose_ok(dd, side, allow), q_init=q, restarts=6)
        if sol is None:
            return False
        q = sol[0]
    return True


def plan(env, side, search=True):
    """Waypoints (world frame). Searches approach yaw / finger pitch for the pick, and yaw for the place,
    keeping the first orientation whose key poses all have collision-free IK. Returns None if none works."""
    po = env.obj_pose(env.obj)
    pt = env.obj_pose(env.target)
    mo, mt = env.meta(env.obj), env.meta(env.target)
    h = 2 * mo.get("half_extents", [0.04] * 3)[2]
    bottom = po[2] - env.bottom_offset(env.obj)
    # upper part of the cup: with the finger pitch the Dex1 palm's top-front corner sits ~5 cm above the
    # pads and ~3.4 cm behind them, so the rim must stay below it; also keeps the fingertips clear of the plate.
    gz = bottom + float(np.clip(0.72 * h, 0.03, 0.09))
    c = np.array([po[0], po[1], gz])
    t_top = pt[2] - env.bottom_offset(env.target) + 2 * mt.get("half_extents", [0, 0, 0.01])[2]
    above = gz - bottom
    place = np.array([pt[0], pt[1], t_top + above + 0.002])
    allow = f"{env.obj}/"
    choice = None
    for pitch in (PITCHES if search else (PITCH_DEG,)):
        for yo in (YAW_OFFSETS if search else (0.0,)):
            R = grasp_frame(env, side, c, yo, pitch)
            if not search or _feasible(env, side, [(c + [0, 0, 0.10], R), (c + [0, 0, 0.02], R), (c, R), (c + [0, 0, 0.06], R)], allow):
                choice = (pitch, R)
                break
        if choice:
            break
    if choice is None:
        env.plan_fail = "no feasible pick orientation"
        return None
    pitch, R = choice
    Rp = None
    for yo in (YAW_OFFSETS if search else (0.0,)):
        Rc = grasp_frame(env, side, place, yo, pitch)            # yaw only: the held cup stays upright
        if not search or _feasible(env, side, [(place + [0, 0, 0.06], Rc), (place, Rc), (place + [0, 0, 0.08], Rc)], allow):
            Rp = Rc
            break
    if Rp is None:
        env.plan_fail = "no feasible place orientation"
        return None
    q, qp = mat_to_quat(R), mat_to_quat(Rp)
    return [
        dict(seg=0, pos=c + np.array([0, 0, 0.10]), quat=q, grip=OPEN, steps=30, joint=True),    # above the cup
        dict(seg=0, pos=c + np.array([0, 0, 0.02]), quat=q, grip=OPEN, steps=20),    # straight down: fingers straddle it
        dict(seg=0, pos=c, quat=q, grip=OPEN, steps=10),
        dict(seg=0, pos=c, quat=q, grip=CLOSED, steps=14),
        dict(seg=0, pos=c + np.array([0, 0, 0.06]), quat=q, grip=CLOSED, steps=20),  # lift
        dict(seg=1, pos=place + np.array([0, 0, 0.06]), quat=qp, grip=CLOSED, steps=40, joint=True),
        dict(seg=1, pos=place, quat=qp, grip=CLOSED, steps=24),
        dict(seg=1, pos=place, quat=qp, grip=OPEN, steps=12),
        dict(seg=1, pos=place + np.array([0, 0, 0.08]), quat=qp, grip=OPEN, steps=20), # straight up, clear of the cup
    ]


def to_relative(wps, ref_obj, ref_tgt):
    """Express waypoint poses in the object (seg 0) / target (seg 1) frames (MimicGen source form)."""
    To, Tt = pose_mat(ref_obj[:3], ref_obj[3:]), pose_mat(ref_tgt[:3], ref_tgt[3:])
    out = []
    for w in wps:
        Tref = To if w["seg"] == 0 else Tt
        Tw = pose_mat(w["pos"], w["quat"])
        out.append({**w, "rel": np.linalg.inv(Tref) @ Tw})
    return out


def from_relative(rel_wps, ref_obj, ref_tgt):
    """Absolute waypoints from object- or target-relative ones."""
    To, Tt = pose_mat(ref_obj[:3], ref_obj[3:]), pose_mat(ref_tgt[:3], ref_tgt[3:])
    out = []
    for w in rel_wps:
        Tref = To if w["seg"] == 0 else Tt
        pos, quat = mat_pose(Tref @ w["rel"])
        out.append({**{k: v for k, v in w.items() if k != "rel"}, "pos": pos, "quat": quat})
    return out


def compute_traj(env, side, wps, pos_noise=0.0, rng=None, max_ik_err=0.012, start=None, check_collisions=True):
    """IK the whole min-jerk TCP path before touching physics. Returns a list of per-step
    dict(q, grip, seg, tcp) or None if any step is unreachable (error > max_ik_err)."""
    rng = rng or np.random.default_rng()
    if start is None:
        p0, R0 = env.tcp_pose(side)
        q_arm = env.d.qpos[env.arm_q[side]].copy()
        g0 = env.grip_open_frac(side)
    else:
        p0, R0, q_arm, g0 = start
    q0 = mat_to_quat(R0)
    base_q = env.d.qpos.copy()
    ik = env.ik[side]
    rest0 = ik.rest
    traj = []
    try:
        for wi, w in enumerate(wps):
            # robust IK at the waypoint (escapes local minima), then follow the straight TCP path with
            # warm-started IK pulled toward that solution, so consecutive steps stay on one arm branch
            allow = f"{env.obj}/"
            if check_collisions:
                sol = solve_feasible(ik, base_q, np.asarray(w["pos"]), quat_to_mat(w["quat"]),
                                     lambda dd: env.plan_pose_ok(dd, side, allow), q_init=q_arm)
                if sol is None:
                    env.plan_fail = f"waypoint {wi} (seg {w['seg']}) no collision-free IK"
                    return None
                q_wp = sol[0]
            else:
                q_wp, pe, _ = ik.solve_robust(base_q, np.asarray(w["pos"]), quat_to_mat(w["quat"]), q_init=q_arm)
                if pe > max_ik_err:
                    return None
            ik.rest = q_wp
            n = int(w["steps"])
            if w.get("joint"):
                # free-space move: min-jerk in joint space (smooth, no branch switches), collision-checked
                qa = q_arm.copy()
                for k in range(1, n + 1):
                    s = min_jerk(k / n)
                    q_s = qa + s * (q_wp - qa)
                    grip = w["grip"] if w["grip"] == g0 else (g0 + s * (w["grip"] - g0))
                    ik.d.qpos[:] = base_q
                    ik.d.qpos[ik.qadr] = q_s
                    if check_collisions and not env.plan_pose_ok(ik.d, side, f"{env.obj}/"):
                        env.plan_fail = f"joint move to waypoint {wi}: collision"
                        return None
                    mujoco.mj_kinematics(env.m, ik.d)
                    tcp = (ik.d.site_xpos[ik.site].copy(), ik.d.site_xmat[ik.site].reshape(3, 3).copy())
                    traj.append(dict(q=q_s.copy(), grip=float(grip), seg=-1, tcp=tcp))   # -1 = free-space transit
                q_arm = q_wp.copy()
                p0, q0, g0 = np.asarray(w["pos"], float), np.asarray(w["quat"], float), w["grip"]
                continue
            for k in range(1, n + 1):
                s = min_jerk(k / n)
                pos = p0 + s * (np.asarray(w["pos"]) - p0)
                if pos_noise > 0:
                    pos = pos + rng.normal(0, pos_noise, 3)
                quat = slerp(q0, w["quat"], s)
                grip = w["grip"] if w["grip"] == g0 else (g0 + s * (w["grip"] - g0))
                q_try, pe, _ = ik.solve(base_q, pos, quat_to_mat(quat), q_init=q_arm, iters=40)
                if pe > max_ik_err:
                    q_try, pe, _ = ik.solve(base_q, pos, quat_to_mat(quat), q_init=q_wp, iters=150)
                if pe > max_ik_err or np.abs(q_try - q_arm).max() > 0.2:
                    env.plan_fail = f"path to waypoint {wi}: ik err {pe:.3f} / jump {np.abs(q_try - q_arm).max():.2f}"
                    return None
                if check_collisions:
                    ik.d.qpos[ik.qadr] = q_try
                    if not env.plan_pose_ok(ik.d, side, f"{env.obj}/"):
                        env.plan_fail = f"path to waypoint {wi}: collision"
                        return None
                q_arm = q_try
                traj.append(dict(q=q_arm.copy(), grip=float(grip), seg=int(w["seg"]), tcp=(pos.copy(), quat_to_mat(quat))))
            p0, q0, g0 = np.asarray(w["pos"], float), np.asarray(w["quat"], float), w["grip"]
    finally:
        ik.rest = rest0
    return traj


def play(env, side, traj, recorder=None, settle_steps=12):
    """Send the planned joint targets at 20 Hz. Returns (success, reason)."""
    other = "right" if side == "left" else "left"
    hold = env.ready_q[other]
    obs = env.observe()
    for st in traj:
        ql, qr = (st["q"], hold) if side == "left" else (hold, st["q"])
        gl, gr = (st["grip"], OPEN) if side == "left" else (OPEN, st["grip"])
        if recorder is not None:
            recorder.add(obs, ql, qr, gl, gr, env.tcp_pose(side), st["seg"])
        obs = env.step(ql, qr, gl, gr)
    last = traj[-1]
    ql, qr = (last["q"], hold) if side == "left" else (hold, last["q"])
    gl, gr = (last["grip"], OPEN) if side == "left" else (OPEN, last["grip"])
    for _ in range(settle_steps):
        if recorder is not None:
            recorder.add(obs, ql, qr, gl, gr, env.tcp_pose(side), 2)
        obs = env.step(ql, qr, gl, gr)
    ok = env.success()
    return ok, "success" if ok else "task check failed"


def run_scripted(env, seed, recorder=None):
    """Reset + plan + play one scripted episode. Returns (ok, reason, side)."""
    env.reset(seed)
    if not env.reset_ok:
        return False, "bad reset", None
    side = choose_side(env)
    wps = plan(env, side)
    traj = compute_traj(env, side, wps) if wps is not None else None
    if traj is None:
        return False, "unreachable", side
    ok, why = play(env, side, traj, recorder)
    return ok, why, side
