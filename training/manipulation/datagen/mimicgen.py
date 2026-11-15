"""MimicGen-style demo generation (Mandlekar et al., 2023) for the standing G1-Dex1 tabletop env.

Source demos (scripted successes, recorded) are split into object-centric segments using the recorded
`segment` labels: 0 = pick (relative to the object), 1 = place (relative to the target); -1 = free-space
transit, 2 = settle. For a new scene:
  1. pick a source demo of the same arm,
  2. transform its pick-segment TCP poses from the source object frame to the new object frame and its
     place-segment poses from the source target frame to the new target frame,
  3. joint-space transit -> follow the transformed pick (IK per step + small noise) -> transit -> place -> settle,
     every step collision-checked at plan time,
  4. execute in physics; keep only successes.
Symmetric categories (cups, plates, bowls, cans, fruit) use translation only (a round object has no meaningful
yaw); other categories also apply the object's yaw change.
"""
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import expert as X  # noqa: E402
from ik import mat_to_quat, quat_to_mat, solve_feasible  # noqa: E402

SYMMETRIC = {"cup", "plate", "bowl", "can", "fruit", "bowl_small"}


def yaw_of(quat):
    R = quat_to_mat(quat)
    return math.atan2(R[1, 0], R[0, 0])


def frame(pose, use_yaw):
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    if use_yaw:
        y = yaw_of(pose[3:])
        T[:2, :2] = [[math.cos(y), -math.sin(y)], [math.sin(y), math.cos(y)]]
    return T


class Source:
    """One source demo loaded from a raw episode (npz + json)."""

    def __init__(self, npz_path, meta):
        z = np.load(npz_path)
        self.side = meta["active_side"]
        self.category = meta["object_category"]
        self.target_category = meta.get("target_category", "plate")
        seg = z["segment"]
        tcp = z["tcp_active"]
        grip = z[f"act_grip_{self.side}"]
        self.segs = {}
        for sid in (0, 1):
            idx = np.where(seg == sid)[0]
            ref = z["obj_pose"][idx[0]] if sid == 0 else z["target_pose"][idx[0]]
            self.segs[sid] = dict(tcp=tcp[idx], grip=grip[idx], ref=ref)


def transform_segment(seg, new_ref, use_yaw):
    Tsrc = frame(seg["ref"], use_yaw)
    Tnew = frame(new_ref, use_yaw)
    M = Tnew @ np.linalg.inv(Tsrc)
    out = []
    for p in seg["tcp"]:
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(p[3:])
        T[:3, 3] = p[:3]
        Tn = M @ T
        out.append((Tn[:3, 3].copy(), Tn[:3, :3].copy()))
    return out


def plan_generated(env, src, rng, noise=0.002):
    """Joint-target trajectory for the current scene from a source demo, or None (infeasible)."""
    side = env.active_side
    ik = env.ik[side]
    base_q = env.d.qpos.copy()
    allow = f"{env.obj}/"
    ok = lambda dd: env.plan_pose_ok(dd, side, allow)
    use_yaw_o = src.category not in SYMMETRIC
    use_yaw_t = src.target_category not in SYMMETRIC
    poses = {0: transform_segment(src.segs[0], env.obj_pose(env.obj), use_yaw_o),
             1: transform_segment(src.segs[1], env.obj_pose(env.target), use_yaw_t)}
    traj = []
    q_arm = env.d.qpos[env.arm_q[side]].copy()
    grip_now = env.grip_open_frac(side)
    rest0 = ik.rest
    try:
        for sid in (0, 1):
            seg_poses, grips = poses[sid], src.segs[sid]["grip"]
            # transit (joint space) to the first pose of the segment
            sol = solve_feasible(ik, base_q, seg_poses[0][0], seg_poses[0][1], ok, q_init=q_arm, restarts=6)
            if sol is None:
                env.plan_fail = f"mimicgen: no IK for segment {sid} start"
                return None
            q_goal = sol[0]
            n = 30 if sid == 0 else 40
            qa = q_arm.copy()
            for k in range(1, n + 1):
                s = X.min_jerk(k / n)
                q_s = qa + s * (q_goal - qa)
                ik.d.qpos[:] = base_q
                ik.d.qpos[ik.qadr] = q_s
                if not ok(ik.d):
                    env.plan_fail = f"mimicgen: transit {sid} collision"
                    return None
                mujoco.mj_kinematics(env.m, ik.d)
                traj.append(dict(q=q_s.copy(), grip=float(grip_now), seg=-1,
                                 tcp=(ik.d.site_xpos[ik.site].copy(), ik.d.site_xmat[ik.site].reshape(3, 3).copy())))
            q_arm = q_goal.copy()
            ik.rest = q_goal
            # follow the transformed segment
            for (pos, R), g in zip(seg_poses, grips):
                pos = pos + rng.normal(0, noise, 3) * (1.0 if g > 0.5 else 0.3)
                q_try, pe, _ = ik.solve(base_q, pos, R, q_init=q_arm, iters=40)
                if pe > 0.012 or np.abs(q_try - q_arm).max() > 0.2:
                    env.plan_fail = f"mimicgen: segment {sid} IK"
                    return None
                ik.d.qpos[ik.qadr] = q_try
                if not ok(ik.d):
                    env.plan_fail = f"mimicgen: segment {sid} collision"
                    return None
                q_arm = q_try
                traj.append(dict(q=q_arm.copy(), grip=float(g), seg=sid, tcp=(pos.copy(), R)))
            grip_now = float(grips[-1])
    finally:
        ik.rest = rest0
    return traj
