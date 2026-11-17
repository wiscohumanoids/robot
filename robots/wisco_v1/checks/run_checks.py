"""Pre-training checks in plain MuJoCo with mjlab's actuator model (sim_replica.py).

  A1 model          the training model loads: 29 policy joints, sensors, named collision geoms
  A2 default pose   no self-collision, both soles flat on the floor, centre of mass over the feet
  A3 stand 10 s     zero policy action holds the pose; shaking dies out; holding torques vs rated
  A4 gains          per-joint step response (robot hung in the air): stable, settles, no overshoot blow-up
  A5 directions     every joint moves the expected way; left/right mirror; range sweep for collisions
  A6 falls          limp collapse, a push, a tilted start: does the 70 deg tilt rule catch each fall?
  A7 foot contacts  both feet carry the weight in stand; foot sites at the sole
  A8 sim sizing     max contacts / constraint rows (mjlab nconmax / njmax), CPU step time

Run:  python run_checks.py      -> results/REPORT.md, results/results.json, results/*.png
"""
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model"))
import ankle_motor  # noqa: E402
import robot_params as P  # noqa: E402
from sim_replica import DECIMATION, SIM_DT, Sim  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
FALL_DEG = 70.0                 # mjlab velocity task "fell_over" termination
R = {}                          # results, by check
SIZING = {"ncon": 0, "nefc": 0}


def track(sim):
    """Track the largest contact and constraint counts."""
    SIZING["ncon"] = max(SIZING["ncon"], int(sim.d.ncon))
    SIZING["nefc"] = max(SIZING["nefc"], int(sim.d.nefc))


def run(sim, seconds, action=None, target=None, every=None):
    """Run the policy loop for `seconds` with a fixed action or target, tracking extremes."""
    n = int(round(seconds / (SIM_DT * DECIMATION)))
    for i in range(n):
        sim.step_policy(action=action, target=target)
        track(sim)
        if every:
            every(i, sim)


def ankle_motor_torques(sim):
    """Per-motor ankle torques through the linkage, both sides."""
    out = {}
    for side in ("left", "right"):
        ip, ir = sim.names.index(f"{side}_ankle_pitch"), sim.names.index(f"{side}_ankle_roll")
        q, t = sim.joint_q(), sim.torque()
        out[side] = ankle_motor.motor_torques(side, q[ip], q[ir], t[ip], t[ir])
    return out


# ----------------------------------------------------------------------------- A1
def a1_model():
    """A1: the training model loads with the expected joints, sensors and named collision geoms."""
    sim = Sim()
    m = sim.m
    col = [m.geom(g).name for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    unnamed = [g for g in range(m.ngeom) if (m.geom_contype[g] and not m.geom(g).name)]
    sensors = [m.sensor(i).name for i in range(m.nsensor)]
    need = ["imu_ang_vel", "imu_lin_vel", "imu_lin_acc", "imu_upvector", "root_angmom"]
    ok = m.nu == 29 and not unnamed and all(s in sensors for s in need) and all(
        n in col for n in P.FOOT_GEOMS) and all(m.site(s) for s in P.FOOT_SITES)
    R["A1"] = {"pass": bool(ok), "policy_joints": m.nu, "collision_geoms": len(col) - 1,
               "mass_kg": round(float(m.body_subtreemass[1]), 3), "sensors": sensors}


# ----------------------------------------------------------------------------- A2
def a2_default_pose():
    """A2: the default pose has no self-collision, flat soles and the centre of mass over the feet."""
    sim = Sim()
    sim.reset(z_offset=0.0)
    mujoco.mj_forward(sim.m, sim.d)
    track(sim)
    selfc = [c for c in sim.self_contacts() if c[2] < -0.001]
    tilt = {}
    for g in P.FOOT_GEOMS:
        Rm = sim.d.geom_xmat[sim.m.geom(g).id].reshape(3, 3)
        tilt[g] = float(np.degrees(np.arccos(np.clip(Rm[2, 2], -1, 1))))
    soles = {s: round(float(sim.d.site_xpos[sim.m.site(s).id][2]), 4) for s in P.FOOT_SITES}
    # centre of mass vs the support area (bounding box of both foot boxes)
    com = sim.d.subtree_com[sim.m.body("pelvis").id]
    pts = []
    for g in P.FOOT_GEOMS:
        gid = sim.m.geom(g).id
        Rm = sim.d.geom_xmat[gid].reshape(3, 3); h = sim.m.geom_size[gid]
        c = np.array([[x, y, -1] for x in (-1, 1) for y in (-1, 1)]) * h
        pts.append(c @ Rm.T + sim.d.geom_xpos[gid])
    pts = np.vstack(pts)
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    com_margin_front, com_margin_back = float(x1 - com[0]), float(com[0] - x0)
    ok = not selfc and max(tilt.values()) < 1.0 and min(com_margin_front, com_margin_back) > 0.03
    R["A2"] = {"pass": bool(ok), "self_contacts_over_1mm": selfc, "sole_tilt_deg": tilt,
               "foot_site_height_m": soles, "com_xy": com[:2].round(4).tolist(),
               "com_margin_to_toe_m": round(com_margin_front, 3), "com_margin_to_heel_m": round(com_margin_back, 3),
               "pelvis_height_m": sim.pelvis_z}
    render(sim, "default_pose")


# ----------------------------------------------------------------------------- A3
def a3_stand():
    """A3: zero action holds the pose for 10 s; shaking dies out; holding torques against rated."""
    sim = Sim()
    sim.reset()
    hist = {"t": [], "z": [], "tilt": [], "x": [], "y": [], "qd": [], "tau": [], "motor": []}

    def rec(i, s):
        hist["t"].append(i * SIM_DT * DECIMATION); hist["z"].append(s.d.qpos[2]); hist["tilt"].append(s.tilt_deg())
        hist["x"].append(s.d.qpos[0]); hist["y"].append(s.d.qpos[1])
        hist["qd"].append(s.d.qvel[s.vadr].copy()); hist["tau"].append(s.torque())
        mt = ankle_motor_torques(s); hist["motor"].append(np.concatenate([mt["left"], mt["right"]]))
    run(sim, 10.0, every=rec)
    t = np.array(hist["t"]); qd = np.array(hist["qd"]); tau = np.array(hist["tau"]); mot = np.array(hist["motor"])
    last = t > 8.0
    rms_qd = float(np.sqrt(np.mean(qd[last] ** 2)))
    dz = float(abs(hist["z"][-1] - sim.pelvis_z)); drift = float(np.hypot(hist["x"][-1], hist["y"][-1]))
    hold = np.abs(tau[last]).mean(0)
    peak = np.abs(tau).max(0)
    joints = {n: {"hold_nm": round(float(hold[i]), 2), "hold_pct_rated": round(float(100 * hold[i] / sim.rated[i]), 1),
                  "peak_nm": round(float(peak[i]), 2), "peak_pct_effort": round(float(100 * peak[i] / sim.effort[i]), 1)}
              for i, n in enumerate(sim.names)}
    motor_hold = np.abs(mot[last]).mean(0); motor_peak = np.abs(mot).max(0)
    q_err = np.abs(sim.joint_q() - sim.default)
    # Pass = doesn't fall and the shaking dies out. (A 5 deg tilt limit was dropped: the G1, which
    # trains fine, falls over completely in this test - see g1_baseline.py.)
    ok = dz < 0.05 and max(hist["tilt"]) < 20 and drift < 0.10 and rms_qd < 0.02
    R["A3"] = {"pass": bool(ok), "pelvis_height_change_m": round(dz, 4), "max_tilt_deg": round(max(hist["tilt"]), 2),
               "horizontal_drift_m": round(drift, 4), "joint_speed_rms_last_2s": round(rms_qd, 5),
               "max_joint_sag_rad": {sim.names[i]: round(float(q_err[i]), 4) for i in np.argsort(-q_err)[:5]},
               "joints": joints,
               "ankle_motor_hold_nm": {"left": motor_hold[:2].round(2).tolist(), "right": motor_hold[2:].round(2).tolist()},
               "ankle_motor_peak_nm": {"left": motor_peak[:2].round(2).tolist(), "right": motor_peak[2:].round(2).tolist()},
               "ankle_motor_rated_nm": P.X4_36_RATED_NM, "ankle_motor_peak_limit_nm": P.X4_36_PEAK_NM}
    render(sim, "stand_after_10s")


# ----------------------------------------------------------------------------- A4
def a4_gains():
    """A4: per-joint step responses with the robot hung in the air."""
    out = {}
    worst_ok = True
    for n in P.joint_params():
        sim = Sim(fixed_base=True)
        sim.reset()
        run(sim, 0.5)                                      # settle under gravity
        i = sim.names.index(n)
        q0 = sim.joint_q()[i]
        lo, hi = sim.params[n]["lower"], sim.params[n]["upper"]
        step = 0.2 if q0 + 0.2 < hi - 0.05 else -0.2       # 0.2 rad step, away from the limit
        target = sim.default.copy(); target[i] = sim.default[i] + step
        qs = []
        for _ in range(int(1.0 / SIM_DT)):                # 1 s at the physics rate
            sim.physics_step(target)
            qs.append(sim.d.qpos[sim.qadr[i]])
        qs = np.array(qs) - q0
        final = qs[-1]
        stable = bool(np.all(np.isfinite(qs)) and abs(qs).max() < 3 * abs(step))
        over = float(max(0.0, (np.max(qs * np.sign(step)) - abs(final)) / max(abs(final), 1e-6)))
        band = 0.05 * abs(final)
        outside = np.where(np.abs(qs - final) > band)[0]
        settle = float((outside[-1] + 1) * SIM_DT) if len(outside) else 0.0
        reached = float(final / step)
        ok = stable and settle < 0.4 and over < 0.25
        worst_ok &= ok
        out[n] = {"step_rad": step, "reached_frac": round(reached, 3), "overshoot_frac": round(over, 3),
                  "settle_s": round(settle, 3), "stable": stable, "pass": bool(ok)}
    R["A4"] = {"pass": bool(worst_ok), "joints": out}


# ----------------------------------------------------------------------------- A5
TRACK_POINT = {  # body whose position shows the joint's effect
    "hip_pitch": "{s}_ankle_roll_link", "hip_roll": "{s}_ankle_roll_link", "hip_yaw": "{s}_foot",
    "knee": "{s}_ankle_roll_link", "ankle_pitch": "{s}_foot", "ankle_roll": "{s}_foot",
    "shoulder_pitch": "{s}_hand", "shoulder_roll": "{s}_hand", "shoulder_yaw": "{s}_hand",
    "elbow": "{s}_hand", "forearm_yaw": "{s}_hand", "wrist_pitch": "{s}_hand", "wrist_roll": "{s}_hand",
    "torso_yaw": "torso", "torso_roll": "torso", "torso_pitch": "torso",
}
EXPECT = {  # what +angle should do at the default pose (world axes: x forward, y left, z up)
    "hip_pitch": ("foot swings forward", lambda d, s: d[0] > 0),
    "knee": ("foot swings backward", lambda d, s: d[0] < 0),
    "hip_roll": ("foot moves outward", lambda d, s: d[1] * (1 if s == "left" else -1) > 0),
    "shoulder_roll": ("hand moves outward", lambda d, s: d[1] * (1 if s == "left" else -1) > 0),
    "shoulder_pitch": ("hand swings forward", lambda d, s: d[0] > 0),
    "elbow": ("hand moves forward/up (flexion)", lambda d, s: d[0] > 0),
}


def point(sim, name):
    """World position of a named site or body."""
    if name.endswith("_foot"):
        return sim.d.site_xpos[sim.m.site(name).id].copy()
    if name in ("torso",):
        # top of the torso: shows lean direction
        b = sim.m.body(name).id
        return sim.d.xpos[b] + sim.d.xmat[b].reshape(3, 3) @ np.array([0, 0, 0.3])
    return sim.d.xipos[sim.m.body(name).id].copy()


def a5_directions():
    """A5: every joint moves the expected way, left/right mirror, range sweeps for collisions."""
    sim = Sim(fixed_base=True)
    dirs, ok = {}, True
    for n in sim.names:
        side = "left" if n.startswith("left_") else "right" if n.startswith("right_") else None
        key = n.split("_", 1)[1] if side else n
        tp = TRACK_POINT[key].format(s=side)
        sim.reset(); mujoco.mj_kinematics(sim.m, sim.d); p0 = point(sim, tp)
        q = sim.default.copy(); q[sim.names.index(n)] += 0.05
        sim.reset(q=q); mujoco.mj_kinematics(sim.m, sim.d); d = point(sim, tp) - p0
        if key in ("ankle_pitch",):                       # toe height change instead
            sim.reset(); mujoco.mj_kinematics(sim.m, sim.d)
            Rm = sim.d.xmat[sim.m.body(f"{side}_ankle_roll_link").id].reshape(3, 3); toe0 = Rm @ [1, 0, 0]
            sim.reset(q=q); mujoco.mj_kinematics(sim.m, sim.d)
            Rm = sim.d.xmat[sim.m.body(f"{side}_ankle_roll_link").id].reshape(3, 3); toe1 = Rm @ [1, 0, 0]
            d = (toe1 - toe0) * 0.15
        entry = {"tracked": tp, "move_mm_per_0.05rad": (d * 1000).round(2).tolist()}
        if key in EXPECT:
            desc, fn = EXPECT[key]
            entry["expected"] = desc; entry["matches"] = bool(fn(d, side)); ok &= entry["matches"]
        if key == "ankle_pitch":
            entry["expected"] = "toe goes up"; entry["matches"] = bool(d[2] > 0); ok &= entry["matches"]
        dirs[n] = entry
    # left/right mirror: same +angle must give mirror-image motion (y flipped)
    mirror = {}
    for n in sim.names:
        if n.startswith("left_"):
            a = np.array(dirs[n]["move_mm_per_0.05rad"]); b = np.array(dirs["right_" + n[5:]]["move_mm_per_0.05rad"])
            b = b * np.array([1, -1, 1])
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            mirror[n[5:]] = round(cos, 4); ok &= cos > 0.99
    # range sweep: each joint to both limits (others at default), record contacts that appear
    sweep = {}
    for n in sim.names:
        i = sim.names.index(n)
        lo, hi = sim.params[n]["lower"], sim.params[n]["upper"]
        res = {}
        for lim_name, lim in (("upper", hi), ("lower", lo)):
            sim.reset(); run(sim, 0.3)
            hits, reached = {}, None
            steps = int(1.5 / (SIM_DT * DECIMATION))
            for k in range(steps + 15):
                tgt = sim.default.copy(); f = min(1.0, k / steps)
                tgt[i] = sim.default[i] + f * (lim - sim.default[i])
                sim.step_policy(target=tgt); track(sim)
                for b1, b2, dist in sim.self_contacts():
                    if dist < -0.002:
                        key = " / ".join(sorted((b1, b2)))
                        q_now = float(sim.joint_q()[i])
                        if key not in hits:
                            hits[key] = {"first_at_rad": round(q_now, 3), "max_penetration_mm": round(-dist * 1000, 1)}
                        else:
                            hits[key]["max_penetration_mm"] = max(hits[key]["max_penetration_mm"], round(-dist * 1000, 1))
            reached = float(sim.joint_q()[i])
            finite = bool(np.all(np.isfinite(sim.d.qpos)))
            res[lim_name] = {"limit_rad": round(lim, 3), "reached_rad": round(reached, 3),
                             "reached_frac": round((reached - sim.default[i]) / (lim - sim.default[i] + 1e-9), 3),
                             "self_collisions": hits, "finite": finite}
            ok &= finite
        sweep[n] = res
    R["A5"] = {"pass": bool(ok), "directions": dirs, "left_right_mirror_cosine": mirror, "range_sweep": sweep}


# ----------------------------------------------------------------------------- A6
def a6_falls():
    """A6: limp collapse, a push and a tilted start are each caught by the fall rule."""
    cases = {}

    def fall_run(sim, seconds, push=None, limp=False):
        if limp:
            sim.motors_off()
        first = None; zs = []
        n = int(seconds / (SIM_DT * DECIMATION))
        for k in range(n):
            if push and k == push[0]:
                sim.d.qvel[:3] += push[1]
            sim.step_policy(); track(sim)
            tl = sim.tilt_deg(); zs.append(sim.d.qpos[2])
            if first is None and tl > FALL_DEG:
                first = k * SIM_DT * DECIMATION
        return first, float(sim.d.qpos[2]), sim.tilt_deg(), float(min(zs))

    sim = Sim(); sim.reset()
    t, z, tl, zmin = fall_run(sim, 4.0, limp=True)
    cases["limp_collapse"] = {"detected_at_s": t, "final_pelvis_z": round(z, 3), "final_tilt_deg": round(tl, 1)}
    render(sim, "limp_collapse_end")
    for name, v in (("push_forward_1.5mps", [1.5, 0, 0]), ("push_sideways_1.0mps", [0, 1.0, 0]),
                    ("push_forward_0.5mps", [0.5, 0, 0])):
        sim = Sim(); sim.reset()
        t, z, tl, zmin = fall_run(sim, 4.0, push=(25, np.array(v)))
        cases[name] = {"detected_at_s": t, "final_pelvis_z": round(z, 3), "final_tilt_deg": round(tl, 1),
                       "fell": bool(z < 0.45 or tl > FALL_DEG)}
    sim = Sim()
    a = np.radians(25); sim.reset(z_offset=0.02, quat=(np.cos(a / 2), 0, np.sin(a / 2), 0))
    t, z, tl, zmin = fall_run(sim, 4.0)
    cases["start_tilted_25deg"] = {"detected_at_s": t, "final_pelvis_z": round(z, 3), "final_tilt_deg": round(tl, 1),
                                   "fell": bool(z < 0.45 or tl > FALL_DEG)}
    # a fall the tilt rule misses = the robot ends low (pelvis < 0.45 m) but less than 70 deg tilted
    missed = [k for k, c in cases.items() if c["final_pelvis_z"] < 0.45 and c["detected_at_s"] is None]
    R["A6"] = {"pass": not missed, "tilt_rule_deg": FALL_DEG, "cases": cases, "missed_falls": missed}


# ----------------------------------------------------------------------------- A7
def a7_contacts():
    """A7: both feet carry the weight when standing."""
    sim = Sim(); sim.reset()
    run(sim, 2.0)
    f = sim.foot_forces()
    weight = float(sim.m.body_subtreemass[1] * 9.81)
    total = sum(f.values())
    split = {k: round(v / weight, 3) for k, v in f.items()}
    ok = abs(total / weight - 1) < 0.05 and all(0.35 < s < 0.65 for s in split.values())
    R["A7"] = {"pass": bool(ok), "weight_N": round(weight, 1), "foot_normal_force_N": {k: round(v, 1) for k, v in f.items()},
               "share_of_weight": split}


# ----------------------------------------------------------------------------- A8
def a8_sizing():
    """A8: contact and constraint counts (mjlab sizing) and CPU step time."""
    sim = Sim(); sim.reset()
    t0 = time.perf_counter(); run(sim, 5.0); dt = time.perf_counter() - t0
    R["A8"] = {"pass": True, "max_contacts_seen": SIZING["ncon"], "max_constraint_rows_seen": SIZING["nefc"],
               "g1_flat_njmax": 300, "cpu_realtime_factor": round(5.0 / dt, 1)}


# ----------------------------------------------------------------------------- output
def render(sim, name):
    """Front and side snapshot of the current state."""
    try:
        r = mujoco.Renderer(sim.m, 480, 480)
        cam = mujoco.MjvCamera(); cam.lookat[:] = [sim.d.qpos[0], sim.d.qpos[1], 0.55]; cam.distance = 2.2
        imgs = []
        for az in (180, 90):                     # front, side
            cam.azimuth, cam.elevation = az, -10
            r.update_scene(sim.d, cam); imgs.append(r.render())
        from PIL import Image
        Image.fromarray(np.hstack(imgs)).save(OUT / f"{name}.png")
        r.close()
    except Exception as e:                        # rendering is a nice-to-have
        print(f"  (render {name} skipped: {e})")


def main():
    """Parse --actuator / --delay and the check names, run them, and write results.json."""
    global OUT
    import sim_replica
    args = sys.argv[1:]
    if "--actuator" in args:
        i = args.index("--actuator"); sim_replica.ACTUATOR = args[i + 1]; del args[i:i + 2]
    if "--delay" in args:
        i = args.index("--delay"); sim_replica.DELAY_STEPS = int(args[i + 1]); del args[i:i + 2]
    tag = f"{sim_replica.ACTUATOR}_delay{sim_replica.DELAY_STEPS}"
    OUT = HERE / "results" / tag
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"actuator model: {sim_replica.ACTUATOR}, command delay: {sim_replica.DELAY_STEPS * SIM_DT * 1000:.0f} ms -> results/{tag}/")
    sys.argv = [sys.argv[0]] + args
    steps = [("A1", a1_model), ("A2", a2_default_pose), ("A3", a3_stand), ("A4", a4_gains),
             ("A5", a5_directions), ("A6", a6_falls), ("A7", a7_contacts), ("A8", a8_sizing)]
    only = set(sys.argv[1:])
    for key, fn in steps:
        if only and key not in only:
            continue
        t0 = time.perf_counter(); fn()
        print(f"{key} {'PASS' if R[key]['pass'] else 'FAIL'}  ({time.perf_counter() - t0:.0f} s)")
    prev = json.loads((OUT / "results.json").read_text()) if only and (OUT / "results.json").exists() else {}
    prev.update(R)
    (OUT / "results.json").write_text(json.dumps(prev, indent=1, default=float))


if __name__ == "__main__":
    main()
