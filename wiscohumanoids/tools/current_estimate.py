"""Wisco V2 walking current-draw estimate (sim -> motor currents -> battery current), with a hand-off report.

Runs the trained walking policy (models/locomotion/wisco_velocity_v2) in plain MuJoCo, empty flat world, through a
set of tests (standing, walking speeds, turning, sidestep, arms moving, hand payload, pushes, stop/start). Every 5 ms
physics step it takes each joint's applied torque and speed, maps them to the motor that drives the joint (ankles and
waist through their 2-motor linkages), and turns them into motor phase current and electrical power with a model built
only from the MyActuator datasheets (MOTORS below). Writes
    wiscohumanoids/hardware/current_draw/REPORT.md, REPORT.pdf (same content, graphs embedded) + plots (*.png)
    + per-test CSV summaries

    .venv-mjlab/bin/python wiscohumanoids/tools/current_estimate.py            (about 3-5 min on the Mac CPU)
    .venv-mjlab/bin/python wiscohumanoids/tools/current_estimate.py --bus-v 48 --regen 0.0 --electronics-w 60
"""
from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WISCO = ROOT / "wiscohumanoids"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(WISCO / "training" / "model"))
sys.path.insert(0, str(WISCO / "robot"))
from sim2sim_play import PREFIX, Policy, Robot  # noqa: E402

SCENE = ROOT / "assets/robots/wisco/mjlab_1.6.0/wisco_velocity_flat.xml"
POLICY = ROOT / "models/locomotion/wisco_velocity_v2"
OUT = WISCO / "hardware" / "current_draw"

# --------------------------------------------------------------------------------------------- motor data
# Every number: MyActuator parameter sheets in wiscohumanoids/training/model/motor_datasheets/ (read 2026-11-12).
# torque_current: (output torque N*m, phase current A rms) points from the sheet: rated, peak and the stall table.
# Current between points is interpolated linearly (through 0, 0); this captures the torque constant dropping at high
# torque (magnetic saturation), which a single Kt would miss.
MOTORS = {
    "X8-150": dict(model="RMD-X8-P20-150-E", gear=19.612, v_rated=48, v_range=(20, 55), i_noload=1.6, rpm_noload=158,
                   rpm_rated=127, t_rated=43, t_peak=120, t_stall=150, kt_sheet=2.97, r_line=0.18, l_line_mh=0.33,
                   ke_v_krpm=13.74, eff_sheet=0.75, mass=1.33,
                   torque_current=[(43, 17.6), (150, 67.18)]),
    "X6-60": dict(model="RMD-X6-P20-60-E", gear=19.612, v_rated=48, v_range=(20, 55), i_noload=0.9, rpm_noload=176,
                  rpm_rated=153, t_rated=20, t_peak=60, t_stall=60, kt_sheet=2.1, r_line=0.41, l_line_mh=0.51,
                  ke_v_krpm=16, eff_sheet=0.727, mass=0.82,
                  torque_current=[(20, 9.5), (30, 12.7), (40, 17.7), (50, 22.6), (60, 28.3)]),
    "X4-36": dict(model="RMD-X4-P36-36-E", gear=36, v_rated=24, v_range=(20, 55), i_noload=0.9, rpm_noload=111,
                  rpm_rated=83, t_rated=10.5, t_peak=34, t_stall=34.5, kt_sheet=1.9, r_line=0.39, l_line_mh=0.07,
                  ke_v_krpm=6, eff_sheet=0.631, mass=0.36,
                  torque_current=[(10.5, 6.1), (17.25, 9.2), (23, 12.7), (28.75, 16.3), (34.5, 21.2)]),
    "X4-10": dict(model="RMD-X4-P12.5-10-E", gear=12.5, v_rated=24, v_range=(20, 55), i_noload=1.0, rpm_noload=317,
                  rpm_rated=238, t_rated=4, t_peak=10, t_stall=8, kt_sheet=0.8, r_line=0.32, l_line_mh=0.14,
                  ke_v_krpm=6, eff_sheet=0.695, mass=0.33,
                  torque_current=[(4, 7.8), (6, 9.1), (7.2, 11.0), (8, 12.1), (10, 19.5)]),
}
# X8 sheet load test (48 V, ~100-126 rpm): torque N*m, speed rpm, output W, bus V, bus (input) A, input W. Used to
# check the electrical model below against a measurement.
X8_LOAD_TEST = [(0.8, 124, 10.16, 47.87, 1.875, 89.78), (1.4, 126, 18.29, 47.85, 2.637, 126.20),
                (4.9, 126, 64.01, 47.82, 3.566, 170.53), (9.5, 125, 124.41, 47.77, 4.861, 232.20),
                (15.1, 124, 196.69, 47.71, 6.635, 316.42), (21.7, 122, 277.00, 47.64, 8.821, 419.96),
                (29.2, 123, 376.19, 47.55, 11.378, 540.68), (37.4, 125, 489.87, 47.45, 13.697, 649.64),
                (45.9, 124, 596.51, 47.34, 17.467, 826.18), (54.9, 123, 707.74, 47.21, 20.403, 962.54),
                (64.8, 118, 801.31, 47.08, 23.271, 1095.12), (74.9, 113, 886.85, 46.93, 28.440, 1333.83),
                (85.2, 106, 946.21, 46.80, 31.721, 1497.06), (95.8, 100, 1003.58, 46.68, 35.273, 1589.47)]

# joint -> motor (wiscohumanoids/training/model/robot_numbers.yaml joint_to_motor). Ankles: 2x X4-36 per ankle through
# the rod linkage; waist roll + pitch: 2x X6-60 through the waist linkage. head_yaw / head_pitch (X4-10) are welded in
# the sim model: counted as idle drivers only.
SINGLE = {
    "X8-150": ["left_hip_pitch", "left_hip_roll", "left_knee", "right_hip_pitch", "right_hip_roll", "right_knee",
               "torso_yaw"],
    "X6-60": ["left_hip_yaw", "right_hip_yaw", "left_shoulder_pitch", "left_shoulder_roll", "right_shoulder_pitch",
              "right_shoulder_roll"],
    "X4-36": ["left_shoulder_yaw", "left_elbow", "left_forearm_yaw", "right_shoulder_yaw", "right_elbow",
              "right_forearm_yaw"],
    "X4-10": ["left_wrist_pitch", "left_wrist_roll", "right_wrist_pitch", "right_wrist_roll"],
}
IDLE_ONLY = [("head_yaw", "X4-10"), ("head_pitch", "X4-10")]

# --------------------------------------------------------------------------------------------- tests
# (name, description, [(seconds, (vx, vy, wz))], options). The first 1.5 s of every test (settling) is not counted.
TESTS = [
    ("stand", "standing still, arms at default", [(12, (0, 0, 0))], {}),
    ("walk_0.25", "walk forward 0.25 m/s", [(12, (0.25, 0, 0))], {}),
    ("walk_0.50", "walk forward 0.5 m/s (typical indoor)", [(12, (0.5, 0, 0))], {}),
    ("walk_0.75", "walk forward 0.75 m/s", [(12, (0.75, 0, 0))], {}),
    ("walk_1.00", "walk forward 1.0 m/s", [(12, (1.0, 0, 0))], {}),
    ("backward_0.4", "walk backward 0.4 m/s", [(12, (-0.4, 0, 0))], {}),
    ("sidestep_0.3", "sidestep 0.3 m/s", [(12, (0, 0.3, 0))], {}),
    ("turn_0.8", "turn in place 0.8 rad/s", [(12, (0, 0, 0.8))], {}),
    ("walk_turn", "walk 0.5 m/s while turning 0.4 rad/s", [(12, (0.5, 0, 0.4))], {}),
    ("walk_0.50_arms", "walk 0.5 m/s with arms moving to random targets", [(12, (0.5, 0, 0))], {"arms": "random"}),
    ("walk_0.50_payload", "walk 0.5 m/s with 1 kg in each hand", [(12, (0.5, 0, 0))], {"payload_kg": 1.0}),
    ("stand_pushes", "standing, 0.5 m/s shove every 2 s (random direction)", [(14, (0, 0, 0))], {"push": 0.5}),
    ("walk_pushes", "walk 0.5 m/s, 0.5 m/s shove every 2 s", [(14, (0.5, 0, 0))], {"push": 0.5}),
    ("stop_start", "kitchen-like: walk 0.4 / stop / turn / stop, repeated",
     [(1.5, (0, 0, 0)), (3, (0.4, 0, 0)), (2, (0, 0, 0)), (2, (0, 0, 0.8)), (2, (0, 0, 0)), (3, (0.4, 0, 0)),
      (2, (0, 0, 0))], {}),
]
SETTLE_S = 1.5
BUS_V_FOR_PLOTS = [48.0]


def current_from_torque(motor, tau):
    """Phase current (A rms) for |output torque|, piecewise linear through (0, 0) and the datasheet points."""
    pts = [(0.0, 0.0)] + sorted(MOTORS[motor]["torque_current"])
    t, i = np.array([p[0] for p in pts]), np.maximum.accumulate(np.array([p[1] for p in pts]))
    tau = np.abs(tau)
    slope_end = (i[-1] - i[-2]) / (t[-1] - t[-2])
    return np.where(tau <= t[-1], np.interp(tau, t, i), i[-1] + (tau - t[-1]) * slope_end)


# Calibration from the X8 load test (least squares over its 14 rows, 2026-11-12): spinning loss is 1.55x the sheet's
# V_rated * I_noload at the same speed, and load-dependent gearbox/bearing loss is 5.4 % of |tau * omega|. Applied to
# all four motor types (same MyActuator planetary family); only the X8 sheet has a load test.
NOLOAD_FACTOR = 1.55
GEAR_LOSS = 0.054


def motor_power(motor, tau, w, idle_w):
    """Electrical input power (W) of one actuator from output torque (N*m) and output speed (rad/s).
       P = tau*w (mechanical out) + copper loss 1.5 * R_line * I_rms^2 (Y winding: 3 phases x R_line/2)
           + gearbox/bearing loss GEAR_LOSS * |tau*w|
           + speed-dependent spinning loss NOLOAD_FACTOR * V_rated * I_noload * |w| / w_noload
           + driver idle power."""
    m = MOTORS[motor]
    i = current_from_torque(motor, tau)
    w_nl = m["rpm_noload"] * 2 * np.pi / 60
    p_nl = NOLOAD_FACTOR * m["v_rated"] * m["i_noload"] * np.minimum(np.abs(w) / w_nl, 1.0)
    return tau * w + GEAR_LOSS * np.abs(tau * w) + 1.5 * m["r_line"] * i ** 2 + p_nl + idle_w, i


def check_against_x8_load_test(idle_w):
    rows = []
    for T, rpm, pout, v, i_in, pin in X8_LOAD_TEST:
        w = rpm * 2 * np.pi / 60
        p, _ = motor_power("X8-150", np.array(T), np.array(w), idle_w)
        rows.append((T, rpm, pin, float(p), 100 * (float(p) - pin) / pin, i_in, float(p) / v))
    return rows


# --------------------------------------------------------------------------------------------- simulation
class Mapper:
    """Joint torques/speeds (MuJoCo) -> per-motor torques/speeds."""

    def __init__(self, m):
        self.m = m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + n)
        self.qadr = {n: m.jnt_qposadr[jid(n)] for n in self._all_joints()}
        self.vadr = {n: m.jnt_dofadr[jid(n)] for n in self._all_joints()}
        act = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
        self.act = {n: act[jid(n)] for n in self._all_joints()}
        import ankle_motor
        self.ankle = ankle_motor
        from linkage import Mechanism
        self.Jw = np.array(Mechanism("waist").jacobian((0.0, 0.0)))     # rows = motors, cols = (roll, pitch)
        self.names = ([(j, mt) for mt, js in SINGLE.items() for j in js]
                      + [(f"{s}_ankle_motor_{k}", "X4-36") for s in ("left", "right") for k in (1, 2)]
                      + [(f"waist_motor_{k}", "X6-60") for k in (1, 2)])

    @staticmethod
    def _all_joints():
        return ([j for js in SINGLE.values() for j in js] + ["torso_roll", "torso_pitch"]
                + [f"{s}_ankle_{a}" for s in ("left", "right") for a in ("pitch", "roll")])

    def __call__(self, d):
        tq = lambda n: d.actuator_force[self.act[n]]
        q = lambda n: d.qpos[self.qadr[n]]
        v = lambda n: d.qvel[self.vadr[n]]
        tau, w = [], []
        for mt, js in SINGLE.items():
            for j in js:
                tau.append(tq(j)); w.append(v(j))
        for s in ("left", "right"):
            p, r = q(f"{s}_ankle_pitch"), q(f"{s}_ankle_roll")
            J = self.ankle.jacobian(s, p, r)                      # rows = motors, cols = (pitch, roll)
            tm = np.linalg.solve(J.T, np.array([tq(f"{s}_ankle_pitch"), tq(f"{s}_ankle_roll")]))
            wm = J @ np.array([v(f"{s}_ankle_pitch"), v(f"{s}_ankle_roll")])
            tau += list(tm); w += list(wm)
        J = self.Jw
        tm = np.linalg.solve(J.T, np.array([tq("torso_roll"), tq("torso_pitch")]))
        wm = J @ np.array([v("torso_roll"), v("torso_pitch")])
        tau += list(tm); w += list(wm)
        return np.array(tau), np.array(w)


def run_test(name, script, opts, seed, idle_w):
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    if opts.get("payload_kg"):
        for h in ("left_hand", "right_hand"):
            m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + h)] += opts["payload_kg"]
    pol = Policy.load(POLICY)
    robot, d = Robot(m, pol, arms=opts.get("arms", "default")), None
    d = mujoco.MjData(m)
    robot.reset(d)
    mp = Mapper(m)
    types = [t for _, t in mp.names]
    period = pol.sim_dt * pol.decimation
    T, W, t_log, x_log, fell = [], [], [], [], False
    t, next_push = 0.0, SETTLE_S + 2.0
    seq = [(SETTLE_S, (0, 0, 0))] + list(script)
    for seconds, cmd in seq:
        for _ in range(int(round(seconds / period))):
            obs = robot.observation(d, np.array(cmd, float))
            a = pol(obs)
            robot.last_action = a.astype(float)
            d.ctrl[robot.act_idx] = pol.c.action_targets(a)
            if len(robot.held_act_idx):
                d.ctrl[robot.held_act_idx] = robot.held_targets(period)
            if opts.get("push") and t >= next_push:
                ang = rng.uniform(0, 2 * np.pi)
                d.qvel[0] += opts["push"] * np.cos(ang); d.qvel[1] += opts["push"] * np.sin(ang)
                next_push += 2.0
            for _ in range(pol.decimation):
                mujoco.mj_step(m, d)
                t += pol.sim_dt
                if t > SETTLE_S:
                    tau, w = mp(d)
                    T.append(tau); W.append(w); t_log.append(t); x_log.append(d.xpos[robot.pelvis][:2].copy())
            if robot.fallen(d):
                fell = True
                break
        if fell:
            break
    T, W = np.array(T), np.array(W)
    P, I = np.zeros_like(T), np.zeros_like(T)
    for k, mt in enumerate(types):
        P[:, k], I[:, k] = motor_power(mt, T[:, k], W[:, k], idle_w)
    x = np.array(x_log)
    dist = float(np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1))) if len(x) > 1 else 0.0
    return dict(name=name, names=[n for n, _ in mp.names], types=types, T=T, W=W, P=P, I=I,
                t=np.array(t_log), dt=pol.sim_dt, dist=dist, fell=fell, mass=float(sum(m.body_mass)))


def window_max(x, n):
    if n <= 1 or len(x) < n:
        return float(np.max(x)) if len(x) else 0.0
    c = np.convolve(x, np.ones(n) / n, mode="valid")
    return float(np.max(c))


def summarize(r, args):
    dt = r["dt"]
    n10, n100, n1000 = int(0.010 / dt), int(0.100 / dt), int(1.0 / dt)
    P = r["P"]
    # bus power: motors that push power back (P < 0, braking) return regen_frac of it to the bus
    P_motor_bus = np.where(P >= 0, P, args.regen * P)
    n_idle = len(IDLE_ONLY)
    bus_p = P_motor_bus.sum(axis=1) + n_idle * args.idle_w + args.electronics_w
    bus_i = bus_p / args.bus_v
    dur = len(bus_i) * dt
    s = dict(name=r["name"], fell=r["fell"], duration=dur, dist=r["dist"], speed=r["dist"] / dur if dur else 0,
             bus_i_mean=float(bus_i.mean()), bus_i_rms=float(np.sqrt((bus_i ** 2).mean())),
             bus_i_peak=float(bus_i.max()), bus_i_peak10=window_max(bus_i, n10), bus_i_peak100=window_max(bus_i, n100),
             bus_i_peak1s=window_max(bus_i, n1000), bus_p_mean=float(bus_p.mean()), bus_p_peak100=window_max(bus_p, n100),
             wh_per_km=(float(bus_p.mean()) * dur / 3600) / (r["dist"] / 1000) if r["dist"] > 0.5 else None,
             cot=float(bus_p.mean()) / (r["mass"] * 9.81 * (r["dist"] / dur)) if r["dist"] > 0.5 else None,
             bus_i=bus_i, motors=[])
    for k, (n, mt) in enumerate(zip(r["names"], r["types"])):
        I, T, W = r["I"][:, k], np.abs(r["T"][:, k]), np.abs(r["W"][:, k])
        md = MOTORS[mt]
        s["motors"].append(dict(
            name=n, type=mt, i_rms=float(np.sqrt((I ** 2).mean())), i_peak=float(I.max()), i_peak10=window_max(I, n10),
            t_rms=float(np.sqrt((T ** 2).mean())), t_peak=float(T.max()), rpm_peak=float(W.max() * 60 / 2 / np.pi),
            p_mean=float(r["P"][:, k].mean()), pct_above_rated=float(100 * np.mean(T > md["t_rated"])),
            pct_of_peak_torque=float(100 * T.max() / md["t_peak"]), pct_rated_current=float(100 * np.sqrt((I ** 2).mean())
                                                                                           / md["torque_current"][0][1])))
    return s


# --------------------------------------------------------------------------------------------- report
# chart style: reference palette (validated: lightness band, chroma, CVD separation), recessive grid, thin marks;
# identity never by colour alone (legend or direct labels on every multi-series chart)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LEG_GROUPS = [("Hips + knees (X8)", ("hip_pitch", "hip_roll", "knee")), ("Ankles (2x X4-36 each)", ("ankle_motor",)),
              ("Waist (yaw X8, roll/pitch 2x X6)", ("torso_yaw", "waist_motor")), ("Hip yaw (X6)", ("hip_yaw",)),
              ("Arms (X6 / X4-36)", ("shoulder", "elbow", "forearm")), ("Wrists (X4-10)", ("wrist",))]


def _style(plt):
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "axes.axisbelow": True})


def plots(results, summaries, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    files = {}
    S = {s["name"]: s for s in summaries if not s["fell"]}
    R = {r["name"]: r for r in results}

    def save(fig, name, key):
        f = out / name; fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig); files[key] = f.name

    # A. battery current vs walking speed (mean, 100 ms peak, 10 ms peak)
    ks = [k for k in ("stand", "walk_0.25", "walk_0.50", "walk_0.75", "walk_1.00") if k in S]
    v = [S[k]["speed"] for k in ks]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for (key, lab), c in zip((("bus_i_mean", "mean"), ("bus_i_peak100", "peak, 100 ms avg"), ("bus_i_peak10", "peak, 10 ms avg")), C):
        y = [S[k][key] for k in ks]
        ax.plot(v, y, color=c, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=lab)
        ax.annotate(f"{y[-1]:.1f} A", (v[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=8)
    ax.set_xlabel("walking speed achieved (m/s)"); ax.set_ylabel(f"battery current (A) at {BUS_V_FOR_PLOTS[0]:.0f} V")
    ax.set_title("Battery current vs walking speed"); ax.set_ylim(0, None); ax.set_xlim(-0.05, max(v) + 0.2); ax.legend(loc="upper left")
    save(fig, "speed_vs_current.png", "speed")

    # B. mean electrical power, every test (single series, sorted)
    order = sorted(S.values(), key=lambda s: s["bus_p_mean"])
    fig, ax = plt.subplots(figsize=(7, 4.6))
    y = np.arange(len(order))
    ax.barh(y, [s["bus_p_mean"] for s in order], height=0.62, color=C[0])
    for i, s in enumerate(order):
        ax.text(s["bus_p_mean"] + 6, i, f"{s['bus_p_mean']:.0f} W  ({s['bus_i_mean']:.1f} A)", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([s["name"] for s in order]); ax.set_xlabel("mean electrical power (W), motors only")
    ax.set_title("Average power by test"); ax.grid(axis="y", visible=False); ax.set_xlim(0, max(s["bus_p_mean"] for s in order) * 1.35)
    save(fig, "power_by_test.png", "power")

    # C. where the power goes: joint groups, for standing / walking speeds / shoves
    ks = [k for k in ("stand", "walk_0.25", "walk_0.50", "walk_0.75", "walk_1.00", "walk_pushes") if k in S]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    left = np.zeros(len(ks))
    for (lab, keys), c in zip(LEG_GROUPS, C):
        vals = np.array([sum(mm["p_mean"] for mm in S[k]["motors"] if any(q in mm["name"] for q in keys)) for k in ks])
        ax.barh(np.arange(len(ks)), vals, left=left, height=0.6, color=c, edgecolor=SURF, linewidth=1.5, label=lab)
        left += vals
    for i, tot in enumerate(left):
        ax.text(tot + 5, i, f"{tot:.0f} W", va="center", fontsize=8, color=INK2)
    ax.set_yticks(np.arange(len(ks))); ax.set_yticklabels(ks); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean electrical power (W)"); ax.set_title("Where the power goes, by joint group")
    ax.set_xlim(0, left.max() * 1.18); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
    save(fig, "power_by_group.png", "groups")

    # D. heat map: RMS current as % of rated, leg + waist motors x tests (sequential, one hue)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#86b6ef", "#2a78d6", "#1c5cab", "#0d2f5c"])
    tests = list(S.keys())
    rows = [i for i, mm in enumerate(summaries[0]["motors"]) if not any(q in mm["name"] for q in ("shoulder", "elbow", "forearm", "wrist"))]
    M = np.array([[S[t]["motors"][i]["pct_rated_current"] for t in tests] for i in rows])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=150, aspect="auto")
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            val = M[a, b]
            ax.text(b, a, f"{val:.0f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if val > 75 else INK, fontweight="bold" if val >= 100 else "normal")
    ax.set_xticks(range(len(tests))); ax.set_xticklabels(tests, rotation=40, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{summaries[0]['motors'][i]['name']} ({summaries[0]['motors'][i]['type']})" for i in rows])
    ax.grid(False); ax.set_title("RMS phase current as % of the motor's rated current (bold = at or above rated)")
    save(fig, "heatmap_pct_rated.png", "heat")

    # E. battery current traces (single series each, mean as a reference line)
    keys = [k for k in ("stand", "walk_0.50", "walk_1.00", "turn_0.8", "walk_pushes", "stop_start") if k in S]
    fig, axs = plt.subplots(len(keys), 1, figsize=(9, 1.7 * len(keys)), sharex=True)
    ymax = max(S[k]["bus_i"].max() for k in keys) * 1.05
    for ax, k in zip(axs, keys):
        ax.plot(R[k]["t"] - R[k]["t"][0], S[k]["bus_i"], lw=0.8, color=C[0])
        ax.axhline(S[k]["bus_i_mean"], color=INK2, ls="--", lw=0.8)
        ax.text(0.995, 0.92, f"{k}: mean {S[k]['bus_i_mean']:.1f} A, 10 ms peak {S[k]['bus_i_peak10']:.1f} A",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
        ax.set_ylim(0, ymax); ax.set_ylabel("A")
    axs[-1].set_xlabel("time (s)"); axs[0].set_title("Battery current over time (dashed = mean)")
    save(fig, "bus_current_traces.png", "traces")

    # F. per-motor RMS / peak current, walk 0.5 and walk 1.0
    for k in ("walk_0.50", "walk_1.00"):
        s = S[k]
        names = [mm["name"] for mm in s["motors"]]
        fig, ax = plt.subplots(figsize=(11, 3.8))
        x = np.arange(len(names))
        ax.bar(x - 0.2, [mm["i_rms"] for mm in s["motors"]], 0.38, color=C[0], label="RMS (heating)")
        ax.bar(x + 0.2, [mm["i_peak10"] for mm in s["motors"]], 0.38, color=C[1], label="peak, 10 ms avg (driver sizing)")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=70, fontsize=7); ax.set_ylabel("phase current (A rms)")
        ax.set_title(f"{k}: phase current per motor"); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
        save(fig, f"motor_currents_{k}.png", f"motors_{k}")

    # G. torque-speed operating points vs datasheet limits, leg/waist motors
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, mt in zip(axs, ("X8-150", "X6-60", "X4-36")):
        for (k, c) in (("walk_1.00", C[0]), ("walk_pushes", C[1])):
            r = R[k]
            cols = [i for i, t in enumerate(r["types"]) if t == mt and not any(a in r["names"][i] for a in ("shoulder", "elbow", "forearm"))]
            if cols:
                ax.scatter(np.abs(r["W"][::4, cols]).ravel() * 60 / 2 / np.pi, np.abs(r["T"][::4, cols]).ravel(), s=2,
                           alpha=0.3, color=c, label=k, rasterized=True)
        md = MOTORS[mt]
        ax.axhline(md["t_rated"], color=INK2, ls="--", lw=1); ax.axhline(md["t_peak"], color=INK, ls="-", lw=1)
        ax.axvline(md["rpm_rated"], color=INK2, ls="--", lw=1)
        ax.text(1, md["t_peak"], f" peak {md['t_peak']} N*m", va="bottom", fontsize=7, color=INK)
        ax.text(1, md["t_rated"], f" rated {md['t_rated']} N*m", va="bottom", fontsize=7, color=INK2)
        ax.text(md["rpm_rated"], ax.get_ylim()[1] if False else md["t_peak"] * 1.08, f"rated {md['rpm_rated']} rpm ", ha="right",
                fontsize=7, color=INK2)
        ax.set_ylim(0, md["t_peak"] * 1.35)
        ax.set_title(f"{mt} (legs / waist)"); ax.set_xlabel("|speed| at output (rpm)"); ax.set_ylabel("|torque| (N*m)")
        ax.legend(loc="upper right", markerscale=5, fontsize=7)
    save(fig, "torque_speed_legs.png", "ts")
    return files


def write_report(results, summaries, val, files, args, out):
    S = {s["name"]: s for s in summaries}
    fallen = [s["name"] for s in summaries if s["fell"]]
    summaries = [s for s in summaries if not s["fell"]]     # worst cases: only tests the robot completed
    L = []
    w = L.append
    w("# Wisco V2: estimated motor and battery current while walking (simulation)\n")
    w(f"Generated {datetime.date.today()} by `wiscohumanoids/tools/current_estimate.py` "
      f"(policy `{POLICY.relative_to(ROOT)}`, plain MuJoCo, flat empty floor). Re-run the script to regenerate.\n")
    if fallen:
        w(f"**Fell (not used for worst cases):** {', '.join(fallen)}. The robot fell in the plain-MuJoCo model at that "
          "command; its current spikes come from the fall, not from walking. Treat it as beyond the usable range of "
          "this policy in this model (training used extra motor limits; see section 6).\n")
    w("**Speed range.** All tests stay inside what this walking policy (checkpoint J3h model_4650) was trained on: "
      "forward up to **1.0 m/s**, backward 0.4 m/s, sidestep, and turning in place. 1.0 m/s is the design walking speed "
      "of this policy; faster walking would need a policy trained for it (and the ankle findings below get worse).\n")
    w("**Read this first.** These are simulation estimates built from datasheet numbers, before any bench data. "
      "Use the averages for battery/energy planning with about +/-25 % uncertainty and the 10 ms / 100 ms peaks (plus "
      "margin) for driver, fuse, connector and BMS sizing. Section 6 lists what is assumed and how to firm it up.\n")
    walk = S["walk_0.50"]; fast = S["walk_1.00"]; st = S["stand"]
    worst = max(summaries, key=lambda s: s["bus_i_peak100"])
    w("## 1. Headline numbers\n")
    w(f"Bus: {args.bus_v:.0f} V, all actuators on one bus; motors only (+{args.electronics_w:.0f} W electronics), "
      f"regeneration counted at {args.regen * 100:.0f} %.\n")
    w("| | Battery current, mean | Peak (100 ms avg) | Peak (10 ms avg) | Electrical power, mean |")
    w("|---|---|---|---|---|")
    for k, lab in (("stand", "Standing"), ("walk_0.50", "Walking 0.5 m/s"), ("walk_1.00", "Walking 1.0 m/s"),
                                  ("walk_pushes", "Walking + shoves")):
        s = S[k]
        if s["fell"]:
            w(f"| {lab} | fell | | | |"); continue
        w(f"| {lab} | {s['bus_i_mean']:.1f} A | {s['bus_i_peak100']:.1f} A | {s['bus_i_peak10']:.1f} A | {s['bus_p_mean']:.0f} W |")
    w(f"\nHighest 100 ms battery current in any test: **{worst['bus_i_peak100']:.1f} A** ({worst['name']}); "
      f"highest 10 ms: **{max(s['bus_i_peak10'] for s in summaries):.1f} A**.")
    if walk["wh_per_km"]:
        w(f"Energy at 0.5 m/s: **{walk['wh_per_km']:.0f} Wh/km** (cost of transport {walk['cot']:.2f}); at 1.0 m/s: "
          f"**{fast['wh_per_km']:.0f} Wh/km** (CoT {fast['cot']:.2f}). Runtime per 1 kWh of usable pack: "
          f"{1000 / walk['bus_p_mean']:.1f} h walking at 0.5 m/s, {1000 / st['bus_p_mean']:.1f} h standing.\n")
    # ---- findings: where the sim asks more than the datasheet allows, and sizing numbers with margin
    w(f"![battery current vs speed]({files['speed']})\n")
    w(f"![where the power goes]({files['groups']})\n")
    w("## Findings for mech / electrical\n")
    n_m = len(summaries[0]["motors"])
    flags = []
    for k in range(n_m):
        rows = [(s["name"], s["motors"][k]) for s in summaries]
        mm0 = rows[0][1]; md = MOTORS[mm0["type"]]
        wt = max(rows, key=lambda x: x[1]["t_peak"]); ws = max(rows, key=lambda x: x[1]["rpm_peak"])
        wr = max(rows, key=lambda x: x[1]["i_rms"])
        if wt[1]["t_peak"] > md["t_peak"]:
            flags.append(f"- **{mm0['name']} ({mm0['type']}): torque above the motor's peak rating**: {wt[1]['t_peak']:.1f} N*m "
                         f"vs {md['t_peak']} N*m peak ({wt[0]}).")
        elif wt[1]["t_peak"] > 0.8 * md["t_peak"]:
            flags.append(f"- {mm0['name']} ({mm0['type']}): torque reaches {wt[1]['pct_of_peak_torque']:.0f} % of peak "
                         f"({wt[1]['t_peak']:.1f} N*m, {wt[0]}).")
        if ws[1]["rpm_peak"] > md["rpm_rated"]:
            flags.append(f"- **{mm0['name']} ({mm0['type']}): speed above rated**: {ws[1]['rpm_peak']:.0f} rpm vs "
                         f"{md['rpm_rated']} rated / {md['rpm_noload']} no-load ({ws[0]}); torque available drops there.")
        elif ws[1]["rpm_peak"] > 0.85 * md["rpm_rated"]:
            flags.append(f"- {mm0['name']} ({mm0['type']}): speed reaches {ws[1]['rpm_peak']:.0f} of {md['rpm_rated']} rpm rated ({ws[0]}).")
        if wr[1]["pct_rated_current"] > 60:
            flags.append(f"- {mm0['name']} ({mm0['type']}): RMS current {wr[1]['i_rms']:.1f} A = {wr[1]['pct_rated_current']:.0f} % "
                         f"of rated ({wr[0]}): watch heating in long runs.")
    w("**Motors near or past their datasheet limits** (worst case over the completed tests):\n")
    w("\n".join(flags) if flags else "- none")
    w("\nThe ankle limits are the known weak point (HARDWARE_ISSUES H3): training clamps each ankle motor to 34 N*m "
      "through the linkage, but this plain-MuJoCo model only limits the joint, so it shows how hard the policy would "
      "like to push the ankles. On the robot those motors will saturate; currents there are capped near the X4-36 "
      "peak phase current (21.5 A).\n")
    top_mean = max(s["bus_i_mean"] for s in summaries if s["name"] != "stand")
    top10 = max(s["bus_i_peak10"] for s in summaries); top100 = max(s["bus_i_peak100"] for s in summaries)
    w(f"**Sizing numbers with margin** (at {args.bus_v:.0f} V, motors only; add electronics):\n")
    w(f"- Continuous battery current: up to {top_mean:.1f} A mean in these tests -> plan for **{1.3 * top_mean:.0f} A** (+30 %).")
    w(f"- Short peaks: {top100:.1f} A over 100 ms, {top10:.1f} A over 10 ms -> plan for **{1.3 * top100:.0f} A / {1.3 * top10:.0f} A** (+30 %).")
    w(f"- Standing: {S['stand']['bus_i_mean']:.1f} A ({S['stand']['bus_p_mean']:.0f} W), mostly copper loss holding the knees and hips.\n")
    w("## 2. All tests (battery side)\n")
    w("| Test | What | Speed achieved | Mean A | RMS A | Peak 1 s | Peak 100 ms | Peak 10 ms | Mean W | Wh/km | Fell |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for (name, desc, _, _) in TESTS:
        s = S[name]
        wk = f"{s['wh_per_km']:.0f}" if s["wh_per_km"] else "-"
        w(f"| {name} | {desc} | {s['speed']:.2f} m/s | {s['bus_i_mean']:.1f} | {s['bus_i_rms']:.1f} | {s['bus_i_peak1s']:.1f} | "
          f"{s['bus_i_peak100']:.1f} | {s['bus_i_peak10']:.1f} | {s['bus_p_mean']:.0f} | {wk} | {'YES' if s['fell'] else 'no'} |")
    w(f"\n![power by test]({files['power']})\n")
    w(f"![battery current]({files['traces']})\n")
    w("## 3. Per motor (phase current, A rms)\n")
    w("RMS = what heats the motor (compare with the rated phase current). Peak = what the driver must deliver "
      "(compare with the peak phase current). Worst case over all tests, then walking 0.5 m/s.\n")
    w("| Motor | Type | Rated / peak phase A | Worst RMS A (test) | Worst peak 10 ms A (test) | Worst peak torque N*m (% of peak) | Max speed rpm (rated) | Walk 0.5: RMS A / peak A |")
    w("|---|---|---|---|---|---|---|---|")
    n_m = len(summaries[0]["motors"])
    for k in range(n_m):
        rows = [(s["name"], s["motors"][k]) for s in summaries]
        mm0 = rows[0][1]; md = MOTORS[mm0["type"]]
        wr = max(rows, key=lambda x: x[1]["i_rms"]); wp = max(rows, key=lambda x: x[1]["i_peak10"])
        wt = max(rows, key=lambda x: x[1]["t_peak"]); ws = max(rows, key=lambda x: x[1]["rpm_peak"])
        wm = S["walk_0.50"]["motors"][k]
        peak_i = {"X8-150": 67.18, "X6-60": 29.1, "X4-36": 21.5, "X4-10": 19.5}[mm0["type"]]
        w(f"| {mm0['name']} | {mm0['type']} | {md['torque_current'][0][1]} / {peak_i} | {wr[1]['i_rms']:.1f} ({wr[0]}) | "
          f"{wp[1]['i_peak10']:.1f} ({wp[0]}) | {wt[1]['t_peak']:.1f} ({wt[1]['pct_of_peak_torque']:.0f} %) | "
          f"{ws[1]['rpm_peak']:.0f} ({md['rpm_rated']}) | {wm['i_rms']:.1f} / {wm['i_peak10']:.1f} |")
    for k in ("walk_0.50", "walk_1.00"):
        w(f"\n![per motor {k}]({files[f'motors_{k}']})")
    w(f"\n![torque-speed]({files['ts']})\n")
    w(f"![percent of rated current]({files['heat']})\n")
    w("Head yaw / head pitch (X4-10) are fixed in the sim model; they are counted as idle drivers only.\n")
    w("## 4. How the numbers are made\n")
    w("1. **Simulation.** The trained walking policy runs at 50 Hz in MuJoCo (5 ms physics steps) on a flat floor. "
      "Legs and waist follow the policy; arms are held by their position controllers (default pose, or random targets "
      "in the `arms` test). Each joint's applied torque comes from the simulated PD loop with the robot's gains, "
      "limited to each joint's torque limit.")
    w("2. **Joint to motor.** Single-motor joints: motor torque/speed = joint torque/speed. Ankles (2x X4-36 each) "
      "and waist roll+pitch (2x X6-60): motor torques solve J^T tau_motor = tau_joint, motor speeds = J q_dot, with J the "
      "linkage Jacobian (ankles: at the current pose, from `ankle_linkage.json`; waist: at the zero pose).")
    w("3. **Torque to phase current.** Piecewise-linear through (0, 0) and the datasheet's torque/current points "
      "(rated, stall table, peak), so the torque constant falls at high torque as on the sheets.")
    w("4. **Electrical power per actuator** = tau x omega (mechanical out; negative = braking) "
      f"+ gearbox/bearing loss {GEAR_LOSS * 100:.1f} % of |tau x omega| + copper loss 1.5 x R_line x I_rms^2 (Y winding) "
      f"+ spinning loss {NOLOAD_FACTOR:.2f} x V_rated x I_noload x speed / no-load speed + driver idle {args.idle_w:.1f} W. "
      "The two loss factors were fitted to the X8 load test below (the only measured efficiency table we have).")
    w(f"5. **Battery current** = (sum of actuator power, braking power returned at {args.regen * 100:.0f} %, + "
      f"{len(IDLE_ONLY)} idle head drivers + {args.electronics_w:.0f} W electronics) / {args.bus_v:.0f} V bus. "
      "Peaks are also given as 10 ms, 100 ms and 1 s moving averages (raw 5 ms values include foot-impact spikes).\n")
    w("### Check against a measurement (X8 load test from its datasheet, 48 V, about 100-126 rpm)\n")
    w("| Torque N*m | rpm | Measured input W | Model W | Error | Measured bus A | Model bus A |")
    w("|---|---|---|---|---|---|---|")
    for T, rpm, pin, p, err, iin, im in val:
        w(f"| {T} | {rpm} | {pin:.0f} | {p:.0f} | {err:+.0f} % | {iin:.2f} | {im:.2f} |")
    errs = [abs(v[4]) for v in val if v[0] >= 5]
    w(f"\nAbove 5 N*m the model is within {max(errs):.0f} % (median {np.median(errs):.0f} %) of the measured input power "
      "(the loss factors were fitted to this same table, so this shows the model form fits; it is not an independent "
      "test). Walking is mostly high torque at low speed, where the copper term dominates.\n")
    w("## 5. Every number used\n")
    w("### Motors (MyActuator parameter sheets, `wiscohumanoids/training/model/motor_datasheets/`)\n")
    w("| | X8-150 | X6-60 | X4-36 | X4-10 |")
    w("|---|---|---|---|---|")
    rowsp = [("Model", "model"), ("Gear ratio", "gear"), ("Rated voltage V", "v_rated"), ("Voltage range V", "v_range"),
             ("No-load current A", "i_noload"), ("No-load speed rpm (output)", "rpm_noload"), ("Rated speed rpm", "rpm_rated"),
             ("Rated torque N*m", "t_rated"), ("Peak torque N*m", "t_peak"), ("Stall torque N*m", "t_stall"),
             ("Module torque constant (sheet) N*m/A", "kt_sheet"), ("Line resistance ohm", "r_line"),
             ("Line inductance mH", "l_line_mh"), ("Back-EMF Vdc/krpm", "ke_v_krpm"), ("Efficiency (sheet)", "eff_sheet"),
             ("Mass kg", "mass"), ("Torque/current points (N*m, A rms)", "torque_current")]
    for lab, key in rowsp:
        w(f"| {lab} | " + " | ".join(str(MOTORS[mt][key]) for mt in ("X8-150", "X6-60", "X4-36", "X4-10")) + " |")
    w("\nCount on the robot: " + ", ".join(f"{mt} x{n}" for mt, n in (("X8-150", 7), ("X6-60", 8), ("X4-36", 10), ("X4-10", 6))) +
      " (31 actuators; 4 of the X4-36 drive the ankles in pairs, 2 of the X6-60 drive the waist in a pair).\n")
    w("### Joint to motor\n")
    for mt, js in SINGLE.items():
        w(f"- {mt}: {', '.join(js)}")
    w("- 2x X4-36 per ankle (pitch + roll through the rod linkage), 2x X6-60 for waist roll + pitch, X4-10: head yaw, head pitch (fixed in sim)\n")
    w("### Robot and simulation\n")
    w(f"- Robot mass in the sim model: {results[0]['mass']:.1f} kg (from CAD, scaled +0-20 % in training for wiring etc.)")
    w("- Joint torque limits = motor peak torque (X8 120, X6 60, X4-36 34, X4-10 10 N*m; ankles/waist in joint space through the linkage)")
    w("- Controller: joint PD on the driver at 50 Hz policy rate, gains from `robot_numbers.yaml` (hips/knees kp 120, hip yaw 50, ankles 35, waist 100)")
    w(f"- Bus voltage {args.bus_v:.0f} V (**assumed**, `robot_numbers.yaml` bus.supply_voltage_v is not set yet), "
      f"regeneration {args.regen * 100:.0f} % (**assumed**: drivers not known to return energy), driver idle "
      f"{args.idle_w:.1f} W each (**assumed**), electronics {args.electronics_w:.0f} W (**not included unless set**)\n")
    w("## 6. Limits of this estimate and how to firm it up\n")
    w("- **Simulated gait.** Real walking has more foot slip, uneven floors and controller effort; add 20-30 % margin to "
      "means and size hardware on the 100 ms / 10 ms peaks plus margin.")
    w("- **Losses** are fitted to one measured table (X8). The X6/X4 use the same loss fractions; their own efficiency "
      "tables would firm this up. Losses while the joint is back-driven (braking) use the same terms.")
    w("- **Torque constant**: the sheets' module constant disagrees with their own rated torque/current pair for some "
      "motors (e.g. X8: 2.97 N*m/A listed, 43 N*m / 17.6 A = 2.44 N*m/A); the model uses the torque/current pairs.")
    w("- **Plain MuJoCo deploy model**: no motor torque-speed droop and no per-motor ankle clamp (training had both); "
      "operating points beyond the rated speed or near peak torque in section 3 are where the real robot would be limited.")
    w("- **Bus voltage / regeneration / electronics** are assumptions; rerun with `--bus-v`, `--regen`, `--electronics-w`.")
    w("- **Thermal**: RMS current over 10-14 s tests; long-duration heating needs a thermal model or a bench run.")
    w("- **Firm it up**: (1) bench each motor type at 3-4 held torques, log driver current and bus current, replace the "
      "torque-current points; (2) log bus current on the robot walking and compare with section 2.\n")
    w("## 7. Re-running\n")
    w("```bash\n.venv-mjlab/bin/python wiscohumanoids/tools/current_estimate.py --bus-v 48 --regen 0 --electronics-w 0\n```")
    w("Writes REPORT.md and REPORT.pdf (same content, graphs embedded; the PDF needs Google Chrome on the Mac).")
    w("The same per-motor numbers are also written as CSV files next to this report (one per test), for spreadsheets; "
      "everything in them is in the appendix below.\n")
    w("## Appendix: per-motor results for every test\n")
    w("Phase current in A rms. RMS = heating over the test; peak 10 ms = highest 10 ms average (driver sizing); raw "
      "peak = single 5 ms sample (includes foot-impact spikes). Torque and speed at the motor output (ankle and waist "
      "motors after the linkage). Mean W = average electrical input power of that actuator. % > rated = share of time "
      "above the rated torque.\n")
    for (name, desc, _, _) in TESTS:
        s = S[name]
        if s["fell"]:
            continue
        w(f"### {name}: {desc}\n")
        w(f"Speed achieved {s['speed']:.2f} m/s; battery current mean {s['bus_i_mean']:.1f} A, peak 100 ms "
          f"{s['bus_i_peak100']:.1f} A, peak 10 ms {s['bus_i_peak10']:.1f} A; mean power {s['bus_p_mean']:.0f} W.\n")
        w("| Motor | Type | RMS A | Peak 10 ms A | Raw peak A | RMS torque N*m | Peak torque N*m (% of peak) | Peak rpm | Mean W | % > rated torque |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for mm in s["motors"]:
            w(f"| {mm['name']} | {mm['type']} | {mm['i_rms']:.1f} | {mm['i_peak10']:.1f} | {mm['i_peak']:.1f} | "
              f"{mm['t_rms']:.1f} | {mm['t_peak']:.1f} ({mm['pct_of_peak_torque']:.0f} %) | {mm['rpm_peak']:.0f} | "
              f"{mm['p_mean']:.1f} | {mm['pct_above_rated']:.1f} |")
        w("")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PRINT_CSS = """
@page { size: A4; margin: 14mm 12mm 16mm 12mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10pt; color: #0b0b0b; line-height: 1.4; }
h1 { font-size: 18pt; margin: 0 0 6pt; } h2 { font-size: 13pt; margin: 16pt 0 6pt; border-bottom: 1px solid #e4e3df; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8pt; page-break-inside: auto; }
tr { page-break-inside: avoid; } th, td { border: 1px solid #e4e3df; padding: 2pt 4pt; text-align: left; vertical-align: top; }
th { background: #f3f2ee; } td:not(:first-child) { font-variant-numeric: tabular-nums; }
img { max-width: 100%; display: block; margin: 6pt auto 10pt; page-break-inside: avoid; }
code, pre { font-family: Menlo, monospace; font-size: 8pt; background: #f3f2ee; } pre { padding: 6pt; white-space: pre-wrap; }
h2.appendix { page-break-before: always; } .appendix-section table { font-size: 7.5pt; }
"""


def to_pdf(out):
    """REPORT.md -> REPORT.html (print styles) -> REPORT.pdf with headless Chrome; images stay embedded."""
    import subprocess
    import markdown
    body = markdown.markdown((out / "REPORT.md").read_text(), extensions=["tables", "fenced_code"])
    body = body.replace("<h2>Appendix", '<h2 class="appendix">Appendix')
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>Wisco V2 current draw estimate</title>" \
           f"<style>{PRINT_CSS}</style></head><body>{body}</body></html>"
    (out / "REPORT.html").write_text(html)
    pdf = out / "REPORT.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    f"--print-to-pdf={pdf}", (out / "REPORT.html").as_uri()], check=True, capture_output=True, timeout=120)
    return pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus-v", type=float, default=48.0)
    ap.add_argument("--regen", type=float, default=0.0, help="fraction of braking power returned to the battery")
    ap.add_argument("--electronics-w", type=float, default=0.0)
    ap.add_argument("--idle-w", type=float, default=1.5, help="per driver, standing still")
    ap.add_argument("--only", default="", help="comma-separated test names")
    ap.add_argument("--no-pdf", action="store_true", help="skip REPORT.pdf (needs Google Chrome)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tests = [t for t in TESTS if not args.only or t[0] in args.only.split(",")]
    results, summaries = [], []
    for i, (name, desc, script, opts) in enumerate(tests):
        r = run_test(name, script, opts, seed=i, idle_w=args.idle_w)
        s = summarize(r, args)
        results.append(r); summaries.append(s)
        print(f"{name:18s} {'FELL ' if r['fell'] else ''}speed {s['speed']:.2f} m/s | bus mean {s['bus_i_mean']:5.1f} A, "
              f"peak100 {s['bus_i_peak100']:5.1f} A, peak10 {s['bus_i_peak10']:5.1f} A | {s['bus_p_mean']:5.0f} W", flush=True)
        with open(OUT / f"{name}_motors.csv", "w", newline="") as f:
            cw = csv.DictWriter(f, fieldnames=list(s["motors"][0].keys())); cw.writeheader(); cw.writerows(s["motors"])
    if len(tests) == len(TESTS):
        val = check_against_x8_load_test(args.idle_w)
        BUS_V_FOR_PLOTS[0] = args.bus_v
        files = plots(results, summaries, OUT)
        write_report(results, summaries, val, files, args, OUT)
        print(f"wrote {OUT / 'REPORT.md'}")
        if not args.no_pdf:
            print(f"wrote {to_pdf(OUT)}")


if __name__ == "__main__":
    main()
