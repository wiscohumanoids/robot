#!/usr/bin/env python3
"""Leg actuator identification for sim-to-real (PACE): fit each joint's armature, friction, damping, encoder bias and
command delay so the trained MuJoCo model reproduces the robot's measured joint motion.

    python training/sim2real/pace.py plan   --scene <trained scene.xml> --out plan.npz [--joints REGEX] [--seconds 12]
    python training/sim2real/pace.py synth  --scene <scene.xml> --plan plan.npz --out log.npz [--truth truth.yaml]
    python training/sim2real/pace.py fit    --scene <scene.xml> --log log.npz --out fitted.yaml [--generations 60]
    python training/sim2real/pace.py report --scene <scene.xml> --log log.npz --params fitted.yaml

Method (Bjelonic, Tischhauser, Hutter, "Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Legged
Robots", 2025; the arm's version is arm_stack/calib/pace.py): the robot hangs on a gantry (fixed base, feet clear),
each joint in turn follows a chirp around the standing pose while the others hold, and the joint commands and measured
positions are logged. The same commands are replayed in the model with the base fixed. For each joint, CMA-ES finds
the parameters that minimize the RMS position error, every candidate of a generation simulated in one threaded
`mujoco.rollout` call. The drive PD is the model's position actuator with the logged kp / kd: the same law the
drives run in mode 5. Delay is modelled as a shift of the command (Menlo's Asimov work reports the timing model as
what made zero-shot transfer work).

`plan` writes the excitation (per-joint chirp segments and hold pose); the robot side plays it at the log rate and
records `q_meas`. `synth` plays it in the model with chosen parameters: the sim twin of the gantry test and how the fit
is checked (it must recover known parameters). `fit` writes per-joint values; copy them into
robots/<robot>/model/robot_numbers.yaml (source: the fit file) and rebuild the training model, then train with
narrower dynamics randomization around them.

Log (.npz): dt (s), joints (names), q_cmd and q_meas (T, n_joints, rad), kp and kd (n_joints), hold (n_joints),
segments (k, 3: joint index, start, stop sample).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ros2_ws/src/wisco_ethercat_ros2_hw"))
from arm_stack.pace.cmaes import CMAES  # noqa: E402

PREFIX = "robot/"
PARAMS = ("armature", "frictionloss", "damping", "encoder_bias", "delay_s")


# ---- model ------------------------------------------------------------------------------------------------------

def fixed_base_model(scene, height=1.2):
    """The scene with its floating base removed and the root lifted `height` m (the robot on a gantry)."""
    spec = mujoco.MjSpec.from_file(str(scene))
    for j in list(spec.joints):
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            body = j.parent
            spec.delete(j)
            body.pos = [0.0, 0.0, height]
    for k in list(spec.keys):
        spec.delete(k)
    return spec.compile()


def actuated_joints(model, pattern=None):
    """(name without prefix, joint id, actuator id) for hinge joints driven by a position actuator."""
    out = []
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        j = int(model.actuator_trnid[a, 0])
        name = model.joint(j).name.removeprefix(PREFIX)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and (pattern is None or re.search(pattern, name)):
            out.append((name, j, a))
    return sorted(out, key=lambda t: t[1])


def gains(model, a):
    """(kp, kd) of a position actuator: force = kp * ctrl - kp * q - kd * qdot."""
    return float(model.actuator_gainprm[a, 0]), float(-model.actuator_biasprm[a, 2])


def set_gains(model, a, kp, kd):
    """Write kp / kd into a position actuator."""
    model.actuator_gainprm[a, 0] = kp
    model.actuator_biasprm[a, 1] = -kp
    model.actuator_biasprm[a, 2] = -kd


def apply_params(model, j, p):
    """Write one joint's armature, friction loss and damping into the model."""
    dof = model.jnt_dofadr[j]
    model.dof_armature[dof] = p["armature"]
    model.dof_frictionloss[dof] = p["frictionloss"]
    model.dof_damping[dof] = p["damping"]


def nominal(model, j):
    """The model's current values for a joint (encoder bias and delay 0)."""
    dof = model.jnt_dofadr[j]
    return {"armature": float(model.dof_armature[dof]), "frictionloss": float(model.dof_frictionloss[dof]),
            "damping": float(model.dof_damping[dof]), "encoder_bias": 0.0, "delay_s": 0.0}


# ---- excitation plan --------------------------------------------------------------------------------------------

def chirp(t, f0, f1):
    """Logarithmic sine sweep from f0 to f1 Hz over t (starts at 0)."""
    T = t[-1] if t[-1] > 0 else 1.0
    k = (f1 / f0) ** (1.0 / T)
    return np.sin(2 * np.pi * f0 * (k ** t - 1) / np.log(k))


def standing_pose(scene, joints, keyframe="init_state"):
    """Each joint's angle in the scene's standing keyframe (0 where there is none)."""
    m = mujoco.MjModel.from_xml_path(str(scene))
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    qpos = m.key_qpos[k] if k >= 0 else m.qpos0
    out = []
    for name, _, _ in joints:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + name)
        jid = jid if jid >= 0 else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        out.append(float(qpos[m.jnt_qposadr[jid]]) if jid >= 0 else 0.0)
    return np.array(out)


def plan(model, joints, dt=0.005, seconds=12.0, f=(0.2, 3.0), amp_max=0.3, ramp=1.0, hold=None):
    """Per-joint chirp segments around the hold pose (kept 0.05 rad inside the limits): q_cmd (T, n)."""
    n = len(joints)
    hold = np.zeros(n) if hold is None else np.array(hold, float)
    for i, (_, j, _) in enumerate(joints):
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
            hold[i] = np.clip(hold[i], lo + 0.05, hi - 0.05) if hi - lo > 0.1 else (lo + hi) / 2
    ts = np.arange(0.0, seconds, dt)
    nr = int(ramp / dt)
    blocks, segments, start = [], [], 0
    for i, (_, j, _) in enumerate(joints):
        amp = amp_max
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
            amp = min(amp_max, 0.8 * min(hold[i] - lo, hi - hold[i]))
        envelope = np.minimum(1.0, np.minimum(ts, ts[-1] - ts) / ramp)       # fade in and out
        q = np.tile(hold, (len(ts) + 2 * nr, 1))
        q[nr:nr + len(ts), i] = hold[i] + amp * envelope * chirp(ts, *f)
        blocks.append(q)
        segments.append((i, start + nr, start + nr + len(ts)))
        start += len(q)
    return {"dt": dt, "q_cmd": np.concatenate(blocks), "hold": hold, "segments": np.array(segments, int)}


# ---- replay -----------------------------------------------------------------------------------------------------

def delayed(signal, delay_s, dt):
    """The command as the drive sees it `delay_s` later (linear interpolation, first value held)."""
    t = np.arange(len(signal)) * dt
    return np.interp(t - delay_s, t, signal, left=signal[0])


def replay(models, joints, log, joint_index, candidates, nthread=None):
    """Simulate segment `joint_index` of the log once per candidate parameter dict; returns q (n_cand, T) of the
    chirped joint, encoder bias added. `models` holds one model copy per candidate (modified in place).
    """
    from mujoco import rollout
    i, s, e = log["segment"]
    dt = float(log["dt"])
    q_cmd = log["q_cmd"][s - int(1.0 / dt):e]            # include the settled lead-in before the chirp
    T = len(q_cmd)
    act_ids = [a for _, _, a in joints]
    ctrl = np.zeros((len(candidates), T, models[0].nu))
    initial = []
    for c, (m, p) in enumerate(zip(models, candidates)):
        m.opt.timestep = dt
        for k, (_, j, a) in enumerate(joints):
            set_gains(m, a, log["kp"][k], log["kd"][k])
        apply_params(m, joints[i][1], p)
        ctrl[c][:, act_ids] = q_cmd
        ctrl[c][:, act_ids[i]] = delayed(q_cmd[:, i], p["delay_s"], dt)
        d = mujoco.MjData(m)
        for k, (_, j, _) in enumerate(joints):
            d.qpos[m.jnt_qposadr[j]] = q_cmd[0, k]
        mujoco.mj_forward(m, d)
        st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        initial.append(st)
    datas = [mujoco.MjData(models[0]) for _ in range(nthread or min(len(models), os.cpu_count() or 1))]
    state, _ = rollout.rollout(models, datas, np.array(initial), ctrl)
    qadr = 1 + models[0].jnt_qposadr[joints[i][1]]        # FULLPHYSICS state = [time, qpos, ...]
    lead = int(1.0 / dt)
    return state[:, lead:, qadr] + np.array([p["encoder_bias"] for p in candidates])[:, None]


def measured(log, joint_index):
    """The measured position of the chirped joint over its segment."""
    _, s, e = log["segment"]
    return log["q_meas"][s:e, joint_index]


def segment_log(log, k):
    """The log narrowed to segment k (adds 'segment')."""
    return {**log, "segment": tuple(int(v) for v in log["segments"][k])}


# ---- fit --------------------------------------------------------------------------------------------------------

def bounds(nom):
    """Search box per parameter around the model's nominal values."""
    arm = max(nom["armature"], 1e-3)
    return {"armature": (0.2 * arm, 5.0 * arm), "frictionloss": (0.0, max(4 * nom["frictionloss"], 3.0)),
            "damping": (0.0, max(4 * nom["damping"], 3.0)), "encoder_bias": (-0.05, 0.05), "delay_s": (0.0, 0.03)}


def fit_joint(model, joints, log, k, generations=60, popsize=16, seed=0):
    """CMA-ES over one segment's joint; returns (params, rms_before, rms_after, cost history)."""
    log_k = segment_log(log, k)
    i = log_k["segment"][0]
    j = joints[i][1]
    nom = nominal(model, j)
    box = bounds(nom)
    lo = np.array([box[p][0] for p in PARAMS])
    hi = np.array([box[p][1] for p in PARAMS])
    scale = hi - lo
    x0 = (np.clip([nom[p] for p in PARAMS], lo, hi) - lo) / scale
    es = CMAES(x0, 0.3, np.zeros(len(PARAMS)), np.ones(len(PARAMS)), popsize=popsize, seed=seed)
    models = [mujoco.MjModel.from_binary_path(_tmp_binary(model)) for _ in range(es.lam)]
    target = measured(log_k, i)

    def to_params(u):
        """Unit-cube point to named parameters."""
        return dict(zip(PARAMS, (lo + np.clip(u, 0, 1) * scale).tolist()))

    def cost(params_list):
        """RMS position error of each candidate over the segment."""
        q = replay(models[:len(params_list)], joints, log_k, i, params_list)
        return np.sqrt(np.mean((q - target) ** 2, axis=1))

    before = float(cost([nom])[0])
    history, best, best_cost = [], nom, before
    for _ in range(generations):
        xs = es.ask()
        cands = [to_params(x) for x in xs]
        cs = cost(cands)
        es.tell(xs, cs)
        if cs.min() < best_cost:
            best, best_cost = cands[int(cs.argmin())], float(cs.min())
        history.append(float(cs.min()))
    return best, before, best_cost, history


_BIN = {}


def _tmp_binary(model):
    """Model copies via a saved binary (MjModel has no public copy constructor in every binding version)."""
    key = id(model)
    if key not in _BIN:
        import tempfile
        path = Path(tempfile.mkdtemp(prefix="pace_")) / "model.mjb"
        mujoco.mj_saveModel(model, str(path), None)
        _BIN[key] = str(path)
    return _BIN[key]


# ---- commands ---------------------------------------------------------------------------------------------------

def load_log(path):
    """A log .npz as a dict (joints as a list of str)."""
    z = np.load(path, allow_pickle=False)
    return {k: (z[k].tolist() if k == "joints" else z[k]) for k in z.files}


def cmd_plan(a):
    """Write the excitation plan for the selected joints."""
    model = fixed_base_model(a.scene)
    joints = actuated_joints(model, a.joints)
    p = plan(model, joints, dt=a.dt, seconds=a.seconds, hold=standing_pose(a.scene, joints))
    kp, kd = zip(*(gains(model, act) for _, _, act in joints))
    np.savez(a.out, dt=p["dt"], joints=np.array([n for n, _, _ in joints]), q_cmd=p["q_cmd"], hold=p["hold"],
             segments=p["segments"], kp=np.array(kp), kd=np.array(kd))
    print(f"{a.out}: {len(joints)} joints, {len(p['q_cmd']) * p['dt']:.0f} s at {1 / p['dt']:.0f} Hz")


def simulate_log(model, joints, pl, truth):
    """Play a plan in the model with per-joint `truth` parameters; returns q_meas (T, n)."""
    q_meas = np.tile(pl["hold"], (len(pl["q_cmd"]), 1)).astype(float)
    for k in range(len(pl["segments"])):
        log_k = segment_log(pl, k)
        i, s, e = log_k["segment"]
        p = {**nominal(model, joints[i][1]), **truth.get(joints[i][0], {})}
        q = replay([mujoco.MjModel.from_binary_path(_tmp_binary(model))], joints, log_k, i, [p])[0]
        q_meas[s:e, i] = q
    return q_meas


def cmd_synth(a):
    """Play a plan in the model with chosen parameters and save it as a log."""
    model = fixed_base_model(a.scene)
    pl = load_log(a.plan)
    joints = [t for t in actuated_joints(model) if t[0] in pl["joints"]]
    truth = yaml.safe_load(open(a.truth))["joints"] if a.truth else {}
    q_meas = simulate_log(model, joints, pl, truth)
    if a.noise:
        q_meas += np.random.default_rng(0).normal(0, a.noise, q_meas.shape)
    np.savez(a.out, **{k: np.array(v) for k, v in pl.items()}, q_meas=q_meas)
    print(f"{a.out}: simulated {len(pl['segments'])} segments")


def cmd_fit(a):
    """Fit every segment of a log and write the per-joint parameters."""
    model = fixed_base_model(a.scene)
    log = load_log(a.log)
    joints = [t for t in actuated_joints(model) if t[0] in log["joints"]]
    if [n for n, _, _ in joints] != list(log["joints"]):
        raise SystemExit("log joints differ from the model's joints (order or names)")
    out = {"scene": str(a.scene), "log": str(a.log), "method": "PACE: per-joint CMA-ES replay, fixed base",
           "joints": {}}
    for k in range(len(log["segments"])):
        name = joints[int(log["segments"][k][0])][0]
        p, before, after, _ = fit_joint(model, joints, log, k, a.generations, a.popsize)
        out["joints"][name] = {**{q: round(v, 6) for q, v in p.items()},
                               "rms_before_rad": round(before, 6), "rms_after_rad": round(after, 6)}
        print(f"{name:22s} rms {before:.4f} -> {after:.4f} rad   " +
              " ".join(f"{q}={v:.4g}" for q, v in p.items()), flush=True)
    yaml.safe_dump(out, open(a.out, "w"), sort_keys=False)
    print(f"wrote {a.out}")


def cmd_report(a):
    """Print each joint's RMS error with the nominal and the fitted parameters."""
    model = fixed_base_model(a.scene)
    log = load_log(a.log)
    joints = [t for t in actuated_joints(model) if t[0] in log["joints"]]
    fitted = yaml.safe_load(open(a.params))["joints"]
    print(f"{'joint':22s} {'nominal rms':>12s} {'fitted rms':>12s}")
    for k in range(len(log["segments"])):
        log_k = segment_log(log, k)
        i = log_k["segment"][0]
        name, j, _ = joints[i]
        nom = nominal(model, j)
        fit = {**nom, **{q: fitted.get(name, {}).get(q, nom[q]) for q in PARAMS}}
        m = [mujoco.MjModel.from_binary_path(_tmp_binary(model)) for _ in range(2)]
        q = replay(m, joints, log_k, i, [nom, fit])
        rms = np.sqrt(np.mean((q - measured(log_k, i)) ** 2, axis=1))
        print(f"{name:22s} {rms[0]:12.4f} {rms[1]:12.4f}")


def main():
    """Parse the subcommand and run it."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--joints", help="regex over joint names (default: every actuated joint)")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--dt", type=float, default=0.005)
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("synth")
    p.add_argument("--scene", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--truth", help="YAML {joints: {name: {param: value}}}; default: the model's own values")
    p.add_argument("--noise", type=float, default=0.0, help="encoder noise std, rad")
    p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("fit")
    p.add_argument("--scene", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--popsize", type=int, default=16)
    p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("report")
    p.add_argument("--scene", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--params", required=True)
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
