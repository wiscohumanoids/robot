#!/usr/bin/env python3
"""Reproducible grasp physics ablation; does not edit production scenes.

python tools/investigate_gripper.py --output reports/gripper_ablation.json
All candidates replay identical 30 Hz arm/gripper commands per scenario. Only
the initial arm pose is set directly; acquisition, holding and release simulate
normally. Slip is measured in the wrist frame, independently of weld flags.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from sim.panthera_env import PantheraSim, mat_to_quat


VARIANTS = {
    "production": {"production": True, "impratio": 100},
    "weld": {},
    "contacts": {},
    "impratio100": {"impratio": 100},
    "impratio1000": {"impratio": 1000},
    "noslip3": {"noslip_iterations": 3},
    "noslip10": {"noslip_iterations": 10},
    "pad_impedance": {"impedance": .999},
    "force6": {"force": 6},
    "friction2": {"friction": 2},
    "impratio100_force6": {"impratio": 100, "force": 6},
    "interpolate": {"interpolate": True},
    "interpolate_force6": {"interpolate": True, "force": 6},
    "interpolate_impratio100": {"interpolate": True, "impratio": 100},
    "interpolate_impratio100_force6": {"interpolate": True, "impratio": 100, "force": 6},
}
SCENARIOS = {
    "seated55": {"pitch": 55, "depth": .018},
    "seated30": {"pitch": 30, "depth": .018},
    "shallow30": {"pitch": 30, "depth": 0},
    "offset8": {"pitch": 55, "depth": .018, "offset": .008},
    "yaw30": {"pitch": 55, "depth": .018, "yaw": 30},
    "yaw45": {"pitch": 55, "depth": .018, "yaw": 45},
    "jump55": {"pitch": 55, "depth": .018, "jump": True},
    "heavy55": {"pitch": 55, "depth": .018, "mass_scale": 10},
    "weak55": {"pitch": 55, "depth": .018, "force": .3},
    "partial_open55": {"pitch": 55, "depth": .018, "release_grip": .65},
}


def configure(sim, variant, case):
    m = sim.model
    cfg = VARIANTS[variant]
    # Preserve the historical baseline after contact-v2 became the default.
    sim.dynamics = "contact-v2" if cfg.get("production") else "weld-v1"
    m.opt.impratio = cfg.get("impratio", 10)
    if variant != "weld":
        sim._update_grasp = lambda: None
        sim._release_grasps()
    for name in ("impratio", "noslip_iterations"):
        if name in cfg:
            setattr(m.opt, name, cfg[name])
    m.actuator_forcerange[sim.grip_act] = [-cfg.get("force", case.get("force", 3)),
                                         cfg.get("force", case.get("force", 3))]
    if "friction" in cfg:
        m.pair_friction[:, :2] = cfg["friction"]
    if "impedance" in cfg:
        m.pair_solimp[:, :2] = cfg["impedance"]
    bid = sim.object_bodies[0]
    m.body_mass[bid] *= case.get("mass_scale", 1)
    m.body_inertia[bid] *= case.get("mass_scale", 1)
    if case.get("mass_scale", 1) != 1:
        # mj_setConst resets MjData; run() initializes after configure(). Do not
        # call it for ordinary variants: stacks already have randomized poses.
        mujoco.mj_setConst(m, sim.data)


def initialize(sim, case, q):
    sim.reset(randomize=False)
    # Place only the target in the commanded path; other cubes remain on table.
    adr = sim.object_qadr[0]
    sim.data.qpos[adr:adr+3] = [.38, .12, .0725]
    yaw = np.deg2rad(case.get("yaw", 0)) / 2
    sim.data.qpos[adr+3:adr+7] = [np.cos(yaw), 0, 0, np.sin(yaw)]
    sim.data.qpos[sim.arm_qadr] = q
    sim.data.qvel[:] = 0
    sim.set_arm_ctrl(q, immediate=True)
    sim.set_gripper(1)
    mujoco.mj_forward(sim.model, sim.data)


def trajectory(case, hold_seconds):
    sim = PantheraSim()
    a = np.deg2rad(case["pitch"])
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                    [-np.sin(a), 0, np.cos(a)]])
    quat = mat_to_quat(rot)
    grasp = np.array([.38, .12 + case.get("offset", 0), .0725]) + case["depth"]*rot[:, 0]
    hover = grasp + [0, 0, .08]
    lift = grasp + [0, 0, .12]
    q, pe, re = sim.ik(hover, quat, max_joint_step=None, iters=200, posture_gain=0)
    if pe > .002 or re > .02:
        raise RuntimeError(f"Initial IK: {pe}, {re}")
    initial = q.copy()
    initialize(sim, case, q)
    target = hover.copy()
    commands = []
    max_error = 0.
    segments = [("settle", hover, 1, .5), ("descend", grasp, 1, 1.),
                ("close", grasp, 0, 1.), ("lift", lift, 0, 1.),
                ("hold", lift, 0, hold_seconds),
                ("shake", lift+[0, .08, 0], 0, .25),
                ("shake", lift+[0, -.08, .03], 0, .4),
                ("shake", lift, 0, .25), ("recover", lift, 0, 2.),
                ("release", lift, case.get("release_grip", 1), 1.)]
    for phase, dest, grip, seconds in segments:
        steps = round(seconds*30)
        start = target.copy()
        for i in range(1, steps+1):
            u = i/steps
            u = u*u*(3-2*u)
            if phase == "lift" and case.get("jump"):
                u = 1.
            target = start*(1-u)+dest*u
            q, pe, re = sim.ik(target, quat, q_init=q, iters=35,
                                max_joint_step=.08, posture_gain=0)
            max_error = max(max_error, pe)
            commands.append((phase, q.copy(), grip, target.copy()))
    return initial, commands, max_error


def sample(sim):
    m, d = sim.model, sim.data
    bid = sim.object_bodies[0]
    gid = m.geom("cube_red").id
    normal, counts = np.zeros(2), np.zeros(2, dtype=int)
    contact_friction = []
    wrench = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        for j, pad in enumerate((sim._pad_L, sim._pad_R)):
            if {c.geom1, c.geom2} == {gid, pad} and c.efc_address >= 0:
                mujoco.mj_contactForce(m, d, i, wrench)
                normal[j] += max(0, wrench[0])
                counts[j] += 1
                contact_friction.append(c.friction.tolist())
    rot = d.xmat[sim._link6].reshape(3, 3)
    return dict(time=float(d.time), z=float(d.xpos[bid, 2]),
                relative=(rot.T @ (d.xpos[bid]-d.xpos[sim._link6])).tolist(),
                rotation=(rot.T @ d.xmat[bid].reshape(3, 3)).tolist(),
                normal=normal.tolist(), contacts=counts.tolist(),
                actuator_force=float(d.actuator_force[sim.grip_act]),
                latched=bool(d.eq_active[sim._grasp_eq[0]]),
                grasped=bool(sim.grasp_flags()[0]),
                contact_friction=contact_friction,
                fingers=d.qpos[sim.finger_qadr].tolist())


def run(variant, case, initial, commands, ik_error, physics_trace=False):
    sim = PantheraSim()
    configure(sim, variant, case)
    initialize(sim, case, initial)
    phases = {}
    elapsed_steps = 0
    step_time = 0.
    dynamics = {}
    last_position = sim.ee_pos()
    last_velocity = np.zeros(3)
    for tick, (phase, q, grip, target) in enumerate(commands, start=1):
        previous_ctrl = sim.data.ctrl[:6].copy()
        sim.set_arm_ctrl(q)
        sim.set_gripper(grip)
        end = round(tick/(30*sim.dt))
        t = time.perf_counter()
        substeps = end-elapsed_steps
        if VARIANTS[variant].get("interpolate") or physics_trace:
            for j in range(1, substeps+1):
                if VARIANTS[variant].get("interpolate"):
                    sim.set_arm_ctrl(previous_ctrl + (q-previous_ctrl)*j/substeps)
                sim.step()
                if physics_trace:
                    pos = sim.ee_pos()
                    vel = (pos-last_position)/sim.dt
                    acc = (vel-last_velocity)/sim.dt
                    dynamics.setdefault(phase, []).append(float(np.linalg.norm(acc)))
                    last_position, last_velocity = pos, vel
        else:
            sim.step(substeps)
        step_time += time.perf_counter()-t
        elapsed_steps = end
        s = sample(sim)
        s["tracking_mm"] = float(1000*np.linalg.norm(sim.ee_pos()-target))
        phases.setdefault(phase, []).append(s)
        if not np.isfinite(sim.data.qpos).all():
            raise RuntimeError("Nonfinite state")
    close = np.array(phases["close"][-1]["relative"])
    summary = {}
    for phase, samples in phases.items():
        last = samples[-1]
        r0, r1 = np.array(samples[0]["rotation"]), np.array(last["rotation"])
        summary[phase] = dict(last=last,
            drift_mm=float(1000*np.linalg.norm(np.array(last["relative"])-samples[0]["relative"])),
            close_slip_mm=float(1000*np.linalg.norm(np.array(last["relative"])-close)),
            rotation_drift_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(r0.T@r1)-1)/2, -1, 1)))),
            min_z=float(min(s["z"] for s in samples)),
            bilateral_fraction=float(np.mean([min(s["normal"]) > .01 for s in samples])),
            mean_normal=np.mean([s["normal"] for s in samples], axis=0).tolist(),
            max_tracking_mm=float(max(s["tracking_mm"] for s in samples)))
    return dict(phases=summary, physics_wall_seconds=step_time,
                tcp_acceleration={p: dict(max_m_s2=max(a), p99_m_s2=float(np.percentile(a, 99)))
                                  for p, a in dynamics.items()},
                simulated_seconds=float(sim.data.time), ik_error_mm=1000*ik_error,
                retained=bool(summary["recover"]["last"]["z"] > .1125 and
                              min(summary["recover"]["last"]["normal"]) > .01),
                released=bool(summary["release"]["last"]["z"] < .11 and
                              max(summary["release"]["last"]["normal"]) < .01),
                warning_counts=sim.data.warning.number.tolist())


def run_stack(variant, seed):
    """Use the production planner's moves with contact-based acquisition checks.

    Mirrors its fixed-start, red-on-green/blue-on-red protocol. The production
    Planner.run itself requires weld activation, so cannot evaluate no-weld
    candidates. No flag is faked here: successful placement is measured from
    released object poses over one second, and failures are retained.
    """
    from tools.collect_scripted import Planner, DemoFailure, grasp_rotation
    from sim.stack_task import stack_metrics, CUBE_EDGE
    planner = Planner(seed, arm_start="fixed")
    sim = planner.sim
    configure(sim, variant, {})
    a = np.deg2rad(55)
    quat = mat_to_quat(np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                                [-np.sin(a), 0, np.cos(a)]]))
    q, pe, re = sim.ik([.38, 0, .24], quat, max_joint_step=None, iters=150)
    if pe > .001 or re > .01:
        return dict(success=False, error="initial IK")
    sim.data.qpos[sim.arm_qadr] = q
    sim.data.qvel[:] = 0
    sim.set_arm_ctrl(q, immediate=True)
    sim.set_gripper(1)
    mujoco.mj_forward(sim.model, sim.data)
    planner.target, planner.quat, planner.qctrl = sim.ee_pos(), sim.ee_quat(), q
    # Interpolate incoming 30 Hz commands within each native physics interval.
    if VARIANTS[variant].get("interpolate"):
        native_step = sim.step
        previous = q.copy()

        def step(n=1):
            nonlocal previous
            goal = sim.data.ctrl[:6].copy()
            for j in range(1, n+1):
                sim.set_arm_ctrl(previous+(goal-previous)*j/n)
                native_step()
            previous = goal

        sim.step = step
    lifts = []
    try:
        planner.hold("settle", .3)
        for level, (block, support) in enumerate(((0, 1), (2, 0)), start=1):
            rot = grasp_rotation(sim, block)
            obj = sim.object_poses()[0][block]
            grasp = obj+.018*rot[:, 0]
            above = grasp+[0, 0, planner.clearance]
            safe = planner.target.copy()
            safe[2] = max(safe[2], above[2]+.015)
            planner.move(f"{level}_clear", safe)
            planner.move(f"{level}_approach", above, mat_to_quat(rot), 1.)
            planner.move(f"{level}_descend", grasp)
            planner.hold(f"{level}_close", .5, 0.)
            if sim.object_names[block] not in sim._pinched():
                raise DemoFailure(f"{level}: no bilateral contact")
            planner.move(f"{level}_lift", above)
            height = float(sim.object_poses()[0][block, 2]-obj[2])
            lifts.append(height)
            if height < .04:
                raise DemoFailure(f"{level}: failed lift")
            held_offset = sim.ee_pos()-sim.object_poses()[0][block]
            dest = sim.object_poses()[0][support]+[0, 0, CUBE_EDGE+.014]
            place = dest+held_offset
            transit = place+[0, 0, planner.clearance]
            lift = planner.target.copy()
            lift[2] = transit[2]
            planner.move(f"{level}_raise", lift)
            planner.move(f"{level}_transfer", transit)
            planner.move(f"{level}_place", place)
            planner.hold(f"{level}_release", .5, 1.)
            planner.move(f"{level}_retreat", transit)
        for _ in range(30):
            planner.tick(planner.target, planner.quat, 1.)
            positions = sim.object_poses()[0]
            if not stack_metrics(positions)["three_stack"] or sim._pinched() or any(
                    sim.data.eq_active[e] for e in sim._grasp_eq):
                raise DemoFailure("released stack not stable for one second")
            if np.max(np.linalg.norm(np.diff(positions[[1, 0, 2], :2], axis=0), axis=1)) > .012:
                raise DemoFailure("stack alignment exceeds 12 mm")
        error = None
    except DemoFailure as exc:
        error = str(exc)
    return dict(success=error is None, error=error, lifts=lifts,
                final_positions=sim.object_poses()[0].tolist(),
                simulated_seconds=float(sim.data.time),
                warning_counts=sim.data.warning.number.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--hold-seconds", type=float, default=10)
    parser.add_argument("--physics-trace", action="store_true")
    parser.add_argument("--stack-seeds", type=int, default=0,
                        help="Run released three-stack protocol on seeds 0..N-1 instead")
    parser.add_argument("--output", type=Path, default=ROOT/"reports/gripper_ablation.json")
    args = parser.parse_args()
    result = dict(mujoco=mujoco.__version__, hold_seconds=args.hold_seconds,
                  retention_definition="Final recover: cube center above 0.1125 m and both pad normal loads >0.01 N; inspect slip separately.",
                  variants={k: VARIANTS[k] for k in args.variants},
                  scenarios={k: SCENARIOS[k] for k in args.scenarios}, runs=[])
    if args.stack_seeds:
        result["stack_seeds"] = args.stack_seeds
        for seed in range(args.stack_seeds):
            for variant in args.variants:
                out = run_stack(variant, seed)
                result["runs"].append(dict(seed=seed, variant=variant, **out))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2)+"\n")
                print(f'seed={seed} {variant} success={out["success"]} error={out["error"]}', flush=True)
        return
    for name in args.scenarios:
        initial, commands, err = trajectory(SCENARIOS[name], args.hold_seconds)
        for variant in args.variants:
            out = run(variant, SCENARIOS[name], initial, commands, err, args.physics_trace)
            result["runs"].append(dict(scenario=name, variant=variant, **out))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+"\n")
            print(f'{name:12} {variant:22} retained={out["retained"]} '
                  f'hold drift={out["phases"]["hold"]["drift_mm"]:.3f} mm '
                  f'released={out["released"]}', flush=True)


if __name__ == "__main__":
    main()
