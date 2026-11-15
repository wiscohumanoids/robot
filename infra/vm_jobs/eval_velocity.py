"""Behaviour test for a velocity (walking) checkpoint: does it actually walk, with the arms moving?

Runs N robots in the task's play config with random arm motion at full difficulty (G1: range 0.6;
Wisco: play mode holds curriculum level 1), pushes off, and fixed command segments. For each segment:
mean achieved body velocity over the second half (all robots), how many robots fell, and whether the
command was followed (>= 50 % of commanded value, same sign). Prints a table; --json writes it.

  python infra/vm_jobs/eval_velocity.py CKPT --task Hsim-G1-Velocity-Flat-v2 --import-path training/locomotion --import-module hsim_tasks
  python infra/vm_jobs/eval_velocity.py CKPT --task Wisco-V2-Velocity-Flat --import-path wiscohumanoids/training/mjlab_task --import-module wisco_mjlab
"""
import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

SEGMENTS = [  # seconds, (vx, vy, wz), label
    (3.0, (0.0, 0.0, 0.0), "stand"),
    (5.0, (0.5, 0.0, 0.0), "forward 0.5"),
    (5.0, (1.0, 0.0, 0.0), "forward 1.0"),
    (5.0, (-0.4, 0.0, 0.0), "backward 0.4"),
    (5.0, (0.0, 0.4, 0.0), "sidestep 0.4"),
    (5.0, (0.0, 0.0, 0.8), "turn 0.8"),
    (5.0, (0.0, 0.0, -0.5), "turn right 0.5"),
    (5.0, (0.3, 0.0, 0.3), "walk 0.3 + turn 0.3"),
    (5.0, (0.2, 0.0, 0.0), "slow 0.2"),
    (3.0, (0.0, 0.0, 0.0), "stand"),
]
TURN_SEGMENTS = [  # --script turns: in-place turns from standstill, both directions, small to large
    (3.0, (0.0, 0.0, 0.0), "stand"),
] + [seg for w in (0.3, 0.5, 0.8, 1.2) for sgn, name in ((1, "left"), (-1, "right"))
     for seg in ((5.0, (0.0, 0.0, sgn * w), f"{name} {w}"), (3.0, (0.0, 0.0, 0.0), "stand"))]

KITCHEN_SEGMENTS = [  # --script kitchen: what the ROS stack sends (Nav2 legs, stops, short approach nudges, waiting)
    (6.0, (0.0, 0.0, 0.0), "stand"),
    (3.0, (0.35, 0.0, 0.0), "walk 0.35"),
    (4.0, (0.0, 0.0, 0.0), "stand"),
    (1.5, (0.0, 0.0, 0.5), "turn 0.5"),
    (3.0, (0.0, 0.0, 0.0), "stand"),
    (0.6, (0.15, 0.0, 0.0), "nudge 0.15"),
    (3.0, (0.0, 0.0, 0.0), "stand"),
    (0.5, (0.0, 0.0, -0.3), "nudge turn -0.3"),
    (3.0, (0.0, 0.0, 0.0), "stand"),
    (3.0, (0.3, 0.0, 0.25), "walk 0.3 + turn"),
    (8.0, (0.0, 0.0, 0.0), "stand"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--task", required=True)
    ap.add_argument("--import-path", default=None)
    ap.add_argument("--import-module", default=None)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--arms", choices=["random", "default"], default="random")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--script", choices=["standard", "turns", "kitchen"], default="standard")
    args = ap.parse_args()
    if args.import_path:
        sys.path.insert(0, str(Path(args.import_path).resolve()))
    if args.import_module:
        importlib.import_module(args.import_module)
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs
    tw = cfg.commands["twist"]
    tw.heading_command, tw.ranges.heading = False, None
    for k in ("rel_heading_envs", "rel_standing_envs", "rel_forward_envs", "rel_world_envs", "rel_turn_envs"):
        if hasattr(tw, k):
            setattr(tw, k, 0.0)
    tw.resampling_time_range = (1e9, 1e9)
    if "arm_motion" in cfg.events:                       # G1 v2 (play cfg has no curriculum)
        if args.arms == "random":                        # full difficulty, same as Wisco play level 1:
            cfg.events["arm_motion"].params["range_frac"] = 1.0          # 100 % of the safe arm offsets
            if "hand_payload" in cfg.events:
                cfg.events["hand_payload"].params["ranges"] = (0.0, 1.0)  # 0-1 kg per hand
        else:
            cfg.events["arm_motion"].params["hold_default_prob"] = 1.0
    if "arm_targets" in cfg.events and args.arms == "default":   # Wisco (play holds level 1)
        cfg.events["arm_targets"].params["hold_prob"] = 1.0
    cfg.events.pop("push_robot", None)
    cfg.episode_length_s = 1e6

    agent = load_rl_cfg(args.task)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=agent.clip_actions)
    runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(env, asdict(agent), device=device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    raw = env.unwrapped
    term = raw.command_manager.get_term("twist")
    robot = raw.scene["robot"]
    dt = raw.step_dt

    env.reset()
    rows = []
    print(f"{args.task}  {Path(args.checkpoint).name}  envs={args.num_envs} arms={args.arms}")
    print(f"{'segment':22s} {'cmd vx vy wz':>17s} {'achieved vx vy wz':>22s} {'fell':>6s}  verdict")
    script = {"turns": TURN_SEGMENTS, "kitchen": KITCHEN_SEGMENTS}.get(args.script, SEGMENTS)
    feet = raw.scene["feet_ground_contact"]
    torso_id = robot.find_bodies("torso_link")[0][0] if hasattr(robot, "find_bodies") else None
    rock_all, tilt_max = [], 0.0
    stand_stats = []   # per stand segment (second half): per-robot speed, foot lift-offs / s, action change rms
    for seconds, cmd, label in script:
        c = torch.tensor(cmd, device=device, dtype=torch.float32)
        steps = int(seconds / dt)
        meas, fell = [], torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        spd, lifts, jit, n2 = 0.0, 0.0, 0.0, 0
        was_air = feet.data.current_air_time > 0
        for i in range(steps):
            term.vel_command_b[:] = c
            with torch.no_grad():
                _, _, dones, _ = env.step(policy(env.get_observations()))
            fell |= dones.bool()
            air = feet.data.current_air_time > 0
            if torso_id is not None:   # torso roll/pitch rate (rocking) and pelvis tilt, all robots still up
                w = robot.data.body_link_ang_vel_w[:, torso_id, :2].norm(dim=1)[~fell]
                rock_all.append(float(w.square().mean()) if len(w) else 0.0)
                g = robot.data.projected_gravity_b[:, :2].norm(dim=1)[~fell]
                tilt_max = max(tilt_max, float(torch.rad2deg(torch.asin(g.clamp(max=1))).max()) if len(g) else 0.0)
            if i >= steps // 2 and not any(cmd):
                ok_ = ~fell
                spd += float((robot.data.root_link_lin_vel_b[:, :2].norm(dim=1)[ok_]).mean())
                lifts += float(((air & ~was_air).float().sum(1)[ok_]).mean())
                da = raw.action_manager.action - raw.action_manager.prev_action
                jit += float(da.square().mean(1).sqrt()[ok_].mean())
                n2 += 1
            was_air = air
            if i >= steps // 2:
                v = robot.data.root_link_lin_vel_b[:, :2]
                w = robot.data.root_link_ang_vel_b[:, 2:3]
                meas.append(torch.cat([v, w], 1)[~fell].mean(0).cpu().numpy() if (~fell).any() else np.full(3, np.nan))
        m = np.nanmean(np.array(meas), axis=0)
        cm = np.array(cmd)
        active = np.abs(cm) > 1e-9
        nf = int(fell.sum())
        if active.any():
            ok = bool(np.all(m[active] * np.sign(cm[active]) >= 0.5 * np.abs(cm[active])))
            verdict = "follows" if ok else "IGNORES / under 50%"
        else:
            ok = bool(np.all(np.abs(m) < 0.15))
            verdict = "stands still" if ok else "DRIFTS"
        if n2:
            st = (spd / n2, lifts / (n2 * dt), jit / n2)
            stand_stats.append(st)
            verdict += f"  [speed {st[0]:.3f} m/s, {st[1]:.2f} foot lifts/s, action jitter {st[2]:.3f}]"
        rows.append({"segment": label, "cmd": list(cmd), "achieved": np.round(m, 3).tolist(), "fell": nf, "ok": ok})
        print(f"{label:22s} {cm[0]:5.2f}{cm[1]:6.2f}{cm[2]:6.2f}   {m[0]:7.2f}{m[1]:7.2f}{m[2]:7.2f} {nf:4d}/{args.num_envs}  {verdict}")
    n_ok = sum(r["ok"] for r in rows)
    falls = sum(r["fell"] for r in rows)
    print(f"SUMMARY: {n_ok}/{len(rows)} segments OK, {falls} falls total over {len(rows)} segments x {args.num_envs} robots")
    if rock_all:
        print(f"ROCKING: torso roll/pitch rate RMS {np.degrees(np.sqrt(np.mean(rock_all))):.1f} deg/s, max pelvis tilt {tilt_max:.1f} deg (lower = steadier)")
    if stand_stats:
        a = np.array(stand_stats).mean(0)
        print(f"STANDING: per-robot speed {a[0]:.3f} m/s, {a[1]:.2f} foot lifts/s, action jitter {a[2]:.3f} (lower = steadier)")
    if args.json:
        args.json.write_text(json.dumps({"task": args.task, "checkpoint": args.checkpoint, "rows": rows,
                                         "segments_ok": n_ok, "falls": falls}, indent=1))


if __name__ == "__main__":
    main()
