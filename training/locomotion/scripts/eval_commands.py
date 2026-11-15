"""Evaluate a trained velocity policy inside mjlab's own environment (GPU, MuJoCo-Warp; CPU works too)
with fixed commands, the same scripts tools/sim2sim_play.py --headless uses on the CPU.

Same table on both sides tells you whether a behavior comes from the policy itself
(both sides agree) or from the CPU deployment (they differ).

    uv run python scripts/eval_commands.py logs/rsl_rl/g1_velocity/<run>/model_9999.pt
    # v2 (legs+waist) policy, small-command deadband test (decision 0004), arms held at default:
    uv run python scripts/eval_commands.py <run>/model_N.pt --task Hsim-G1-Velocity-Flat-v2 \
        --script deadband --arms default --json deadband_v2.json

--script deadband: small commands the v1 policy ignored. A segment "responds" when the measured
velocity reaches >= 50 % of the command on every commanded axis (second half of the segment).
--arms (v2 tasks only): default = arms held at the default pose; random = the training arm motion.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # training/locomotion: hsim_tasks
import hsim_tasks  # noqa: E402,F401  (registers Hsim-* tasks)
import mjlab.tasks  # noqa: E402,F401  (registers mjlab tasks)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

SCRIPT = [  # same as tools/sim2sim_play.py
    (2.0, (0.0, 0.0, 0.0), "stand"),
    (4.0, (0.5, 0.0, 0.0), "forward 0.5"),
    (4.0, (1.0, 0.0, 0.0), "forward 1.0"),
    (3.0, (0.0, 0.0, 1.0), "turn left 1.0"),
    (3.0, (0.0, 0.6, 0.0), "sidestep left 0.6"),
    (3.0, (-0.4, 0.0, 0.0), "backward 0.4"),
    (2.0, (0.0, 0.0, 0.0), "stand"),
]
DEADBAND_SCRIPT = [  # same as tools/sim2sim_play.py --script deadband
    (2.0, (0.0, 0.0, 0.0), "stand"),
    (4.0, (0.0, 0.0, 0.3), "turn 0.3"),
    (4.0, (0.0, 0.0, 0.5), "turn 0.5"),
    (4.0, (0.0, 0.0, 0.8), "turn 0.8"),
    (4.0, (0.0, 0.0, 1.0), "turn 1.0"),
    (4.0, (0.0, 0.0, -0.5), "turn right 0.5"),
    (4.0, (0.0, 0.2, 0.0), "sidestep 0.2"),
    (4.0, (0.0, 0.4, 0.0), "sidestep 0.4"),
    (4.0, (0.1, 0.0, 0.0), "forward 0.1"),
    (4.0, (0.2, 0.0, 0.0), "forward 0.2"),
    (4.0, (0.2, 0.0, 0.3), "forward 0.2 + turn 0.3"),
    (2.0, (0.0, 0.0, 0.0), "stand"),
]
SCRIPTS = {"standard": SCRIPT, "deadband": DEADBAND_SCRIPT}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-Unitree-G1")
    ap.add_argument("--script", choices=sorted(SCRIPTS), default="standard")
    ap.add_argument("--arms", choices=["default", "random"], default="default")
    ap.add_argument("--json", type=Path, help="write the results here")
    args = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    tw = env_cfg.commands["twist"]
    # We set the command ourselves: no heading control, no resampling, no special env types.
    tw.heading_command, tw.ranges.heading = False, None
    tw.rel_heading_envs = tw.rel_standing_envs = tw.rel_forward_envs = 0.0
    if hasattr(tw, "rel_world_envs"):
        tw.rel_world_envs = 0.0
    if hasattr(tw, "rel_turn_envs"):
        tw.rel_turn_envs = 0.0
    tw.resampling_time_range = (1e9, 1e9)
    if "arm_motion" in env_cfg.events:
        if args.arms == "default":
            env_cfg.events["arm_motion"].params["hold_default_prob"] = 1.0
        else:   # training's full-difficulty arm motion (play cfg has no curriculum)
            env_cfg.events["arm_motion"].params["range_frac"] = 0.6
    env_cfg.events.pop("push_robot", None)

    agent_cfg = load_rl_cfg(args.task)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    raw = env.unwrapped
    term = raw.command_manager.get_term("twist")
    robot = raw.scene["robot"]
    dt = raw.step_dt
    deadband = args.script == "deadband"

    env.reset()
    rows = []
    print(f"{'segment':24s} {'cmd vx vy wz':>18s} {'measured vx vy wz':>22s}")
    for seconds, cmd, label in SCRIPTS[args.script]:
        c = torch.tensor(cmd, device=device, dtype=torch.float32)
        term.vel_command_b[:] = c
        steps = int(seconds / dt)
        meas, fell = [], False
        for i in range(steps):
            term.vel_command_b[:] = c
            obs = env.get_observations()
            with torch.no_grad():
                _, _, dones, _ = env.step(policy(obs))
            if bool(dones[0]) and i < steps - 1:
                fell = True
            if i >= steps // 2:
                v = robot.data.root_link_lin_vel_b[0, :2].cpu().numpy()
                w = robot.data.root_link_ang_vel_b[0, 2].item()
                meas.append([v[0], v[1], w])
        m = np.mean(meas, axis=0)
        cm = np.array(cmd)
        active = np.abs(cm) > 1e-9
        responds = (bool(np.all(m[active] * np.sign(cm[active]) >= 0.5 * np.abs(cm[active]))) and not fell) if active.any() else None
        rows.append({"segment": label, "cmd": list(cmd), "measured": m.round(3).tolist(), "responds": responds, "reset": fell})
        verdict = "" if responds is None or not deadband else ("responds" if responds else "IGNORED / under 50 %")
        print(f"{label:24s} {cmd[0]:6.2f}{cmd[1]:6.2f}{cmd[2]:6.2f}   {m[0]:7.2f}{m[1]:7.2f}{m[2]:7.2f}"
              f"  {verdict}{'  (episode reset: fell?)' if fell else ''}")
    scored = [r for r in rows if r["responds"] is not None]
    summary = {"task": args.task, "checkpoint": args.checkpoint, "script": args.script, "arms": args.arms,
               "responds": sum(r["responds"] for r in scored), "commanded_segments": len(scored), "rows": rows}
    if deadband:
        print(f"deadband: responds to {summary['responds']}/{summary['commanded_segments']} small commands")
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
