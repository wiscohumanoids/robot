#!/usr/bin/env python3
"""Practice a local skill on the real arm (or, with --robot sim, through the skills without privileged help).

    python tools/practice_rig.py eval   --episodes 10 --peg "red peg" --hole "orange block"
    python tools/practice_rig.py demos  --episodes 20 ...       # the person drives with the keyboard (learning/teleop.py)
    python tools/practice_rig.py dagger --rounds 2 --episodes 10 ...
    python tools/practice_rig.py rl     --episodes 60 ...       # with the person's corrections and the watchdog

The environment is learning/envs.py SkillEnv: the insert skill's own handoff (find, line up, lower), the policy's
steps, success from the arm's pose (the peg's bottom below the hole's top). Between attempts the peg is lifted back
out to the handoff; when something goes wrong the supervisor resets through the agent (Claude, with
--reset-instruction) or asks the person. Every attempt is an episode in data/episodes/insert/, with full-resolution
final images for Claude's labels (python -m learning.label), synced to the VM for the daily cycle
(tools/learning_vm.sh push).

Before the first session: tools/rig_check.py passes, the peg and the hole are learned (the agent's learn tool), and
someone is at the stop button. The rig's speed limits (config/rig.yaml motion, safety) apply to every step.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.build import config_for_scene, load_config, load_dotenv, make_segmenter, make_skills  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['demos', 'dagger', 'rl', 'eval'])
    ap.add_argument('--robot', choices=['rig', 'sim'], default='rig')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--peg', default='red peg')
    ap.add_argument('--hole', default='orange block')
    ap.add_argument('--extra-mm', type=float, default=3.0, help='extra random handoff error for practice')
    ap.add_argument('--extra-deg', type=float, default=4.0)
    ap.add_argument('--policy', default='latest')
    ap.add_argument('--reset-instruction', help='an instruction the agent runs (Claude) to put the scene back')
    ap.add_argument('--no-keyboard', action='store_true')
    a = ap.parse_args()
    load_dotenv()
    if a.robot == 'rig':
        from robots.real_robot import RealRobot
        cfg = load_config('config/rig.yaml')
        robot = RealRobot(cfg)
    else:
        from robots.sim_robot import SimRobot
        cfg = config_for_scene(load_config(), 'agent_peg_tight.xml')
        robot = SimRobot(cfg, seed=0)
    seg = make_segmenter('dino_local', cfg)
    skills = make_skills(robot, seg, cfg, seg.memory)
    from learning.envs import SkillEnv
    from learning.practice import Practice
    from learning.supervisor import AgentSupervisor
    env = SkillEnv(skills, a.peg, a.hole, a.extra_mm / 1000, a.extra_deg)
    teleop = None
    if not a.no_keyboard and sys.stdin.isatty():
        from learning.teleop import KeyboardTeleop
        teleop = KeyboardTeleop()
    planner = None
    if a.reset_instruction:
        from llm.claude import ClaudePlanner
        planner = ClaudePlanner()
    sup = AgentSupervisor(skills, planner, a.reset_instruction, teleop=teleop)
    pr = Practice(env, sup, on_event=lambda k, d: print(json.dumps(d, default=str), flush=True)
                  if k == 'practice' else None)
    try:
        if a.stage == 'demos':
            pr.demos(a.episodes)
        elif a.stage == 'dagger':
            pr.dagger(a.rounds, a.episodes, a.steps)
        elif a.stage == 'rl':
            pr.rl(a.episodes)
        else:
            pr.evaluate(a.episodes, a.policy, save_it=True)
    finally:
        if teleop is not None:
            teleop.close()
        if hasattr(robot, 'stop'):
            robot.stop()


if __name__ == '__main__':
    main()
