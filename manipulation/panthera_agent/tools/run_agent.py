#!/usr/bin/env python3
"""S4: one instruction through the LLM agent, in the sim.

    .venv/bin/python tools/run_agent.py "stack the red cube on the green cube"
    .venv/bin/python tools/run_agent.py --task stack_three --seed 4          # a test task, judged from ground truth
    .venv/bin/python tools/run_agent.py --task unstack --segmenter server    # real segmentation (VM tunnel up)
    python tools/run_agent.py --robot rig "put the red brick on the blue one"  # the real arm (R3; rig computer)

Needs ANTHROPIC_API_KEY (the repo's .env). PANTHERA_AGENT_MODEL / PANTHERA_AGENT_EFFORT or --model / --effort pick
the model (default claude-opus-5-5, medium effort). The run is logged to runs/agent_<time>/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.build import load_config, load_dotenv, make_segmenter, make_skills  # noqa: E402
from core.camera import pose_error  # noqa: E402
from core.tasks import TASKS  # noqa: E402
from llm.claude import ClaudePlanner  # noqa: E402
from llm.loop import AgentRun  # noqa: E402
from robots.sim_robot import SimRobot  # noqa: E402


def run_one(instruction, task, seed, config, segmenter, planner, log_dir, camera_error=None, max_turns=40):
    """Set up the sim (task scene and set-up if given), run the agent, judge the result if a task."""
    import numpy as np
    scene = task.scene if task else config['scene']
    robot = SimRobot({**config, 'scene': scene}, seed=seed)
    if task and task.setup:
        task.setup(robot)
    if camera_error:
        robot.camera_errors['overhead'] = pose_error(camera_error[0], camera_error[1], np.random.default_rng(seed))
    skills = make_skills(robot, make_segmenter(segmenter, config), config)
    outcome = AgentRun(skills, planner, config, log_dir, max_turns=max_turns).run(instruction)
    if task:
        outcome['metrics'] = task.check(robot.truth_objects())
        outcome['judged_success'] = bool(outcome['metrics']['success'])
    with open(pathlib.Path(log_dir) / 'outcome.json', 'w') as f:
        json.dump(dict(instruction=instruction, task=task.name if task else None, seed=seed, scene=scene,
                       segmenter=segmenter, **outcome), f, indent=2, default=str)
    return outcome


def run_rig(args, config):
    """The real arm: the perception server for masks, no ground truth to judge by."""
    from robots.real_robot import RealRobot
    log_dir = ROOT / 'runs' / f'rig_{time.strftime("%Y%m%d_%H%M%S")}'
    planner = ClaudePlanner(model=args.model, effort=args.effort)
    robot = RealRobot(config)
    try:
        skills = make_skills(robot, make_segmenter('server', config), config)
        print(f'{planner.model} ({planner.effort}): {args.instruction}\nlog: {log_dir}', flush=True)
        outcome = AgentRun(skills, planner, config, log_dir, max_turns=args.max_turns).run(args.instruction)
        print(json.dumps(outcome, indent=2, default=str))
    finally:
        robot.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('instruction', nargs='?', help='what to do (or --task)')
    ap.add_argument('--task', choices=sorted(TASKS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--segmenter', default='sim_truth', choices=['sim_truth', 'server'])
    ap.add_argument('--camera-error', type=float, nargs=2, metavar=('METRES', 'DEGREES'))
    ap.add_argument('--model')
    ap.add_argument('--effort', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    ap.add_argument('--max-turns', type=int, default=40)
    ap.add_argument('--robot', choices=['sim', 'rig'], default='sim')
    ap.add_argument('--config', help='default: config/sim.yaml, or config/rig.yaml with --robot rig')
    args = ap.parse_args()
    if not args.instruction and not args.task:
        ap.error('give an instruction or --task')
    load_dotenv()
    config = load_config(args.config or f'config/{args.robot}.yaml')
    if args.robot == 'rig':
        if args.task:
            ap.error('--task is for the sim (it is judged from the simulator\'s ground truth)')
        return run_rig(args, config)
    task = TASKS[args.task] if args.task else None
    instruction = args.instruction or task.instruction
    log_dir = ROOT / 'runs' / f'agent_{time.strftime("%Y%m%d_%H%M%S")}'
    planner = ClaudePlanner(model=args.model, effort=args.effort)
    print(f'{planner.model} ({planner.effort}): {instruction}\nlog: {log_dir}', flush=True)
    outcome = run_one(instruction, task, args.seed, config, args.segmenter, planner, log_dir, args.camera_error,
                      args.max_turns)
    print(json.dumps(outcome, indent=2, default=str))


if __name__ == '__main__':
    main()
