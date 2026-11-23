#!/usr/bin/env python3
"""S5: the LLM agent on a fixed test set (tasks x seeds), success judged from the simulator's ground truth.

    .venv/bin/python tools/evaluate.py --tasks stack_red_on_green unstack --seeds 5       # 10 runs
    .venv/bin/python tools/evaluate.py --seeds 3 --camera-error 0.005 1.0                  # every task, miscalibrated

Each run is logged under runs/eval_<time>/<task>_seed<N>/; summary.json has the success rate per task, how often the
model's own claim agreed with the ground truth, turns and token use. Every run calls the API: about 10-20 turns,
each with one or two images.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.build import SEGMENTERS, load_config, load_dotenv  # noqa: E402
from core.tasks import TASKS  # noqa: E402
from llm.claude import ClaudePlanner  # noqa: E402
from tools.run_agent import run_one  # noqa: E402


def tokens(log_dir: pathlib.Path) -> dict:
    """Summed token usage of a run."""
    total = {}
    for line in (log_dir / 'steps.jsonl').read_text().splitlines():
        for k, v in (json.loads(line).get('usage') or {}).items():
            total[k] = total.get(k, 0) + (v or 0)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tasks', nargs='+', choices=sorted(TASKS), default=sorted(TASKS))
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--first-seed', type=int, default=100)      # away from the seeds used while developing
    ap.add_argument('--segmenter', default='sim_truth', choices=list(SEGMENTERS))
    ap.add_argument('--camera-error', type=float, nargs=2, metavar=('METRES', 'DEGREES'))
    ap.add_argument('--model')
    ap.add_argument('--effort', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    ap.add_argument('--max-turns', type=int, default=40)
    ap.add_argument('--planner', choices=['claude', 'oracle'], default='claude',
                    help='oracle: the scripted sim stand-in (llm/scripted.py ORACLES; free, for the pipeline itself)')
    ap.add_argument('--fresh-memory', action='store_true',
                    help='each run starts with no learned objects (measures learning on first sight)')
    ap.add_argument('--policy', action='store_true', help='insert uses the trained local policy (models/insert)')
    args = ap.parse_args()
    load_dotenv()
    config = load_config()
    planner = ClaudePlanner(model=args.model, effort=args.effort) if args.planner == 'claude' else None
    policies = None
    if args.policy:
        from learning.policy import load
        pol = load('insert')
        policies = {'insert': pol.act} if pol is not None else None
    out = ROOT / 'runs' / f'eval_{time.strftime("%Y%m%d_%H%M%S")}'
    rows = []
    if args.planner == 'oracle':
        from llm.scripted import ORACLES
        skipped = [n for n in args.tasks if n not in ORACLES]
        if skipped:
            print(f'no oracle script for {", ".join(skipped)}: skipped', flush=True)
        args.tasks = [n for n in args.tasks if n in ORACLES]
    for name in args.tasks:
        task = TASKS[name]
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            log_dir = out / f'{name}_seed{seed}'
            try:
                memory = None
                if args.fresh_memory:
                    import tempfile
                    from perception.memory import ObjectMemory
                    memory = ObjectMemory(tempfile.mkdtemp(prefix='memory_'))
                p = planner
                if p is None:
                    from llm.scripted import ORACLES, OracleScript
                    p = (lambda steps: (lambda skills: OracleScript(skills, steps)))(ORACLES[name])
                o = run_one(task.instruction, task, seed, config, args.segmenter, p, log_dir,
                            args.camera_error, args.max_turns, memory=memory, policies=policies)
            except Exception as e:                       # a crashed run counts as a failure and is recorded
                o = dict(done=False, success=False, judged_success=False, turns=0, stop=f'error: {e}')
            row = dict(task=name, seed=seed, judged=o.get('judged_success', False), claimed=o.get('success', False),
                       turns=o.get('turns'), stop=o.get('stop'), tokens=tokens(log_dir) if log_dir.exists() else {})
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {}
    for name in args.tasks:
        r = [x for x in rows if x['task'] == name]
        summary[name] = dict(success=sum(x['judged'] for x in r), runs=len(r),
                             claim_agrees=sum(x['judged'] == x['claimed'] for x in r),
                             mean_turns=sum(x['turns'] or 0 for x in r) / max(1, len(r)))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(dict(model=planner.model if planner else 'oracle (scripted)',
                                                      effort=planner.effort if planner else None,
                                                      segmenter=args.segmenter, camera_error=args.camera_error,
                                                      per_task=summary, runs=rows), indent=2))
    for name, s in summary.items():
        print(f'{name}: {s["success"]}/{s["runs"]} (the model\'s claim matched the ground truth '
              f'{s["claim_agrees"]}/{s["runs"]}, '
              f'{s["mean_turns"]:.1f} turns)')
    print(out)


if __name__ == '__main__':
    main()
