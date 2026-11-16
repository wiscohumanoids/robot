#!/usr/bin/env python3
"""Evaluate grasp/lift/stack success over repeatable random ACT rollouts."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop.dataset_contract import DEFAULT_CHECKPOINT, DEFAULT_DATASET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT / 'best')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/act/evaluation_30hz')
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--seconds', type=float, default=15.)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--action-steps', type=int)
    parser.add_argument('--save-failures', action='store_true', help='retain physical traces for failed sustained pickups')
    parser.add_argument('--temporal-ensemble', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.episodes < 1 or args.seconds <= 0:
        parser.error('episodes and seconds must be positive')
    contract = args.checkpoint / 'deployment.json'
    fps = json.loads(contract.read_text())['fps'] if contract.is_file() else 10
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(args.episodes):
        seed = args.seed + i
        path = args.output / f'seed_{seed}.json'
        inference_args = []
        if args.action_steps is not None:
            inference_args += ['--action-steps', str(args.action_steps)]
        if args.temporal_ensemble is not None:
            inference_args += ['--temporal-ensemble' if args.temporal_ensemble else '--no-temporal-ensemble']
        trace_path = args.output / f'seed_{seed}_trace.npz'
        if args.save_failures:
            inference_args += ['--trace', str(trace_path)]
        with (args.output / f'seed_{seed}.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'rollout_act.py'),
                            '--checkpoint', str(args.checkpoint), '--dataset', str(args.dataset),
                            '--seed', str(seed), '--steps', str(round(args.seconds * fps)),
                            '--report', str(path), '--no-display', '--no-realtime', *inference_args],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = json.loads(path.read_text())
        if args.save_failures and record['milestones']['sustained_pickup']:
            trace_path.unlink(missing_ok=True)
        records.append(record)
        print(json.dumps({'episode': i+1, 'seed': seed, **record['milestones']}), flush=True)
    summary = {'checkpoint': str(args.checkpoint.resolve()), 'fps': fps,
               'episodes': args.episodes, 'seconds': args.seconds,
               'rates': {key: sum(r['milestones'][key] for r in records) / len(records)
                         for key in ('grasped', 'lifted', 'two_stacked', 'success', 'sustained_pickup')},
               'records': records}
    summary['rates']['grasp_and_lift'] = sum(
        r['milestones']['grasped'] and r['milestones']['lifted'] for r in records) / len(records)
    summary['rates']['pickup_seconds_capped_mean'] = sum(
        r['pickup_seconds'] if r['pickup_seconds'] is not None else args.seconds for r in records) / len(records)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
