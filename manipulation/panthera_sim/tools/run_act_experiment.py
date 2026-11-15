#!/usr/bin/env python3
"""Run ACT training followed by a fixed-seed rollout evaluation of its best model."""
import argparse
import datetime
import json
import subprocess
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop.dataset_contract import DEFAULT_CHECKPOINT, DEFAULT_DATASET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--action-steps', type=int, default=30)
    parser.add_argument('--rollout-eval-freq', type=int, default=10000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / 'run_status.json'
    def status(phase, **extra):
        value = {'phase': phase, 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(status_path)
    command = [sys.executable, str(ROOT / 'train_act.py'), '--dataset', str(args.dataset),
               '--output', str(args.output), '--steps', str(args.steps), '--batch-size', '12',
               '--workers', '4', '--checkpoint-freq', '2000', '--eval-freq', '2000',
               '--keep-checkpoints', '3', '--val-batches', '0', '--log-freq', '100',
               '--early-stop-patience', '0', '--rollout-eval-freq', str(args.rollout_eval_freq),
               '--rollout-action-steps', str(args.action_steps)]
    if args.resume:
        command += ['--resume', str(args.resume)]
    try:
        baseline = args.baseline or args.resume
        if baseline is not None and not (args.output / 'best_rollout/selection.json').is_file():
            status('baseline_evaluation')
            baseline_output = args.output / 'baseline_evaluation'
            subprocess.run([sys.executable, str(ROOT / 'tools/evaluate_act.py'),
                            '--checkpoint', str(baseline), '--dataset', str(args.dataset),
                            '--output', str(baseline_output), '--episodes', '8',
                            '--action-steps', str(args.action_steps)], cwd=ROOT, check=True)
            summary = json.loads((baseline_output / 'summary.json').read_text())
            destination = args.output / 'best_rollout'
            destination.mkdir(parents=True, exist_ok=True)
            for source_file in baseline.iterdir():
                if source_file.is_file() and source_file.suffix in ('.json', '.safetensors', '.pt'):
                    shutil.copy2(source_file, destination / source_file.name)
            (destination / 'inference.json').write_text(json.dumps({
                'temporal_ensemble': False, 'action_steps': args.action_steps}, indent=2) + '\n')
            (destination / 'selection.json').write_text(json.dumps({
                'source_checkpoint': str(baseline.resolve()), 'rates': summary['rates'], 'seed': 20261001}, indent=2) + '\n')
        status('training', command=command)
        subprocess.run(command, cwd=ROOT, check=True)
        status('evaluating')
        subprocess.run([sys.executable, str(ROOT / 'tools/evaluate_act.py'),
                        '--checkpoint', str(args.output / 'best_rollout'), '--dataset', str(args.dataset),
                        '--output', str(args.output / 'evaluation'), '--episodes', '8', '--seed', '20261201'], cwd=ROOT, check=True)
        status('complete')
    except BaseException as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
