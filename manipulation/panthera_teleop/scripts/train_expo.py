#!/usr/bin/env python3
"""Train the EXPO editor/critic from verified replay, or exercise it in a synthetic bandit.

This never opens robot devices, sends inference requests, or launches AWS jobs.
The synthetic mode is an algorithm/compute check, not a trained Panthera policy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--replay', type=Path)
    source.add_argument('--synthetic', action='store_true')
    p.add_argument('--output', required=True, type=Path, help='New directory; existing outputs are preserved')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--utd-ratio', type=int, default=20)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--prefix-length', type=int, default=17,
                   help='Expected executed chunk length (default 17); old C=1 replay requires explicit 1')
    p.add_argument('--minimum-free-gib', type=float, default=2.)
    return p


def main():
    args = parser().parse_args()
    if min(args.batch_size, args.steps, args.utd_ratio, args.hidden_dim, args.save_every) < 1:
        raise ValueError('Batch size, steps, update ratio, hidden size, and save interval must be positive')
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; previous runs are preserved')
    if not 1 <= args.prefix_length <= 50:
        raise ValueError('prefix-length must be between 1 and the frozen pi0.5 horizon of 50')
    from infra.rl.resources import require_idle_gpu
    # Acquire the guard before importing torch or constructing a CUDA object.
    with require_idle_gpu(args.device, minimum_free_gib=args.minimum_free_gib,
                          disk_path=args.output) as resources:
        import numpy as np
        import torch
        from infra.rl.learner import EXPOConfig, EXPOLearner

        rng = np.random.default_rng(args.seed)
        replay = metadata = None
        training_objective = reward_semantics = None
        if args.synthetic:
            config = EXPOConfig(obs_dim=2318, action_dim=14, chunk_size=args.prefix_length,
                                hidden_dim=args.hidden_dim, utd_ratio=args.utd_ratio,
                                initial_temperature=.001, learn_temperature=False, seed=args.seed)
            mode = 'synthetic_bandit_algorithm_check_NOT_PANTHERA'
        else:
            from infra.rl.replay import REWARD_SEMANTICS, grouped_split, load_replay
            replay, metadata = load_replay(args.replay)
            if metadata['spec']['C'] != args.prefix_length:
                raise ValueError('Replay chunk length differs from --prefix-length; choose the matching explicit value')
            training_objective = metadata['training_objective']
            reward_semantics = REWARD_SEMANTICS[training_objective]
            split = metadata.get('splits')
            if split:
                train_indices = np.asarray(split['train_indices'], dtype=np.int64)
                evaluation_indices = np.asarray(split['eval_indices'], dtype=np.int64)
            else:
                train_indices, evaluation_indices = grouped_split(replay['source_episode_ids'], seed=args.seed)
            if not len(train_indices):
                raise ValueError('Replay has no training episodes after source-grouped holdout')
            from infra.rl.contracts import ActionNormalizer
            normalizer = ActionNormalizer.from_stats(metadata['normalization']['action'])
            native_lower = np.zeros(14, dtype=np.float32)
            native_upper = np.full(14, 2., dtype=np.float32)
            normalized_lower = normalizer.normalize(native_lower)
            normalized_upper = normalizer.normalize(native_upper)
            action_lower, action_upper = [None] * 14, [None] * 14
            for index in (6, 13):
                action_lower[index] = float(normalized_lower[index])
                action_upper[index] = float(normalized_upper[index])
            config = EXPOConfig(obs_dim=replay['obs'].shape[1], action_dim=14,
                                chunk_size=replay['actions'].shape[1],
                                hidden_dim=args.hidden_dim, utd_ratio=args.utd_ratio, seed=args.seed,
                                action_lower=action_lower, action_upper=action_upper)
            mode = 'frozen_base_expo_residual_rl'

        # No output may overwrite either the running SFT job or earlier RL work.
        args.output.mkdir(parents=True, exist_ok=False)
        report = {'mode': mode, 'base_policy_updated': False, 'synthetic': args.synthetic,
                  'config': asdict(config), 'batch_size': args.batch_size, 'requested_steps': args.steps,
                  'resources_before_cuda': resources, 'replay': str(args.replay) if args.replay else None,
                  'replay_metadata': metadata, 'seed': args.seed,
                  'training_objective': training_objective, 'reward_semantics': reward_semantics}
        if not args.synthetic:
            report['training_transitions'] = len(train_indices)
            report['held_out_transitions'] = len(evaluation_indices)
            report['evaluation_note'] = 'Source-grouped holdout is excluded from optimization; robot success is not estimated from critic loss.'
        (args.output / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        learner = EXPOLearner(config, device=args.device)
        if not args.synthetic:
            learner.checkpoint_metadata = {
                'training_objective': training_objective, 'reward_semantics': reward_semantics}
        if str(args.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()

        def synthetic_batch(size):
            # One-decision contextual control over the configured chunk: genuine reward, known
            # stochastic base policy, and bounded random behavior exploration.
            obs = np.zeros((size, config.obs_dim), dtype=np.float32)
            goal = np.full((size, config.chunk_size, 14), .03, dtype=np.float32)
            obs[:, :14] = goal[:, 0] / .05
            base = rng.normal(0, .004, (size, 8, config.chunk_size, 14)).astype(np.float32)
            actions = rng.uniform(-.055, .055, (size, config.chunk_size, 14)).astype(np.float32)
            reward = -np.mean(((actions - goal) / .05) ** 2, axis=(1, 2))
            return dict(obs=obs, next_obs=obs.copy(), actions=actions, base_actions=base,
                        next_base_actions=base.copy(), reward_return=reward,
                        bootstrap_discount=np.zeros(size, dtype=np.float32))

        fixed = synthetic_batch(256) if args.synthetic else None

        def evaluate_synthetic():
            with torch.no_grad():
                result = learner.select_actions(fixed['obs'], fixed['base_actions'], deterministic=True)
                action = result['actions'].detach().cpu().numpy()
            return float(-np.mean(((action - .03) / .05) ** 2))

        if args.synthetic:
            report['synthetic_reward_before'] = evaluate_synthetic()
            report['synthetic_base_reward'] = float(-np.mean(((fixed['base_actions'][:, 0] - .03) / .05) ** 2))
        started = time.monotonic()
        log = args.output / 'metrics.jsonl'
        with log.open('x') as output:
            for step in range(1, args.steps + 1):
                if args.synthetic:
                    batch = synthetic_batch(args.batch_size)
                else:
                    indices = rng.choice(train_indices, size=args.batch_size, replace=True)
                    batch = {key: value[indices] for key, value in replay.items()}
                metrics = learner.update(batch)
                row = {'step': step, 'elapsed_s': time.monotonic()-started, **metrics}
                output.write(json.dumps(row, allow_nan=False) + '\n')
                output.flush()
                if step == 1 or step % 10 == 0 or step == args.steps:
                    print(json.dumps(row, allow_nan=False), flush=True)
                if step % args.save_every == 0 or step == args.steps:
                    learner.save(args.output / f'checkpoint-{step:06d}.pt')
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize()
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
            report['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
        report.update(completed_steps=args.steps, elapsed_s=time.monotonic()-started,
                      checkpoint=f'checkpoint-{args.steps:06d}.pt')
        if args.synthetic:
            report['synthetic_reward_after'] = evaluate_synthetic()
            report['synthetic_reward_improved'] = report['synthetic_reward_after'] > report['synthetic_reward_before']
        (args.output / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
