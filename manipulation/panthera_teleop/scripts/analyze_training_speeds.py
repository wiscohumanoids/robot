#!/usr/bin/env python3
"""Measure saved training motions without connecting to robot hardware.

Run with work/postprocess-venv/bin/python. Only numeric Parquet columns are
loaded; embedded camera images are not decoded. No safety settings are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


GROUPS = {'joints': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
          'grippers': [6, 13]}
PERCENTILES = [50, 95, 99, 99.9, 100]


def quantiles(values):
    return dict(zip(map(str, PERCENTILES),
                    np.percentile(values, PERCENTILES).tolist()))


def summarize(values, names, episodes, frames, identifiers):
    absolute = np.abs(values)
    result = {'groups': {}, 'per_axis': {}}
    for group, axes in GROUPS.items():
        subset = absolute[:, axes]
        result['groups'][group] = {
            'axis_sample_percentiles_rad_s': quantiles(subset),
            'frame_max_percentiles_rad_s': quantiles(subset.max(axis=1))}
    for axis, name in enumerate(names):
        row = int(np.argmax(absolute[:, axis]))
        episode = int(episodes[row])
        result['per_axis'][name] = {
            'percentiles_rad_s': quantiles(absolute[:, axis]),
            'peak': {'signed_speed_rad_s': float(values[row, axis]),
                     'episode_index': episode, 'source_id': identifiers[episode],
                     'frame_index': int(frames[row])}}
    return result


def analyze(root, contract_path):
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    manifest = json.loads((root / 'processing/manifest.json').read_text())
    publication = json.loads((root / 'publication.json').read_text())
    contract = json.loads(contract_path.read_text())
    assert publication['repo_id'] == contract['dataset_repo']
    assert publication['commit'] == contract['dataset_revision']
    assert info['fps'] == manifest['fps'] == contract['fps']
    assert info['splits']['train'] == f"0:{info['total_episodes']}"
    names = info['features']['action']['names']
    assert len(names) == 14 and names[6] == 'left.gripper_rad'
    assert names[13] == 'right.gripper_rad'
    columns = ['observation.state', 'action', 'observation.velocity',
               'episode_index', 'frame_index', 'index']
    files = sorted((root / 'data').rglob('*.parquet'))
    tables = [pq.read_table(p, columns=columns).to_pydict() for p in files]
    data = {key: np.concatenate([np.asarray(t[key]) for t in tables])
            for key in columns}
    order = np.argsort(data['index'])
    data = {key: value[order] for key, value in data.items()}
    assert len(order) == info['total_frames']
    assert np.array_equal(data['index'], np.arange(len(order)))
    for key in columns[:3]:
        assert data[key].shape == (len(order), 14)
        assert np.isfinite(data[key]).all()
    identifiers = {e['episode_index']: e['id'] for e in manifest['episodes']}
    assert set(data['episode_index']) == set(identifiers)
    for episode in identifiers:
        frames = data['frame_index'][data['episode_index'] == episode]
        assert np.array_equal(frames, np.arange(len(frames)))
    fps = info['fps']
    required = (data['action'] - data['observation.state']) * fps
    same_episode = np.diff(data['episode_index']) == 0
    consecutive = np.diff(data['frame_index'])[same_episode]
    assert np.all(consecutive == 1)
    # Do not bridge episode boundaries when measuring action-to-action motion.
    action_change = np.diff(data['action'], axis=0)[same_episode]
    prior_rows = np.flatnonzero(same_episode)
    report = {
        'dataset_root': str(root.resolve()), 'dataset_repo': publication['repo_id'],
        'dataset_revision': publication['commit'], 'episodes': info['total_episodes'],
        'frames': len(order), 'fps': fps, 'units': 'rad/s',
        'method': {
            'next_pose_required_speed': '(action - observation.state) * fps',
            'consecutive_action_speed': 'diff(action) * fps within each episode',
            'reported_velocity': 'observation.velocity from motor feedback',
            'percentiles': 'absolute values; joints and grippers separated',
            'clipping': 'independent one-timestep demand, not a closed-loop rollout'},
        'numeric_checks_passed': True,
        'source_files_sha256': {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [info_path, root / 'processing/manifest.json', root / 'publication.json']},
        'next_pose_required_speed': summarize(required, names, data['episode_index'],
                                               data['frame_index'], identifiers),
        'reported_velocity': summarize(data['observation.velocity'], names,
                                       data['episode_index'], data['frame_index'], identifiers),
        'consecutive_action_speed': summarize(action_change * fps, names,
                                               data['episode_index'][prior_rows],
                                               data['frame_index'][prior_rows], identifiers),
        'cap_comparison': {}}
    for cap in [.3, .5, .6, .8, 1., 1.5, 2., 3., 3.2, 4., 5., 6., 9.]:
        report['cap_comparison'][str(cap)] = {}
        for group, axes in GROUPS.items():
            exceeded = np.abs(required[:, axes]) > cap
            report['cap_comparison'][str(cap)][group] = {
                'axis_sample_percent': float(exceeded.mean() * 100),
                'frame_percent': float(exceeded.any(axis=1).mean() * 100)}
    jump = contract['max_target_jump_rad']
    report['target_jump_check'] = {
        'limit_rad': jump,
        'consecutive_action_transitions': len(action_change),
        'max_consecutive_action_jump_rad': float(np.abs(action_change).max()),
        'transitions_exceeding_limit': int((np.abs(action_change) > jump).any(axis=1).sum()),
        'episodes_with_exceeding_transition': len(set(data['episode_index'][prior_rows][
            (np.abs(action_change) > jump).any(axis=1)]))}
    return report


def full_rate_feedback(root, trimmed_parent):
    """Check feedback in exactly the cropped source episodes used by training."""
    manifest = json.loads((root / 'processing/manifest.json').read_text())
    names = manifest['state_names']
    velocities, episodes, sequences = [], [], []
    identifiers = {}
    max_pose_derivative = np.zeros(14)
    for episode in manifest['episodes']:
        identifier = episode['id']
        paths = list(trimmed_parent.glob(f'trimmed-full-rate*/{identifier}/states.jsonl'))
        assert len(paths) == 1, (identifier, paths)
        rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
        assert len(rows) == episode['trimmed_states']
        def vector(row, field, gripper_field):
            return [v for side in ('left', 'right') for v in
                    row['arms'][side][field] + [row['arms'][side]['gripper'][gripper_field]]]
        velocity = np.array([vector(row, 'velocity_rad_s', 'velocity') for row in rows])
        pose = np.array([vector(row, 'position_rad', 'position') for row in rows])
        dt = np.diff(np.array([row['monotonic_ns'] for row in rows], dtype=np.int64)) / 1e9
        assert np.all(dt > 0) and np.isfinite(velocity).all() and np.isfinite(pose).all()
        max_pose_derivative = np.maximum(max_pose_derivative,
                                        np.abs(np.diff(pose, axis=0) / dt[:, None]).max(axis=0))
        velocities.append(velocity)
        episodes.extend([episode['episode_index']] * len(rows))
        sequences.extend(row['sequence'] for row in rows)
        identifiers[episode['episode_index']] = identifier
    result = summarize(np.concatenate(velocities), names, np.array(episodes),
                       np.array(sequences), identifiers)
    # Source sequence numbers are not 20 Hz frame indices.
    for axis in result['per_axis'].values():
        peak = axis['peak']
        peak['source_sequence'] = peak.pop('frame_index')
    result['samples'] = len(episodes)
    result['method'] = 'Motor velocity feedback across all trimmed full-rate source states'
    result['host_timestamp_pose_derivative_max_rad_s'] = dict(zip(names, max_pose_derivative.tolist()))
    result['pose_derivative_note'] = (
        'Adjacent host receive times need not match individual motor update times. '
        'Encoder quantization and repeated readings make these derivatives noisy; '
        'use 20 Hz pose differences for comparison with policy timesteps.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--contract', type=Path, default=Path('infra/inference/contract.json'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--trimmed-parent', type=Path,
                        help='Optionally check trimmed-full-rate* source directories under this parent')
    args = parser.parse_args()
    report = analyze(args.dataset, args.contract)
    if args.trimmed_parent:
        report['full_rate_reported_velocity'] = full_rate_feedback(args.dataset, args.trimmed_parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({
        'episodes': report['episodes'], 'frames': report['frames'],
        'next_pose_required_speed': report['next_pose_required_speed']['groups'],
        'target_jump_check': report['target_jump_check'],
        'report': str(args.output.resolve())}, indent=2))
