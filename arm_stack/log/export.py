"""`arm export pace` and `arm export lerobot` (spec: Logging and data formats).

pace     command and feedback arrays plus sample rate, sweep range and
         amplitude: the inputs RP1's PACE recorder stores. Written as .npz with
         a .json sidecar.
lerobot  teleop sessions to LeRobot format (v2.x layout: meta/info.json,
         meta/episodes.jsonl, meta/tasks.jsonl, data/chunk-000/*.parquet and a
         dataset card naming the calibration version) for the
         WiscoHumanoidsLab Hugging Face org. Uploading is left to
         `huggingface-cli upload`, run by a person.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np

from .recorder import Session, command_array, load_session


def _aligned(session: Session):
    """Measured state and the command in force at each state sample."""
    st = session.state
    if not st:
        raise ValueError('session has no joint_state messages')
    cmd = command_array(session, 'q')
    if len(cmd['t_ns']) == 0:
        raise ValueError('session has no position commands')
    idx = np.searchsorted(cmd['t_ns'], st['t_ns'], side='right') - 1
    ok = idx >= 0
    return st['t_ns'][ok], cmd['q'][idx[ok]], st['q'][ok], st['qd'][ok], st['tau'][ok]


def export_pace(session_path: str, out_path: str) -> str:
    s = load_session(session_path)
    t_ns, q_cmd, q, qd, tau = _aligned(s)
    dt = float(np.median(np.diff(t_ns))) * 1e-9
    sweep_range = np.stack([q_cmd.min(axis=0), q_cmd.max(axis=0)], axis=1)
    amplitude = 0.5 * (sweep_range[:, 1] - sweep_range[:, 0])
    out_path = out_path if out_path.endswith('.npz') else out_path + '.npz'
    np.savez_compressed(out_path, t=(t_ns - t_ns[0]) * 1e-9, command=q_cmd, position=q, velocity=qd,
                        torque=tau, sample_rate_hz=1.0 / dt, sweep_range_rad=sweep_range,
                        amplitude_rad=amplitude)
    side = {'source': os.path.basename(session_path), 'calibration_version': s.meta['calibration_version'],
            'calibration_hash': s.meta.get('calibration_hash'), 'sample_rate_hz': 1.0 / dt,
            'joints': s.meta.get('joint_names'), 'sweep_range_rad': sweep_range.tolist(),
            'amplitude_rad': amplitude.tolist(), 'samples': int(len(t_ns))}
    with open(out_path[:-4] + '.json', 'w') as f:
        json.dump(side, f, indent=2)
    return out_path


def export_lerobot(session_paths: List[str], out_dir: str, repo_id: str = 'WiscoHumanoidsLab/arm01-teleop',
                   fps: int = 50, task: str = 'teleop') -> str:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise ImportError('arm export lerobot needs pyarrow (pip install pyarrow)') from e
    sessions = [load_session(p) for p in session_paths]
    versions = sorted({s.meta['calibration_version'] for s in sessions})
    names = sessions[0].meta.get('joint_names')
    os.makedirs(os.path.join(out_dir, 'meta'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'data', 'chunk-000'), exist_ok=True)
    total = 0
    episodes = []
    for e, s in enumerate(sessions):
        t_ns, q_cmd, q, _, _ = _aligned(s)
        t = (t_ns - t_ns[0]) * 1e-9
        grid = np.arange(0, t[-1], 1.0 / fps)
        idx = np.searchsorted(t, grid)
        idx = np.clip(idx, 0, len(t) - 1)
        n = len(grid)
        table = pa.table({
            'observation.state': [row.tolist() for row in q[idx].astype(np.float32)],
            'action': [row.tolist() for row in q_cmd[idx].astype(np.float32)],
            'timestamp': grid.astype(np.float32),
            'frame_index': np.arange(n, dtype=np.int64),
            'episode_index': np.full(n, e, dtype=np.int64),
            'index': np.arange(total, total + n, dtype=np.int64),
            'task_index': np.zeros(n, dtype=np.int64),
        })
        pq.write_table(table, os.path.join(out_dir, 'data', 'chunk-000', 'episode_%06d.parquet' % e))
        episodes.append({'episode_index': e, 'tasks': [task], 'length': n,
                         'calibration_version': s.meta['calibration_version']})
        total += n
    dim = len(names) if names else int(q.shape[1])
    info = {
        'codebase_version': 'v2.1', 'robot_type': sessions[0].meta.get('arm', 'arm01'),
        'total_episodes': len(sessions), 'total_frames': total, 'total_tasks': 1, 'fps': fps,
        'chunks_size': 1000, 'total_chunks': 1, 'splits': {'train': '0:%d' % len(sessions)},
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'features': {
            'observation.state': {'dtype': 'float32', 'shape': [dim], 'names': names},
            'action': {'dtype': 'float32', 'shape': [dim], 'names': names},
            'timestamp': {'dtype': 'float32', 'shape': [1]},
            'frame_index': {'dtype': 'int64', 'shape': [1]},
            'episode_index': {'dtype': 'int64', 'shape': [1]},
            'index': {'dtype': 'int64', 'shape': [1]},
            'task_index': {'dtype': 'int64', 'shape': [1]},
        },
        'calibration_versions': versions,
    }
    with open(os.path.join(out_dir, 'meta', 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(out_dir, 'meta', 'episodes.jsonl'), 'w') as f:
        f.writelines(json.dumps(e) + '\n' for e in episodes)
    with open(os.path.join(out_dir, 'meta', 'tasks.jsonl'), 'w') as f:
        f.write(json.dumps({'task_index': 0, 'task': task}) + '\n')
    with open(os.path.join(out_dir, 'README.md'), 'w') as f:
        f.write('---\nlicense: other\ntags: [LeRobot, arm01]\n---\n\n# %s\n\n' % repo_id)
        f.write('Teleop sessions from %s at %d fps.\n\n' % (info['robot_type'], fps))
        f.write('Calibration version(s): %s. Calibration hash(es): %s.\n' % (
            ', '.join(str(v) for v in versions),
            ', '.join(sorted({str(s.meta.get('calibration_hash')) for s in sessions}))))
        f.write('Joint config hash: %s. Git sha: %s.\n' % (sessions[0].meta.get('joint_config_hash'),
                                                         sessions[0].meta.get('git_sha')))
    return out_dir
