"""Successful local-skill episodes as a LeRobot dataset, for an ACT fine-tune with LeRobot's own trainer on the VM.

    python -m learning.export_lerobot --skill insert --out ~/hsim_data/panthera_agent/lerobot/insert
    lerobot-train --policy.type=act --dataset.repo_id=local/panthera_insert --dataset.root=<out> \\
        --output_dir=~/hsim_runs/act_insert --steps=20000

The profile manipulation/profiles/panthera_local.yaml states the contract: one wrist camera at 64 x 64, a 5-number
state (TCP relative to the handoff, yaw turned, jaw width), a 5-number action (relative step and done), 10 Hz.
ACT's own checkpoints then run through `learning/policy.py`-compatible wrappers only after a conversion; the small
LocalPolicy is what the skills load today. Needs LeRobot installed (the VM's pi0.5 environment has it at a pinned
commit; this Mac's environment does not), and its dataset API has changed between versions: written against
LeRobotDataset.create / add_frame / save_episode (LeRobot 0.3-0.4).
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from learning.episodes import EpisodeStore


def export(skill: str, out: pathlib.Path, repo_id: str | None = None) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    store = EpisodeStore(skill)
    eps = store.filter(success=True)
    features = {
        'observation.images.wrist': {'dtype': 'image', 'shape': (64, 64, 3), 'names': ['height', 'width', 'channel']},
        'observation.state': {'dtype': 'float32', 'shape': (5,), 'names': ['dx', 'dy', 'dz', 'dyaw', 'width']},
        'action': {'dtype': 'float32', 'shape': (5,), 'names': ['dx', 'dy', 'dz', 'dyaw', 'done']},
    }
    ds = LeRobotDataset.create(repo_id=repo_id or f'local/panthera_{skill}', fps=10, root=str(out), features=features,
                               robot_type='panthera_ht_local', use_videos=False)
    frames = 0
    for e in eps:
        z = store.load(e['id'])
        tgt = z['action'].copy()
        has = ~np.isnan(z['expert']).any(axis=1)
        tgt[has] = z['expert'][has]
        for t in range(len(tgt)):
            ds.add_frame({'observation.images.wrist': z['wrist'][t], 'observation.state': z['state'][t],
                          'action': tgt[t].astype(np.float32), 'task': f'{skill} (local skill)'})
            frames += 1
        ds.save_episode()
    return dict(episodes=len(eps), frames=frames, root=str(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skill', default='insert')
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    print(export(a.skill, a.out))


if __name__ == '__main__':
    main()
