#!/usr/bin/env python3
"""Find identical ACT observations with contradictory future-action labels."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.dataset as pads


def collision_report(states,actions,images,action_std,chunk=30):
    groups=defaultdict(list)
    for index,state in enumerate(states):
        digest=hashlib.sha256(np.asarray(state,dtype=np.float32).tobytes())
        for camera in images:digest.update(camera[index]['bytes'])
        groups[digest.digest()].append(index)
    collisions=[];bound_sum=0.;valid_count=0
    for index in range(len(states)):
        valid_count+=min(chunk,len(states)-index)*actions.shape[1]
    for ids in groups.values():
        if len(ids)<2:continue
        chunks=np.stack([actions[np.minimum(np.arange(i,i+chunk),len(actions)-1)] for i in ids])
        span=np.ptp(chunks,axis=0)
        if not span.any():continue
        scaled=chunks/action_std
        for horizon in range(chunk):
            active=np.array(ids)+horizon<len(actions)
            values=scaled[active,horizon]
            if len(values):bound_sum+=float(np.abs(values-np.median(values,axis=0)).sum())
        collisions.append(dict(frames=ids,max_arm_chunk_difference_rad=float(span[:,:6].max()),
                               max_gripper_chunk_difference_mm=float(span[:,6].max()*1000),
                               max_first_action_joint_difference_rad=float(span[0,:6].max())))
    return dict(frames=len(states),contradictory_groups=collisions,
                contradictory_frames=sum(len(g['frames']) for g in collisions),
                whole_episode_l1_lower_bound_from_exact_duplicates=bound_sum/valid_count)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    stats=json.loads((a.dataset/'meta/stats.json').read_text())
    std=np.array(stats['action']['std'])
    dataset=pads.dataset(a.dataset/'data',format='parquet')
    result={}
    for ep in a.episodes:
        rows=dataset.to_table(filter=pads.field('episode_index')==ep).sort_by('frame_index').to_pydict()
        images=[rows['observation.images.'+cam] for cam in ('shoulder','wrist')]
        result[str(ep)]=collision_report(np.array(rows['observation.state']),np.array(rows['action']),images,std)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
