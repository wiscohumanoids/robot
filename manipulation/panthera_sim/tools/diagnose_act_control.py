#!/usr/bin/env python3
"""Compare ACT checkpoints and execution modes on identical simulated starts."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from train_act_rl import PantheraStackEnv
from sim.stack_task import stack_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', default=['ensemble', 'queue3', 'queue1'])
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env=None; results=[]
    try:
        for checkpoint in args.checkpoints:
            contract=json.loads((checkpoint/'deployment.json').read_text());hz=contract['fps']
            config=PreTrainedConfig.from_pretrained(checkpoint);config.device=str(device)
            policy=ACTPolicy.from_pretrained(checkpoint,config=config).to(device).eval()
            pre,post=make_pre_post_processors(config,pretrained_path=str(checkpoint))
            if env is not None:env.close()
            env=PantheraStackEnv(args.seed,round(15*hz)+1,hz,None,.99,0.,0.,None,0.)
            for mode in args.modes:
                if mode=='ensemble':
                    policy.config.temporal_ensemble_coeff=.01
                    policy.config.n_action_steps=1
                    policy.temporal_ensembler=ACTTemporalEnsembler(.01,config.chunk_size)
                else:
                    policy.config.temporal_ensemble_coeff=None
                    policy.config.n_action_steps=int(mode.removeprefix('queue'))
                for seed in range(args.seed,args.seed+args.episodes):
                    env.reset(seed);policy.reset();stable=0
                    record={'checkpoint':str(checkpoint),'mode':mode,'seed':seed,
                            'grasped':False,'lifted':False,'two_stacked':False,'success':False}
                    trace={key:[] for key in ('q','action','ee','objects','grasped')}
                    panels=[]
                    with torch.inference_mode():
                        for step in range(round(15*hz)):
                            state,shoulder,wrist=env.observation()
                            obs=prepare_observation_for_inference({'observation.state':state,
                                'observation.images.shoulder':shoulder,'observation.images.wrist':wrist},
                                device,task='stack the three colored cubes',robot_type='panthera_ht_sim')
                            action=post(policy.select_action(pre(obs))).cpu().numpy().reshape(-1)
                            env.step(action)
                            positions,_=env.sim.object_poses();metrics=stack_metrics(positions)
                            grasped=env._grasped()
                            record['grasped'] |= grasped;record['lifted'] |= bool(metrics['lifted_cubes'])
                            record['two_stacked'] |= bool(metrics['two_stack'])
                            stable=stable+1 if metrics['three_stack'] and not grasped else 0
                            record['success'] |= stable>=round(.5*hz)
                            for key,value in [('q',env.sim.q),('action',action),('ee',env.sim.ee_pos()),
                                              ('objects',positions),('grasped',grasped)]:trace[key].append(value)
                            if seed==args.seed and step in (0,30,90,180,300,449):
                                panels.append(np.concatenate([shoulder,wrist],axis=1))
                    label=checkpoint.parent.name+'_'+checkpoint.name
                    stem=f'{label}_{mode}_{seed}'
                    np.savez_compressed(args.output/f'{stem}.npz',**{k:np.asarray(v) for k,v in trace.items()})
                    if panels:
                        from PIL import Image
                        Image.fromarray(np.concatenate(panels,axis=0)).save(args.output/f'{stem}.jpg')
                    record['final_gripper_m']=float(trace['action'][-1][6])
                    record['final_nearest_cube_m']=float(np.linalg.norm(positions-env.sim.ee_pos(),axis=1).min())
                    results.append(record)
                    with (args.output/'results.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
                    print(json.dumps(record),flush=True)
                rows=results[-args.episodes:]
                print('SUMMARY '+json.dumps({'checkpoint':str(checkpoint),'mode':mode,
                      **{k:sum(r[k] for r in rows)/len(rows) for k in ('grasped','lifted','two_stacked','success')}}),flush=True)
            del policy
    finally:
        if env is not None:env.close()
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
