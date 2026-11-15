#!/usr/bin/env python3
"""Controlled ACT objective probes on the single IK demonstration."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES
from tools.overfit_act import DemoDataset, save, rollout
from tools.act_pickup import PhysicalErrors
from train_act import seed_everything,evaluate_inference

CHECKPOINT=ROOT/'outputs/act/ik_overfit_20260923_fp32/checkpoint'
DATASET=ROOT/'outputs/lerobot/panthera_scripted_stack_30hz'
TRAJECTORY=ROOT/'outputs/scripted_stack_rendered_30hz/episode_0000/trajectory.npz'
RAW=ROOT/'data/scripted_stack/episode_0000'


def prep(pre,raw):
    batch=pre(raw)
    batch['diagnostic.phase']=raw['diagnostic.phase'].to('cuda')
    return batch


def load(checkpoint):
    policy=ACTPolicy.from_pretrained(checkpoint).to('cuda')
    policy.config.use_amp=False
    pre,post=make_pre_post_processors(policy.config,pretrained_path=str(checkpoint))
    return policy,pre,post


class IndexedDemo(DemoDataset):
    def __getitem__(self,index):
        result=super().__getitem__(index)
        result['diagnostic.phase']=torch.tensor(index/len(self),dtype=torch.float32)
        return result


def add_phase(policy,initialize=True,route='latent'):
    if route=='state':
        policy.model.register_parameter('diagnostic_phase_weight',torch.nn.Parameter(torch.zeros(policy.config.dim_model,device='cuda')))
        def state_hook(module,inputs,output):
            return output+policy._diagnostic_phase[:,None]*policy.model.diagnostic_phase_weight[None,:]
        policy._phase_center=.5;policy._phase_scale=4.
        policy._phase_hook=policy.model.encoder_robot_state_input_proj.register_forward_hook(state_hook)
        return
    # Diagnostic oracle only: inject elapsed phase into an otherwise zero latent.
    # Zero this column initially so the starting predictions are unchanged.
    if initialize:
        with torch.no_grad():policy.model.encoder_latent_input_proj.weight[:,0].zero_()
    def hook(module,inputs):
        latent=inputs[0].clone()
        latent[:,0]=policy._diagnostic_phase
        return (latent,)
    policy._phase_center=.5
    policy._phase_scale=4.
    policy._phase_hook=policy.model.encoder_latent_input_proj.register_forward_pre_hook(hook)


def zero_prediction(policy,batch):
    # eval mode selects the deployed zero latent; keep autograd enabled.
    policy.eval()
    if hasattr(policy,'_phase_hook'):policy._diagnostic_phase=(batch['diagnostic.phase'].to('cuda')-policy._phase_center)*policy._phase_scale
    obs={k:batch[k] for k in policy.config.input_features}
    obs[OBS_IMAGES]=[obs[k] for k in policy.config.image_features]
    return policy.model(obs)[0]


@torch.no_grad()
def metrics(policy,batch,post):
    pred=zero_prediction(policy,batch)
    valid=(~batch['action_is_pad']).unsqueeze(-1).expand_as(pred)
    physical=PhysicalErrors('absolute')
    # PhysicalErrors uses state only for relative actions.
    physical.add(post(pred[:,0]),post(batch['action'][:,0]),batch['observation.state'])
    return dict(l1=float((pred-batch['action']).abs().masked_select(valid).mean()),
                _target_errors_mm=physical.positions,**physical.summary())


def fit(args):
    dataset=IndexedDemo(DATASET,0,np.load(TRAJECTORY))
    if args.scope=='conflict':
        ids=[17,39]
    elif args.scope=='fixed':
        ids=[0,100,180,200,400,465,490,550,620,670,850,898]
    else:
        ids=list(range(len(dataset)))
    history=[]
    for objective in args.objectives:
        seed_everything(20260924)
        policy,pre,post=load(args.checkpoint)
        if objective=='phase':
            add_phase(policy,route=args.phase_route)
            if args.scope=='conflict':
                policy._phase_center=float(np.mean(ids)/len(dataset))
                policy._phase_scale=float(len(dataset)/np.std(ids))
        params=policy.parameters()
        if objective=='phase' and args.phase_route=='state' and args.phase_lr is not None:
            params=[{'params':[v for n,v in policy.named_parameters() if n!='model.diagnostic_phase_weight']},
                    {'params':[policy.model.diagnostic_phase_weight],'lr':args.phase_lr}]
        optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=1e-4)
        loader=torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,ids),batch_size=12,shuffle=args.scope=='full')
        validation=torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,ids if args.scope!='full' else np.linspace(0,len(dataset)-1,192,dtype=int)),batch_size=12)
        iterator=iter(loader); started=time.monotonic()
        out=args.output/objective;out.mkdir(parents=True,exist_ok=True)
        save(out/'protocol.json',dict(objective=objective,scope=args.scope,indices=ids,lr=args.lr,
                                     steps=args.steps,checkpoint=str(args.checkpoint),dropout=policy.config.dropout,phase_route=args.phase_route,phase_lr=args.phase_lr,
                                     phase_center=getattr(policy,'_phase_center',None),phase_scale=getattr(policy,'_phase_scale',None),
                                     amp=False,seed=20260924,optimizer='fresh AdamW; weight decay 1e-4'))
        for step in range(args.steps+1):
            if step:
                try: raw=next(iterator)
                except StopIteration:iterator=iter(loader);raw=next(iterator)
                batch=prep(pre,raw);optimizer.zero_grad(set_to_none=True)
                if objective=='vae':
                    policy.train();loss,logs=policy.forward(batch)
                else:
                    pred=zero_prediction(policy,batch)
                    error=(pred-batch['action']).abs()
                    # Match stock ACT's padding reduction exactly.
                    loss=(error*(~batch['action_is_pad']).unsqueeze(-1)).mean()
                    logs={'l1_loss':float(loss.detach())}
                if not torch.isfinite(loss):raise RuntimeError(f'nonfinite at {step}')
                loss.backward()
                grad=float(torch.nn.utils.clip_grad_norm_(policy.parameters(),10.))
                optimizer.step()
            if step%args.eval_freq==0 or step==args.steps:
                vals=[]
                for raw in validation:vals.append(metrics(policy,prep(pre,raw),post))
                row=dict(step=step,objective=objective,elapsed=time.monotonic()-started,
                         **{k:float(np.mean([v[k] for v in vals])) for k in vals[0] if not k.startswith('_')})
                row['validation_first_target_p95_mm']=float(np.percentile(np.concatenate([v['_target_errors_mm'] for v in vals]),95))
                if step:row.update(train_logs=logs,gradient_norm=grad)
                history.append(row)
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print('FIT',json.dumps(row),flush=True)
        ck=out/'checkpoint';policy.save_pretrained(ck);pre.save_pretrained(ck);post.save_pretrained(ck)
        for name in ('deployment.json','inference.json'):
            (ck/name).write_bytes((args.checkpoint/name).read_bytes())
        if args.scope=='full':
            errors=PhysicalErrors('absolute'); error_sum=count=0
            with torch.inference_mode():
                for raw in torch.utils.data.DataLoader(dataset,batch_size=12):
                    batch=prep(pre,raw);pred=zero_prediction(policy,batch)
                    valid=(~batch['action_is_pad']).unsqueeze(-1).expand_as(pred)
                    error_sum+=float((pred-batch['action']).abs().masked_select(valid).sum());count+=int(valid.sum())
                    errors.add(post(pred[:,0]),post(batch['action'][:,0]),raw['observation.state'])
            full=dict(l1=error_sum/count,**errors.summary())
            def phase_callback(policy,step):policy._diagnostic_phase=torch.tensor([step/len(dataset)*4-2],device='cuda')
            result=rollout(RAW,np.load(TRAJECTORY),policy,pre,post,out/'rollout.mp4',
                           before_action=phase_callback if objective=='phase' else None)
            save(out/'result.json',dict(offline=full,rollout=result))
            print('ROLLOUT',objective,json.dumps(result),flush=True)
        del policy,optimizer,pre,post
        torch.cuda.empty_cache()
    save(args.output/'history.json',history)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scope',choices=('fixed','full','conflict'),default='fixed')
    p.add_argument('--objectives',nargs='+',choices=('vae','zero','phase'),default=['vae','zero'])
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--eval-freq',type=int,default=100)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--phase-route',choices=('latent','state'),default='state')
    p.add_argument('--phase-lr',type=float)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);fit(args)

if __name__=='__main__':main()
