#!/usr/bin/env python3
"""Evaluate a saved objective-probe checkpoint, with optional phase oracle."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.diagnose_act_overfit import load,add_phase,RAW,TRAJECTORY,CHECKPOINT
from tools.overfit_act import rollout,save
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--samples',type=int)
p.add_argument('--phase',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
policy,pre,post=load(a.checkpoint)
trajectory=np.load(TRAJECTORY)
if a.phase:
 protocol=json.loads((a.checkpoint.parent/'protocol.json').read_text())
 route=protocol.get('phase_route','latent')
 add_phase(policy,initialize=False,route=route)
 if route=='state':
  from safetensors.torch import load_file
  with torch.no_grad():policy.model.diagnostic_phase_weight.copy_(load_file(a.checkpoint/'model.safetensors')['model.diagnostic_phase_weight'])
 policy._phase_center=protocol.get('phase_center',.5)
 policy._phase_scale=protocol.get('phase_scale',4.)
def callback(policy,step):
 phase=min(step,len(trajectory['q'])-2)/(len(trajectory['q'])-1)
 policy._diagnostic_phase=torch.tensor([(phase-policy._phase_center)*policy._phase_scale],device='cuda')
r=rollout(RAW,trajectory,policy,pre,post,a.output/'rollout.mp4',
          before_action=callback if a.phase else None,render_samples=a.samples)
save(a.output/'result.json',r)
print(r,flush=True)
