#!/usr/bin/env python3
"""Separate renderer/physics repeatability from ACT numerical repeatability."""
import argparse
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
import torch
from tools.overfit_act import exact_start,save
from tools.diagnose_act_overfit import load,RAW,CHECKPOINT
from teleop.dataset_contract import PhysicsClock
from teleop.render_vla_dataset import shoulder_camera,wrist_camera
from lerobot.policies.utils import prepare_observation_for_inference

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--steps',type=int,default=240)
p.add_argument('--no-shadow',action='store_true')
p.add_argument('--no-reflection',action='store_true')
p.add_argument('--samples',type=int)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
policy,pre,post=load(CHECKPOINT)
actions=np.load(ROOT/'outputs/act/ik_overfit_20260923_fp32/rollout_005000.npz')['action']
trials=[]
for trial in range(2):
 sim,_=exact_start(RAW)
 if a.samples is not None:sim.model.vis.quality.offsamples=a.samples
 renderers=[mujoco.Renderer(sim.model,256,256) for _ in range(2)]
 cameras=[shoulder_camera(sim.model),wrist_camera(sim.model)]
 clock=PhysicsClock(30,sim.dt)
 values={k:[] for k in ('images','state','qpos','qvel','prediction','repeat_prediction')}
 try:
  for step in range(a.steps):
   images=[]
   for renderer,camera in zip(renderers,cameras):
    renderer.update_scene(sim.data,camera)
    if a.no_shadow:renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
    if a.no_reflection:renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
    images.append(renderer.render().copy())
   if step%30==0:
    state=np.r_[sim.q,sim.data.ctrl[sim.grip_act]].astype(np.float32)
    obs=pre(prepare_observation_for_inference({'observation.state':state,'observation.images.shoulder':images[0],
      'observation.images.wrist':images[1]},torch.device('cuda'),task='stack the three colored cubes',robot_type='panthera_ht_sim'))
    with torch.inference_mode():
     pred=post(policy.predict_action_chunk(obs)).cpu().numpy()
     repeat=post(policy.predict_action_chunk(obs)).cpu().numpy()
    for key,value in zip(values,(images,state,sim.data.qpos,sim.data.qvel,pred,repeat)):
     values[key].append(np.asarray(value).copy())
   act=actions[step]
   sim.set_arm_ctrl(np.clip(act[:6],sim.arm_range[:,0],sim.arm_range[:,1]))
   sim.set_gripper(float(np.clip(act[6]/.04,0,1)))
   sim.step(clock.next_steps())
 finally:
  for r in renderers:r.close()
 values={k:np.array(v) for k,v in values.items()}
 np.savez_compressed(a.output/f'trial_{trial}.npz',**values)
 trials.append(values)
summary={}
for key in trials[0]:
 x,y=trials[0][key],trials[1][key]
 differences=np.abs(x.astype(float)-y.astype(float))
 mask=np.any(differences!=0,axis=tuple(range(1,x.ndim)))
 summary[key]={'max_abs_difference':float(differences.max()),'first_difference_control_step':int(np.flatnonzero(mask)[0]*30) if mask.any() else None,
 'per_query_max_difference':differences.reshape(len(x),-1).max(1).tolist()}
summary['same_input_repeated_inference_max_difference']=max(float(np.abs(t['prediction']-t['repeat_prediction']).max()) for t in trials)
save(a.output/'summary.json',summary)
print(summary,flush=True)
