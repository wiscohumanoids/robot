#!/usr/bin/env python3
"""One-demo ACT diagnostic using existing IK data and an exact regenerated start."""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import mujoco
import numpy as np
import pyarrow.dataset as pads
import torch
from PIL import Image
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from tools.collect_scripted import Planner
from tools.act_pickup import SustainedPickup
from tools.lerobot_image_cache import _dataset_signature
from teleop.dataset_contract import PhysicsClock, file_hash, validate_rendered
from teleop.render_vla_dataset import shoulder_camera, wrist_camera
from sim.stack_task import stack_metrics
from train_act import seed_everything, evaluate_inference


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


class DemoDataset(torch.utils.data.Dataset):
    """Predecode just the selected episode; same successor targets and padding."""
    def __init__(self, root, episode, trajectory):
        columns = ['observation.state', 'action', 'frame_index',
                   'observation.images.shoulder', 'observation.images.wrist']
        table = pads.dataset(root / 'data', format='parquet').to_table(
            columns=columns, filter=pads.field('episode_index') == episode)
        rows = table.sort_by('frame_index').to_pydict()
        self.state = torch.tensor(np.array(rows['observation.state']), dtype=torch.float32)
        self.actions = torch.tensor(np.array(rows['action']), dtype=torch.float32)
        expected = np.c_[trajectory['q'][:-1], trajectory['ctrl'][:-1, 6]]
        np.testing.assert_allclose(self.state.numpy(), expected, atol=1e-7)
        np.testing.assert_allclose(self.actions.numpy(), trajectory['ctrl'][1:], atol=1e-7)
        self.images = {k: torch.from_numpy(np.stack([
            np.asarray(Image.open(io.BytesIO(v['bytes'])).convert('RGB')).transpose(2,0,1)
            for v in rows[k]])) for k in columns if '.images.' in k}

    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        ids = torch.arange(index, index+30)
        return {'observation.state': self.state[index],
                'action': self.actions[ids.clamp(max=len(self)-1)],
                'action_is_pad': ids >= len(self),
                **{k: v[index].float()/255 for k,v in self.images.items()}}


class FirstFrame(Planner):
    class Captured(Exception):
        pass

    def tick(self, *args):
        super().tick(*args)
        raise self.Captured


def exact_start(raw_path):
    meta = json.loads((raw_path / 'meta.json').read_text())
    if meta['generator_sha256'] != file_hash(ROOT / 'tools/collect_scripted.py'):
        raise ValueError('IK generator changed; cannot regenerate exact initial state')
    planner = FirstFrame(meta['seed'])
    try:
        planner.run()
    except FirstFrame.Captured:
        pass
    raw = np.load(raw_path / 'data.npz')
    errors = {}
    for key, value in planner.episode.rows[0].items():
        if key in raw:
            error = float(np.max(np.abs(np.asarray(value) - raw[key][0])))
            errors[key] = error
            np.testing.assert_allclose(value, raw[key][0], rtol=0, atol=1e-12,
                                       err_msg=f'Initial state mismatch: {key}')
    return planner.sim, errors


@torch.inference_mode()
def rollout(raw_path, trajectory, policy=None, pre=None, post=None, video=None, before_action=None, render_samples=None):
    sim, errors = exact_start(raw_path)
    clock = PhysicsClock(30, sim.dt)
    if render_samples is not None: sim.model.vis.quality.offsamples=render_samples
    renderers = [mujoco.Renderer(sim.model, 256, 256) for _ in range(2)] if policy else []
    cameras = [shoulder_camera(sim.model), wrist_camera(sim.model)] if policy else []
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'),30,(512,256)) if video else None
    if writer and not writer.isOpened():
        raise RuntimeError('Could not open video')
    if policy:
        policy.eval(); policy.reset()
    pickup = SustainedPickup(30)
    stable = 0
    result = dict(grasped=False, sustained_pickup=False, success=False,
                  initial_state_max_error=max(errors.values()), initial_state_errors=errors)
    trace = {k: [] for k in ('q','action','objects','grasp_flags')}
    # Allow 20% extra completion time in policy rollouts.
    steps = int((len(trajectory['q'])-1)*(1.2 if policy else 1))
    try:
        for step in range(steps):
            if policy:
                images=[]
                for renderer,camera in zip(renderers,cameras):
                    renderer.update_scene(sim.data,camera); images.append(renderer.render().copy())
                if writer: writer.write(cv2.cvtColor(np.concatenate(images,axis=1),cv2.COLOR_RGB2BGR))
                state = np.r_[sim.q,sim.data.ctrl[sim.grip_act]].astype(np.float32)
                obs=pre(prepare_observation_for_inference({'observation.state':state,
                    'observation.images.shoulder':images[0], 'observation.images.wrist':images[1]},
                    torch.device(policy.config.device), task='stack the three colored cubes',robot_type='panthera_ht_sim'))
                if before_action is not None: before_action(policy,step)
                action=post(policy.select_action(obs)).cpu().numpy().reshape(-1)
            else:
                action=trajectory['ctrl'][step+1]
            sim.set_arm_ctrl(np.clip(action[:6],sim.arm_range[:,0],sim.arm_range[:,1]))
            sim.set_gripper(float(np.clip(action[6]/.04,0,1)))
            sim.step(clock.next_steps())
            positions=sim.object_poses()[0]
            flags=sim.grasp_flags()
            metrics=stack_metrics(positions)
            stable=stable+1 if metrics['three_stack'] and not flags.any() else 0
            result['grasped'] |= bool(flags.any())
            result['sustained_pickup'] |= pickup.update(positions[:,2],flags,step)
            result['success'] |= stable>=15
            for key,value in zip(trace,(sim.q,action,positions,flags)): trace[key].append(np.array(value).copy())
        result.update(steps=steps, final_three_stack=bool(metrics['three_stack']),
                      final_positions=positions.tolist())
        if video: np.savez_compressed(video.with_suffix('.npz'),**{k:np.array(v) for k,v in trace.items()})
    finally:
        for renderer in renderers: renderer.close()
        if writer: writer.release()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'outputs/lerobot/panthera_scripted_stack_30hz')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode',type=int,default=0)
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--eval-freq',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=12)
    p.add_argument('--resume',type=Path)
    p.add_argument('--evaluate-only',type=Path,metavar='CHECKPOINT')
    p.add_argument('--dropout',type=float,default=None,help='explicit diagnostic override')
    p.add_argument('--lr',type=float,default=None,help='explicit diagnostic override')
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args()
    if args.steps < 1 or args.eval_freq < 1 or args.batch_size < 1:
        p.error('steps, eval-freq and batch-size must be positive')
    if args.dropout is not None and not 0 <= args.dropout < 1:
        p.error('dropout must be in [0, 1)')
    if args.lr is not None and args.lr <= 0:
        p.error('lr must be positive')
    torch.set_num_threads(2)
    seed_everything(20260924)
    args.output.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((args.dataset/'meta/provenance.json').read_text())
    rendered=Path(provenance['rendered_root'])
    manifest=validate_rendered(rendered)
    assert provenance['converter_sha256']==file_hash(ROOT/'teleop/build_lerobot_dataset.py')
    assert provenance['render_manifest_sha256']==file_hash(rendered/'manifest.json')
    name=manifest['episodes'][args.episode]
    source=json.loads((rendered/name/'source.json').read_text())
    raw_path=Path(source['source'])
    trajectory=np.load(rendered/name/'trajectory.npz')
    replay=rollout(raw_path,trajectory)
    save(args.output/'replay.json',replay)
    print('REPLAY',json.dumps(replay),flush=True)
    if not replay['success']: raise RuntimeError('Demonstrated target replay failed; diagnose before training')
    if args.evaluate_only:
        policy=ACTPolicy.from_pretrained(args.evaluate_only).to('cuda')
        prior=json.loads((args.evaluate_only.parent/'protocol.json').read_text())
        if prior['episode_index'] != args.episode or prior['deployment']['dataset_signature'] != _dataset_signature(args.dataset):
            raise ValueError('Evaluation checkpoint belongs to a different demo or dataset')
        pre,post=make_pre_post_processors(policy.config,pretrained_path=str(args.evaluate_only))
        result=rollout(raw_path,trajectory,policy,pre,post,args.output/'rollout.mp4')
        save(args.output/'evaluation.json',result)
        print('RELOADED EVALUATION',json.dumps(result),flush=True)
        return
    dataset=DemoDataset(args.dataset,args.episode,trajectory)
    metadata=LeRobotDatasetMetadata('local/ik_overfit',root=args.dataset)
    if metadata.fps != 30:
        raise ValueError('This diagnostic requires the 30 Hz IK dataset')
    if len(dataset) < args.batch_size:
        raise ValueError('batch-size exceeds demo length')
    features=dataset_to_policy_features(metadata.features)
    cfg=ACTConfig(input_features={k:v for k,v in features.items() if v.type!=FeatureType.ACTION},
                  output_features={k:v for k,v in features.items() if v.type==FeatureType.ACTION},
                  chunk_size=30,n_action_steps=30,device='cuda',use_amp=True,push_to_hub=False)
    policy=ACTPolicy(cfg).to('cuda') if args.resume is None else ACTPolicy.from_pretrained(args.resume).to('cuda')
    cfg=policy.config
    cfg.use_amp=args.amp
    if args.dropout is not None:
        cfg.dropout=args.dropout
        for module in policy.modules():
            if isinstance(module,torch.nn.Dropout): module.p=args.dropout
            if isinstance(module,torch.nn.MultiheadAttention): module.dropout=args.dropout
    pre,post=make_pre_post_processors(cfg,dataset_stats=metadata.stats) if args.resume is None else make_pre_post_processors(cfg,pretrained_path=str(args.resume))
    optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=cfg.optimizer_lr,weight_decay=cfg.optimizer_weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=cfg.use_amp)
    start_step=0
    if args.resume:
        prior=json.loads((args.resume.parent/'protocol.json').read_text())
        if prior['episode_index'] != args.episode or prior['deployment']['dataset_signature'] != _dataset_signature(args.dataset):
            raise ValueError('Resume checkpoint belongs to a different demo or dataset')
        saved=torch.load(args.resume/'overfit_training.pt',weights_only=False,map_location='cpu')
        optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler']); start_step=saved['step']
    if args.lr is not None:
        for group in optimizer.param_groups: group['lr']=args.lr
    loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=0)
    val=torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,np.linspace(0,len(dataset)-1,192,dtype=int)),batch_size=args.batch_size)
    deployment=dict(version=2,fps=30,dataset=str(args.dataset.resolve()),dataset_signature=_dataset_signature(args.dataset),state_gripper='command',max_joint_step=None)
    protocol=dict(episode=name,episode_index=args.episode,raw_source=str(raw_path),frames=len(dataset),
                  training='ACT; standard VAE and full-dataset normalization; gradients from selected demo only',
                  resumed_from=str(args.resume) if args.resume else None,
                  dropout=cfg.dropout, amp=cfg.use_amp, learning_rates=[g['lr'] for g in optimizer.param_groups],
                  validation='same training demo, zero latent; no future action inputs',
                  initial_state='regenerated first recorded physics state; all recorded fields verified <=1e-12',
                  execution='30-action queue at 30 Hz; full demo duration plus 20 percent',
                  success='released three-stack held for 0.5 seconds',deployment=deployment)
    save(args.output/'protocol.json',protocol)
    if start_step >= args.steps:
        p.error('steps must exceed the resume checkpoint step')
    iterator=iter(loader); started=time.monotonic(); history=[]; passed=0
    for step in range(start_step+1,args.steps+1):
        policy.train()
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader); batch=next(iterator)
        batch=pre(batch); optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16,enabled=cfg.use_amp): loss,logs=policy.forward(batch)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(),10.)
        scaler.step(optimizer); scaler.update()
        if step%100==0:
            status=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.monotonic()-started,logs=logs)
            save(args.output/'status.json',status); print('TRAIN',json.dumps(status),flush=True)
        if step%args.eval_freq==0 or step==args.steps:
            metrics=evaluate_inference(policy,val,pre,post)
            result=rollout(raw_path,trajectory,policy,pre,post,args.output/f'rollout_{step:06d}.mp4')
            row=dict(step=step,**metrics,rollout=result); history.append(row)
            with (args.output/'evaluations.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print('EVALUATION',json.dumps(row),flush=True)
            checkpoint=args.output/'checkpoint'; checkpoint.mkdir(exist_ok=True)
            policy.save_pretrained(checkpoint); pre.save_pretrained(checkpoint); post.save_pretrained(checkpoint)
            save(checkpoint/'deployment.json',deployment)
            save(checkpoint/'inference.json',dict(temporal_ensemble=False,action_steps=30))
            torch.save(dict(step=step,optimizer=optimizer.state_dict(),scaler=scaler.state_dict()),checkpoint/'overfit_training.pt')
            passed=passed+1 if result['success'] else 0
            if passed>=2: break
    full=evaluate_inference(policy,torch.utils.data.DataLoader(dataset,batch_size=args.batch_size),pre,post)
    save(args.output/'result.json',dict(protocol=protocol,replay=replay,final_step=step,
        full_training_demo_inference=full,evaluations=history,consecutive_successful_evaluations=passed))

if __name__=='__main__': main()
