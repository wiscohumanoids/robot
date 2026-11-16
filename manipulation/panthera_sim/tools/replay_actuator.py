"""Replay an actuator trial in a viewer or render paired overview/close-up video."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('episode',type=Path)
    p.add_argument('--video',type=Path)
    p.add_argument('--seed',type=int,default=0,help='Fallback for early rollouts without metadata')
    p.add_argument('--stride',type=int,default=2,help='Render every Nth saved frame')
    p.add_argument('--start-stage',help='Start at a recorded stage, e.g. pin_3_approach')
    args=p.parse_args()
    if args.stride<1:p.error('--stride must be positive')
    if args.video:os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    import numpy as np
    from sim.actuator_env import ActuatorSim, ASSETS
    meta=json.loads((args.episode/'metadata.json').read_text()) if (args.episode/'metadata.json').exists() else {}
    with_pins = meta.get('with_pins', False)
    scene = ASSETS/('scene_pins.xml' if with_pins else 'scene.xml')
    if meta.get('scene_sha256') and meta['scene_sha256'] != hashlib.sha256(scene.read_bytes()).hexdigest():
        raise ValueError('Scene changed since recording; restore the recorded scene before replay')
    sim=ActuatorSim(seed=meta.get('seed',args.seed), with_pins=with_pins)
    states=np.load(args.episode/'carrier_rollout.npz')
    qpos=states['qpos'];fps=float(states['fps'])
    if qpos.shape[1]!=sim.model.nq:raise ValueError('Rollout and model dimensions differ')
    stages=json.loads((args.episode/'stages.json').read_text())
    start_frame = 0
    if args.start_stage:
        matches = [stage['frame'] for stage in stages if stage['name'] == args.start_stage]
        if not matches:
            raise ValueError(f'Unknown recorded stage: {args.start_stage}')
        start_frame = matches[0]
    if args.video:
        import cv2
        args.video.parent.mkdir(parents=True,exist_ok=True)
        renderer=mujoco.Renderer(sim.model,480,640)
        option=mujoco.MjvOption();option.geomgroup[3]=0;option.geomgroup[1]=0
        option.sitegroup[:] = 0
        close_camera = 'assembly_close'
        if with_pins:
            close_camera = mujoco.MjvCamera()
            close_camera.lookat[:] = sim.data.xpos[sim.model.body('small_carrier').id]+[0, 0, .022]
            close_camera.distance = .24
            close_camera.azimuth = 45
            close_camera.elevation = -40
        writer=cv2.VideoWriter(str(args.video),cv2.VideoWriter_fourcc(*'mp4v'),fps/args.stride,(1280,480))
        if not writer.isOpened():raise RuntimeError('Could not open video writer')
        try:
            for i in range(start_frame,len(qpos),args.stride):
                sim.data.qpos[:]=qpos[i];mujoco.mj_forward(sim.model,sim.data)
                views=[]
                for camera in ['overview',close_camera]:
                    renderer.update_scene(sim.data,camera=camera,scene_option=option)
                    views.append(renderer.render().copy())
                image=np.hstack(views)
                stage=next((s['name'] for s in reversed(stages) if s['frame']<=i),'reset')
                cv2.rectangle(image,(0,0),(1280,48),(20,27,34),-1)
                cv2.putText(image,f'CONTACT-ONLY TRIAL  /  {stage.replace("_"," ")}  /  {i/fps:.1f}s',
                            (15,31),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1,cv2.LINE_AA)
                writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        finally:writer.release();renderer.close()
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(sim.model,sim.data) as viewer:
            viewer.opt.geomgroup[3]=0
            start=time.monotonic()
            while viewer.is_running():
                i=min(start_frame+int((time.monotonic()-start)*fps),len(qpos)-1)
                sim.data.qpos[:]=qpos[i];mujoco.mj_forward(sim.model,sim.data)
                viewer.sync();time.sleep(1/fps)


if __name__=='__main__':main()
