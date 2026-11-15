"""Render the three synchronized camera streams from a bimanual IK rollout."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import mujoco
import numpy as np
from sim.bimanual_actuator_env import BimanualActuatorSim, CAMERAS


def render(episode, output, stride=1):
    if stride < 1: raise ValueError('stride must be positive')
    result = json.loads((episode/'result.json').read_text())
    states = np.load(episode/'rollout.npz')
    sim = BimanualActuatorSim(result['seed'], result.get('xy_jitter_m', .002),
                             result.get('yaw_jitter_deg', 2.))
    fixture = sim.model.body('small_carrier').id
    sim.model.body_pos[fixture] = states['initial_fixture_position']
    sim.model.body_quat[fixture] = states['initial_fixture_quaternion']
    stages = json.loads((episode/'stages.json').read_text())
    option = mujoco.MjvOption(); option.geomgroup[3] = 0; option.sitegroup[:] = 0
    renderer = mujoco.Renderer(sim.model, 360, 480)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), 20, (1440,360))
    if not writer.isOpened():
        renderer.close(); raise RuntimeError('Unable to open video output')
    try:
        for i in range(0, len(states['qpos']), stride):
            sim.data.qpos[:] = states['qpos'][i]
            mujoco.mj_forward(sim.model, sim.data)
            panels = []
            for camera in CAMERAS:
                renderer.update_scene(sim.data, camera=camera, scene_option=option)
                panel = renderer.render().copy()
                cv2.putText(panel, camera, (12,24), cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                label = stages[int(states['stage_id'][i])]['name']
                cv2.putText(panel, label, (12,335), cv2.FONT_HERSHEY_SIMPLEX,.45,(70,220,255),1)
                cv2.putText(panel, f'{i/20:.1f}s / {stride}x playback', (12,48), cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                panels.append(panel)
            writer.write(cv2.cvtColor(np.concatenate(panels,axis=1),cv2.COLOR_RGB2BGR))
    finally:
        writer.release(); renderer.close()
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--stride', type=int, default=1, help='Playback speed multiplier')
    a = p.parse_args()
    print(render(a.episode, a.output or a.episode/'bimanual.mp4', a.stride))
