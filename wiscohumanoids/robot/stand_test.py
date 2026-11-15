"""Quick check: load robot.xml, hold the 'stand' keyframe with joint PD for 5 s, print holding torques.
   pip install mujoco numpy ;  python stand_test.py        (view: python -m mujoco.viewer --mjcf=robot.xml)"""
import json, os, numpy as np, mujoco
here = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(here, 'robot.xml')); d = mujoco.MjData(m)
rep = json.load(open(os.path.join(here, 'model_report.json')))['joints']
names = [m.actuator(i).name for i in range(m.nu)]
kp = np.array([{'X8': 300, 'X6': 200, '2xX6': 300, '2xX4-36': 600, 'X4-36': 60, 'X4-10': 20}[rep[n]['motor']] for n in names]); kd = 0.06*kp
lim = m.actuator_ctrlrange[:, 1]
mujoco.mj_resetDataKeyframe(m, d, m.key('stand').id); q0 = d.qpos.copy(); hist = []
for i in range(2500):
    d.ctrl[:] = np.clip(kp*(q0[7:]-d.qpos[7:])-kd*d.qvel[6:], -lim, lim); mujoco.mj_step(m, d)
    if i > 2250: hist.append(d.ctrl.copy())
print('pelvis after 5 s:', d.qpos[:3].round(3), '(started at z = %.3f)' % q0[2])
for n, t in zip(names, np.mean(hist, 0)):
    if abs(t) > 1: print(f'  {n:22s} {t:7.2f} N*m  {100*abs(t)/rep[n]["rated"]:5.1f}% of rated')
