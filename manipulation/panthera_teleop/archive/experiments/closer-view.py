"""Bounded viewing moves only: each arm can move +20 mm X, +5 mm Z once.
Commands on stdin: left, right. Holds after each. No gripper commands.
"""
import importlib.util
from pathlib import Path
import json
import select
import sys
import time
import numpy as np
from scipy.optimize import least_squares
import pinocchio as pin
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('step',HERE/'noncontact-step.py')
step=importlib.util.module_from_spec(spec);spec.loader.exec_module(step)
guide=step.guide
OFFSET=np.array([.020,0.,.005])

def plan(model,frame,q,lower,upper):
 data=model.createData();initial=step.transform(model,data,frame,q)
 def residual(x):
  pose=step.transform(model,data,frame,x)
  return np.r_[pose.translation-initial.translation-OFFSET,pin.log3(initial.rotation.T@pose.rotation)]
 result=least_squares(residual,q,bounds=(np.maximum(lower,q-.35),np.minimum(upper,q+.35)),xtol=1e-12,ftol=1e-12,gtol=1e-12)
 if not result.success or np.linalg.norm(residual(result.x)[:3])>.0002 or np.linalg.norm(residual(result.x)[3:])>.001:raise RuntimeError('No accurate bounded viewing move')
 for a in np.linspace(0,1,101):
  d=step.transform(model,data,frame,q+a*(result.x-q)).translation-initial.translation
  if np.linalg.norm(d)>.022 or d[2]<-.0002 or d[0]<-.0002:raise RuntimeError('Viewing move outside envelope')
 return result.x

def main():
 if '--run' not in sys.argv:
  print('Use --run to hold current poses; stdin left/right each permits one bounded viewing move.');return
 configs=guide.setup.prepare();sys.path.insert(0,str(guide.setup.SDK/'scripts'))
 from Panthera_lib import Panthera
 robots={};done=set();targets={}
 try:
  for side in ('left','right'):
   robots[side]=Panthera(str(configs[side]));guide.setup.wait_for_joint_feedback(robots[side],side)
  targets={s:r.get_current_pos().copy() for s,r in robots.items()}
  def update():
   samples={s:guide.compensation(r,s) for s,r in robots.items()}
   for s,r in robots.items():
    if np.max(np.abs(samples[s][1]-targets[s]))>.05:raise RuntimeError(f'{s}: tracking error; stop')
    if not r.Joint_Pos_Vel(targets[s],np.full(6,.03),max_tqu=[5.,8.,8.,3.,2.,2.],iswait=False):raise RuntimeError('Command rejected')
   return {s:q for s,(_,q) in samples.items()}
  start=time.monotonic()
  while time.monotonic()-start<2.:update();time.sleep(.01)
  print('READY: holding both arms. Awaiting one-arm viewing command.',flush=True)
  while True:
   update()
   if select.select([sys.stdin],[],[],0)[0]:
    side=sys.stdin.readline().strip()
    if side not in robots or side in done:
     print('REJECTED: expected unused left/right command',flush=True);continue
    r=robots[side];q=r.get_current_pos().copy();frame=r.end_effector_frame_id
    end=plan(r.model,frame,q,r.joint_limits['lower'],r.joint_limits['upper'])
    initial=step.transform(r.model,r.model.createData(),frame,q)
    duration=max(6.,float(np.max(np.abs(end-q)))*1.5/.025)
    print(f'MOVING {side}: +20 mm forward, +5 mm up, {duration:.1f} seconds',flush=True)
    start=time.monotonic()
    while time.monotonic()-start<duration:
     t=min(1.,(time.monotonic()-start)/duration);a=3*t*t-2*t*t*t
     targets[side]=q+a*(end-q);update();time.sleep(.01)
    targets[side]=end;start=time.monotonic()
    while time.monotonic()-start<2.:update();time.sleep(.01)
    finalq=r.get_current_pos().copy();actual=step.transform(r.model,r.model.createData(),frame,finalq)
    delta=actual.translation-initial.translation;error=float(np.linalg.norm(delta-OFFSET));done.add(side)
    result={'side':side,'delta_m':delta.tolist(),'error_m':error,'before_q':q.tolist(),'after_q':finalq.tolist(),'target_q':end.tolist()}
    (HERE/f'closer-view-{side}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE; HOLDING '+json.dumps(result),flush=True)
    if error>.002:
     done.update(robots);print('Further viewing moves disabled: tracking error >2 mm',flush=True)
   time.sleep(.01)
 except KeyboardInterrupt:print('Stopping hold; support arms.',flush=True)
 finally:guide.stop_robots(robots)
if __name__=='__main__':main()
