"""Two supervised 5 cm diagonal viewing stages per arm, then hold.
Stdin commands left/right. No gripper commands. Targets are referenced to
saved, verified closer-view poses so restarting does not erase progress.
"""
import importlib.util,json,select,sys,time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('step',HERE/'noncontact-step.py');step=importlib.util.module_from_spec(spec);spec.loader.exec_module(step)
guide=step.guide

def plan(model,frame,q,reference,lower,upper,side,stage):
 data=model.createData();start=step.transform(model,data,frame,q);ref=step.transform(model,data,frame,reference)
 offset=stage*np.array([np.sqrt(.05**2-.045**2),-.045 if side=='left' else .045,0.]);goal=ref.translation+offset
 def residual(x):
  pose=step.transform(model,data,frame,x)
  return np.r_[pose.translation-goal,pin.log3(ref.rotation.T@pose.rotation)]
 result=least_squares(residual,reference,bounds=(np.maximum(lower,q-1.6),np.minimum(upper,q+1.6)),xtol=1e-12,ftol=1e-12,gtol=1e-12)
 if not result.success or np.linalg.norm(residual(result.x)[:3])>.0002 or np.linalg.norm(residual(result.x)[3:])>.001:raise RuntimeError('No accurate bounded approach')
 for a in np.linspace(0,1,201):
  d=step.transform(model,data,frame,q+a*(result.x-q)).translation-start.translation
  if np.linalg.norm(d)>.250 or start.translation[2]+d[2]<max(.100,min(start.translation[2],goal[2])-.002) or abs(d[1])>.120:raise RuntimeError('Approach path outside clearance envelope')
 return result.x,goal,ref.translation

def main():
 if '--run' not in sys.argv:print('Use --run to hold; then left/right for bounded viewing approach.');return
 references={'left':np.array(json.loads((HERE/'approach-carrier-left.json').read_text())['after_q'])}
 configs=guide.setup.prepare();sys.path.insert(0,str(guide.setup.SDK/'scripts'));from Panthera_lib import Panthera
 robots={};done={'left':0,'right':0}
 try:
  for s in ('left','right'):
   robots[s]=Panthera(str(configs[s]));guide.setup.wait_for_joint_feedback(robots[s],s)
  # Both arms must have settled before latching a position-hold target.
  previous=None;stable=0;deadline=time.monotonic()+10
  while time.monotonic()<deadline:
   for r in robots.values():r.send_get_motor_state_cmd();r.motor_send_cmd()
   time.sleep(.1)
   current={s:guide.compensation(r,s)[1] for s,r in robots.items()}
   quiet=previous is not None and all(np.max(np.abs(current[s]-previous[s]))<.002 and np.max(np.abs(robots[s].get_current_vel()))<.05 for s in robots)
   stable=stable+1 if quiet else 0;previous=current
   if stable>=8:break
  else:raise RuntimeError('Arms still moving during startup; refusing to latch hold')
  targets={s:q.copy() for s,q in current.items()};references['right']=targets['right'].copy()
  print('STABLE START '+json.dumps({s:q.tolist() for s,q in targets.items()}),flush=True)
  def update():
   states={s:guide.compensation(r,s) for s,r in robots.items()}
   for s,r in robots.items():
    if np.max(np.abs(states[s][1]-targets[s]))>.06:raise RuntimeError(f'{s}: tracking deviation; actual={states[s][1].tolist()} target={targets[s].tolist()}; stopping')
    if not r.Joint_Pos_Vel(targets[s],np.full(6,.10),max_tqu=[5.,8.,8.,3.,2.,2.],iswait=False):raise RuntimeError('Command rejected')
  start=time.monotonic()
  while time.monotonic()-start<2:update();time.sleep(.01)
  print('READY: holding; awaiting left/right',flush=True)
  while True:
   update()
   if select.select([sys.stdin],[],[],0)[0]:
    s=sys.stdin.readline().strip()
    if s not in robots or done[s]>=2:print('REJECTED unused left/right required',flush=True);continue
    r=robots[s];q=r.get_current_pos().copy();frame=r.end_effector_frame_id
    end,goal,ref=plan(r.model,frame,q,references[s],r.joint_limits['lower'],r.joint_limits['upper'],s,done[s]+1)
    duration=max(8.,float(np.max(np.abs(end-q)))*1.5/.08)
    print(f'MOVING {s} 5 cm stage {done[s]+1}/2 toward center; duration {duration:.1f}s; max joint change {max(abs(end-q)):.3f}',flush=True)
    start=time.monotonic()
    while time.monotonic()-start<duration:
     t=min(1.,(time.monotonic()-start)/duration);a=3*t*t-2*t*t*t;targets[s]=q+a*(end-q);update();time.sleep(.01)
    targets[s]=end;start=time.monotonic()
    while time.monotonic()-start<2:update();time.sleep(.01)
    actualq=r.get_current_pos().copy();actual=step.transform(r.model,r.model.createData(),frame,actualq).translation;error=float(np.linalg.norm(actual-goal));done[s]+=1
    result={'side':s,'stage':done[s],'target_q':end.tolist(),'after_q':actualq.tolist(),'delta_from_previous_view_m':(actual-ref).tolist(),'target_error_m':error}
    (HERE/f'ten-cm-view-{s}-{done[s]}.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE; HOLDING '+json.dumps(result),flush=True)
    if error>.002:done.update({'left':2,'right':2});print('Further commands disabled: tracking error >2mm',flush=True)
   time.sleep(.01)
 except KeyboardInterrupt:print('Stopping hold; support arms.',flush=True)
 finally:guide.stop_robots(robots)
if __name__=='__main__':main()
