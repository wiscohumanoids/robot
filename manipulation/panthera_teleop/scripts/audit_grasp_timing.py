"""Compare arm and gripper delays specifically around recorded closures."""
import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin


def fit_delay(command_times, commands, measured_times, measured, maximum=.15):
    """Least-squares delay on host times, without fitting away spatial bias."""
    delays=np.arange(0,maximum+.0001,.001)
    valid=(measured_times >= command_times[0]+maximum)&(measured_times <= command_times[-1])
    if valid.sum()<5:
        return None
    times=measured_times[valid];values=measured[valid]
    costs=[]
    for delay in delays:
        expected=np.column_stack([np.interp(times-delay,command_times,commands[:,i])
                                  for i in range(commands.shape[1])])
        costs.append(float(np.mean((values-expected)**2)))
    best=int(np.argmin(costs))
    return dict(best_fit_ms=float(delays[best]*1000),rmse_without_delay=float(np.sqrt(costs[0])),
                rmse_with_delay=float(np.sqrt(costs[best])),at_search_limit=best==len(delays)-1)


def events_from_closing(t,g):
    rate=np.diff(g)/np.diff(t)
    candidates=np.flatnonzero(rate < -.05)+1
    groups=[]
    for i in candidates:
        if not groups or t[i]-t[groups[-1][-1]] > .25:
            groups.append([max(0,i-1),i])
        else:
            groups[-1].append(i)
    return [(group[0],group[-1]) for group in groups if g[group[0]]-g[group[-1]] > .1]


def audit(path,output):
    path=Path(path);output=Path(output);output.mkdir(exist_ok=True,parents=True)
    meta=json.loads((path/'metadata.json').read_text());ctx=meta['context']
    rows=[json.loads(l) for l in (path/'states.jsonl').open()]
    rows=[r for r in rows if r.get('control_source')=='policy' and r.get('command_target_rad')]
    t=np.asarray([(r['monotonic_ns']-meta['start_monotonic_ns'])/1e9 for r in rows])
    command_times=t+np.asarray([r.get('control_compute_s',0) for r in rows])
    targets=np.asarray([r['command_target_rad'] for r in rows]);result=[]
    for side,o in [('left',0),('right',7)]:
        cfg=ctx['robot_configs'][side];model=pin.buildModelFromXML(ctx['urdf_sources'][side])
        data=model.createData();frame=model.getFrameId(cfg['urdf']['end_effector_link'])
        idx=[model.joints[model.getJointId(n)].idx_q for n in cfg['kinematics']['joint_names']]
        def positions(joints):
            pos=[]
            for angles in joints:
                q=np.zeros(model.nq);q[idx]=angles
                pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
                pos.append(data.oMf[frame].translation.copy())
            return np.asarray(pos)
        measured=np.asarray([r['arms'][side]['position_rad'] for r in rows])
        actual=positions(measured);desired=positions(targets[:,o:o+6])
        g=targets[:,o+6];mg=np.asarray([r['arms'][side]['gripper']['position'] for r in rows])
        # At each feedback time, use the most recent emitted command, not the
        # new command that is only sent later in this same control iteration.
        previous=np.searchsorted(command_times,t,side='right')-1
        previous=np.clip(previous,0,len(rows)-1)
        ee=np.linalg.norm(actual-desired[previous],axis=1)*1000
        for number,(start,end) in enumerate(events_from_closing(command_times,g),1):
            threshold=g[start]-.8*(g[start]-g[end])
            commanded80=start+int(np.flatnonzero(g[start:end+1]<=threshold)[0])
            reach=np.flatnonzero((np.arange(len(rows))>=start)&(t<=t[end]+.3)&(mg<=threshold))
            actual80=int(reach[0]) if len(reach) else None
            mask=(t>=t[start]-.2)&(t<=t[end]+.3)
            arm_delay=fit_delay(command_times[mask],desired[mask],t[mask],actual[mask])
            grip_delay=fit_delay(command_times[mask],g[mask,None],t[mask],mg[mask,None])
            rates=np.linalg.norm(np.diff(desired[mask],axis=0),axis=1)/np.diff(command_times[mask])
            excursion=float(np.linalg.norm(np.ptp(desired[mask],axis=0))*1000)
            if arm_delay is not None:
                arm_delay['command_pose_excursion_mm']=excursion
                arm_delay['rmse_reduction_percent']=float(100*(1-arm_delay['rmse_with_delay']/
                                                             max(arm_delay['rmse_without_delay'],1e-12)))
                arm_delay['well_excited']=bool(excursion>=3 and not arm_delay['at_search_limit'] and
                                               arm_delay['rmse_reduction_percent']>=15)
            torque=np.asarray([abs(rows[i]['arms'][side]['gripper']['torque']) for i in range(start,end+1)])
            at=int(actual80 if actual80 is not None else commanded80)
            feedback_age=1000*(rows[at]['unix_ns']/1e9-
                np.asarray(rows[at]['arms'][side]['motor_feedback_time'][:6]))
            result.append(dict(side=side,closure=number,start_host_s=float(t[start]),end_host_s=float(t[end]),
                start_gripper_rad=float(g[start]),end_gripper_rad=float(g[end]),
                commanded_80_percent_time_s=float(command_times[commanded80]),
                actual_80_percent_time_s=float(t[actual80]) if actual80 is not None else None,
                gripper_80_percent_delay_ms=float((t[actual80]-command_times[commanded80])*1000)
                                          if actual80 is not None else None,
                ee_error_at_actual_80_percent_mm=float(ee[actual80]) if actual80 is not None else None,
                ee_error_at_command_80_percent_mm=float(ee[commanded80]),
                maximum_ee_error_in_closure_mm=float(ee[start:end+1].max()),
                arm_position_delay_fit=arm_delay,gripper_delay_fit=grip_delay,
                arm_max_command_speed_mm_s=float(rates.max()*1000),
                arm_feedback_age_ms=feedback_age.tolist(),
                peak_gripper_torque_abs_nm=float(torque.max()),
                percent_gripper_torque_near_limit=float((torque>=.48).mean()*100)))
    report=dict(id=meta['id'],recording=str(path.resolve()),events=result,
        method='Closures drop >0.1 rad, commanded negative rate >0.05 rad/s; gap <0.25 s. Most recent command at feedback timestamp; 80% travel is a timing marker, not proof of contact. Delay fits use host times, include feedback age and fixed pose bias, and are approximate. FK is tool_link, not externally measured jaw/TCP or object position.')
    (output/f'{path.name}-timing.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('recordings',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    results=[audit(path,args.output) for path in args.recordings]
    (args.output/'timing-summary.json').write_text(json.dumps(results,indent=2))
    for r in results:
        print(r['id'])
        for e in r['events']:
            print(e['side'],e['closure'],'EE@80%',e['ee_error_at_actual_80_percent_mm'],
                  'arm delay',e['arm_position_delay_fit'],'gripper delay',e['gripper_delay_fit'])


if __name__=='__main__':
    main()
