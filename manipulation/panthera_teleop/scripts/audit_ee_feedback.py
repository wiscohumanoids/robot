"""Measure URDF tool-frame tracking error in saved policy/replay recordings.

Feedback is read before the command logged on each row is sent. Compare it with
the previous row's command in the same uninterrupted policy segment. This is an
encoder/FK tracking audit, not a camera measurement of tool-to-object accuracy.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pinocchio as pin


def stats(values):
    values = np.asarray(values)
    if not len(values):
        return None
    return dict(samples=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.percentile(values,95)), maximum=float(np.max(values)))


def audit(path, output):
    path, output = Path(path), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((path/'metadata.json').read_text())
    context=metadata['context']
    models={}
    for side in ('left','right'):
        model=pin.buildModelFromXML(context['urdf_sources'][side])
        cfg=context['robot_configs'][side]
        indices=[model.joints[model.getJointId(n)].idx_q for n in cfg['kinematics']['joint_names']]
        frame=model.getFrameId(cfg['urdf']['end_effector_link'])
        if frame >= model.nframes:
            raise ValueError('Recorded tool frame is absent from the recorded URDF')
        models[side]=(model,model.createData(),indices,frame)
    def fk(side,joints):
        model,data,indices,frame=models[side]
        q=np.zeros(model.nq);q[indices]=joints
        pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
        pose=data.oMf[frame]
        return pose.translation.copy(),pose.rotation.copy()
    records=[];previous=None;before_previous=None
    for line in (path/'states.jsonl').open():
        row=json.loads(line)
        if (previous is not None and row.get('control_source')=='policy' and
            previous.get('control_source')=='policy' and
            row.get('recording_segment')==previous.get('recording_segment') and
            row.get('inference_sequence')==previous.get('inference_sequence') and
            0 < row['monotonic_ns']-previous['monotonic_ns'] < 100000000):
            for side,offset in (('left',0),('right',7)):
                arm=row['arms'][side]
                target=previous.get('command_target_rad')
                if target is None:
                    continue
                target=np.asarray(target[offset:offset+7])
                measured=np.r_[arm['position_rad'],arm['gripper']['position']]
                p,r=fk(side,measured[:6]);tp,tr=fk(side,target[:6])
                error=(p-tp)*1000
                rotation=np.degrees(np.arccos(np.clip((np.trace(tr.T@r)-1)/2,-1,1)))
                dt=(row['monotonic_ns']-previous['monotonic_ns'])/1e9
                prior_target=before_previous.get('command_target_rad') if before_previous else None
                same_sequence=(before_previous is not None and
                               before_previous.get('inference_sequence')==previous.get('inference_sequence'))
                command_rate=((target[6]-prior_target[offset+6])/dt
                              if prior_target is not None and same_sequence else 0.)
                # Label an approach/close interval using the recorded command
                # and actual gripper velocity, rather than assuming zero=grasp.
                closing=(arm['gripper']['velocity'] < -.1 or command_rate < -.1)
                logged=arm.get('tool_pose_base',{}).get('position_m',p)
                records.append(dict(time_s=(row['monotonic_ns']-metadata['start_monotonic_ns'])/1e9,
                    side=side,sequence=row.get('inference_sequence'),closing=closing,
                    ee_error_mm=float(np.linalg.norm(error)),rotation_error_deg=float(rotation),
                    x_error_mm=float(error[0]),y_error_mm=float(error[1]),z_error_mm=float(error[2]),
                    joint_error_max_rad=float(np.max(np.abs(measured[:6]-target[:6]))),
                    gripper_target_rad=float(target[6]),gripper_measured_rad=float(measured[6]),
                    gripper_error_rad=float(measured[6]-target[6]),
                    gripper_torque_nm=arm['gripper']['torque'],
                    fk_logged_disagreement_mm=float(np.linalg.norm(p-np.asarray(logged))*1000),
                    **{f'joint{i+1}_error_rad':float(measured[i]-target[i]) for i in range(6)}))
        before_previous,previous=previous,row
    result=dict(recording=str(path.resolve()),id=metadata['id'],complete=metadata['complete'],
                outcome=metadata.get('outcome'),checkpoint=context['checkpoint'],sides={})
    for side in ('left','right'):
        rows=[r for r in records if r['side']==side]
        close=[r for r in rows if r['closing']]
        fields=('ee_error_mm','rotation_error_deg','joint_error_max_rad','fk_logged_disagreement_mm')
        result['sides'][side]=dict(all_policy={f:stats([r[f] for r in rows]) for f in fields},
            closing={f:stats([r[f] for r in close]) for f in fields},
            mean_signed_ee_error_mm={axis:float(np.mean([r[f'{axis}_error_mm'] for r in rows]))
                                     if rows else None for axis in ('x','y','z')},
            mean_signed_joint_error_rad=[float(np.mean([r[f'joint{i+1}_error_rad'] for r in rows]))
                                         if rows else None for i in range(6)],
            gripper_absolute_error_rad=stats([abs(r['gripper_error_rad']) for r in rows]),
            closing_gripper_absolute_error_rad=stats([abs(r['gripper_error_rad']) for r in close]),
            closing_torque_abs_nm=stats([abs(r['gripper_torque_nm']) for r in close]))
    result['method']='Measured encoder FK minus previous emitted-command FK in each arm base; only adjacent policy rows from one inference sequence. Tool link from recorded URDF. No object/TCP calibration.'
    if records:
        with (output/f'{path.name}-samples.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (output/f'{path.name}-audit.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('recordings',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();results=[audit(path,args.output) for path in args.recordings]
    (args.output/'summary.json').write_text(json.dumps(results,indent=2))
    for r in results:
        print(r['id'])
        for side,s in r['sides'].items():
            print(side,'all EE mm',s['all_policy']['ee_error_mm'],'closing EE mm',s['closing']['ee_error_mm'])


if __name__=='__main__':
    main()
