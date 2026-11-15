"""One 5 mm base-Z lift of the left-mapped arm, then hold both arms.
No part approach, gripper closing, or automatic return trajectory.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('guide', HERE/'hand-guide.py')
guide = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guide)
KP = np.array([30.,50.,60.,25.,15.,10.])
KD = np.array([3.,5.,6.,2.5,1.5,1.])


def transform(model, data, frame, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame].copy()


def plan_lift(model, frame, q, lower, upper):
    data = model.createData()
    initial = transform(model, data, frame, q)
    target = initial.translation + np.array([0.,0.,0.005])
    def residual(x):
        pose = transform(model, data, frame, x)
        return np.r_[pose.translation-target, pin.log3(initial.rotation.T @ pose.rotation)]
    result = least_squares(residual, q, bounds=(np.maximum(lower,q-.05),np.minimum(upper,q+.05)), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not result.success or np.linalg.norm(residual(result.x)[:3]) > 0.0002 or np.linalg.norm(residual(result.x)[3:]) > 0.001:
        raise RuntimeError('No accurate bounded lift solution')
    for alpha in np.linspace(0,1,101):
        pose = transform(model, data, frame, q+alpha*(result.x-q))
        delta = pose.translation-initial.translation
        if np.linalg.norm(delta) > .006 or delta[2] < -.0002:
            raise RuntimeError('Lift trajectory exceeds test envelope')
    return result.x


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--position-mode',action='store_true',help='Use SDK position/velocity mode for the bounded non-contact test')
    args=parser.parse_args()
    if not args.run:
        print('Use --run only with the robot area clear. This moves one arm 5 mm upward and holds both.')
        return
    configs=guide.setup.prepare()
    sys.path.insert(0,str(guide.setup.SDK/'scripts'))
    from Panthera_lib import Panthera
    robots={}
    try:
        for side in ('left','right'):
            robots[side]=Panthera(str(configs[side]))
            guide.setup.wait_for_joint_feedback(robots[side],side)
        targets={side:robot.get_current_pos().copy() for side,robot in robots.items()}
        frames={side:robot.end_effector_frame_id for side,robot in robots.items()}
        def update(desired, ramp=1.):
            samples={side:guide.compensation(robot,side) for side,robot in robots.items()}
            for side,robot in robots.items():
                torque,q=samples[side]
                if np.max(np.abs(q-desired[side]))>.08:
                    raise RuntimeError(f'{side}: tracking error exceeds 0.08 rad; stopping test')
                if args.position_mode:
                    accepted=robot.Joint_Pos_Vel(desired[side],np.full(6,.03),max_tqu=[5.,8.,8.,3.,2.,2.],iswait=False)
                else:
                    accepted=robot.pos_vel_tqe_kp_kd(desired[side],guide.ZERO,torque,KP*ramp,KD)
                if not accepted:
                    raise RuntimeError(f'{side}: command rejected')
            return {side:q for side,(_,q) in samples.items()}
        start=time.monotonic()
        while time.monotonic()-start<3.:
            update(targets,min(1.,(time.monotonic()-start)/1.))
            time.sleep(.01)
        baseline={side:robot.get_current_pos().copy() for side,robot in robots.items()}
        # Anchor the small test to freshly settled measured positions.
        targets={side:q.copy() for side,q in baseline.items()}
        left=robots['left']
        end=plan_lift(left.model,frames['left'],baseline['left'],left.joint_limits['lower'],left.joint_limits['upper'])
        print('START SINGLE 5 mm UPWARD TEST; maximum joint step',float(np.max(np.abs(end-baseline['left']))),flush=True)
        start=time.monotonic()
        while time.monotonic()-start<3.:
            t=min(1.,(time.monotonic()-start)/3.)
            alpha=3*t*t-2*t*t*t
            targets['left']=baseline['left']+alpha*(end-baseline['left'])
            update(targets)
            time.sleep(.01)
        targets['left']=end
        start=time.monotonic()
        while time.monotonic()-start<2.:
            update(targets)
            time.sleep(.01)
        measured={side:robot.get_current_pos().copy() for side,robot in robots.items()}
        deltas={}
        for side,robot in robots.items():
            data=robot.model.createData()
            a=transform(robot.model,data,frames[side],baseline[side])
            b=transform(robot.model,data,frames[side],measured[side])
            deltas[side]=(b.translation-a.translation).tolist()
        error=float(np.linalg.norm(np.array(deltas['left'])-[0,0,.005]))
        result={'status':'test_complete_holding' if error<.001 else 'failed_tracking_holding','control_mode':'position' if args.position_mode else 'compliant','tcp_displacement_error_m':error,'baseline_joints':{s:q.tolist() for s,q in baseline.items()},'target_joints':{s:q.tolist() for s,q in targets.items()},'measured_joints':{s:q.tolist() for s,q in measured.items()},'measured_tcp_delta_m':deltas,'note':'TCP from encoders/model, not independently calibrated vision. No assembly attempted.'}
        result_name='noncontact-position-result.json' if args.position_mode else 'noncontact-step-result.json'
        (HERE/result_name).write_text(json.dumps(result,indent=2)+'\n')
        print('TEST COMPLETE; HOLDING. Measured model TCP displacement:',json.dumps(deltas),flush=True)
        last_log=0.
        while True:
            q=update(targets)
            now=time.monotonic()
            if now-last_log>=2.:
                print('HOLDING; max joint errors', {s:round(float(np.max(np.abs(q[s]-targets[s]))),5) for s in robots},flush=True)
                last_log=now
            time.sleep(.01)
    except KeyboardInterrupt:
        print('Stopping position hold; support the arms.',flush=True)
    finally:
        guide.stop_robots(robots)

if __name__=='__main__':main()
