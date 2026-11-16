"""Contact-only carrier and four-gear assembly with an IK feasibility audit.

The audit moves candidate states for planning only; these are never counted as
physical assembly successes. The carrier rollout uses actuators and contacts,
without object teleportation, grasp welds, or success snaps.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import math
import hashlib
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from sim.actuator_env import ActuatorSim, NOMINAL, ASSETS, PIN_GEARS
from sim.panthera_env import mat_to_quat


def gear_grasp(sim, name):
    """Tool pose and radial direction for the exposed rim of an assembled gear."""
    target, _ = sim.assembled_pose(name)
    outward = target-sim.data.xpos[sim.model.body('small_carrier').id]
    outward[2] = 0
    outward /= np.linalg.norm(outward)
    x = np.array([0., 0., -1.])
    y = np.cross([0., 0., 1.], -outward)
    if name in ('gear_1', 'gear_2'):
        y = -y
    rotation = np.column_stack([x, y, np.cross(x, y)])
    return outward*.018+np.array([0., 0., .0041]), rotation, outward


def gear_placement(sim, name):
    """Released placement tolerances; these do not imply pin retention."""
    bid = sim.model.body(name).id
    target, _ = sim.assembled_pose(name)
    delta = sim.data.xpos[bid]-target
    normal = sim.data.xmat[bid].reshape(3, 3)[:, 2]
    tilt = float(np.rad2deg(np.arccos(np.clip(normal[2], -1, 1))))
    return dict(final_xy_error_mm=float(np.linalg.norm(delta[:2])*1000),
                final_z_error_mm=float(delta[2]*1000), tilt_deg=tilt,
                success=bool(np.linalg.norm(delta[:2]) < .0015 and
                             -.003 < delta[2] < .001 and tilt < 4))


def pin_target_pose(sim, name):
    """Follow the actual upper carrier holes, including seating displacement."""
    bid = sim.model.body('large_carrier').id
    parts = sim.manifest['parts']
    offset = np.array(parts[name]['assembled_origin'])-parts['large_carrier']['assembled_origin']
    return (sim.data.xpos[bid]+sim.data.xmat[bid].reshape(3, 3)@offset,
            sim.data.xquat[bid].copy())


def pin_placement(sim, name):
    bid = sim.model.body(name).id
    target, quat = pin_target_pose(sim, name)
    delta = sim.data.xpos[bid]-target
    carrier_rotation = sim.data.xmat[sim.model.body('large_carrier').id].reshape(3, 3)
    delta = carrier_rotation.T@delta
    axis = sim.data.xmat[bid].reshape(3, 3)[:, 2]
    tilt = float(np.rad2deg(np.arccos(np.clip(axis@carrier_rotation[:, 2], -1, 1))))
    height = sim.manifest['parts'][name]['bounds'][1][2]

    def bore(body_name, center_xy, planes):
        body = sim.model.body(body_name).id
        rotation = sim.data.xmat[body].reshape(3, 3)
        base = rotation.T@(sim.data.xpos[bid]-sim.data.xpos[body])
        direction = rotation.T@axis
        if direction[2] <= 0:
            return False, None
        distances = (np.asarray(planes)-base[2])/direction[2]
        centers = base[None, :]+distances[:, None]*direction
        deviation = float(np.linalg.norm(centers[:, :2]-center_xy, axis=1).max())
        return bool(np.all(distances >= 0) and np.all(distances <= height)), deviation

    parts = sim.manifest['parts']
    carrier_hole = (np.array(parts[name]['assembled_origin'])-
                    parts['large_carrier']['assembled_origin'])[:2]
    through_carrier, carrier_error = bore('large_carrier', carrier_hole, [.01429, .01725])
    through_gear, gear_error = bore(PIN_GEARS[name], np.zeros(2), [0., .0082])
    # Evaluate the shaft at BOTH ends of each bore. A slightly inclined pin
    # can have an offset base while still fitting; body-origin proximity alone
    # cannot distinguish a correctly inserted pin from one resting on a face.
    engaged = (through_carrier and through_gear and carrier_error < .00017 and gear_error < .00015)
    return dict(part=name, gear=PIN_GEARS[name], lateral_error_mm=float(np.linalg.norm(delta[:2])*1000),
                axial_error_mm=float(delta[2]*1000), tilt_deg=tilt,
                through_carrier=through_carrier, through_gear=through_gear,
                carrier_bore_error_mm=None if carrier_error is None else carrier_error*1000,
                gear_bore_error_mm=None if gear_error is None else gear_error*1000,
                success=bool(engaged and abs(delta[2]) < .00075 and tilt < 3))


def pin_entry_ready(sim, name):
    """A shallow pin tip within the upper bore mouth can be pressed farther.

    This is only an entry gate. Full-depth/both-bore success remains strict.
    """
    carrier = sim.model.body('large_carrier').id
    pin = sim.model.body(name).id
    rotation = sim.data.xmat[carrier].reshape(3, 3)
    base = rotation.T@(sim.data.xpos[pin]-sim.data.xpos[carrier])
    axis = rotation.T@sim.data.xmat[pin].reshape(3, 3)[:, 2]
    if axis[2] < np.cos(np.deg2rad(5.)):
        return False
    distance = (.01725-base[2])/axis[2]
    height = sim.manifest['parts'][name]['bounds'][1][2]
    hole = (np.array(sim.manifest['parts'][name]['assembled_origin'])-
            sim.manifest['parts']['large_carrier']['assembled_origin'])[:2]
    center = base[:2]+distance*axis[:2]
    return bool(-.0005 <= distance <= height and np.linalg.norm(center-hole) < .00025)


def ideal_gripper_clearance(sim, pitches=(0.,)):
    """Isolate tool fit from arm reachability, at exact desired tool poses."""
    root = ET.parse(ASSETS/'scene.xml').getroot()
    root.find('compiler').set('meshdir', str(ASSETS))
    world = root.find('worldbody')
    for node in list(world):
        if node.tag == 'body' and node.get('name') == 'link1' or node.tag == 'geom' and node.get('class'):
            world.remove(node)
    for tag in ['actuator','equality','contact','keyframe']:
        node=root.find(tag)
        if node is not None:root.remove(node)
    for body in world.findall('body'):
        name=body.get('name')
        if name != 'small_carrier':
            body.set('pos',' '.join(map(str,sim.assembled_pose(name)[0])))
            body.remove(body.find('freejoint'))
    probe=ET.SubElement(world,'body',name='tool_probe')
    ET.SubElement(probe,'freejoint',name='tool_probe')
    for side,sign in [('L',1),('R',-1)]:
        ET.SubElement(probe,'geom',name=side+'_pad_probe',type='box',size='.0225 .002 .01',pos=f'-.0225 {sign*.0125} 0')
        ET.SubElement(probe,'geom',name=side+'_bulk_probe',type='box',size='.035 .015 .015',pos=f'-.058 {sign*.03} 0')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model)
    probe_id=model.body('tool_probe').id
    out=[]
    for name in ['gear_1','gear_2','gear_3','gear_4']:
        target,_=sim.assembled_pose(name)
        outward=target-sim.data.xpos[sim.model.body('small_carrier').id]
        outward[2]=0;outward/=np.linalg.norm(outward)
        for pitch,offset,opening in [(p,o,g) for p in pitches for o,g in [(0.,.011),(.009,.007),(.012,.003),(.018,.0082)]]:
            angle=np.deg2rad(pitch)
            x=-outward*np.cos(angle)-np.array([0.,0.,1.])*np.sin(angle)
            y=np.cross([0.,0.,1.],-outward)
            rotation=np.column_stack([x,y,np.cross(x,y)])
            data.qpos[:3]=target+outward*offset+[0,0,.0041]
            data.qpos[3:7]=mat_to_quat(rotation)
            for side,sign in [('L',1),('R',-1)]:
                model.geom_pos[model.geom(side+'_pad_probe').id,1]=sign*(opening+.0015)
                model.geom_pos[model.geom(side+'_bulk_probe').id,1]=sign*(opening+.019)
            mujoco.mj_kinematics(model,data);mujoco.mj_collision(model,data)
            contacts=[]
            for c in data.contact:
                b1,b2=model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]
                if probe_id in (b1,b2) and c.dist<-.0001:
                    other=b2 if b1==probe_id else b1
                    if other!=model.body(name).id:
                        contacts.append(dict(other=model.body(other).name,penetration_mm=float(-c.dist*1000)))
            contacts.sort(key=lambda c:-c['penetration_mm'])
            out.append(dict(part=name,pitch_deg=pitch,grasp_offset_mm=offset*1000,clear=not contacts,
                            tip_pos=data.qpos[:3].tolist(),tip_quat=data.qpos[3:7].tolist(),
                            collision_count=len(contacts),deepest_contacts=contacts[:2]))
    return out


def audit(sim):
    """Check true CAD poses and sample straight insertion paths every 0.5 mm."""
    sim.reset(randomize=False)
    results = {}
    for name in ['large_carrier', 'gear_1', 'gear_2', 'gear_3', 'gear_4']:
        target, quat = sim.assembled_pose(name)
        modes = ['vertical'] if name == 'large_carrier' else ['vertical', 'radial']
        for mode in modes:
            worst, blocked = 0., 0
            for distance in np.linspace(.045, 0, 91):
                pos = target.copy()
                if mode == 'vertical':
                    pos[2] += distance
                else:
                    direction = target[:2]-sim.data.xpos[sim.model.body('small_carrier').id, :2]
                    pos[:2] += distance*direction/np.linalg.norm(direction)
                sim.set_part_pose(name, pos, quat)
                mujoco.mj_forward(sim.model, sim.data)
                contacts = sim.part_contacts(name, include_robot=False)
                depth = min([c['distance_m'] for c in contacts]+[0.])
                worst = min(worst, depth)
                blocked += depth < -.0001
            results[name+'_'+mode] = dict(blocked_samples=int(blocked), samples=91,
                max_penetration_mm=-worst*1000, clear=bool(blocked == 0))
        sim.set_part_pose(name, target, quat)
        mujoco.mj_forward(sim.model, sim.data)
    if sim.with_pins:
        for name in PIN_GEARS:
            target, quat = sim.assembled_pose(name)
            worst, blocked = 0., 0
            for distance in np.linspace(.025, 0, 51):
                sim.set_part_pose(name, target+[0, 0, distance], quat)
                mujoco.mj_forward(sim.model, sim.data)
                depth = min([c['distance_m'] for c in sim.part_contacts(name, include_robot=False)]+[0.])
                worst = min(worst, depth)
                blocked += depth < -.0001
            results[name+'_vertical'] = dict(blocked_samples=int(blocked), samples=51,
                max_penetration_mm=-worst*1000, clear=bool(blocked == 0))
    # Test the installed gripper at each side-insertion target, including an
    # outer-rim grasp that keeps as much finger material outside as possible.
    ik = []
    for name in ['gear_1', 'gear_2', 'gear_3', 'gear_4']:
        target, quat = sim.assembled_pose(name)
        outward = target-sim.data.xpos[sim.model.body('small_carrier').id]
        outward[2] = 0
        outward /= np.linalg.norm(outward)
        x = -outward
        z = np.array([0., 0., 1.])
        y = np.cross(z, x)
        rotation = np.column_stack([x, y, z])
        for offset in (0., .009, .012):
            tip = target+outward*offset+np.array([0, 0, .0041])
            rng = np.random.default_rng(42)
            starts = [sim._q_nominal, sim.q]+[rng.uniform(sim.arm_range[:,0]+.02, sim.arm_range[:,1]-.02) for _ in range(18)]
            solutions = [sim.ik(tip, mat_to_quat(rotation), max_joint_step=None,
                                iters=160, q_init=start, posture_gain=0) for start in starts]
            q, pe, re = min(solutions, key=lambda r:r[1]+.35*r[2])
            sim.data.qpos[sim.arm_qadr] = q
            opening = .011 if offset == 0 else (.007 if offset == .009 else .003)
            sim.data.qpos[sim.finger_qadr] = [opening, -opening]
            mujoco.mj_kinematics(sim.model, sim.data)
            mujoco.mj_collision(sim.model, sim.data)
            # Recompute actual error after the solve, not its last iteration.
            error = sim._ee_error(tip, mat_to_quat(rotation))
            collisions = []
            robot_bodies = {sim.model.body(n).id for n in ['link1','link2','link3','link4','link5','link6','L_finger','R_finger']}
            for c in sim.data.contact:
                b1,b2 = sim.model.geom_bodyid[c.geom1],sim.model.geom_bodyid[c.geom2]
                if (b1 in robot_bodies) != (b2 in robot_bodies) and c.dist < -.0001:
                    other = b2 if b1 in robot_bodies else b1
                    # Grasp contact with the held gear is intentional.
                    if other != sim.model.body(name).id:
                        collisions.append(dict(geom1=sim.model.geom(c.geom1).name,
                            geom2=sim.model.geom(c.geom2).name, penetration_mm=float(-c.dist*1000)))
            collisions.sort(key=lambda c:-c['penetration_mm'])
            ik.append(dict(part=name, grasp_offset_mm=offset*1000,
                position_error_mm=float(np.linalg.norm(error[:3])*1000),
                rotation_error_deg=float(np.rad2deg(np.linalg.norm(error[3:]))),
                ik_starts=len(starts), collision_count=len(collisions), deepest_collisions=collisions[:3]))
    candidates=ideal_gripper_clearance(sim,pitches=(0,10,20,35,55,75,90))
    for candidate in candidates:
        if not candidate['clear']:
            continue
        choices=[]
        for flip in (False,True):
            quat=np.array(candidate['tip_quat'])
            if flip:mujoco.mju_mulQuat(quat,quat.copy(),np.array([0.,1.,0.,0.]))
            rng=np.random.default_rng(42)
            starts=[sim._q_nominal]+[rng.uniform(sim.arm_range[:,0]+.02,sim.arm_range[:,1]-.02) for _ in range(9)]
            for start in starts:
                q,pe,re=sim.ik(candidate['tip_pos'],quat,iters=160,q_init=start,max_joint_step=None,posture_gain=0)
                choices.append((pe+.35*re,pe,re,flip,q,quat))
                if pe<.0001 and re<.001:break
        _,pe,re,flip,q,quat=min(choices,key=lambda c:c[0])
        candidate.update(ik_position_error_mm=pe*1000,ik_rotation_error_deg=float(np.rad2deg(re)),
                         roll_flip=flip,ik_solutions_tested=len(choices),q=q.tolist(),tip_quat=quat.tolist(),
                         endpoint_reachable=bool(pe<.0005 and re<.01))
    return dict(part_paths=results, gear_gripper_ik=ik, gripper_grasp_search=candidates,
                warning='Kinematic planning checks only; no grasps or physical insertion successes inferred.')


class CarrierTrial:
    def __init__(self, sim, out, video=False):
        self.sim, self.out = sim, out
        self.qctrl = sim.q.copy()
        self.joint_bias = np.zeros_like(self.qctrl)
        self.target, self.quat = sim.ee_pose()
        self.grip = 1.
        self.frames = []
        self.controls = []
        self.tracking = []
        self.stage = 'reset'
        self.stages = []
        self.renderer = self.writer = None
        if video:
            import cv2
            self.renderer = mujoco.Renderer(sim.model, 600, 960)
            self.writer = cv2.VideoWriter(str(out/'carrier_attempt.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (960,600))
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0

    def tick(self, pos, quat, grip):
        s = self.sim
        # A position servo needs a small setpoint bias to carry gravity and
        # overcome joint friction. Without it the tool sags ~0.7 mm, while
        # the gear teeth have only ~0.1 mm axial clearance along insertion.
        # Integrate against the previous IK solution, with bounded windup;
        # retain all original actuator gains, force limits and contacts.
        self.joint_bias = np.clip(self.joint_bias+.25*(self.qctrl-s.q), -.015, .015)
        q, pe, re = s.ik(pos, quat, q_init=self.qctrl, iters=100,
                        max_joint_step=.09, posture_gain=0,
                        pos_tol=1e-6, rot_tol=1e-5, min_damping=.001)
        self.qctrl = q
        s.set_arm_ctrl(q+self.joint_bias)
        s.set_gripper(grip)
        s.step(50)
        self.target, self.quat, self.grip = np.array(pos), np.array(quat), grip
        self.frames.append(s.data.qpos.copy())
        self.controls.append(s.data.ctrl.copy())
        error = s._ee_error(pos, quat)
        self.tracking.append([pe, re, np.linalg.norm(error[:3]), np.linalg.norm(error[3:])])
        if self.writer:
            import cv2
            self.renderer.update_scene(s.data, camera='overview', scene_option=self.option)
            image = self.renderer.render().copy()
            cv2.putText(image, 'ACTUATOR / contact-only assembly', (20,30), cv2.FONT_HERSHEY_SIMPLEX,.65,(235,235,235),1,cv2.LINE_AA)
            cv2.putText(image, self.stage.replace('_',' '), (20,58),cv2.FONT_HERSHEY_SIMPLEX,.6,(70,200,255),1,cv2.LINE_AA)
            self.writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        if not np.isfinite(s.data.qpos).all():
            raise RuntimeError('Nonfinite physics state')

    def move(self, stage, pos, quat=None, grip=None, duration=None):
        self.stage = stage
        print('Stage:', stage, flush=True)
        self.stages.append(dict(name=stage, frame=len(self.frames)))
        pos = np.asarray(pos)
        quat = self.quat.copy() if quat is None else np.array(quat)
        self.stages[-1].update(target_position=pos.tolist(), target_quaternion=quat.tolist())
        grip = self.grip if grip is None else grip
        p0,q0,g0 = self.target.copy(),self.quat.copy(),self.grip
        if q0@quat<0:quat=-quat
        duration = duration or max(1., np.linalg.norm(pos-p0)/.035, 2*np.arccos(np.clip(q0@quat,-1,1))/.7)
        steps = math.ceil(duration*20)
        for i in range(1,steps+1):
            u=i/steps;u=u*u*(3-2*u)
            q=(1-u)*q0+u*quat;q/=np.linalg.norm(q)
            self.tick((1-u)*p0+u*pos,q,(1-u)*g0+u*grip)
        for _ in range(10):self.tick(pos,quat,grip)
        pe, re = self.tracking[-1][:2]
        self.stages[-1].update(ik_position_error_mm=pe*1000,
                              ik_rotation_error_deg=float(np.rad2deg(re)),
                              tracking_position_error_mm=self.tracking[-1][2]*1000)
        if pe > .0001 or re > .001:
            raise RuntimeError(f'{stage}: IK target unreachable ({pe*1000:.3f} mm, '
                               f'{np.rad2deg(re):.3f} degrees)')

    def gear_trial(self, name='gear_3'):
        """Pinch the exposed outer rim with a vertical tool, then insert radially."""
        s=self.sim; bid=s.model.body(name).id
        start=s.data.xpos[bid].copy()
        target,target_quat=s.assembled_pose(name)
        offset,rotation,outward=gear_grasp(s,name)
        small_rotation=s.data.xmat[s.model.body('small_carrier').id].reshape(3,3)
        initial_rotation=s.data.xmat[bid].reshape(3,3)
        relative=initial_rotation@small_rotation.T
        pickup=start+relative@offset
        pickup_quat=mat_to_quat(relative@rotation)
        insert=target+offset
        insert_quat=mat_to_quat(rotation)
        self.move(name+'_approach',pickup+[0,0,.060],pickup_quat,1.)
        self.move(name+'_grasp',pickup,pickup_quat,1.)
        self.move(name+'_close',pickup,pickup_quat,0.,2.)
        self.move(name+'_lift',pickup+[0,0,.05],pickup_quat,0.)
        lift=float(s.data.xpos[bid,2]-start[2])
        result=dict(part=name,lift_m=lift,grasp='outer rim, offset 18 mm, pitch 90 degrees')
        if lift<.03:
            result.update(success=False,failure='gear_pickup_failed')
            return result
        # Estimate the actual in-hand transform after pickup: the shallow rim
        # grasp can tilt the gear on its rack. Correct that measured tilt before
        # entering the slot, rather than assuming an ideal rigid grasp pose.
        desired_rotation=np.zeros(9)
        mujoco.mju_quat2Mat(desired_rotation,target_quat)
        correction=desired_rotation.reshape(3,3)@s.data.xmat[bid].reshape(3,3).T
        tool_rotation=s.data.site_xmat[s.ee_site].reshape(3,3)
        offset=correction@(s.ee_pos()-s.data.xpos[bid])
        insert=target+offset
        insert_quat=mat_to_quat(correction@tool_rotation)
        result['measured_grasp_correction']=True
        above=insert+outward*.030;above[2]=.100
        self.move(name+'_transfer',above,insert_quat,0.)
        self.move(name+'_align_side_slot',insert+outward*.030,insert_quat,0.)
        self.move(name+'_radial_insertion',insert,insert_quat,0.,4.)
        result['pre_release_error_mm']=float(np.linalg.norm(s.data.xpos[bid]-target)*1000)
        release_grip=min(1.,float(s.data.qpos[s.finger_qadr[0]]+.004)/.04)
        self.move(name+'_release',insert,insert_quat,release_grip,1.)
        self.move(name+'_withdraw',insert+outward*.030,insert_quat,release_grip)
        self.move(name+'_open_clear',self.target,grip=1.,duration=.5)
        self.move(name+'_raise_clear',insert+[0,0,.045])
        self.move(name+'_settle',self.target,duration=2.)
        # CAD leaves 2.24 mm axial clearance below each gear. Without pins,
        # a released gear may settle onto the lower plate. This is a placement
        # metric, never a retained or functioning gear-train success claim.
        result.update(gear_placement(s, name))
        if not result['success']:result['failure']='gear_not_seated_after_release'
        return result

    def pin_trial(self, name, *, allow_partial_entry=False):
        """Guide a pin into its bore, release, then press with a finger edge."""
        s = self.sim
        bid = s.model.body(name).id
        start = s.data.xpos[bid].copy()
        target, target_quat = pin_target_pose(s, name)
        _, rotation, outward = gear_grasp(s, PIN_GEARS[name])
        quat = mat_to_quat(rotation)
        pickup = start+np.array([0., 0., .013])
        self.move(name+'_approach', pickup+[0, 0, .040], quat, 1.)
        self.move(name+'_grasp', pickup, quat, 1.)
        self.move(name+'_close', pickup, quat, 0., 1.)
        self.move(name+'_lift', pickup+[0, 0, .040], quat, 0.)
        lift = float(s.data.xpos[bid, 2]-start[2])
        result = dict(part=name, lift_m=lift)
        if lift < .025:
            result.update(success=False, failure='pin_pickup_failed')
            return result
        desired_rotation = s.data.xmat[s.model.body('large_carrier').id].reshape(3, 3)
        correction = desired_rotation@s.data.xmat[bid].reshape(3, 3).T
        offset = correction@(s.ee_pos()-s.data.xpos[bid])
        insert_quat = mat_to_quat(correction@s.data.site_xmat[s.ee_site].reshape(3, 3))
        self.move(name+'_transfer', target+offset+[0, 0, .040], insert_quat, 0.)
        insert = target+offset+[0, 0, .014]
        self.move(name+'_guide_into_bore', insert, insert_quat, 0., 4.)
        # First engage the upper carrier, then align with the released gear's
        # measured bore. Its small settling tilt otherwise catches the blunt
        # pin on the gear face. The free carrier can settle laterally as the
        # pin is guided; no part pose or constraint is set by this controller.
        gear = s.model.body(PIN_GEARS[name]).id
        gear_rotation = s.data.xmat[gear].reshape(3, 3).copy()
        pin_rotation = s.data.xmat[bid].reshape(3, 3).copy()
        local_offset = pin_rotation.T@(s.ee_pos()-s.data.xpos[bid])
        local_rotation = pin_rotation.T@s.data.site_xmat[s.ee_site].reshape(3, 3)
        origins = s.manifest['parts']
        pin_in_gear = np.array(origins[name]['assembled_origin'])-origins[PIN_GEARS[name]]['assembled_origin']
        bore_target = s.data.xpos[gear]+gear_rotation@pin_in_gear
        insert_quat = mat_to_quat(gear_rotation@local_rotation)
        self.move(name+'_align_gear', bore_target+gear_rotation@(local_offset+[0, 0, .014]), insert_quat, 0., 3.)
        insert = bore_target+gear_rotation@(local_offset+[0, 0, .008])
        self.move(name+'_enter_gear', insert, insert_quat, 0., 4.)
        gear_rotation = s.data.xmat[gear].reshape(3, 3)
        pin_bottom = gear_rotation.T@(s.data.xpos[bid]-s.data.xpos[gear])
        result['guided_depth_in_gear_mm'] = float((origins[PIN_GEARS[name]]['bounds'][1][2]-pin_bottom[2])*1000)
        guided = pin_placement(s, name)
        shallow_entry = allow_partial_entry and pin_entry_ready(s, name)
        result['shallow_entry_before_press'] = bool(shallow_entry and not guided['through_carrier'])
        full_guide = (guided['through_carrier'] and guided['carrier_bore_error_mm'] is not None and
                      guided['carrier_bore_error_mm'] <= .25 and guided['tilt_deg'] <= 5.)
        if not (full_guide or shallow_entry):
            result.update(success=False, failure='pin_not_guided_by_carrier')
            return result
        release_grip = min(1., float(s.data.qpos[s.finger_qadr[0]]+.004)/.04)
        self.move(name+'_release', insert, insert_quat, release_grip, 1.)
        self.move(name+'_clear_pin', insert+[0, 0, .025], insert_quat, release_grip)
        for attempt in range(2):
            self.press_pin(name, outward, quat, retry=attempt > 0)
            result['press_attempts'] = attempt+1
            if pin_placement(s, name)['success']:
                break
        result.update(pin_placement(s, name))
        if not result['success']:
            result['failure'] = 'pin_not_seated_after_release'
        return result

    def press_pin(self, name, outward, quat, *, retry=False):
        s = self.sim
        bid = s.model.body(name).id
        height = s.manifest['parts'][name]['bounds'][1][2]
        prefix = name+('_retry' if retry else '')
        # A 5 mm offset clears the central hub but covers the complete pin
        # head. A larger offset loads one edge and can lever the pin out.
        axis = s.data.xmat[bid].reshape(3, 3)[:, 2]
        head = s.data.xpos[bid]+axis*height
        self.move(prefix+'_position_press', head+outward*.005+[0, 0, .004], quat, 0.)
        axis = s.data.xmat[bid].reshape(3, 3)[:, 2]
        head = s.data.xpos[bid]+axis*height
        stop, _ = s.assembled_pose(name)
        press = head+outward*.005
        press[2] = stop[2]+axis[2]*height+.0001
        self.move(prefix+'_press', press, quat, 0., 5.)
        self.move(prefix+'_press_retreat', self.target+[0, 0, .040], quat, 0.)
        self.move(prefix+'_settle', self.target, quat, 1., 2.)

    def run(self, try_gear=False, all_gears=False, with_pins=False):
        s=self.sim
        if with_pins and not s.with_pins:
            raise ValueError('Pin insertion requires ActuatorSim(with_pins=True)')
        all_gears = all_gears or with_pins
        s.reset()
        self.qctrl = s.q.copy()
        self.joint_bias[:] = 0
        self.target, self.quat = s.ee_pose()
        self.grip = 1.
        bid=s.model.body('large_carrier').id
        start=s.data.xpos[bid].copy()
        yaw=2*np.arctan2(s.data.xquat[bid,3],s.data.xquat[bid,0])
        c,si=np.cos(yaw),np.sin(yaw)
        rz=np.array([[c,-si,0],[si,c,0],[0,0,1]])
        down=np.array([[0,0,1],[0,1,0],[-1,0,0.]])
        quat=mat_to_quat(rz@down)
        grasp=start+np.array([0,0,.004])
        target,tquat=s.assembled_pose('large_carrier')
        target_yaw=2*np.arctan2(tquat[3],tquat[0])
        c,si=np.cos(target_yaw),np.sin(target_yaw)
        tq=mat_to_quat(np.array([[c,-si,0],[si,c,0],[0,0,1]])@down)
        result=dict(seed=s.seed, mode='physical_contact_only', grasp_welds=False,
                    small_carrier_fixture=True, pins_installed=False)
        try:
            self.move('approach_carrier',grasp+[0,0,.055],quat)
            self.move('lower_to_grasp',grasp,quat)
            self.move('close_gripper',grasp,quat,0.,2.)
            self.move('lift_carrier',grasp+[0,0,.055],quat,0.)
            lift=float(s.data.xpos[bid,2]-start[2]); result['lift_m']=lift
            if lift<.035:
                result.update(success=False,failure='carrier_pickup_failed')
                return result
            self.move('transfer_above_small_carrier',target+[0,0,.045],tq,0.)
            self.move('seat_carrier',target+[0,0,.004],tq,0.,3.)
            result['pre_release_error_mm']=float(np.linalg.norm(s.data.xpos[bid]-target)*1000)
            self.move('release',self.target,grip=1.,duration=1.)
            self.move('retreat',target+[0,0,.05],tq,1.)
            self.move('settle',self.target,duration=2.)
            delta=s.data.xpos[bid]-target
            angle=2*np.arccos(np.clip(abs(s.data.xquat[bid]@tquat),0,1))
            result.update(final_position_error_mm=float(np.linalg.norm(delta)*1000),
                final_xy_error_mm=float(np.linalg.norm(delta[:2])*1000),
                final_orientation_error_deg=float(np.rad2deg(angle)),
                success=bool(np.linalg.norm(delta)<.0015 and angle<np.deg2rad(4)))
            if not result['success']:result['failure']='carrier_not_seated_after_release'
            result['carrier_success'] = result['success']
            if result['success'] and (try_gear or all_gears):
                result['gear_trials'] = []
                for name in (['gear_1','gear_2','gear_3','gear_4'] if all_gears else ['gear_3']):
                    gear = self.gear_trial(name)
                    result['gear_trials'].append(gear)
                    if not gear['success']:
                        break
                if not all_gears:
                    result['gear_trial'] = result['gear_trials'][0]
                result['carrier_after_gear_error_mm']=float(np.linalg.norm(s.data.xpos[bid]-target)*1000)
                result['final_gears'] = {g['part']: gear_placement(s, g['part']) for g in result['gear_trials']}
                carrier_angle = 2*np.arccos(np.clip(abs(s.data.xquat[bid]@tquat),0,1))
                result['carrier_after_gear_orientation_deg'] = float(np.rad2deg(carrier_angle))
                result['success'] = bool(all(g['success'] for g in result['gear_trials']) and
                                         all(g['success'] for g in result['final_gears'].values()) and
                                         result['carrier_after_gear_error_mm'] < 1.5 and
                                         carrier_angle < np.deg2rad(4))
                if not result['success']:
                    result['failure'] = 'assembly_not_seated_after_release'
            if result['success'] and with_pins:
                result['pin_trials'] = []
                for name in ['pin_3', 'pin_4', 'pin_2', 'pin_1']:
                    pin = self.pin_trial(name)
                    result['pin_trials'].append(pin)
                    if not pin['success']:
                        break
                result['final_pins'] = {name: pin_placement(s, name) for name in PIN_GEARS}
                result['final_gears'] = {name: gear_placement(s, name) for name in PIN_GEARS.values()}
                result['carrier_after_pins_error_mm'] = float(np.linalg.norm(s.data.xpos[bid]-target)*1000)
                carrier_angle = 2*np.arccos(np.clip(abs(s.data.xquat[bid]@tquat), 0, 1))
                result['carrier_after_pins_orientation_deg'] = float(np.rad2deg(carrier_angle))
                result['pins_installed'] = all(p['success'] for p in result['final_pins'].values())
                result['success'] = bool(result['pins_installed'] and
                                         all(g['success'] for g in result['final_gears'].values()) and
                                         result['carrier_after_pins_error_mm'] < 1.5 and
                                         carrier_angle < np.deg2rad(4))
                if not result['success']:
                    result['failure'] = 'pin_assembly_not_seated_after_release'
            return result
        except RuntimeError as exc:
            result.update(success=False, failure=str(exc))
            return result
        finally:
            np.savez_compressed(self.out/'carrier_rollout.npz',qpos=np.asarray(self.frames),ctrl=np.asarray(self.controls),
                                tracking=np.asarray(self.tracking),fps=20)
            (self.out/'stages.json').write_text(json.dumps(self.stages,indent=2)+'\n')
            (self.out/'metadata.json').write_text(json.dumps(dict(seed=s.seed,fps=20,
                scene=str(s.scene_path.relative_to(ROOT)),scene_sha256=hashlib.sha256(s.scene_path.read_bytes()).hexdigest(),
                mujoco_version=mujoco.__version__,xy_jitter_m=s.xy_jitter,yaw_jitter_deg=s.yaw_jitter_deg,
                mode='physical_contact_only',with_pins=s.with_pins,pins_installed=result['pins_installed'],
                tracking_columns=['ik_position_error_m','ik_rotation_error_rad',
                                  'actual_position_error_m','actual_rotation_error_rad'],
                joint_bias_limit_rad=.015),indent=2)+'\n')
            if self.writer:self.writer.release();self.renderer.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/actuator')
    p.add_argument('--video',action='store_true')
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--trial-only',action='store_true')
    p.add_argument('--try-gear',action='store_true',help='After carrier success, physically attempt the near-side gear')
    p.add_argument('--all-gears',action='store_true',help='Physically place the carrier and all four gears')
    p.add_argument('--with-pins',action='store_true',help='Place the carrier, all four gears, and their four pins')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sim=ActuatorSim(seed=args.seed, with_pins=args.with_pins)
    result={}
    if not args.trial_only:
        result['audit']=audit(sim)
        (args.output/'audit.json').write_text(json.dumps(result['audit'],indent=2)+'\n')
        print(json.dumps(result['audit'],indent=2),flush=True)
    if not args.audit_only:
        sim.reset()
        result['carrier_trial']=CarrierTrial(sim,args.output,args.video).run(
            try_gear=args.try_gear, all_gears=args.all_gears, with_pins=args.with_pins)
        print(json.dumps(result['carrier_trial'],indent=2),flush=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if 'carrier_trial' in result and not result['carrier_trial']['success']:
        raise SystemExit(1)


if __name__=='__main__':main()
