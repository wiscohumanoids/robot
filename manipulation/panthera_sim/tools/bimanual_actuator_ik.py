"""Kept-session-inspired IK: hold carrier, half-place, push gear, place/press pin.

Simulation only. Uses privileged reset/grasp geometry; contact seating is
physical, with no object teleports, welds or success snaps during the rollout.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from sim.bimanual_actuator_env import BimanualActuatorSim, CAMERAS, STATE_NAMES
from sim.panthera_env import mat_to_quat
from tools.try_actuator_assembly import CarrierTrial, gear_grasp, gear_placement, pin_placement


def smoothstep(u):
    """Quintic ease with zero velocity and acceleration at both endpoints."""
    return u*u*u*(10+u*(-15+6*u))


class BimanualTrial(CarrierTrial):
    def __init__(self, sim, out, video=False, placement_error=.00035, holder_angle_deg=45.):
        super().__init__(sim, out, False)
        self.arm = 'right'
        self.commands = {n: a.q.copy() for n, a in sim.arms.items()}
        self.ideal = {n: a.q.copy() for n, a in sim.arms.items()}
        self.bias = {n: np.zeros(6) for n in sim.arms}
        self.velocity = {n: np.zeros(6) for n in sim.arms}
        self.poses = {n: a.ee_pose() for n, a in sim.arms.items()}
        self.grips = {n: 1. for n in sim.arms}
        self.target, self.quat = self.poses[self.arm]
        self.placement_error = placement_error
        self.holder_angle_deg = holder_angle_deg
        self.contact_mode = False
        self.states, self.actions, self.stage_ids, self.contact_loads = [], [], [], []
        self.renderer = self.writer = None
        if video:
            import cv2
            self.renderer = mujoco.Renderer(sim.model, 300, 400)
            self.writer = cv2.VideoWriter(str(out/'bimanual.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (1200,300))

    def select(self, side):
        self.arm = side
        self.target, self.quat = (v.copy() for v in self.poses[side])
        self.grip = self.grips[side]

    def tool_load(self, side):
        prefix = '' if side == 'left' else 'right_'
        bodies = {self.sim.model.body(prefix+n).id for n in ('link6', 'L_finger', 'R_finger')}
        load, force = 0., np.zeros(6)
        for i, c in enumerate(self.sim.data.contact):
            b1, b2 = self.sim.model.geom_bodyid[[c.geom1, c.geom2]]
            if (b1 in bodies) != (b2 in bodies):
                mujoco.mj_contactForce(self.sim.model, self.sim.data, i, force)
                load += max(0., force[0])
        return load

    def tick(self, pos, quat, grip):
        s = self.sim
        self.poses[self.arm] = (np.asarray(pos).copy(), np.asarray(quat).copy())
        self.grips[self.arm] = grip
        errors = {}
        for side, a in s.arms.items():
            # Warm-start both arms every tick: the supporting arm stays active.
            p, q = self.poses[side]
            desired, pe, re = a.ik(p, q, q_init=self.ideal[side], iters=80,
                                  max_joint_step=.03, posture_gain=0, min_damping=.001,
                                  pos_tol=.00002 if self.contact_mode else .00015,
                                  rot_tol=.001)
            # Bound joint velocity and acceleration on the actual command,
            # including integral compensation, rather than just IK iterations.
            if not (side == self.arm and self.contact_mode):
                self.bias[side] = np.clip(self.bias[side]+.15*(self.ideal[side]-a.q), -.012, .012)
            self.ideal[side] = desired.copy()
            desired = np.clip(desired+self.bias[side], *a.arm_range.T)
            velocity = np.clip((desired-self.commands[side])/.05, -.6, .6)
            velocity = np.clip(velocity, self.velocity[side]-.1, self.velocity[side]+.1)
            self.commands[side] = np.clip(self.commands[side]+.05*velocity, *a.arm_range.T)
            self.velocity[side] = velocity
            a.set_arm_ctrl(self.commands[side])
            a.set_gripper(self.grips[side])
            errors[side] = (pe, re)
        s.step(50)
        self.target, self.quat, self.grip = np.asarray(pos).copy(), np.asarray(quat).copy(), grip
        self.frames.append(s.data.qpos.copy())
        self.controls.append(s.data.ctrl.copy())
        a = s.arms[self.arm]
        err = a._ee_error(pos, quat)
        self.tracking.append([*errors[self.arm], np.linalg.norm(err[:3]), np.linalg.norm(err[3:])])
        self.states.append(s.state()); self.actions.append(s.action())
        self.stage_ids.append(len(self.stages)-1)
        self.contact_loads.append([self.tool_load(side) for side in s.arms])
        if self.writer:
            import cv2
            panels = []
            for camera in CAMERAS:
                self.renderer.update_scene(s.data, camera=camera, scene_option=self.option)
                panel = self.renderer.render().copy()
                cv2.putText(panel, camera, (10,20), cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                cv2.putText(panel, self.stage, (10,285), cv2.FONT_HERSHEY_SIMPLEX,.4,(80,220,255),1)
                panels.append(panel)
            self.writer.write(cv2.cvtColor(np.concatenate(panels,axis=1), cv2.COLOR_RGB2BGR))
        if not np.isfinite(s.data.qpos).all():
            raise RuntimeError('Nonfinite physics state')

    def move(self, stage, pos, quat=None, grip=None, duration=None):
        self.stage = stage
        pos = np.asarray(pos, dtype=float)
        quat = self.quat.copy() if quat is None else np.asarray(quat, dtype=float)
        grip = self.grip if grip is None else grip
        p0, q0, g0 = self.target.copy(), self.quat.copy(), self.grip
        if q0@quat < 0: quat = -quat
        angle = 2*np.arccos(np.clip(q0@quat, -1, 1))
        # Quintic peak speed is 1.875 times average speed.
        duration = max(duration or 0., 1., 1.875*np.linalg.norm(pos-p0)/.04, 1.875*angle/.6)
        steps = math.ceil(duration*20)
        self.stages.append(dict(name=stage, arm=self.arm, frame=len(self.frames),
                                contact_mode=self.contact_mode, duration_s=steps/20,
                                target_position=pos.tolist(), target_quaternion=quat.tolist()))
        print(f'{self.arm}: {stage}', flush=True)
        for i in range(1, steps+1):
            u = smoothstep(i/steps)
            if angle < 1e-8:
                q = q0.copy()
            else:
                half = angle/2
                q = (np.sin((1-u)*half)*q0+np.sin(u*half)*quat)/np.sin(half)
            self.tick(p0+u*(pos-p0), q, g0+u*(grip-g0))
        for _ in range(12): self.tick(pos, quat, grip)
        pe, re, tracking, _ = self.tracking[-1]
        self.stages[-1].update(ik_position_error_mm=pe*1000, ik_rotation_error_deg=float(np.rad2deg(re)),
                               tracking_position_error_mm=tracking*1000)
        # Relax ordinary waypoints, while rejecting truly unreachable motions.
        if pe > .003 or re > .04 or tracking > .008:
            raise RuntimeError(f'{stage}: unresolved motion ({pe*1000:.2f} mm IK, {tracking*1000:.2f} mm tracking)')

    def push(self, stage, end, direction, tangent, travel, duration=5., search_radius=.0003):
        """Force-limited advancing push plus small lateral search; no bore tracking.

        A blocked push pauses axial travel and searches around the approximate
        target. It never silently substitutes a perfectly measured part pose.
        """
        self.stage = stage
        print(f'{self.arm}: {stage}', flush=True)
        end = np.asarray(end)
        start = end-np.asarray(direction)*travel
        self.stages.append(dict(name=stage, arm=self.arm, frame=len(self.frames), contact_mode=True,
                                target_position=end.tolist(), start_position=start.tolist(),
                                travel_m=travel, direction=np.asarray(direction).tolist(),
                                search_radius_mm=search_radius*1000,
                                force_limit_n=20., maximum_force_n=35.))
        self.contact_mode = True
        progress = 0.
        maximum_load = 0.
        try:
            for i in range(math.ceil(duration*20)):
                load = self.tool_load(self.arm); maximum_load = max(maximum_load, load)
                if load > 35.:
                    raise RuntimeError(f'{stage}: contact load exceeded 35 N')
                progress = min(1., progress+max(0., 1-load/20.)/(duration*20))
                search = search_radius*np.sin(6*np.pi*i/(duration*20))*np.sin(np.pi*progress)
                p = start+(end-start)*smoothstep(progress)+tangent*search
                self.tick(p, self.quat, self.grip)
            self.stages[-1].update(progress=progress, maximum_contact_load_n=maximum_load)
        finally:
            self.contact_mode = False

    def press_pin(self, name, outward, quat, *, retry=False):
        s = self.sim
        bid = s.model.body(name).id
        height = s.manifest['parts'][name]['bounds'][1][2]
        head = s.data.xpos[bid]+s.data.xmat[bid].reshape(3,3)[:,2]*height
        prefix = name+('_retry' if retry else '')
        approach = head+outward*.005+[0,0,.004]
        self.move(prefix+'_position_press', approach, quat, 0.)
        stop, _ = s.assembled_pose(name)
        end = approach.copy(); end[2] = stop[2]+height+.0001
        travel = max(0., approach[2]-end[2])
        self.push(prefix+'_seating_press', end, np.array([0.,0.,-1.]),
                  np.cross([0,0,1.], outward), travel, 6., search_radius=.00004)
        self.move(prefix+'_press_retreat', self.target+[0,0,.04], quat, 0.)
        self.move(prefix+'_supported_settle', self.target, quat, 1., 2.)

    def hold_carrier(self):
        s = self.sim; a = s.arms['right']; bid = s.model.body('large_carrier').id
        start = s.data.xpos[bid].copy()
        rotation = s.data.xmat[bid].reshape(3,3).copy()
        down = np.array([[0,0,-1],[0,-1,0],[-1,0,0.]])
        angle = np.deg2rad(self.holder_angle_deg)
        tilt = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        # The wrist leans back toward its base, clearing the working arm.
        quat = mat_to_quat(rotation@tilt@down)
        pickup = start+rotation@np.array([-.01*np.sin(angle),0,.004])
        self.move('carrier_approach', pickup+[0,0,.055], quat)
        self.move('carrier_grasp', pickup, quat)
        self.move('carrier_close', pickup, quat, 0., 2.)
        self.move('carrier_lift', pickup+[0,0,.055], quat, 0.)
        lift = float(s.data.xpos[bid,2]-start[2])
        if lift < .035: raise RuntimeError('carrier_pickup_failed')
        target, target_quat = s.assembled_pose('large_carrier')
        desired = np.zeros(9); mujoco.mju_quat2Mat(desired, target_quat)
        correction = desired.reshape(3,3)@s.data.xmat[bid].reshape(3,3).T
        offset = correction@(a.ee_pos()-s.data.xpos[bid])
        quat = mat_to_quat(correction@s.data.site_xmat[a.ee_site].reshape(3,3))
        self.move('carrier_transfer', target+offset+[0,0,.04], quat, 0.)
        self.move('carrier_supported_hold', target+offset, quat, 0., 3.)
        return dict(lift_m=lift, held_by='right', holder_angle_deg=self.holder_angle_deg,
                    fixture_under_small_carrier=True)

    def place_push_gear(self):
        s = self.sim; a = s.arms['left']; name = 'gear_3'; bid = s.model.body(name).id
        start = s.data.xpos[bid].copy()
        target, target_quat = s.assembled_pose(name)
        offset, rotation, outward = gear_grasp(s, name)
        initial = s.data.xmat[bid].reshape(3,3)
        fixture = s.data.xmat[s.model.body('small_carrier').id].reshape(3,3)
        relative = initial@fixture.T
        pickup = start+relative@offset; quat = mat_to_quat(relative@rotation)
        self.move('gear_approach', pickup+[0,0,.05], quat, 1.)
        self.move('gear_grasp', pickup, quat)
        self.move('gear_close', pickup, quat, 0., 1.5)
        self.move('gear_lift', pickup+[0,0,.05], quat)
        lift = float(s.data.xpos[bid,2]-start[2])
        if lift < .03: raise RuntimeError('gear_pickup_failed')
        desired = np.zeros(9); mujoco.mju_quat2Mat(desired, target_quat)
        correction = desired.reshape(3,3)@s.data.xmat[bid].reshape(3,3).T
        offset = correction@(a.ee_pos()-s.data.xpos[bid])
        quat = mat_to_quat(correction@s.data.site_xmat[a.ee_site].reshape(3,3))
        tangent = np.cross([0,0,1.], outward)
        error = tangent*np.random.default_rng(s.seed+1000).uniform(-self.placement_error, self.placement_error)
        insert = target+offset+error
        # Transfer in the slot plane: crossing above the carrier would sweep
        # the long working wrist through the supporting forearm.
        low = self.target.copy(); low[2] = insert[2]
        self.move('gear_lower_to_transfer_plane', low, quat)
        self.move('gear_transfer', insert+outward*.03, quat)
        self.move('gear_slot_approach', insert+outward*.03, quat)
        self.move('gear_half_insert', insert+outward*.004, quat, duration=4.)
        # Keep a shallow rim pinch while pushing the partially engaged gear.
        # Releasing at this point lets the 2.24 mm axial clearance tip its
        # unsupported edge into the plate; contact search cannot fix that jam.
        half = gear_placement(s, name)
        self.push('gear_supported_seating_push', insert, -outward, tangent, .004, 8.)
        release = min(1., float(s.data.qpos[a.finger_qadr[0]]+.004)/.04)
        self.move('gear_release_after_push', self.target, grip=release)
        self.move('gear_clear_rim', self.target+outward*.03, grip=release)
        self.move('gear_close_for_push', self.target, grip=0.)
        # Closed fingers contact the outer tooth rim with their inner edge.
        press = target+outward*.0207+np.array([0,0,.0005])
        self.move('gear_push_approach', press+outward*.012, quat, 0.)
        before = gear_placement(s, name)
        self.push('gear_seating_push', press, -outward, tangent, .012, 8.)
        self.move('gear_push_retreat', self.target+outward*.03, quat, 0.)
        # The CAD has 2.24 mm of clearance below the gear. A gentle downward
        # rim press lets it settle flat before the pin enters two separate bores.
        top = target+outward*.022+np.array([0,0,.0082])
        self.move('gear_flatten_approach', top+[0,0,.004], quat, 0.)
        self.push('gear_flatten_press', top-[0,0,.00224], np.array([0.,0.,-1.]),
                  tangent, .00624, 6., search_radius=.00004)
        self.move('gear_flatten_retreat', self.target+outward*.02, quat, 0.)
        self.move('gear_clear_for_pin', self.target+[0,0,.04], quat, 1.)
        after = gear_placement(s, name)
        return dict(lift_m=lift, commanded_lateral_error_mm=float(np.linalg.norm(error)*1000),
                    half_insert_remaining_mm=4., half_placement=half, before_push=before, after_push=after,
                    success=after['success'])

    def run(self, **unused):
        result = dict(seed=self.sim.seed, method='bimanual_half_place_then_push', success=False,
                      xy_jitter_m=self.sim.xy_jitter, yaw_jitter_deg=self.sim.yaw_jitter_deg,
                      placement_error_bound_mm=self.placement_error*1000,
                      holder_angle_deg=self.holder_angle_deg,
                      cameras=CAMERAS, state_names=STATE_NAMES, state_dimension=14,
                      gripper_units='per-finger opening in metres; real recordings use motor radians',
                      object_teleportation_during_rollout=False, grasp_welds=False,
                      source_camera_assignment='unverified; synthetic cameras use arm names')
        try:
            self.select('left')
            _, rotation, _ = gear_grasp(self.sim, 'gear_3')
            self.move('left_clear_support_workspace', [.24, -.14, .14], mat_to_quat(rotation), 1.)
            self.select('right')
            result['carrier'] = self.hold_carrier()
            self.select('left')
            result['gear'] = self.place_push_gear()
            if not result['gear']['success']:
                raise RuntimeError('gear_not_seated_after_push')
            # Reuse physical pickup and guided partial placement. Its final
            # release/closed-finger press follows the observed second push.
            # Adapter selects the left arm while retaining shared part geometry.
            original = self.sim
            adapter = LeftTaskView(original)
            self.sim = adapter
            try:
                result['pin'] = self.pin_trial('pin_2', allow_partial_entry=True)
            finally:
                self.sim = original
            self.move('final_supported_hold', self.target, duration=2.)
            result['final_gear'] = gear_placement(self.sim, 'gear_3')
            result['final_pin'] = pin_placement(self.sim, 'pin_2')
            carrier_pos, carrier_quat = self.sim.assembled_pose('large_carrier')
            bid = self.sim.model.body('large_carrier').id
            carrier_error = np.linalg.norm(self.sim.data.xpos[bid]-carrier_pos)
            angle = 2*np.arccos(np.clip(abs(self.sim.data.xquat[bid]@carrier_quat),0,1))
            result['final_carrier_error_mm'] = float(carrier_error*1000)
            result['success'] = bool(result['pin']['success'] and result['final_gear']['success']
                                     and result['final_pin']['success'] and carrier_error < .0015
                                     and angle < np.deg2rad(4))
            result['terminal_condition'] = 'supported assembly; carrier gripper remains closed'
            if not result['success']:
                result['failure'] = result['pin'].get('failure', 'final_supported_assembly_audit_failed')
        except RuntimeError as exc:
            result['failure'] = str(exc)
        finally:
            if self.writer: self.writer.release(); self.renderer.close()
            if self.actions:
                joint_columns = [0,1,2,3,4,5,7,8,9,10,11,12]
                q = np.asarray(self.actions)[:, joint_columns]
                velocity = np.diff(q, axis=0)/.05
                acceleration = np.diff(velocity, axis=0)/.05
                result['maximum_joint_command_speed_rad_s'] = float(np.abs(velocity).max()) if len(velocity) else 0.
                result['maximum_joint_command_acceleration_rad_s2'] = float(np.abs(acceleration).max()) if len(acceleration) else 0.
            np.savez_compressed(self.out/'rollout.npz', qpos=np.asarray(self.frames), ctrl=np.asarray(self.controls),
                                state=np.asarray(self.states), action=np.asarray(self.actions),
                                stage_id=np.asarray(self.stage_ids), contact_load_n=np.asarray(self.contact_loads),
                                tracking=np.asarray(self.tracking), fps=20,
                                initial_fixture_position=self.sim.model.body_pos[self.sim.model.body('small_carrier').id],
                                initial_fixture_quaternion=self.sim.model.body_quat[self.sim.model.body('small_carrier').id])
            (self.out/'stages.json').write_text(json.dumps(self.stages, indent=2))
            (self.out/'result.json').write_text(json.dumps(result, indent=2))
        return result


class LeftTaskView:
    """Only legacy pin helpers need single-arm addresses; physics stays shared."""
    def __init__(self, sim):
        self.parent = sim

    def __getattr__(self, name):
        arm = self.parent.arms['left']
        if name in ('ee_pos', 'ee_pose', 'ee_site', 'finger_qadr', 'arm_qadr', 'q', 'ik'):
            return getattr(arm, name)
        return getattr(self.parent, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/bimanual_ik/seed_0')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--xy-jitter', type=float, default=.002)
    parser.add_argument('--yaw-jitter-deg', type=float, default=2.)
    parser.add_argument('--placement-error-mm', type=float, default=.35)
    parser.add_argument('--holder-angle-deg', type=float, default=45.,
                        help='Carrier wrist lean toward the assembly (0 gives upright grip)')
    args = parser.parse_args()
    if args.xy_jitter < 0 or args.yaw_jitter_deg < 0 or args.placement_error_mm < 0:
        parser.error('Jitter/error ranges must be nonnegative')
    if not 0 <= args.holder_angle_deg <= 45:
        parser.error('Holder angle must be between 0 and 45 degrees')
    args.output.mkdir(parents=True, exist_ok=True)
    sim = BimanualActuatorSim(args.seed, args.xy_jitter, args.yaw_jitter_deg)
    result = BimanualTrial(sim, args.output, args.video, args.placement_error_mm/1000,
                           args.holder_angle_deg).run()
    print(json.dumps(result, indent=2))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
