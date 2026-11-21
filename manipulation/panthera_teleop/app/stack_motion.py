"""Plan a two-block stack in the selected Panthera base frame and execute it from feedback."""

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import numpy as np

from remote_inference import GRIPPER_ENCODER_TOLERANCE_RAD, SmoothActionTrajectory

SIM_ROOT = Path(__file__).resolve().parents[2] / 'panthera_sim'


@dataclass
class StackPlan:
    initial: np.ndarray
    waypoints: dict
    open_rad: float
    closed_rad: float
    gripper_limits: tuple
    tcp_path: np.ndarray

    @property
    def phases(self):
        return tuple(self.waypoints)


def _vector(value, length, label):
    value = np.asarray(value, dtype=float)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f'{label} must contain {length} finite values')
    return value


def build_plan(initial, lower, upper, source, base, *, gripper_open_rad, gripper_closed_rad,
               table_z_m=None):
    """Preflight a level-bench transfer between two stud-centroid rows, in base-frame meters.

    The hardware joint zeroes and axes must match the stock Panthera-HT model.
    The two encoder endpoints are measured with empty jaws on the selected arm.
    """
    import mujoco
    if str(SIM_ROOT) not in sys.path:
        sys.path.insert(0, str(SIM_ROOT))
    from sim.mega_bloks import BODY_HEIGHT_M, PITCH_M, STUD_HEIGHT_M
    from sim.mega_bloks_motion import _slerp
    from sim.mega_bloks_stack_motion import StackMotion, _placement_target, _row_heading, _seated_center
    from sim.panthera_env import PantheraSim

    initial = _vector(initial, 7, 'Initial state').copy()
    lower = _vector(lower, 7, 'Lower limits')
    upper = _vector(upper, 7, 'Upper limits')
    gripper_overshoot = abs(initial[-1] - np.clip(initial[-1], lower[-1], upper[-1]))
    if (np.any(lower >= upper) or np.any(initial[:6] < lower[:6]) or np.any(initial[:6] > upper[:6])
            or gripper_overshoot > GRIPPER_ENCODER_TOLERANCE_RAD):
        raise ValueError('Measured start outside SDK limits')
    endpoints = _vector([gripper_open_rad, gripper_closed_rad], 2, 'Gripper calibration')
    if (np.any(endpoints < lower[-1]) or np.any(endpoints > upper[-1]) or
            abs(gripper_open_rad - gripper_closed_rad) <= 2 * GRIPPER_ENCODER_TOLERANCE_RAD):
        raise ValueError('Gripper calibration needs distinct open and closed targets inside SDK limits')
    rows = []
    for label, studs in [('Source', source), ('Base', base)]:
        studs = np.asarray(studs, dtype=float)
        if studs.shape != (2, 3) or not np.isfinite(studs).all():
            raise ValueError(f'{label} needs two finite XYZ stud centers')
        spacing = np.linalg.norm(studs[1, :2] - studs[0, :2])
        if abs(spacing - PITCH_M) > .005 or abs(studs[0, 2] - studs[1, 2]) > .004:
            raise ValueError(f'{label} studs do not match a level 2×1 block')
        rows.append(studs)
    source, base = rows
    stud_to_center = np.array([0., 0., (STUD_HEIGHT_M + BODY_HEIGHT_M) / 2])
    src = source.mean(axis=0) - stud_to_center
    dst = base.mean(axis=0) - stud_to_center
    if abs(src[2] - dst[2]) > .005 or np.linalg.norm(src[:2] - dst[:2]) < 2 * PITCH_M:
        raise ValueError('Place two separated blocks on the same level bench')

    model = PantheraSim()
    model.arm_range = np.column_stack((np.maximum(lower[:6], model.arm_range[:, 0]),
                                       np.minimum(upper[:6], model.arm_range[:, 1])))
    if np.any(initial[:6] < model.arm_range[:, 0]) or np.any(initial[:6] > model.arm_range[:, 1]):
        raise ValueError('Measured joints fall outside the Panthera kinematics model')
    model.data.qpos[model.arm_qadr] = initial[:6]
    mujoco.mj_forward(model.model, model.data)
    motion = StackMotion(model)
    yaw = motion.choose_stack_yaw(src, source, dst, base, [_row_heading(source) + np.pi / 2])
    if yaw is None:
        raise ValueError('No reachable pickup and aligned release for this pair')
    pitch = motion._planned_pitch_deg
    pick, pick_quat = motion._gripper_pose(src, yaw, pitch)
    desired, _ = _placement_target(source, _seated_center(dst, base), base)
    place = motion._stack_plan(desired, motion._planned_turn, yaw, pitch)
    up = np.array([0., 0., motion.clearance])
    lift = pick + up
    lift[2] = max(lift[2], place.above[2])
    xyz, quat = model.ee_pose()
    start_hover = xyz.copy()
    start_hover[2] = max(xyz[2], (pick + up)[2])
    phases = [
        ('Open jaws', xyz, quat, gripper_open_rad),
        ('Raise clear', start_hover, quat, gripper_open_rad),
        ('Approach', pick + up, pick_quat, gripper_open_rad),
        ('Descend', pick, pick_quat, gripper_open_rad),
        ('Close jaws', pick, pick_quat, gripper_closed_rad),
        ('Lift', lift, pick_quat, gripper_closed_rad),
        ('Carry', place.above, place.quat, gripper_closed_rad),
        ('Align', place.preinsert, place.quat, gripper_closed_rad),
        ('Lower', place.release, place.quat, gripper_closed_rad),
        ('Release', place.release, place.quat, gripper_open_rad),
        ('Retreat', place.above, place.quat, gripper_open_rad),
    ]
    q = initial[:6].copy()
    measured_plane = np.array([src[2], dst[2]]) - BODY_HEIGHT_M / 2
    table_z = max(measured_plane) if table_z_m is None else float(table_z_m)
    if not math.isfinite(table_z) or np.max(np.abs(measured_plane - table_z)) > .008:
        raise ValueError('Observed blocks disagree with the calibrated bench height')
    # Compare the distal collision meshes against the measured plane. The base
    # and shoulder can sit on a separate mount below the tabletop.
    distal_meshes = {}
    arm_geoms = set()
    for geom_id, mesh_id in enumerate(model.model.geom_dataid):
        body_name = model.model.body(model.model.geom_bodyid[geom_id]).name
        if body_name.startswith('link') or body_name in ('L_finger', 'R_finger'):
            arm_geoms.add(geom_id)
        if model.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_name = model.model.mesh(mesh_id).name
        if mesh_name == 'base_link_col':
            arm_geoms.add(geom_id)
        if mesh_name.endswith('_col') and body_name in (
                'link3', 'link4', 'link5', 'link6', 'L_finger', 'R_finger'):
            start = model.model.mesh_vertadr[mesh_id]
            distal_meshes[geom_id] = model.model.mesh_vert[start:start + model.model.mesh_vertnum[mesh_id]]
    fingers = [model.model.geom(name).id for name in
               ('L_finger_pad', 'R_finger_pad', 'L_finger_bulk', 'R_finger_bulk')]

    def check_clearance(joints, phase):
        model.data.qpos[model.arm_qadr] = joints
        mujoco.mj_forward(model.model, model.data)
        for contact in model.data.contact:
            if all(g in arm_geoms for g in contact.geom) and contact.dist < 0:
                raise ValueError(f'{phase} puts the arm in contact with itself')
        z_axes = model.data.geom_xmat.reshape(-1, 3, 3)[:, 2, :]
        half_height = np.sum(np.abs(z_axes[fingers]) * model.model.geom_size[fingers], axis=1)
        lowest = np.min(model.data.geom_xpos[fingers, 2] - half_height)
        for geom_id, vertices in distal_meshes.items():
            lowest = min(lowest, model.data.geom_xpos[geom_id, 2] + min(vertices @ z_axes[geom_id]))
        if lowest < table_z + .002:
            raise ValueError(f'{phase} puts the arm within 2 mm of the bench')

    waypoints = {}
    tcp_path = [xyz.copy()]
    for phase, end, rotation, gripper in phases:
        angle = 2 * math.acos(np.clip(abs(quat @ rotation), 0., 1.))
        count = max(1, math.ceil(np.linalg.norm(end - xyz) / .005), math.ceil(angle / .05))
        targets = []
        for t in np.linspace(0., 1., count + 1)[1:]:
            pos = xyz + t * (end - xyz)
            candidate, perr, rerr = model.ik(pos, _slerp(quat, rotation, t),
                                              q_init=q, max_joint_step=.15, posture_gain=0.)
            if (not np.all(np.isfinite(candidate)) or not math.isfinite(perr + rerr)
                    or perr > .0015 or rerr > .025 or np.any(candidate < lower[:6])
                    or np.any(candidate > upper[:6])):
                raise ValueError(f'{phase} path exceeds joint limits or IK tracking tolerance')
            substeps = max(1, math.ceil(np.max(np.abs(candidate - q)) / .025))
            for u in np.linspace(0., 1., substeps + 1)[1:]:
                check_clearance(q + u * (candidate - q), phase)
            q = candidate
            targets.append(np.r_[q, gripper])
            tcp_path.append(model.ee_pos().copy())
        waypoints[phase] = np.asarray(targets)
        xyz, quat = end, rotation
    return StackPlan(initial, waypoints, gripper_open_rad, gripper_closed_rad,
                     (lower[-1], upper[-1]), np.asarray(tcp_path))


def execute_plan(plan, *, read, command, check_active, on_phase):
    """Execute one preflighted path. The caller stops the drive on exit or failure.

    read() and command(target) use six joint angles and one gripper encoder in
    SDK radians. check_active() raises when the operator stops or loses the lease.
    """
    measured = _vector(read(), 7, 'Arm feedback')
    if np.max(np.abs(plan.initial - measured)) > .02:
        raise RuntimeError('Arm moved since preview; build a new plan')
    target = measured.copy()
    target[-1] = np.clip(target[-1], *plan.gripper_limits)
    held = False
    # Empty-stop encoder noise must fall outside the accepted pinching range.
    margin = max(.05, GRIPPER_ENCODER_TOLERANCE_RAD / abs(plan.open_rad - plan.closed_rad))
    for phase, points in plan.waypoints.items():
        on_phase(phase)
        # The shared trajectory interpolator handles at most 50 bimanual samples.
        # The duplicate arm is used only to evaluate its bounded interpolation.
        for i in range(0, len(points), 50):
            chunk = points[i:i + 50]
            started = time.monotonic()
            path = SmoothActionTrajectory(np.tile(chunk, (1, 2)), started)
            path.speed[:] = np.tile([.30] * 6 + [.6], 2)
            path.acceleration[:] = np.tile([1.] * 6 + [2.], 2)
            last = started
            done = False
            while not done:
                check_active()
                measured = _vector(read(), 7, 'Arm feedback')
                if np.max(np.abs(measured[:6] - target[:6])) > .06:
                    raise RuntimeError(f'Joint tracking lost during {phase}')
                fraction = (measured[-1] - plan.closed_rad) / (plan.open_rad - plan.closed_rad)
                if held and phase != 'Release' and not margin < fraction < 1. - margin:
                    raise RuntimeError('Gripper encoder lost the grasp during transfer')
                now = time.monotonic()
                both, done = path.tick(now, now - last, np.tile(target, 2))
                target = both[:7]
                check_active()
                command(target)
                last = now
                time.sleep(1 / 60)

        # Let the physical joints settle before lowering, lifting or releasing.
        deadline = time.monotonic() + 2.
        stable_since = None
        while True:
            check_active()
            measured = _vector(read(), 7, 'Arm feedback')
            now = time.monotonic()
            if now > deadline:
                raise RuntimeError(f'{phase} failed to reach its measured endpoint')
            arm_error = np.max(np.abs(measured[:6] - target[:6]))
            if arm_error > .06:
                raise RuntimeError(f'Joint tracking lost during {phase}')
            fraction = (measured[-1] - plan.closed_rad) / (plan.open_rad - plan.closed_rad)
            closed = phase == 'Close jaws' or (held and phase != 'Release')
            grip_ok = margin < fraction < 1. - margin if closed else abs(measured[-1] - plan.open_rad) < .02
            if held and phase != 'Release' and not grip_ok:
                raise RuntimeError('Gripper encoder lost the grasp during transfer')
            if arm_error <= .012 and grip_ok:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= .25:
                    break
            else:
                stable_since = None
            check_active()
            command(target)
            time.sleep(1 / 60)
        if phase == 'Close jaws':
            held = True
        elif phase == 'Release':
            held = False
