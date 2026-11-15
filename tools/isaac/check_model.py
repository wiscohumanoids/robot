#!/usr/bin/env python3
"""Compare the Isaac Lab arm01 articulation against Pinocchio (ArmModel).

Spec: "Load the URDF into Pinocchio and Isaac Lab; check joint axes,
directions, limits, and masses against CAD". tools/check_model_pinocchio.py
checks the URDF itself; this loads the USD made by tools/isaac/import_urdf.py
and prints every place Isaac Lab disagrees with the Pinocchio reference:

  joints   names and order (Isaac orders DOFs breadth first; PACE maps by
           name, so an order difference is reported but is not a failure)
  axes     each revolute joint's axis from the USD (axis token rotated by the
           joint's localRot0) in the parent link frame, vs Pinocchio
  limits   data.joint_pos_limits vs ArmModel lower/upper
  masses   PhysX body masses and COMs vs ArmModel links (link5 includes the
           merged parallelogram crank and rod)
  FK       link origins at random in-range poses vs Pinocchio FK, which also
           catches a flipped joint direction

  python tools/isaac/check_model.py [--usd build/isaac/arm01/arm01.usd] [--poses 20]

Run with the Isaac Lab Python; Pinocchio must be importable there too
(pip install pin). Exits 1 on any mismatch.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(HERE))      # tools/, for check_model_pinocchio

AXIS_TOL = 1e-4          # 1 - cos
LIMIT_TOL = 1e-4         # rad
MASS_TOL = 1e-4          # kg
COM_TOL = 1e-4           # m
FK_TOL = 1e-4            # m


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def usd_joint_axes(stage, root):
    """{joint name: (parent link name, axis in the parent link frame)} for revolute joints under root."""
    from pxr import Gf, Usd, UsdPhysics
    out = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        j = UsdPhysics.RevoluteJoint(prim)
        tok = j.GetAxisAttr().Get()
        a = Gf.Vec3d(*{'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}[str(tok)])
        q = j.GetLocalRot0Attr().Get() or Gf.Quatf(1.0)
        rot = Gf.Rotation(Gf.Quatd(q.GetReal(), Gf.Vec3d(q.GetImaginary())))
        v = rot.TransformDir(a)
        b0 = j.GetBody0Rel().GetTargets()
        out[prim.GetName()] = (b0[0].name if b0 else None, np.array([v[0], v[1], v[2]]))
    return out


def main():
    from arm_stack.pace.isaac_cfg import DRIVEN_JOINTS, URDF_PATH, USD_PATH, arm01_articulation_cfg

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--usd', default=USD_PATH)
    p.add_argument('--urdf', default=URDF_PATH)
    p.add_argument('--poses', type=int, default=20)
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        sys.exit('isaaclab is not importable; run this with the Isaac Lab Python')
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.headless = True
    app = AppLauncher(args).app

    import isaaclab.sim as sim_utils
    import omni.usd
    import torch
    from isaaclab.assets import Articulation

    from check_model_pinocchio import reference

    ref = reference(args.urdf, list(DRIVEN_JOINTS))
    m = ref['model']
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.002, device=args.device))
    robot = Articulation(arm01_articulation_cfg(usd_path=args.usd, prim_path='/World/Robot'))
    sim.reset()
    robot.update(0.0)
    problems = []

    def bad(msg):
        problems.append(msg)
        print('  MISMATCH ' + msg)

    # ------------------------------------------------------------------ joints
    print('== joints')
    names = list(robot.joint_names)
    print('  Isaac order %s' % names)
    print('  Pinocchio  %s' % ref['names'])
    if sorted(names) != sorted(ref['names']):
        bad('joint sets differ: Isaac only %s, Pinocchio only %s' % (
            sorted(set(names) - set(ref['names'])), sorted(set(ref['names']) - set(names))))
    elif names != ref['names']:
        print('  note: DOF order differs (PACE maps joints by name)')
    ids, _ = robot.find_joints(ref['names'], preserve_order=True)

    # ------------------------------------------------------------------ axes
    print('== axes (parent link frame)')
    stage = omni.usd.get_context().get_stage()
    axes = usd_joint_axes(stage, '/World/Robot')
    for i, n in enumerate(ref['names']):
        if n not in axes:
            bad('%s: no revolute joint prim in the USD' % n)
            continue
        parent, a = axes[n]
        want = ref['axes_parent'][i]
        c = float(np.dot(a, want) / np.linalg.norm(a))
        print('  %-15s Isaac %s  Pinocchio %s  (parent %s)' % (n, np.round(a, 5), np.round(want, 5), parent))
        if 1 - c > AXIS_TOL:
            bad('%s axis %s vs %s (%s)' % (n, np.round(a, 5), np.round(want, 5), 'flipped' if c < -0.999 else 'off'))
    extra = sorted(set(axes) - set(ref['names']))
    if extra:
        bad('extra revolute joints in the USD (mimic joints not merged?): %s' % extra)

    # ------------------------------------------------------------------ limits
    print('== limits (rad)')
    lim = getattr(robot.data, 'joint_pos_limits', None)
    lim = _np(lim if lim is not None else robot.data.joint_limits)[0]
    for i, n in enumerate(ref['names']):
        lo, hi = lim[ids[i]]
        print('  %-15s Isaac [%.5f, %.5f]  Pinocchio [%.5f, %.5f]' % (n, lo, hi, ref['lower'][i], ref['upper'][i]))
        if abs(lo - ref['lower'][i]) > LIMIT_TOL or abs(hi - ref['upper'][i]) > LIMIT_TOL:
            bad('%s limits [%.5f, %.5f] vs [%.5f, %.5f]' % (n, lo, hi, ref['lower'][i], ref['upper'][i]))

    # ------------------------------------------------------------------ masses
    print('== masses (kg) and COMs (m, link frame)')
    bids, _ = robot.find_bodies(ref['links'], preserve_order=True)
    masses = _np(robot.root_physx_view.get_masses())[0]
    coms = _np(robot.root_physx_view.get_coms())[0]
    for l, b in zip(ref['links'], bids):
        mi, mp = float(masses[b]), ref['link_mass'][l]
        ci, cp = coms[b, :3], ref['link_com'][l]
        print('  %-6s Isaac %.4f  Pinocchio %.4f   COM Isaac %s  Pinocchio %s' % (
            l, mi, mp, np.round(ci, 5), np.round(cp, 5)))
        if abs(mi - mp) > MASS_TOL:
            bad('%s mass %.4f vs %.4f' % (l, mi, mp))
        if np.linalg.norm(ci - cp) > COM_TOL:
            bad('%s COM %s vs %s' % (l, np.round(ci, 5), np.round(cp, 5)))
    total_i = float(sum(masses[b] for b in bids))
    total_p = sum(ref['link_mass'].values())
    print('  moving total Isaac %.4f  Pinocchio %.4f' % (total_i, total_p))
    if abs(total_i - total_p) > MASS_TOL:
        bad('moving mass total %.4f vs %.4f' % (total_i, total_p))

    # ------------------------------------------------------------------ FK / directions
    print('== FK at %d random poses (link origins, base frame)' % args.poses)
    rng = np.random.default_rng(0)
    worst = 0.0
    root_pos = _np(robot.data.root_link_pos_w if hasattr(robot.data, 'root_link_pos_w') else robot.data.root_pos_w)[0]
    for _ in range(args.poses):
        q = rng.uniform(ref['lower'], ref['upper'])
        pos = np.zeros((1, robot.num_joints))
        pos[0, ids] = q
        t = torch.as_tensor(pos, dtype=torch.float32, device=robot.data.joint_pos.device)
        robot.write_joint_state_to_sim(t, torch.zeros_like(t))
        sim.forward()
        robot.update(0.0)
        bp = robot.data.body_link_pos_w if hasattr(robot.data, 'body_link_pos_w') else robot.data.body_pos_w
        bp = _np(bp)[0] - root_pos
        for l, b in zip(ref['links'], bids):
            want = m.fk(q, l).translation
            worst = max(worst, float(np.linalg.norm(bp[b] - want)))
    print('  worst link origin error %.2e m' % worst)
    if worst > FK_TOL:
        bad('FK differs by up to %.4f m (axis direction, origin or zero offset)' % worst)

    print('\n%d mismatches' % len(problems))
    for s in problems:
        print('  - ' + s)
    app.close()
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
