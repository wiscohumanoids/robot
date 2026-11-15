#!/usr/bin/env python3
"""Convert urdf/arm01/arm01.urdf to USD with Isaac Lab's UrdfConverter.

Run with the Isaac Lab Python (Isaac Sim 5.x + Isaac Lab 2.x):

  python tools/isaac/import_urdf.py                 # -> build/isaac/arm01/arm01.usd
  python tools/isaac/import_urdf.py --out-dir D --force

Settings: fixed base (the stand's post is bolted down), fixed joints merged, a
force-type position drive on every joint with zero gains (the ArticulationCfg's
ImplicitActuators in arm_stack/pace/isaac_cfg.py set them), convex-hull
colliders, self collision off.

Mimic joints: the wrist pitch parallelogram's crank and rod are URDF mimic
joints of wrist_pitch. They are rewritten as fixed joints before conversion
(arm_stack.pace.isaac_cfg.prepare_urdf_for_isaac) and merge into link5, so the
USD has only the 7 driven joints and wrist_pitch is the single DOF that drives
the wrist pitch, exactly as Pinocchio's reduced model (ArmModel) has it. The
crank and rod meshes stay visible as part of link5 but no longer counter-rotate
when the wrist pitches (display only; their 82 g is in link5 either way).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)


def main():
    from arm_stack.pace.isaac_cfg import MIMIC_JOINTS, URDF_PATH, USD_DIR, prepare_urdf_for_isaac

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--urdf', default=URDF_PATH)
    parser.add_argument('--out-dir', default=USD_DIR)
    parser.add_argument('--name', default='arm01.usd')
    parser.add_argument('--force', action='store_true', help='reconvert even if the USD is up to date')
    parser.add_argument('--instanceable', action='store_true', help='make the link meshes instanceable')
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        sys.exit('isaaclab is not importable; run this with the Isaac Lab Python (see arm_stack/pace/isaac_backend.py)')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    app = AppLauncher(args).app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    prepared = prepare_urdf_for_isaac(args.urdf, os.path.join(args.out_dir, 'arm01_isaac.urdf'))
    print('prepared %s (mimic joints %s fixed, merged into link5)' % (prepared, ', '.join(MIMIC_JOINTS)))
    cfg = UrdfConverterCfg(
        asset_path=prepared,
        usd_dir=args.out_dir,
        usd_file_name=args.name,
        force_usd_conversion=args.force,
        make_instanceable=args.instanceable,
        fix_base=True,
        merge_fixed_joints=True,
        convert_mimic_joints_to_normal_joints=False,   # none are left after prepare_urdf_for_isaac
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type='force',
            target_type='position',
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        collider_type='convex_hull',
        self_collision=False,
    )
    conv = UrdfConverter(cfg)
    print('USD: %s' % conv.usd_path)
    app.close()


if __name__ == '__main__':
    main()
