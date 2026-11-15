"""Isaac Lab asset config for arm01 (spec stage 9: PACE replay in Isaac Lab).

Written against the Isaac Lab 2.x API (isaaclab 0.4x: `isaaclab.assets`,
`isaaclab.actuators`, `isaaclab.sim`). Nothing here imports isaaclab at module
level, so the pure-Python parts (paths, URDF preparation) work without it:

  prepare_urdf_for_isaac()   the URDF the USD is converted from (mimic joints
                             fixed, package:// paths made absolute)
  arm01_articulation_cfg()   an ArticulationCfg with one ImplicitActuator per
                             driven joint (needs isaaclab)
  ARM01_CFG                  that config with the joint-config gains, built on
                             first access (needs isaaclab)

Mimic joints. The wrist pitch is a parallelogram: J6's crank (`wrist_crank_joint`)
and the coupler rod (`wrist_rod_joint`) are URDF mimic joints of `wrist_pitch`
(multipliers +1 and -1), and the URDF is a tree, so the loop is never closed:
crank and rod are display bodies hanging off link5. Only `wrist_pitch` is driven
(and its 1:1 ratio is exact to 0.2%). Pinocchio (ArmModel) locks both mimic
joints at zero and merges their 82 g into link5. The Isaac asset does the same:
`prepare_urdf_for_isaac` rewrites the two mimic joints as fixed joints, and the
converter's `merge_fixed_joints=True` folds the crank and rod into link5. The
USD then has exactly the 7 driven DOFs in the joint config, with the same link5
inertia as Pinocchio, and no PhysX mimic constraint whose solver compliance would
leak into the J6 fit. (Isaac Sim 5's importer could instead keep them as PhysX
mimic joints, `convert_mimic_joints_to_normal_joints=False` on the raw URDF, but
that adds two DOFs that no actuator drives and a constraint Pinocchio does not
model.) The flange and tool0 frames are massless and merge into link7 too.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKAGE = 'wisco_ethercat_ros2_hw'
URDF_PATH = os.path.join(REPO, 'urdf', 'arm01', 'arm01.urdf')
ARM_CONFIG = os.path.join(REPO, 'config', 'arms', 'arm01.yaml')
# generated, not checked in (/build/ is ignored): tools/isaac/import_urdf.py writes it
USD_DIR = os.path.join(REPO, 'build', 'isaac', 'arm01')
USD_PATH = os.path.join(USD_DIR, 'arm01.usd')
PREPARED_URDF = os.path.join(USD_DIR, 'arm01_isaac.urdf')

DRIVEN_JOINTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'forearm_yaw', 'wrist_pitch',
                 'wrist_yaw')
MIMIC_JOINTS = ('wrist_crank_joint', 'wrist_rod_joint')   # follow wrist_pitch; merged into link5


def prepare_urdf_for_isaac(src: str = URDF_PATH, dst: str = PREPARED_URDF, repo: str = REPO) -> str:
    """Write the URDF the Isaac importer converts: mimic joints fixed, mesh paths absolute.

    Returns dst. The source URDF is not modified.
    """
    tree = ET.parse(src)
    root = tree.getroot()
    for joint in root.findall('joint'):
        if joint.find('mimic') is None:
            continue
        mimic = joint.find('mimic')
        if mimic.get('joint') not in DRIVEN_JOINTS:
            raise ValueError('%s mimics %s, which is not a driven joint' % (joint.get('name'), mimic.get('joint')))
        joint.set('type', 'fixed')
        for tag in ('mimic', 'axis', 'limit', 'dynamics', 'safety_controller'):
            for el in joint.findall(tag):
                joint.remove(el)
    prefix = 'package://%s/' % PACKAGE
    base = repo.replace('\\', '/').rstrip('/') + '/'
    for mesh in root.iter('mesh'):
        fn = mesh.get('filename', '')
        if fn.startswith(prefix):
            mesh.set('filename', base + fn[len(prefix):])
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tree.write(dst, xml_declaration=True, encoding='utf-8')
    return dst


def urdf_joint_limits(path: str = URDF_PATH) -> Dict[str, Dict[str, float]]:
    out = {}
    for joint in ET.parse(path).getroot().findall('joint'):
        lim = joint.find('limit')
        if lim is not None and joint.find('mimic') is None:
            out[joint.get('name')] = {k: float(lim.get(k)) for k in ('lower', 'upper', 'effort', 'velocity')
                                      if lim.get(k) is not None}
    return out


def config_gains(path: str = ARM_CONFIG) -> Dict[str, Dict[str, float]]:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {r['name']: dict(r.get('gains') or {}) for r in raw['joints']}


def arm01_articulation_cfg(usd_path: str = USD_PATH, joint_names: Sequence[str] = DRIVEN_JOINTS,
                           stiffness: Optional[Dict[str, float]] = None, damping: Optional[Dict[str, float]] = None,
                           armature: float = 0.0, prim_path: str = '{ENV_REGEX_NS}/Robot',
                           init_joint_pos: Optional[Dict[str, float]] = None):
    """ArticulationCfg for arm01: fixed base, one ImplicitActuator group per joint.

    Gains default to the joint config's Mode P gains; PACE overwrites stiffness,
    damping and armature per env at run time. Joint friction is left at 0: PhysX's
    joint friction is a unitless coefficient, so the backend applies Coulomb
    friction (N m) as an effort instead.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    gains = config_gains()
    limits = urdf_joint_limits()
    stiffness = stiffness or {n: float(gains.get(n, {}).get('kp', 0.0)) for n in joint_names}
    damping = damping or {n: float(gains.get(n, {}).get('kd', 0.0)) for n in joint_names}
    actuators = {
        n: ImplicitActuatorCfg(
            joint_names_expr=[n],
            stiffness=stiffness[n],
            damping=damping[n],
            armature=armature,
            friction=0.0,
            effort_limit_sim=limits[n]['effort'],
            velocity_limit_sim=limits[n]['velocity'],
        )
        for n in joint_names
    }
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, fix_root_link=True,
                solver_position_iteration_count=8, solver_velocity_iteration_count=1),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), joint_pos=dict(init_joint_pos or {n: 0.0 for n in joint_names}),
            joint_vel={'.*': 0.0}),
        actuators=actuators,
    )


def __getattr__(name):  # PEP 562: build ARM01_CFG only when asked for (needs isaaclab)
    if name == 'ARM01_CFG':
        cfg = arm01_articulation_cfg()
        globals()['ARM01_CFG'] = cfg
        return cfg
    raise AttributeError(name)
