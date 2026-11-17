#!/usr/bin/env python3
"""Check the arm01 URDF against its joint config, with Pinocchio only (no Isaac).

Goal: "Load the URDF into Pinocchio and Isaac Lab; check joint axes,
directions, limits, and masses against CAD". This is the Pinocchio half;
tools/isaac/check_model.py compares the Isaac Lab articulation against the same
reference (`reference()` below).

Checks:
  joint order   config joints (bus order) = the URDF's driven joints in chain
                order = ArmModel's order; no driven joint missing from either
  limits        URDF limits = the measured ROM per joint (ROM_DEG); Pinocchio
                reads the same values
  mimic         the parallelogram joints mimic wrist_pitch 1:1 (+1 / -1), with
                its axis and limits, and ArmModel locks them
  axes          every URDF axis is a unit vector; Pinocchio's axis matches it;
                J5 is parallel to J3 at zero (straight elbow)
  masses        totals (URDF, Pinocchio full, reduced ArmModel), per link, the
                config's weighed masses where given, inertia tensors physical
  efforts       URDF effort / velocity limits = actuator model peak / max

  python3 tools/check_model_pinocchio.py [--arm arm01] [--strict]

Exits 1 on any FAIL (warnings, such as unweighed placeholder masses, do not).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Operating ROM, degrees: the CAD export's (robots/arm01/cad/arm01_capture.json, written by
# robots/arm01/cad/import_full_assembly.py): J1 -90..+180, J2 0..+190, J3 +-90, J4 0..+130,
# J5 +-90, J6 +-35, J7 +-90.
with open(os.path.join(os.path.normpath(os.path.join(REPO, '..', '..', '..', 'robots', 'arm01')), 'cad', 'arm01_capture.json')) as _f:
    ROM_DEG = {k: tuple(v) for k, v in json.load(_f)['operating_limits_deg'].items()}
LIMIT_TOL_DEG = 0.01
AXIS_NORM_TOL = 1e-4
MASS_TOL = 1e-6           # kg, bookkeeping (totals should match exactly)
WEIGHED_TOL = 0.05        # relative, URDF vs weighed mass
MOVABLE = ('revolute', 'continuous', 'prismatic')


class Report:
    """Collects check results by section and prints them."""
    def __init__(self):
        """No results yet."""
        self.fails, self.warns, self.section = [], [], None

    def head(self, title):
        """Start a section."""
        self.section = title
        print('\n== %s' % title)

    def ok(self, msg):
        """Print a pass."""
        print('  ok    ' + msg)

    def fail(self, msg):
        """Record and print a failure."""
        self.fails.append('%s: %s' % (self.section, msg))
        print('  FAIL  ' + msg)

    def warn(self, msg):
        """Record and print a warning."""
        self.warns.append('%s: %s' % (self.section, msg))
        print('  warn  ' + msg)

    def check(self, cond, msg, warn_only=False):
        """Record a check as ok, warning or failure."""
        if cond:
            self.ok(msg)
        elif warn_only:
            self.warn(msg)
        else:
            self.fail(msg)
        return cond


def parse_urdf(path):
    """Links (inertials) and joints (axes, limits) from the URDF."""
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall('link'):
        inert = link.find('inertial')
        if inert is None:
            links[link.get('name')] = None
            continue
        I = inert.find('inertia')
        m = float(inert.find('mass').get('value'))
        t = {k: float(I.get(k)) for k in ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')}
        links[link.get('name')] = (m, np.array([[t['ixx'], t['ixy'], t['ixz']],
                                                 [t['ixy'], t['iyy'], t['iyz']],
                                                 [t['ixz'], t['iyz'], t['izz']]]))
    joints = {}
    for j in root.findall('joint'):
        axis = j.find('axis')
        lim = j.find('limit')
        mim = j.find('mimic')
        org = j.find('origin')
        r_, p_, y_ = [float(v) for v in ((org.get('rpy') if org is not None else None) or '0 0 0').split()]
        Rx = np.array([[1, 0, 0], [0, np.cos(r_), -np.sin(r_)], [0, np.sin(r_), np.cos(r_)]])
        Ry = np.array([[np.cos(p_), 0, np.sin(p_)], [0, 1, 0], [-np.sin(p_), 0, np.cos(p_)]])
        Rz = np.array([[np.cos(y_), -np.sin(y_), 0], [np.sin(y_), np.cos(y_), 0], [0, 0, 1]])
        joints[j.get('name')] = dict(
            type=j.get('type'), parent=j.find('parent').get('link'), child=j.find('child').get('link'),
            axis=np.array([float(v) for v in (axis.get('xyz') if axis is not None else '1 0 0').split()]),
            origin_rot=Rz @ Ry @ Rx,
            lower=float(lim.get('lower', 'nan')) if lim is not None else float('nan'),
            upper=float(lim.get('upper', 'nan')) if lim is not None else float('nan'),
            effort=float(lim.get('effort', 'nan')) if lim is not None else float('nan'),
            velocity=float(lim.get('velocity', 'nan')) if lim is not None else float('nan'),
            mimic=None if mim is None else (mim.get('joint'), float(mim.get('multiplier', 1.0)),
                                            float(mim.get('offset', 0.0))))
    return links, joints


def chain_order(joints, root='base_link'):
    """Driven (movable, non-mimic) joints, depth first from the root."""
    out = []

    def walk(link):
        for name, j in joints.items():
            if j['parent'] == link:
                if j['type'] in MOVABLE and j['mimic'] is None:
                    out.append(name)
                walk(j['child'])
    walk(root)
    return out


def reference(urdf_path, joint_names):
    """What any other loader (Isaac Lab) should reproduce: from Pinocchio's ArmModel."""
    from arm_stack.model.robot import ArmModel
    m = ArmModel(urdf_path, joint_names)
    axes = []
    for k, jid in enumerate(m.jid):
        R = m.model.jointPlacements[jid].rotation
        axes.append(R @ m.axis_in_child(k))           # in the parent link frame
    links = m.link_names()
    return dict(model=m, names=list(m.joint_names), axes_parent=np.array(axes), lower=m.lower.copy(),
                upper=m.upper.copy(), links=links,
                link_mass={l: float(m.model.inertias[j].mass) for l, j in zip(links, m.jid)},
                link_com={l: np.array(m.model.inertias[j].lever) for l, j in zip(links, m.jid)})


def main():
    """Check the arm model in Pinocchio: inertials, ranges, gravity and the CAD capture."""
    import pinocchio as pin
    from arm_stack.config import load_arm, load_yaml, resolve

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--arm', default='arm01')
    p.add_argument('--models', default='config/actuator_models.yaml')
    p.add_argument('--strict', action='store_true', help='also exit 1 on warnings')
    args = p.parse_args()
    arm = load_arm(args.arm, args.models)
    models = load_yaml(resolve(args.models)).get('models', {})
    urdf = arm.urdf
    links, joints = parse_urdf(urdf)
    r = Report()
    print('URDF %s\nconfig %s' % (os.path.relpath(urdf, REPO), os.path.relpath(arm.path, REPO)))

    # ------------------------------------------------------------------ joint order
    r.head('joint order')
    chain = chain_order(joints)
    cfg = arm.names
    r.check(cfg == chain, 'config order %s %s URDF chain order %s' % (cfg, '==' if cfg == chain else '!=', chain))
    missing = [n for n in cfg if n not in joints]
    r.check(not missing, 'every config joint is in the URDF' + ('' if not missing else ': missing %s' % missing))
    extra = [n for n in chain if n not in cfg]
    r.check(not extra, 'every driven URDF joint is in the config' + ('' if not extra else ': extra %s' % extra))
    ref = reference(urdf, cfg)
    m = ref['model']
    full = pin.buildModelFromUrdf(urdf)
    pin_order = [n for n in full.names[1:] if n in cfg]
    r.check(pin_order == cfg, 'Pinocchio joint order (by depth) %s config order' % ('==' if pin_order == cfg else '!='))
    r.check(m.nominal.nq == len(cfg), 'ArmModel has %d DOFs (locked: %s)' % (m.nominal.nq, ', '.join(m.locked_joints)))
    buses = [j.bus_position for j in arm.joints]
    r.check(buses == list(range(len(buses))), 'bus positions 0..%d in joint order' % (len(buses) - 1))

    # ------------------------------------------------------------------ limits
    r.head('limits vs measured ROM (deg)')
    for i, n in enumerate(cfg):
        lo, hi = np.degrees(joints[n]['lower']), np.degrees(joints[n]['upper'])
        want = ROM_DEG.get(n)
        if want is None:
            r.fail('%s has no ROM entry' % n)
            continue
        good = abs(lo - want[0]) <= LIMIT_TOL_DEG and abs(hi - want[1]) <= LIMIT_TOL_DEG
        r.check(good, '%-15s URDF [%8.3f, %8.3f]  ROM [%6.1f, %6.1f]' % (n, lo, hi, want[0], want[1]))
        pl, pu = np.degrees(m.lower[i]), np.degrees(m.upper[i])
        if abs(pl - lo) > 1e-6 or abs(pu - hi) > 1e-6:
            r.fail('%s Pinocchio reads [%.3f, %.3f]' % (n, pl, pu))

    # ------------------------------------------------------------------ mimic
    r.head('mimic joints (wrist parallelogram)')
    mimics = {n: j for n, j in joints.items() if j['mimic'] is not None}
    for n, j in mimics.items():
        src, mult, off = j['mimic']
        r.check(src in cfg, '%s mimics %s (driven)' % (n, src))
        r.check(abs(abs(mult) - 1.0) < 1e-9 and off == 0.0, '%s multiplier %+g offset %g (1:1)' % (n, mult, off))
        s = joints[src]
        cosang = abs(np.dot(j['axis'], s['axis'])) / (np.linalg.norm(j['axis']) * np.linalg.norm(s['axis']))
        r.check(cosang > 1 - 1e-6, '%s axis parallel to %s (cos %.7f)' % (n, src, cosang))
        rng = sorted([mult * s['lower'], mult * s['upper']])
        r.check(abs(rng[0] - j['lower']) < 1e-6 and abs(rng[1] - j['upper']) < 1e-6,
                '%s limits [%.4f, %.4f] = %+g x %s limits' % (n, j['lower'], j['upper'], mult, src))
        r.check(n in m.locked_joints, '%s locked in ArmModel (merged into %s)' % (n, joints[src]['parent']))

    # ------------------------------------------------------------------ axes
    r.head('joint axes')
    for i, n in enumerate(cfg):
        a = joints[n]['axis']
        norm = np.linalg.norm(a)
        r.check(abs(norm - 1.0) < AXIS_NORM_TOL, '%-15s axis (% .5f % .5f % .5f) |a| = %.6f' % (n, a[0], a[1], a[2], norm))
        pa = ref['axes_parent'][i]
        a_parent = joints[n]['origin_rot'] @ (a / norm)        # the URDF axis, in the parent link frame
        if np.dot(pa, a_parent) < 1 - 1e-9:
            r.fail('%s Pinocchio axis %s differs from the URDF %s' % (n, np.round(pa, 5), np.round(a_parent, 5)))
    for n, j in mimics.items():
        norm = np.linalg.norm(j['axis'])
        r.check(abs(norm - 1.0) < AXIS_NORM_TOL, '%-15s axis |a| = %.6f (mimic)' % (n, norm))
    if 'shoulder_yaw' in cfg and 'forearm_yaw' in cfg:
        # the export's elbow zero is not straight: J4 = 0 leaves the forearm a few
        # degrees off the upper arm (its contact onset is near -5 deg)
        d0 = full.createData()
        pin.forwardKinematics(full, d0, pin.neutral(full))
        w = [d0.oMi[full.getJointId(n)].rotation @ np.array(full.joints[full.getJointId(n)].S)[3:, 0]
             for n in ('shoulder_yaw', 'forearm_yaw')]
        c = abs(np.dot(w[0], w[1])) / np.linalg.norm(w[0]) / np.linalg.norm(w[1])
        print('  J5 axis vs J3 axis at q = 0: %.3f deg (elbow zero)' % np.degrees(np.arccos(min(c, 1.0))))
    dirs = {j.name: j.direction for j in arm.joints}
    r.check(all(d in (1, -1) for d in dirs.values()),
            'config directions are +-1: %s' % ', '.join('%s %+d' % kv for kv in dirs.items()))
    for fr in ('flange', 'tool0'):
        r.check(full.existFrame(fr), 'frame %s exists' % fr)

    # ------------------------------------------------------------------ masses
    r.head('masses (kg)')
    total_urdf = sum(v[0] for v in links.values() if v is not None)
    base = links.get('base_link')
    moving_urdf = total_urdf - (base[0] if base else 0.0)
    total_pin_full = sum(full.inertias[i].mass for i in range(full.njoints))
    total_pin_red = sum(m.nominal.inertias[i].mass for i in range(m.nominal.njoints))
    moving_pin = sum(ref['link_mass'].values())
    print('  URDF total %.4f (base_link %.4f, moving %.4f)' % (total_urdf, base[0] if base else 0.0, moving_urdf))
    r.check(abs(total_pin_full - total_urdf) < MASS_TOL, 'Pinocchio full model total %.4f = URDF' % total_pin_full)
    r.check(abs(total_pin_red - total_urdf) < MASS_TOL,
            'ArmModel (mimic links merged) total %.4f = URDF' % total_pin_red)
    r.check(abs(moving_pin - moving_urdf) < MASS_TOL,
            'ArmModel moving links sum %.4f = URDF moving mass' % moving_pin)
    merged = {}
    for n, j in mimics.items():
        # the mimic link's mass lands on the nearest driven ancestor
        link = j['parent']
        while link not in ref['links'] and link != 'base_link':
            link = next(jj['parent'] for jj in joints.values() if jj['child'] == link)
        merged[link] = merged.get(link, 0.0) + links[j['child']][0]
    print('  %-8s %8s %8s %8s %8s' % ('link', 'URDF', '+merged', 'ArmModel', 'weighed'))
    for l in ref['links']:
        u = links[l][0]
        want = u + merged.get(l, 0.0)
        got = ref['link_mass'][l]
        w = arm.link_masses_kg.get(l)
        print('  %-8s %8.4f %8.4f %8.4f %8s' % (l, u, merged.get(l, 0.0), got, '-' if w is None else '%.4f' % w))
        if abs(got - want) > MASS_TOL:
            r.fail('%s ArmModel mass %.4f != URDF %.4f + merged %.4f' % (l, got, u, merged.get(l, 0.0)))
        if w is not None and abs(want - w) > WEIGHED_TOL * w:
            r.fail('%s URDF+merged %.4f kg vs weighed %.4f kg (> %.0f%%)' % (l, want, w, 100 * WEIGHED_TOL))
    unweighed = [l for l in ref['links'] if arm.link_masses_kg.get(l) is None]
    r.check(not unweighed, 'weighed masses in the config for %d/%d links%s' % (
        len(ref['links']) - len(unweighed), len(ref['links']),
        '' if not unweighed else ' (placeholders: %s)' % ', '.join(unweighed)), warn_only=True)
    bad = 0
    for name, v in links.items():
        if v is None:
            continue
        mass, I = v
        ev = np.linalg.eigvalsh(I)
        tri = ev[0] + ev[1] >= ev[2] - 1e-12
        if mass <= 0 or ev.min() <= 0 or not tri:
            bad += 1
            r.fail('%s inertia not physical (mass %.4g, principal %s)' % (name, mass, np.round(ev, 8)))
    if not bad:
        r.ok('every link has mass > 0 and a positive-definite inertia obeying the triangle inequality')

    # ------------------------------------------------------------------ efforts
    r.head('effort / velocity limits vs actuator model')
    for spec in arm.joints:
        md = models[spec.model]
        j = joints[spec.name]
        good = abs(j['effort'] - md['peak_effort_nm']) < 1e-6 and abs(j['velocity'] - md['max_velocity_rad_s']) < 1e-6
        r.check(good, '%-15s %s: URDF %.1f N m %.1f rad/s, model %.1f N m %.1f rad/s' % (
            spec.name, spec.model, j['effort'], j['velocity'], md['peak_effort_nm'], md['max_velocity_rad_s']))

    print('\n%d fail, %d warn' % (len(r.fails), len(r.warns)))
    for f in r.fails:
        print('  FAIL ' + f)
    for w in r.warns:
        print('  warn ' + w)
    sys.exit(1 if r.fails or (args.strict and r.warns) else 0)


if __name__ == '__main__':
    main()
