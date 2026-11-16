#!/usr/bin/env python3
"""Convert the CAD team's `full_assembly_urdf` export into robots/arm01/urdf/arm01.urdf.

The export (Onshape capture, audited inertials and harness) names its joints J1..J7 plus
the two passive wrist-loop joints, and uses the CAD world frame as base_link. The arm
stack, its configs and calibration files use the names and base frame below, so this
script renames, re-bases and adds the tool frames; geometry, joint senses, zeros, limits
and inertials are the export's, unchanged.

    python3 robots/arm01/cad/import_full_assembly.py ~/Downloads/full_assembly_urdf \\
        --out robots/arm01/urdf/arm01.urdf

* Names: J1 shoulder_pitch, J2 shoulder_roll, J3 shoulder_yaw, J4 elbow, J5 forearm_yaw,
  output_pivot wrist_pitch, J7 wrist_yaw. J6 (the wrist crank the actuator turns) and
  coupler_hinge become mimic joints of wrist_pitch (+1 / -1), as before. Links:
  shoulder_flexion link1, shoulder_abduction link2, upper_arm link3, forearm link4,
  wrist_base link5, wrist_output link6, tool_link link7, input_crank wrist_crank,
  coupler wrist_rod.
* Base frame: the stack's base_link sits at the top centre of the post, CAD (0, 0, 541)
  mm, x forward (CAD -y), y left (CAD +x), z up: base = Rz(+90 deg) (cad - [0, 0, 0.541]).
  That transform goes into the J1 origin and base_link's visual and inertial.
* flange: on the J7 axis at the output face (5.24 mm from the J7 origin, from the earlier
  STEP extract of the same capture), z out of the face toward the tool, x toward the peg.
  tool0: centre of the peg's end face (fitted to the tool_link mesh), z out of the tip,
  x perpendicular to the peg and the J7 axis.
* The wrist is a near-parallelogram: output / crank ratio 0.996..1.004 over +-35 deg, so
  the 1:1 mimic is good to about 0.15 deg at the ends of travel (export README).
"""

import argparse
import os
import shutil
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

LINKS = {
    'base_link': 'base_link', 'shoulder_flexion': 'link1', 'shoulder_abduction': 'link2',
    'upper_arm': 'link3', 'forearm': 'link4', 'wrist_base': 'link5', 'input_crank': 'wrist_crank',
    'coupler': 'wrist_rod', 'wrist_output': 'link6', 'tool_link': 'link7',
}
JOINTS = {
    'J1': 'shoulder_pitch', 'J2': 'shoulder_roll', 'J3': 'shoulder_yaw', 'J4': 'elbow',
    'J5': 'forearm_yaw', 'J6': 'wrist_crank_joint', 'coupler_hinge': 'wrist_rod_joint',
    'output_pivot': 'wrist_pitch', 'J7': 'wrist_yaw',
}
MIMIC = {'wrist_crank_joint': 1.0, 'wrist_rod_joint': -1.0}     # of wrist_pitch
ROM_NAME = dict(JOINTS, J6='wrist_pitch')    # J6's range is the driven wrist pitch's
LIMITS_FROM = {'wrist_pitch': 'J6', 'wrist_crank_joint': 'J6', 'wrist_rod_joint': 'J6'}
POST_TOP_CAD = np.array([0.0, 0.0, 0.541])
RZ90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
FLANGE_ALONG_J7_M = 0.005243        # J7 origin to the output face, on the J7 axis
MESH_PREFIX = 'package://arm01_description/meshes/'


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_rpy(R):
    p = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(p)) < 1e-9:
        raise ValueError('gimbal lock')
    r, y = np.arctan2(R[2, 1], R[2, 2]), np.arctan2(R[1, 0], R[0, 0])
    assert np.allclose(rpy_matrix(r, p, y), R, atol=1e-9)
    return r, p, y


def origin_matrix(el):
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = rpy_matrix(*[float(v) for v in el.get('rpy', '0 0 0').split()])
        T[:3, 3] = [float(v) for v in el.get('xyz', '0 0 0').split()]
    return T


def set_origin(el, T):
    el.set('xyz', ' '.join('%.9g' % v for v in T[:3, 3]))
    el.set('rpy', ' '.join('%.9g' % v for v in matrix_rpy(T[:3, :3])))


def frame(origin, z, x_hint):
    z = z / np.linalg.norm(z)
    x = x_hint - (x_hint @ z) * z
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = origin
    return T


def stl_samples(path, count=400000, seed=0):
    """Area-weighted surface samples (long cylinder triangles have no mid-length vertices)."""
    b = open(path, 'rb').read()
    n = struct.unpack('<I', b[80:84])[0]
    tri = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]))
    V = tri['v'].reshape(-1, 3, 3).astype(float)
    area = 0.5 * np.linalg.norm(np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    k = rng.choice(len(V), count, p=area / area.sum())
    a, c = rng.random((2, count))
    m = a + c > 1
    a[m], c[m] = 1 - a[m], 1 - c[m]
    return V[k, 0] + a[:, None] * (V[k, 1] - V[k, 0]) + c[:, None] * (V[k, 2] - V[k, 0])


def peg_tip(tool_mesh):
    """The peg is a 20 mm shaft sticking out of tool_link along about +y: fit its axis on
    the last 60 mm and take the centroid of its end face as the tip."""
    v = stl_samples(tool_mesh)
    s = v[:, 1]
    shaft = v[(s > s.max() - 0.060) & (s < s.max() - 0.002)]
    c = shaft.mean(0)
    _, _, vt = np.linalg.svd(shaft - c, full_matrices=False)
    axis = vt[0] * np.sign(vt[0][1])
    radius = np.median(np.linalg.norm(np.cross(shaft - c, axis), axis=1))
    if not 0.008 < radius < 0.012:
        raise ValueError('peg shaft radius %.1f mm, expected 10 mm: is this the peg?' % (1000 * radius))
    along = (v - c) @ axis
    tip = v[along > along.max() - 0.0003].mean(0)
    return tip, axis, radius


def write_capture(src, path, T_base_cad):
    """The export's CAD capture, in this repo's names and base frame: the reference
    test/arm_stack/test_fk.py checks the URDF against."""
    import csv
    import json
    md = json.load(open(os.path.join(src, 'model_data.json')))
    with open(os.path.join(src, 'operating_limits_and_rom.csv')) as f:
        rom = {r['joint']: r for r in csv.DictReader(f)}

    def num(v):
        return None if v in (None, '') else float(v)
    out = {
        'source': 'full_assembly_urdf export, Onshape microversion %s' % md.get('baseline_microversion'),
        'units': 'm, rad (deg where named)',
        'capture_q_rad': {JOINTS[j['name']]: float(np.radians(j['capture_deg'])) for j in md['joints']},
        'links_in_base': {LINKS[l['name']]: (T_base_cad @ np.array(l['capture_world'])).tolist()
                          for l in md['links']},
        'four_bar_in_link5': {k: v for k, v in md['four_bar'].items()
                              if k in ('input_origin', 'output_origin', 'input_pin', 'output_pin', 'axis')},
        'operating_limits_deg': {ROM_NAME[j]: [float(r['operating_lower_deg']), float(r['operating_upper_deg'])]
                                 for j, r in rom.items()},
        'contact_onset_deg': {ROM_NAME[j]: [num(r['conditional_contact_lower_deg']),
                                          num(r['conditional_contact_upper_deg'])] for j, r in rom.items()},
        'moving_mass_kg': md['moving_mass_kg'],
    }
    with open(path, 'w', newline='\n') as f:
        json.dump(out, f, indent=1)
        f.write('\n')


def convert(src, out_urdf, mesh_dir):
    tree = ET.parse(os.path.join(src, 'full_assembly.urdf'))
    robot = tree.getroot()
    robot.set('name', 'arm01')
    T_base_cad = np.eye(4)
    T_base_cad[:3, :3] = RZ90
    T_base_cad[:3, 3] = -RZ90 @ POST_TOP_CAD

    joints = {j.get('name'): j for j in robot.findall('joint')}
    links = {l.get('name'): l for l in robot.findall('link')}
    if set(joints) != set(JOINTS) or set(links) != set(LINKS):
        raise ValueError('unexpected export: joints %s, links %s' % (sorted(joints), sorted(links)))
    j6_limit = dict(joints['J6'].find('limit').attrib)

    for name, link in links.items():
        new = LINKS[name]
        link.set('name', new)
        for tag in ('visual', 'collision'):
            for el in link.findall(tag):
                mesh = el.find('geometry/mesh')
                mesh.set('filename', MESH_PREFIX + new + '.stl')
                mesh.attrib.pop('scale', None)
                mat = el.find('material')
                if mat is not None:
                    mat.set('name', new + '_color')
                if name == 'base_link':
                    o = el.find('origin')
                    set_origin(o, T_base_cad @ origin_matrix(o))
        if name == 'base_link':
            inertial = link.find('inertial')
            o = inertial.find('origin')
            T = T_base_cad @ origin_matrix(o)
            I = inertial.find('inertia')
            J = np.array([[float(I.get('ixx')), float(I.get('ixy')), float(I.get('ixz'))],
                          [float(I.get('ixy')), float(I.get('iyy')), float(I.get('iyz'))],
                          [float(I.get('ixz')), float(I.get('iyz')), float(I.get('izz'))]])
            J = T[:3, :3] @ J @ T[:3, :3].T
            T[:3, :3] = np.eye(3)
            set_origin(o, T)
            for k, (a, b) in {'ixx': (0, 0), 'ixy': (0, 1), 'ixz': (0, 2), 'iyy': (1, 1),
                              'iyz': (1, 2), 'izz': (2, 2)}.items():
                I.set(k, '%.9g' % J[a, b])
        shutil.copyfile(os.path.join(src, 'meshes', name + '.stl'), os.path.join(mesh_dir, new + '.stl'))

    for name, joint in joints.items():
        new = JOINTS[name]
        joint.set('name', new)
        for tag in ('parent', 'child'):
            el = joint.find(tag)
            el.set('link', LINKS[el.get('link')])
        if name == 'J1':
            o = joint.find('origin')
            set_origin(o, T_base_cad @ origin_matrix(o))
        if new in LIMITS_FROM:
            joint.find('limit').attrib.update(j6_limit)
        if new in MIMIC:
            ET.SubElement(joint, 'mimic', joint='wrist_pitch', multiplier='%g' % MIMIC[new], offset='0')

    tip, peg_axis, radius = peg_tip(os.path.join(src, 'meshes', 'tool_link.stl'))
    z7 = np.array([0.0, 0.0, 1.0])            # J7 axis in tool_link, + toward the tool side
    frames = {
        'flange': frame(FLANGE_ALONG_J7_M * z7, z7, peg_axis),
        'tool0': frame(tip, peg_axis, np.array([1.0, 0.0, 0.0])),   # x across the peg and J7
    }
    for f, T in frames.items():
        j = ET.SubElement(robot, 'joint', name=f + '_joint', type='fixed')
        ET.SubElement(j, 'parent', link='link7')
        ET.SubElement(j, 'child', link=f)
        o = ET.SubElement(j, 'origin')
        set_origin(o, T)
        ET.SubElement(robot, 'link', name=f)

    write_capture(src, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arm01_capture.json'),
                  T_base_cad)

    ET.indent(tree, space='  ')
    body = ET.tostring(robot, encoding='unicode')
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arm01_ledger.txt')) as f:
        ledger = f.read()
    with open(out_urdf, 'w', newline='\n') as f:
        f.write('<?xml version="1.0"?>\n<!--\n' + ledger.rstrip('\n') + '\n-->\n' + body + '\n')
    return dict(tip=tip, peg_axis=peg_axis, peg_radius=radius)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('export', help='the unzipped full_assembly_urdf folder')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'urdf', 'arm01.urdf'))
    args = ap.parse_args(argv)
    mesh_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)
    info = convert(args.export, args.out, mesh_dir)
    print('wrote %s (meshes in %s)' % (args.out, mesh_dir))
    print('tool0 tip in link7 %s mm, peg axis %s, shaft radius %.2f mm'
          % (np.round(1000 * info['tip'], 2), np.round(info['peg_axis'], 4), 1000 * info['peg_radius']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
