#!/usr/bin/env python3
"""Offline URDF sanity checks (no ROS graph needed).

Reports:
  * left/right joint pairs whose axes, limits, effort or velocity differ
    (usually a CAD export pose or a copy/paste slip), and
  * joints in the joint table whose URDF effort or velocity limit disagrees
    with the actuator model they are wired to.

  ros2 run wisco_ethercat_ros2_hw check_urdf.py
  ros2 run wisco_ethercat_ros2_hw check_urdf.py --strict   # exit 1 on findings
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

from wisco_ethercat_ros2_hw.hardware_config import load_yaml

PACKAGE = 'wisco_ethercat_ros2_hw'


def _share_dir():
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory(PACKAGE)
    except Exception:  # noqa: BLE001 - running from a source checkout
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_joints(urdf_xml):
    out = {}
    for joint in ET.fromstring(urdf_xml).findall('joint'):
        if joint.get('type') not in ('revolute', 'continuous', 'prismatic'):
            continue
        axis = joint.find('axis')
        limit = joint.find('limit')
        out[joint.get('name')] = {
            'axis': [float(v) for v in (axis.get('xyz') if axis is not None else '1 0 0').split()],
            'lower': float(limit.get('lower', 'nan')) if limit is not None else float('nan'),
            'upper': float(limit.get('upper', 'nan')) if limit is not None else float('nan'),
            'effort': float(limit.get('effort', 'nan')) if limit is not None else float('nan'),
            'velocity': float(limit.get('velocity', 'nan')) if limit is not None else float('nan'),
        }
    return out


def mirror_findings(joints, axis_tol=0.05, rel_tol=0.05):
    findings = []
    for name in sorted(joints):
        if not name.startswith('left_'):
            continue
        other = 'right_' + name[len('left_'):]
        if other not in joints:
            findings.append('%s has no %s' % (name, other))
            continue
        a, b = joints[name], joints[other]
        # A mirrored joint may flip the sign of an axis component, never its size.
        if any(abs(abs(x) - abs(y)) > axis_tol for x, y in zip(a['axis'], b['axis'])):
            findings.append('axis: %s %s vs %s %s' % (
                name, _fmt(a['axis']), other, _fmt(b['axis'])))
        range_a = sorted([abs(a['lower']), abs(a['upper'])])
        range_b = sorted([abs(b['lower']), abs(b['upper'])])
        if any(abs(x - y) > 1e-3 for x, y in zip(range_a, range_b)):
            findings.append('limits: %s [%.3f, %.3f] vs %s [%.3f, %.3f]' % (
                name, a['lower'], a['upper'], other, b['lower'], b['upper']))
        for key, unit in (('effort', 'Nm'), ('velocity', 'rad/s')):
            if abs(a[key] - b[key]) > rel_tol * max(abs(a[key]), abs(b[key]), 1e-9):
                findings.append('%s: %s %.2f %s vs %s %.2f %s' % (
                    key, name, a[key], unit, other, b[key], unit))
    return findings


def model_findings(joints, table, models, rel_tol=0.05):
    findings = []
    for row in table.get('joints') or []:
        name, model_name = row['name'], row['model']
        if name not in joints or model_name not in models:
            continue
        j, m = joints[name], models[model_name]
        if abs(j['effort'] - m['peak_effort_nm']) > rel_tol * m['peak_effort_nm']:
            findings.append('%s: URDF effort %.2f Nm, model %s peak %.2f Nm' % (
                name, j['effort'], model_name, m['peak_effort_nm']))
        if abs(j['velocity'] - m['max_velocity_rad_s']) > rel_tol * m['max_velocity_rad_s']:
            findings.append('%s: URDF velocity %.2f rad/s, model %s max %.2f rad/s' % (
                name, j['velocity'], model_name, m['max_velocity_rad_s']))
    return findings


def _fmt(v):
    return '(%s)' % ', '.join('%.3f' % x for x in v)


def main():
    share = _share_dir()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--urdf', default=os.path.join(share, 'urdf', 'test_bench.urdf.xacro'))
    parser.add_argument('--joints', default=os.path.join(share, 'config', 'joints.yaml'))
    parser.add_argument('--models', default=os.path.join(share, 'config', 'actuator_models.yaml'))
    parser.add_argument('--strict', action='store_true', help='exit 1 if anything is reported')
    args = parser.parse_args()

    if args.urdf.endswith('.xacro'):
        import xacro
        urdf_xml = xacro.process_file(args.urdf).toxml()
    else:
        with open(args.urdf) as f:
            urdf_xml = f.read()
    joints = read_joints(urdf_xml)

    sections = [
        ('left/right mismatches', mirror_findings(joints)),
        ('URDF limits vs actuator model', model_findings(
            joints, load_yaml(args.joints), load_yaml(args.models).get('models', {}))),
    ]
    total = 0
    for title, findings in sections:
        print('%s: %d' % (title, len(findings)))
        for f in findings:
            print('  - ' + f)
        total += len(findings)
    sys.exit(1 if (args.strict and total) else 0)


if __name__ == '__main__':
    main()
