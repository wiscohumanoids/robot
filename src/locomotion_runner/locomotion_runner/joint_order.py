"""Loads the single canonical joint order from humanoid_interfaces's share
directory, instead of hard-coding a 23-entry list in this package too. See
humanoid_interfaces/config/canonical_joint_order.yaml -- the same ~10-line
loader is duplicated (not imported as a shared library) in
manipulation_runner, wbc_stub, lowlevel_control's Python tooling, and safety,
since a full shared-library package for 10 lines of yaml-loading was judged
not worth the extra build target. If this grows, factor it out.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory

NUM_JOINTS = 23


def load_canonical_joint_order():
    share_dir = get_package_share_directory('humanoid_interfaces')
    path = os.path.join(share_dir, 'config', 'canonical_joint_order.yaml')
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    names = data['joint_names']
    assert len(names) == NUM_JOINTS, (
        f'canonical_joint_order.yaml has {len(names)} joints, expected {NUM_JOINTS} '
        '-- update NUM_JOINTS everywhere it is duplicated if this is intentional.')
    return names
