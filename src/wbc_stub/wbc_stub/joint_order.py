"""See locomotion_runner/joint_order.py -- identical loader, duplicated per
the note in that file rather than factored into a shared library package."""
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
    assert len(names) == NUM_JOINTS
    return names
