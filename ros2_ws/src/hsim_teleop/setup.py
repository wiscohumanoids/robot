"""Package setup: installs the viewer teleop and bridge nodes."""
from setuptools import setup

setup(
    name='hsim_teleop',
    version='0.1.0',
    packages=['hsim_teleop'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/hsim_teleop']),
        ('share/hsim_teleop', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'viewer_teleop = hsim_teleop.viewer_teleop:main',
        'viewer_bridge = hsim_teleop.viewer_bridge:main',
    ]},
)
