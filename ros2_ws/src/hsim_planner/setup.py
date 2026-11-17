"""Package setup: installs the planner node and the `ask` command."""
from setuptools import setup

setup(
    name='hsim_planner',
    version='0.1.0',
    packages=['hsim_planner'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/hsim_planner']),
        ('share/hsim_planner', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'planner_node = hsim_planner.planner_node:main',
        'ask = hsim_planner.ask:main',
    ]},
)
