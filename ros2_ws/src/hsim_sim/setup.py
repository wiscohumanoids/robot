"""Package setup: installs the simulation nodes."""
from glob import glob
from setuptools import setup

setup(
    name='hsim_sim',
    version='0.1.0',
    packages=['hsim_sim'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/hsim_sim']),
        ('share/hsim_sim', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'sim_stub = hsim_sim.sim_stub:main',
        'mujoco_sim = hsim_sim.mujoco_sim:main',
        'stereo_camera = hsim_sim.stereo_camera:main',
        'ros2_control_backend = hsim_sim.ros2_control_backend:main',
        'ethercat_sim_backend = hsim_sim.ethercat_sim_backend:main',
        'backend_supervisor = hsim_sim.backend_supervisor:main',
    ]},
)
