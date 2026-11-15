from glob import glob
from setuptools import setup

setup(
    name='hsim_state_estimation',
    version='0.1.0',
    packages=['hsim_state_estimation'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/hsim_state_estimation']),
        ('share/hsim_state_estimation', ['package.xml']),
        ('share/hsim_state_estimation/config', ['config/ekf.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'state_estimation_stub = hsim_state_estimation.state_estimation_stub:main',
        'leg_odometry = hsim_state_estimation.leg_odometry:main',
    ]},
)
