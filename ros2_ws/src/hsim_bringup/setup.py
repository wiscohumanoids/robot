"""Package setup: installs the bring-up helpers, launch files and config."""
from glob import glob
from setuptools import setup

setup(
    name='hsim_bringup',
    version='0.1.0',
    packages=['hsim_bringup'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/hsim_bringup']),
        ('share/hsim_bringup', ['package.xml']),
        ('share/hsim_bringup/launch', glob('launch/*.launch.py')),
        ('share/hsim_bringup/config', glob('config/*.yaml')),
        ('share/hsim_bringup/config/robots', glob('config/robots/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'contract_check = hsim_bringup.contract_check:main',
        'drive_check = hsim_bringup.drive_check:main',
    ]},
)
