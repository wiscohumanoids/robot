"""Package setup: installs the navigation launch file and configs."""
from glob import glob
from setuptools import setup

setup(
    name='hsim_navigation',
    version='0.1.0',
    packages=['hsim_navigation'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/hsim_navigation']),
        ('share/hsim_navigation', ['package.xml']),
        ('share/hsim_navigation/config', glob('config/*.yaml')),
        ('share/hsim_navigation/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
)
