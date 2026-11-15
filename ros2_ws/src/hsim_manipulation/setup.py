from glob import glob
from setuptools import setup

setup(
    name='hsim_manipulation',
    version='0.1.0',
    packages=['hsim_manipulation'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/hsim_manipulation']),
        ('share/hsim_manipulation', ['package.xml']),
        ('share/hsim_manipulation/launch', glob('launch/*.py')),
        ('share/hsim_manipulation/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'manipulation_node = hsim_manipulation.manipulation_node:main',
    ]},
)
