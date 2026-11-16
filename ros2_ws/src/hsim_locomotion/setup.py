from glob import glob
from setuptools import setup

setup(
    name='hsim_locomotion',
    version='0.1.0',
    packages=['hsim_locomotion'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/hsim_locomotion']),
        ('share/hsim_locomotion', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'locomotion_stub = hsim_locomotion.locomotion_stub:main',
        'policy_node = hsim_locomotion.policy_node:main',
    ]},
)
