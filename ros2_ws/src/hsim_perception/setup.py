"""Package setup: installs the perception nodes and the world model."""
from setuptools import setup

setup(
    name='hsim_perception',
    version='0.1.0',
    packages=['hsim_perception'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/hsim_perception']),
        ('share/hsim_perception', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'perception_stub = hsim_perception.perception_stub:main',
        'perception_client = hsim_perception.perception_client:main',
        'world_model = hsim_perception.world_model:main',
    ]},
)
