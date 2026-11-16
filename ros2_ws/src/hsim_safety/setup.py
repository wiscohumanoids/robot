from setuptools import setup

setup(
    name='hsim_safety',
    version='0.1.0',
    packages=['hsim_safety'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/hsim_safety']),
        ('share/hsim_safety', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'safety_monitor = hsim_safety.safety_monitor:main',
        'estop = hsim_safety.estop_cli:main',
    ]},
)
