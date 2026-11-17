"""Package setup: installs the HiPNUC IMU driver, its bench check, config and launch file."""
from glob import glob
from setuptools import setup

setup(
    name='wisco_hipnuc_imu',
    version='0.1.0',
    packages=['wisco_hipnuc_imu'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/wisco_hipnuc_imu']),
        ('share/wisco_hipnuc_imu', ['package.xml']),
        ('share/wisco_hipnuc_imu/config', glob('config/*')),
        ('share/wisco_hipnuc_imu/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'imu_node = wisco_hipnuc_imu.imu_node:main',
        'imu_check = wisco_hipnuc_imu.imu_check:main',
    ]},
)
