from setuptools import find_packages, setup

package_name = 'cmd_vel_mux'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='WiscoHumanoids',
    maintainer_email='wiscohumanoids@example.com',
    description='50 Hz arbiter: sole publisher of /cmd_vel (teleop > navigation, with timeouts).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mux_node = cmd_vel_mux.mux_node:main',
        ],
    },
)
