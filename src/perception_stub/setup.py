from setuptools import find_packages, setup

package_name = 'perception_stub'

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
    description='STUB perception: hardcoded cube on /object_poses at 30 Hz.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = perception_stub.perception_node:main',
        ],
    },
)
