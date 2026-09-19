from setuptools import find_packages, setup

package_name = 'behavior_tree_stub'

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
    description='STUB behavior tree: sequential SkillSequence executor over the nav and manipulation actions.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bt_node = behavior_tree_stub.bt_node:main',
        ],
    },
)
