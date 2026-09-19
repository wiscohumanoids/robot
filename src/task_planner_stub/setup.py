from setuptools import find_packages, setup

package_name = 'task_planner_stub'

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
    description='STUB task planner: /user_intent -> canned /skill_sequence.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner_node = task_planner_stub.planner_node:main',
        ],
    },
)
