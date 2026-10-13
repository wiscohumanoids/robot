from setuptools import find_packages, setup

package_name = 'integration_tests'

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
    description='Integration and stress tests for the task layer (test harness, not part of the stack).',
    license='Apache-2.0',
    tests_require=['pytest'],
)
