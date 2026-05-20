from setuptools import setup
from glob import glob

package_name = 'rover_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sanjay',
    maintainer_email='otonomyrobotics@gmail.com',
    description='Bringup for PX4 SITL rover_360cam + stella_vslam.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'fisheye_to_equirect = rover_bringup.fisheye_to_equirect:main',
        ],
    },
)
