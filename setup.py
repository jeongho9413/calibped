from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'calibped'

setup(
    name=package_name,
    version='1.1.0',
    packages=find_packages(exclude=['test']),
    package_data={
        'calibped': ['PIDNet/**/*'],
    },
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'torchvision',
        # 'opencv-python',
        'Pillow',
        'matplotlib',
        'numpy',
    ],
    zip_safe=True,
    maintainer='Jeongho Ahn',
    maintainer_email='totoroahn@gmail.com',
    description='TODO: LiDAR-camera calibration for segmenting pedestrian point clouds on ROS2 Humble.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'calibped_node = calibped.calibped_node:main',
        ],
    },
)
