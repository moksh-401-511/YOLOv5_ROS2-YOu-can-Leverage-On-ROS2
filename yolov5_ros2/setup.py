from setuptools import setup

import os
from glob import glob

package_name = 'yolov5_ros2'
sub_package1 = 'models'
sub_package2 = 'utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,sub_package1,sub_package2],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('./launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='moksh',
    maintainer_email='mokshmmever@gmail.com',
    description='YOLOv5 compatible with ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'obj_detect = '+ package_name +'.obj_detect:main',
        ],
    },
)
