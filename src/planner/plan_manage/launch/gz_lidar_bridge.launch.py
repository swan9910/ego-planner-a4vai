"""
Gazebo LiDAR Bridge + PointCloud Transformer Launch File

Bridges Gazebo gpu_lidar pointcloud to ROS2 and transforms to world frame.

Pipeline:
  Gazebo /lidar/points -> [ros_gz_bridge] -> ROS2 /lidar/points (body frame)
  ROS2 /lidar/points -> [pointcloud_transformer] -> /lidar/points_world (world frame)

Usage:
  ros2 launch ego_planner gz_lidar_bridge.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():

    # 1) ros_gz_bridge: Gazebo lidar pointcloud -> ROS2
    gz_lidar_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_lidar_bridge',
        arguments=[
            '/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked'
        ],
        output='screen'
    )

    # 2) Pointcloud transformer: body frame -> world frame
    #    Requires TF: world -> lidar frame (published by offboard/foxglove)
    transformer_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
        'pointcloud_transformer_fast.py'
    )

    pointcloud_transformer = ExecuteProcess(
        cmd=['python3', transformer_script],
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(gz_lidar_bridge)
    ld.add_action(pointcloud_transformer)

    return ld
