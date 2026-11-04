import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default = os.path.join(get_package_share_directory("pallet_rl_server"),
                           "config", "server.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default,
                              description="settings file"),
        Node(package="pallet_rl_server", executable="server",
             name="pallet_rl_server", output="screen",
             parameters=[{"config": LaunchConfiguration("config")}]),
    ])
