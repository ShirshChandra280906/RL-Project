#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Tomaž Canabrava
# Modified: Remote GPU inference launch file

"""
Launch file for TurtleBot3 DRL with Remote GPU Inference.

This launch file sets up the DRL environment with remote GPU inference support.
Before running this launch file, ensure:
1. SSH tunnel is established: ssh -L 8000:localhost:8000 rl-group13@192.168.3.214
2. FastAPI server is running on the remote GPU server at port 8000

Usage:
    ros2 launch turtlebot3_gazebo turtlebot3_drl_remote.launch.py stage:=stage1

Arguments:
    stage: The simulation stage (stage1 through stage10)
    gpu_server_url: URL of the remote GPU server (default: http://localhost:8000)
    request_timeout: Timeout for inference requests in seconds (default: 5.0)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    gazebo_pkg_dir = get_package_share_directory('turtlebot3_gazebo')
    
    # Launch configuration variables
    stage = LaunchConfiguration('stage')
    gpu_server_url = LaunchConfiguration('gpu_server_url')
    request_timeout = LaunchConfiguration('request_timeout')
    
    # Declare launch arguments
    declare_stage_arg = DeclareLaunchArgument(
        'stage',
        default_value='stage1',
        description='Simulation stage to launch (stage1-stage10)'
    )
    
    declare_gpu_server_url_arg = DeclareLaunchArgument(
        'gpu_server_url',
        default_value='http://localhost:8000',
        description='URL of the remote GPU server for inference'
    )
    
    declare_request_timeout_arg = DeclareLaunchArgument(
        'request_timeout',
        default_value='5.0',
        description='Timeout for inference requests in seconds'
    )
    
    # GPU Client Node - handles communication with remote GPU server
    gpu_client_node = Node(
        package='turtlebot3_drl',
        executable='gpu_client',
        name='gpu_client_node',
        output='screen',
        parameters=[{
            'gpu_server_url': gpu_server_url,
            'request_timeout': request_timeout,
            'retry_count': 3,
            'retry_delay': 0.5,
        }]
    )
    
    # Environment Node - manages simulation environment
    environment_node = Node(
        package='turtlebot3_drl',
        executable='environment',
        name='environment_node',
        output='screen'
    )
    
    # Gazebo Goals Node - manages goal positions
    gazebo_goals_node = Node(
        package='turtlebot3_drl',
        executable='gazebo_goals',
        name='gazebo_goals_node',
        output='screen'
    )
    
    # Remote Agent Node - DRL agent using remote GPU inference
    # Note: This node is started separately for training/testing
    # Using: ros2 run turtlebot3_drl remote_agent
    
    return LaunchDescription([
        declare_stage_arg,
        declare_gpu_server_url_arg,
        declare_request_timeout_arg,
        gpu_client_node,
        environment_node,
        gazebo_goals_node,
    ])
