import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_name = 'complete_robot'
    
    # 1. Gazebo 실행
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch'), '/gazebo.launch.py']),
        )

    # 2. URDF 파일 로드
    urdf_file_name = 'ms_robot.urdf.xacro'
    urdf_path = os.path.join(get_package_share_directory(pkg_name), 'urdf', urdf_file_name)
    robot_description_content = xacro.process_file(urdf_path).toxml()
    
    # 3. Robot State Publisher 실행
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content}]
    )

    # 4. Gazebo에 로봇 모델 스폰
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'any_axis_arm'],
        output='screen'
    )

    # 5. Joint State Broadcaster 컨트롤러 실행
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    # 6. Joint Trajectory Controller 컨트롤러 실행
    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
    )
    
    return LaunchDescription([
        gazebo,
        robot_state_publisher_node,
        spawn_entity_node,
        joint_state_broadcaster_spawner,
        robot_controller_spawner,
    ])